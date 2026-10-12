import os
import time
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from backend import (
    AccessError, AuthContext, AuthError, AuthService, ConflictError, DENIED, DbDirectory, Forbidden, GuardianService,
    Repo, StudentRecord, build_snapshot, coach_student_detail, coach_view, parent_view, student_view
)
from engine import (
    ConceptGraph, DEFAULT_EXAM, EngagementState, QuestionBank, StudentModel, engagement_from_dict, model_kind,
    student_from_dict
)


ID = Field(pattern=r"^[a-z0-9_-]{2,64}$")


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class StudentIn(BaseModel):
    id: str = ID
    name: str = Field(min_length=1, max_length=200)
    batch: str = Field(default="", max_length=64)
    is_minor: bool = True
    adult_consents_parent: bool = False
    exam_date: date | None = None
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class UserIn(BaseModel):
    id: str = ID
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)
    role: Literal["parent", "coach"]
    batches: list[str] = Field(default_factory=list, max_length=20)


class CodeIn(BaseModel):
    student_id: str = ID


class RedeemIn(BaseModel):
    code: str = Field(max_length=32)


class ExamIn(BaseModel):
    exam_id: str = Field(pattern=r"^[a-z0-9_]{2,40}$")


class PasswordIn(BaseModel):
    password: str = Field(max_length=128)


class DbSnapshots:
    def __init__(self, repo: Repo, graph, bank, org: str, now: float, registry=None):
        self.repo, self.graph, self.bank, self.org, self.now, self.registry = repo, graph, bank, org, now, registry

    def __getitem__(self, sid: str):
        prof = self.repo.get_doc(self.org, sid, "profile")
        exam_id = prof[0].get("exam_id", DEFAULT_EXAM) if prof else DEFAULT_EXAM
        graph, bank = self.graph, self.bank
        if exam_id != DEFAULT_EXAM and self.registry is not None and exam_id in self.registry.modules:
            graph, bank = self.registry.get(exam_id).graph, self.registry.get(exam_id).bank
        else:
            exam_id = DEFAULT_EXAM
        sm_doc, eng_doc, hist_doc = (self.repo.get_doc(self.org, sid, k) for k in (model_kind(exam_id), "engagement", "exam_hist"))
        sm = student_from_dict(graph, bank, sm_doc[0]) if sm_doc else StudentModel(graph, bank)
        eng = engagement_from_dict(eng_doc[0]) if eng_doc else EngagementState()
        return build_snapshot(sid, sm, eng, hist_doc[0]["items"] if hist_doc else [], graph, self.now)


def create_app(repo: Repo, auth: AuthService, directory: DbDirectory, guardians: GuardianService, graph, bank, clock=time.time,
               expose_docs: bool = False, registry=None) -> FastAPI:
    app = FastAPI(title="AI Study Companion API", docs_url="/docs" if expose_docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if expose_docs else None)
    bearer = HTTPBearer(auto_error=False)

    def ctx(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> AuthContext:
        if creds is None:
            raise HTTPException(401, "Not authenticated.", headers={"WWW-Authenticate": "Bearer"})
        try:
            return auth.authenticate(creds.credentials)
        except AuthError:
            raise HTTPException(401, "Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"})

    def need(c: AuthContext, role: str) -> None:
        if c.role != role:
            raise HTTPException(403, "Your account type can't do this.")

    def today(): return date.fromtimestamp(clock())

    @app.exception_handler(AccessError)
    async def _access(_: Request, e: AccessError):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(e)}, status_code=404)

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, e: ConflictError):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Conflict, please retry."}, status_code=409)

    @app.get("/healthz")
    def health(): return {"ok": True}

    @app.post("/auth/login")
    def login(b: LoginIn):
        try:
            return {"access_token": auth.login(b.email, b.password), "token_type": "bearer"}
        except AuthError as e:
            raise HTTPException(401, str(e))

    @app.post("/auth/logout")
    def logout(creds: HTTPAuthorizationCredentials = Depends(bearer), c: AuthContext = Depends(ctx)):
        auth.logout(creds.credentials); return {"ok": True}

    @app.get("/me")
    def me(c: AuthContext = Depends(ctx)): return {"user_id": c.user_id, "org": c.org, "role": c.role}

    @app.post("/admin/students", status_code=201)
    def add_student(b: StudentIn, c: AuthContext = Depends(ctx)):
        need(c, "org_admin")
        try:
            auth.create_user(c.org, b.id, b.email, b.password, "student")
            directory.add_student(StudentRecord(b.id, c.org, b.batch, b.name, b.is_minor, b.adult_consents_parent, b.exam_date))
        except ValueError as e:
            raise HTTPException(422, str(e))
        except IntegrityError:
            raise HTTPException(409, "That id or email already exists.")
        return {"id": b.id}

    @app.post("/admin/users", status_code=201)
    def add_user(b: UserIn, c: AuthContext = Depends(ctx)):
        need(c, "org_admin")
        try:
            auth.create_user(c.org, b.id, b.email, b.password, b.role, b.batches)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except IntegrityError:
            raise HTTPException(409, "That id or email already exists.")
        return {"id": b.id}

    @app.post("/admin/guardian-codes", status_code=201)
    def issue_code(b: CodeIn, c: AuthContext = Depends(ctx)):
        need(c, "org_admin")
        try:
            code, exp = guardians.issue(c, b.student_id)
        except KeyError:
            raise HTTPException(404, DENIED)
        except Forbidden as e:
            raise HTTPException(403, str(e))
        directory._log(clock(), c.user_id, b.student_id, "issue_guardian_code", True, "ok")
        return {"code": code, "expires_at": exp, "note": "Shown once. Give it to the parent in person."}

    @app.get("/admin/audit")
    def audit(c: AuthContext = Depends(ctx)):
        need(c, "org_admin")
        return [vars(e) for e in repo.audit(org=c.org)[-200:]]

    @app.delete("/admin/students/{sid}", status_code=204)
    def admin_erase(sid: str, c: AuthContext = Depends(ctx)):
        need(c, "org_admin")
        try:
            repo.erase_student(c.org, sid)
        except KeyError:
            raise HTTPException(404, DENIED)

    @app.post("/parent/redeem")
    def redeem(b: RedeemIn, c: AuthContext = Depends(ctx)):
        try:
            return {"linked_student": guardians.redeem(c, b.code, directory)}
        except Forbidden as e:
            raise HTTPException(403, str(e))
        except AuthError as e:
            raise HTTPException(400, str(e))

    @app.get("/parent/students/{sid}/report")
    def parent_report(sid: str, c: AuthContext = Depends(ctx)):
        need(c, "parent")
        return parent_view(directory, c.user_id, sid, DbSnapshots(repo, graph, bank, c.org, clock(), registry), today(), clock())

    @app.get("/coach/overview")
    def coach_overview(batch: str | None = None, c: AuthContext = Depends(ctx)):
        need(c, "coach")
        return coach_view(directory, c.user_id, DbSnapshots(repo, graph, bank, c.org, clock(), registry), today(), clock(), batch)

    @app.get("/coach/students/{sid}")
    def coach_student(sid: str, c: AuthContext = Depends(ctx)):
        need(c, "coach")
        return coach_student_detail(directory, c.user_id, sid, DbSnapshots(repo, graph, bank, c.org, clock(), registry), today(), clock())

    @app.get("/student/me/dashboard")
    def my_dashboard(c: AuthContext = Depends(ctx)):
        need(c, "student")
        return student_view(directory, c.user_id, c.user_id, DbSnapshots(repo, graph, bank, c.org, clock(), registry), today(), clock())

    @app.get("/exams")
    def list_exams(c: AuthContext = Depends(ctx)):
        return [{"id": m.spec.id, "name": m.spec.name, "verified_on": m.spec.source.verified_on, "confidence": m.spec.source.confidence}
                for m in (registry.modules.values() if registry else [])]

    @app.put("/student/me/exam")
    def choose_exam(b: ExamIn, c: AuthContext = Depends(ctx)):
        need(c, "student")
        if registry is None or b.exam_id not in registry.modules:
            raise HTTPException(404, "Unknown exam.")
        cur = repo.get_doc(c.org, c.user_id, "profile")
        repo.put_doc(c.org, c.user_id, "profile", {"exam_id": b.exam_id}, cur[1] if cur else None)
        return {"exam_id": b.exam_id}

    @app.get("/student/me/export")
    def my_export(c: AuthContext = Depends(ctx)):
        need(c, "student")
        return repo.export_student(c.org, c.user_id)

    @app.delete("/student/me", status_code=204)
    def erase_me(b: PasswordIn, c: AuthContext = Depends(ctx)):
        need(c, "student")
        if not auth.verify_user_password(c.user_id, b.password):
            raise HTTPException(403, "Password confirmation failed.")
        repo.erase_student(c.org, c.user_id)

    return app


def build():
    secret = os.environ.get("APP_SECRET", "").encode()
    if len(secret) < 32:
        raise RuntimeError("APP_SECRET must be set to at least 32 random bytes (e.g. `python -c 'import secrets;print(secrets.token_hex(32))'`).")
    repo = Repo(os.environ.get("APP_DB", "sqlite+pysqlite:///study_companion.db"))
    return create_app(repo, AuthService(repo, secret), DbDirectory(repo=repo), GuardianService(repo, secret),
                      ConceptGraph.from_data("algebra"),
                      QuestionBank.from_data("algebra"), expose_docs=os.environ.get("EXPOSE_DOCS") == "1")


app = build() if os.environ.get("APP_SECRET") else None

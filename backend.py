import base64
import hashlib
import hmac
import json
import os
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean, quantiles

import jwt
from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine, delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from engine import EngagementState, behaviour_signals, current_stress, local, rank


STRONG, DEVELOPING = 0.8, 0.4
DENIED = "Not found or not permitted."


class AccessError(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: str
    role: str
    org: str
    batches: frozenset = frozenset()


@dataclass
class StudentRecord:
    id: str
    org: str
    batch: str
    name: str
    is_minor: bool = True
    adult_consents_parent: bool = False
    exam_date: date | None = None


@dataclass(frozen=True)
class Policy:
    parent_sees_wellbeing_note: bool = False
    coach_sees_safeguarding_flags: bool = True


@dataclass
class Snapshot:
    student_id: str
    mastery: dict[str, float]
    has_evidence: dict[str, bool]
    names: dict[str, str]
    weights: dict[str, float]
    minutes_by_day: dict[date, float]
    daily_goal_min: int
    mocks: list[tuple[date, float, float]]
    error_counts: dict[str, int]
    last_active: date | None
    signals: list[str]
    focus: list[str] = field(default_factory=list)
    safeguarding_flag: bool = False
    wellbeing_note: bool = False
    today: date | None = None


@dataclass
class AuditEntry:
    ts: float
    viewer: str
    role: str
    org: str
    student: str | None
    action: str
    allowed: bool
    reason: str


def audit_ref(raw: str | None, exists: bool) -> str | None:
    if raw is None or exists:
        return raw
    import hashlib
    return "unknown:" + hashlib.sha256(str(raw).encode()).hexdigest()[:8]


class Directory:
    def __init__(self, policies: dict[str, Policy] | None = None):
        self.users: dict[str, User] = {}
        self.students: dict[str, StudentRecord] = {}
        self.links: set[tuple[str, str]] = set()
        self.policies = policies or {}
        self.audit: list[AuditEntry] = []

    def policy(self, org: str) -> Policy:
        return self.policies.get(org, Policy())

    def add_user(self, u: User): self.users[u.id] = u
    def add_student(self, s: StudentRecord): self.students[s.id] = s

    def link_parent(self, parent_id: str, student_id: str) -> None:
        p, s = self.users.get(parent_id), self.students.get(student_id)
        if not p or not s or p.role != "parent" or p.org != s.org:
            raise AccessError(DENIED)
        if not s.is_minor and not s.adult_consents_parent:
            raise AccessError("This student is 18+, so they need to agree to share with a parent first.")
        self.links.add((parent_id, student_id))

    def students_in_org(self, org: str) -> list[StudentRecord]:
        return [s for s in self.students.values() if s.org == org]

    def _log(self, now, viewer, student, action, allowed, reason):
        u = self.users.get(viewer)
        self.audit.append(AuditEntry(now, audit_ref(viewer, u is not None), u.role if u else "unknown", u.org if u else "unknown",
                                     audit_ref(student, student is None or student in self.students), action, allowed, reason))

    def authorize(self, viewer_id: str, student_id: str, action: str, now: float) -> StudentRecord:
        u, s = self.users.get(viewer_id), self.students.get(student_id)
        reason = None
        if u is None or s is None:
            reason = "unknown_user_or_student"
        elif u.org != s.org:
            reason = "cross_org"
        elif u.role == "student":
            reason = None if u.id == s.id else "student_not_self"
        elif u.role == "parent":
            if (u.id, s.id) not in self.links: reason = "not_linked"
            elif not s.is_minor and not s.adult_consents_parent: reason = "adult_no_consent"
        elif u.role == "coach":
            if u.batches and s.batch not in u.batches: reason = "outside_coach_batches"
        else:
            reason = "bad_role"
        self._log(now, viewer_id, student_id, action, reason is None, reason or "ok")
        if reason:
            raise AccessError(DENIED)
        return s


def _today(s: Snapshot, fallback: date) -> date:
    return s.today or fallback


def readiness(s: Snapshot) -> int:
    tw = sum(s.weights.values()) or 1.0
    return int(5 * round(100 * sum(s.weights[c] * v for c, v in s.mastery.items()) / tw / 5))


def band(s: Snapshot, c: str) -> str:
    if not s.has_evidence.get(c):
        return "not started"
    v = s.mastery[c]
    return "strong" if v >= STRONG else "developing" if v >= DEVELOPING else "needs attention"


def heatmap(s: Snapshot) -> list[dict]:
    return [{"concept": c, "name": s.names[c], "band": band(s, c)} for c in s.mastery]


def minutes_last_7(s: Snapshot, today: date) -> list[tuple[str, float]]:
    return [((today - timedelta(days=i)).isoformat(), s.minutes_by_day.get(today - timedelta(days=i), 0.0)) for i in range(6, -1, -1)]


def mock_trend(s: Snapshot) -> dict:
    pts = [{"date": d.isoformat(), "score_pct": round(f * 100)} for d, f, _ in sorted(s.mocks)]
    if len(s.mocks) < 3:
        return {"points": pts, "direction": "not enough data yet"}
    vals = [f for _, f, _ in sorted(s.mocks)]
    delta = mean(vals[-2:]) - mean(vals[:-2])
    return {"points": pts, "direction": "improving" if delta > 0.05 else "dipping" if delta < -0.05 else "steady"}


def weak_areas(s: Snapshot, n: int = 3) -> list[str]:
    cand = [c for c in s.mastery if band(s, c) in ("needs attention", "developing")]
    return [s.names[c] for c in sorted(cand, key=lambda c: s.mastery[c])[:n]]


def error_patterns(s: Snapshot) -> list[str]:
    tot = sum(s.error_counts.values())
    if not tot:
        return []
    label = {"careless": "small slips (signs, arithmetic, rushing)", "conceptual": "ideas that need re-learning",
             "knowledge_gap": "topics not yet learned", "time_pressure": "running short of time", "guess": "quick guesses"}
    return [f"{round(100 * n / tot)}% of lost marks were {label[k]}." for k, n in sorted(s.error_counts.items(), key=lambda kv: -kv[1]) if n and k in label][:3]


def parent_tips(s: Snapshot, today: date) -> list[str]:
    tips, tot = [], sum(s.error_counts.values())
    if tot and s.error_counts.get("careless", 0) / tot >= 0.4:
        tips.append("Many lost marks are small slips. A 'check your answer' habit will help more than extra hours.")
    if tot and (s.error_counts.get("conceptual", 0) + s.error_counts.get("knowledge_gap", 0)) / tot >= 0.5:
        tips.append("Some ideas need re-learning. Encourage asking for explanations rather than doing more questions.")
    if "late_night_study" in s.signals:
        tips.append("Late-night studying is showing up. Protecting sleep helps memory more than the extra hour.")
    if "overwork" in s.signals:
        tips.append("Some study days are very long. Short breaks and a day off make the rest of the week better.")
    if s.last_active and (today - s.last_active).days >= 4:
        tips.append("Activity has dropped recently. A calm, curious chat usually works better than checking scores.")
    tips.append("Praise effort and consistency rather than scores. It keeps motivation steady.")
    return tips


@dataclass
class Alert:
    severity: int
    code: str
    text: str


def risk_alerts(s: Snapshot, today: date, exam_date: date | None, policy: Policy) -> list[Alert]:
    a: list[Alert] = []
    if s.safeguarding_flag and policy.coach_sees_safeguarding_flags:
        a.append(Alert(3, "follow_up", "Follow-up recommended: please check in with this student personally. Details are not shared."))
    if s.last_active:
        gap = (today - s.last_active).days
        if gap >= 10: a.append(Alert(3, "inactive", f"No activity for {gap} days."))
        elif gap >= 4: a.append(Alert(2, "inactive", f"No activity for {gap} days."))
    if "score_drop" in s.signals: a.append(Alert(2, "score_drop", "Recent mock scores are lower than earlier ones."))
    if "accuracy_drop" in s.signals: a.append(Alert(2, "accuracy_drop", "Practice accuracy has fallen recently."))
    week = sum(m for _, m in minutes_last_7(s, today))
    if s.last_active and week < 0.4 * 7 * s.daily_goal_min and (today - s.last_active).days < 4:
        a.append(Alert(2, "behind_goal", f"Study time this week ({week:.0f} min) is under 40% of the goal."))
    if exam_date and 0 < (exam_date - today).days <= 21 and readiness(s) < 50:
        a.append(Alert(3, "exam_soon_low_readiness", f"Exam in {(exam_date - today).days} days; estimated readiness {readiness(s)}%."))
    if not s.mocks and exam_date and 0 < (exam_date - today).days <= 30:
        a.append(Alert(1, "no_mock", "No mock exam taken yet."))
    if "late_night_study" in s.signals or "overwork" in s.signals:
        a.append(Alert(1, "heavy_load", "Heavy or late-night study pattern; worth a conversation about rest."))
    return sorted(a, key=lambda x: -x.severity)


def risk_level(alerts: list[Alert]) -> str:
    m = max((x.severity for x in alerts), default=0)
    return {3: "high", 2: "medium", 1: "low", 0: "none"}[m]


VISIBLE = {
    "parent": ["Estimated readiness and topic strengths", "Time spent each day", "Mock exam scores and trend", "Patterns in mistakes",
               "Practical tips for supporting your child"],
    "coach": ["Per-student progress, activity and mock scores in your batches", "Topic strengths across the batch", "At-risk alerts with reasons"],
}
HIDDEN = ["The student's chat messages and what they typed", "Individual answers and hints used", "Anything said when feeling stressed",
          "Other students' data (including peer comparisons)"]


def parent_view(d: Directory, viewer_id: str, student_id: str, snaps: dict[str, Snapshot], today: date, now: float) -> dict:
    u = d.users.get(viewer_id)
    if u is None or u.role != "parent":
        d._log(now, viewer_id, student_id, "parent_view", False, "wrong_role")
        raise AccessError(DENIED)
    rec = d.authorize(viewer_id, student_id, "parent_view", now)
    s = snaps[student_id]
    today = _today(s, today)
    out = {
        "student": rec.name, "readiness_pct": readiness(s), "readiness_note": "An estimate from practice so far, not a grade.",
        "heatmap": heatmap(s), "minutes_last_7_days": minutes_last_7(s, today),
        "days_practised_this_week": sum(1 for _, m in minutes_last_7(s, today) if m > 0),
        "mock_trend": mock_trend(s), "weak_areas": weak_areas(s), "mistake_patterns": error_patterns(s),
        "days_to_exam": (rec.exam_date - today).days if rec.exam_date else None,
        "tips": parent_tips(s, today), "what_you_can_see": VISIBLE["parent"], "what_you_cannot_see": HIDDEN,
    }
    if d.policy(rec.org).parent_sees_wellbeing_note and s.wellbeing_note:
        out["wellbeing_note"] = "Your child may be having a heavy week. A calm, low-pressure check-in could help."
    return out


def coach_view(d: Directory, viewer_id: str, snaps: dict[str, Snapshot], today: date, now: float, batch: str | None = None) -> dict:
    u = d.users.get(viewer_id)
    if u is None or u.role != "coach":
        d._log(now, viewer_id, None, "coach_view", False, "wrong_role")
        raise AccessError(DENIED)
    pol = d.policy(u.org)
    roster = [s for s in d.students_in_org(u.org) if (not u.batches or s.batch in u.batches) and (batch is None or s.batch == batch)]
    d._log(now, viewer_id, None, f"coach_view(batch_filter={'yes' if batch else 'no'}, n={len(roster)})", True, "ok")
    rows, alerts = [], []
    for rec in roster:
        s = snaps[rec.id]
        st_today = _today(s, today)
        al = risk_alerts(s, st_today, rec.exam_date, pol)
        rows.append({"student_id": rec.id, "name": rec.name, "batch": rec.batch, "readiness_pct": readiness(s),
                     "minutes_last_7_days": round(sum(m for _, m in minutes_last_7(s, st_today))),
                     "last_active_days_ago": (st_today - s.last_active).days if s.last_active else None,
                     "latest_mock_pct": round(sorted(s.mocks)[-1][1] * 100) if s.mocks else None, "risk": risk_level(al)})
        alerts += [{"student_id": rec.id, "name": rec.name, "severity": x.severity, "code": x.code, "summary": x.text} for x in al]
    n = len(roster)
    concepts = {}
    for rec in roster:
        for c in snaps[rec.id].mastery:
            concepts.setdefault(c, []).append(band(snaps[rec.id], c) == "strong")
    names = {c: next(iter(snaps[r.id].names[c] for r in roster)) for c in concepts} if roster else {}
    strength = {names[c]: round(100 * sum(v) / len(v)) for c, v in concepts.items()}
    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    return {
        "org": u.org, "n_students": n,
        "active_last_7_days_pct": round(100 * sum(1 for r in rows if r["last_active_days_ago"] is not None and r["last_active_days_ago"] < 7) / n) if n else 0,
        "avg_readiness_pct": round(mean(r["readiness_pct"] for r in rows)) if rows else 0,
        "share_strong_by_topic_pct": strength, "weakest_topics": sorted(strength, key=strength.get)[:3],
        "risk_counts": {k: sum(1 for r in rows if r["risk"] == k) for k in ("high", "medium", "low", "none")},
        "students": sorted(rows, key=lambda r: (order[r["risk"]], r["name"])),
        "alerts": sorted(alerts, key=lambda a: (-a["severity"], a["name"])),
        "what_you_can_see": VISIBLE["coach"], "what_you_cannot_see": HIDDEN[:3],
    }


def student_view(d: Directory, viewer_id: str, student_id: str, snaps: dict[str, Snapshot], today: date, now: float) -> dict:
    rec = d.authorize(viewer_id, student_id, "student_view", now)
    s = snaps[student_id]
    today = _today(s, today)
    return {"student": rec.name, "readiness_pct": readiness(s), "heatmap": heatmap(s), "weak_areas": weak_areas(s),
            "mock_trend": mock_trend(s), "minutes_last_7_days": minutes_last_7(s, today)}


def coach_student_detail(d: Directory, viewer_id: str, student_id: str, snaps: dict[str, Snapshot], today: date, now: float) -> dict:
    u = d.users.get(viewer_id)
    if u is None or u.role != "coach":
        d._log(now, viewer_id, student_id, "coach_student_detail", False, "wrong_role")
        raise AccessError(DENIED)
    rec = d.authorize(viewer_id, student_id, "coach_student_detail", now)
    s = snaps[student_id]
    today = _today(s, today)
    al = risk_alerts(s, today, rec.exam_date, d.policy(rec.org))
    return {"student": rec.name, "batch": rec.batch, "readiness_pct": readiness(s), "heatmap": heatmap(s),
            "weak_areas": weak_areas(s), "mock_trend": mock_trend(s), "minutes_last_7_days": minutes_last_7(s, today),
            "mistake_patterns": error_patterns(s), "risk": risk_level(al),
            "alerts": [{"code": a.code, "severity": a.severity, "summary": a.text} for a in al]}


LOST = ("knowledge_gap", "conceptual", "careless", "time_pressure", "guess")


def build_snapshot(student_id: str, student, eng: EngagementState, exam_hist: list[dict], graph, now: float) -> Snapshot:
    minutes: dict[date, float] = {}
    for a in eng.log:
        d = local(eng, a.ts).date()
        minutes[d] = minutes.get(d, 0.0) + a.minutes
    errs: dict[str, int] = {}
    for e in exam_hist:
        for k in LOST:
            errs[k] = errs.get(k, 0) + e["counts"].get(k, 0)
    days = set(minutes)
    return Snapshot(
        student_id=student_id, mastery=student.mastery_vector(now),
        has_evidence={c: student.state(c).last_seen is not None for c in graph.g},
        names={c: graph.name(c) for c in graph.g}, weights={c: graph.weight(c) for c in graph.g},
        minutes_by_day=minutes, daily_goal_min=eng.daily_goal_min,
        mocks=[(local(eng, e["ts"]).date(), e["score_frac"], e["accuracy"]) for e in exam_hist],
        error_counts=errs, last_active=max(days) if days else None, signals=behaviour_signals(eng, now),
        focus=list(student.focus),
        safeguarding_flag=any(lv == "acute" and now - ts <= 7 * 86400 for ts, lv, _ in eng.stress_events),
        wellbeing_note=rank(current_stress(eng, now)) >= rank("elevated"), today=local(eng, now).date())


PERSONAS = ["steady", "strong", "disengaged", "dip", "overwork", "behind", "flagged"]


def synthetic_snapshot(i: int, graph, today: date, persona: str | None = None, seed: int = 0) -> Snapshot:
    rng = random.Random(seed * 1000 + i)
    persona = persona or PERSONAS[i % len(PERSONAS)]
    order = graph.topological_order()
    k = {"steady": 6, "strong": 10, "disengaged": 5, "dip": 7, "overwork": 7, "behind": 3, "flagged": 4}[persona]
    mastery, evid = {}, {}
    for j, c in enumerate(order):
        mastery[c] = min(0.97, max(0.05, rng.gauss(0.88 if j < k else 0.5 if j == k else 0.2, 0.05)))
        evid[c] = j <= k + 1
    mins: dict[date, float] = {}
    active_days = {"disengaged": range(10, 20), "behind": (1, 3, 5), "steady": range(0, 7), "strong": range(0, 7)}.get(persona, range(0, 7))
    for off in active_days:
        mins[today - timedelta(days=off)] = {"behind": 10, "strong": 40, "overwork": 400 if off < 3 else 60}.get(persona, 35) + rng.randint(-5, 5)
    mocks_f = {"steady": [0.45, 0.5, 0.56], "strong": [0.7, 0.78, 0.85], "dip": [0.7, 0.72, 0.74, 0.5, 0.45],
               "overwork": [0.5, 0.52], "behind": [], "disengaged": [0.4, 0.42], "flagged": [0.3, 0.28]}[persona]
    mocks = [(today - timedelta(days=7 * (len(mocks_f) - n)), f, min(1.0, f + 0.1)) for n, f in enumerate(mocks_f)]
    sig = {"dip": ["score_drop"], "overwork": ["overwork", "late_night_study"]}.get(persona, [])
    errs = {"steady": {"careless": 4, "conceptual": 2}, "dip": {"conceptual": 5, "time_pressure": 2},
            "flagged": {"knowledge_gap": 6}, "strong": {"careless": 2}}.get(persona, {})
    last = max(mins) if mins else None
    return Snapshot(f"syn{i}", mastery, evid, {c: graph.name(c) for c in order}, {c: graph.weight(c) for c in order},
                    mins, 30, mocks, errs, last, sig, safeguarding_flag=(persona == "flagged"), wellbeing_note=(persona == "flagged"))


DAY = 86400.0
MIN_COHORT = 30


class ConsentError(Exception):
    pass


def difficulty_adjusted(result) -> float:
    total = sum(max(q.difficulty, 0.05) for q in result.questions)
    got = sum(max(q.difficulty, 0.05) for q in result.questions if q.correct)
    return got / total if total else 0.0


@dataclass
class PeerRecord:
    pid: str
    org: str
    exam: str
    scope: str
    ts: float
    value: float
    n_items: int
    source: str = "real"


@dataclass
class Standing:
    status: str
    message: str = ""
    percentile: int | None = None
    band: str | None = None
    cohort_size: int | None = None
    quartiles: tuple[float, float, float] | None = None
    delta_points: float | None = None
    synthetic: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("status", "message", "percentile", "band", "cohort_size", "quartiles", "delta_points", "synthetic")}


BANDS = [(75, "leading group"), (25, "middle group"), (0, "building group")]


def band_for(pct: int) -> str:
    return next(name for floor, name in BANDS if pct >= floor)


class PeerStore:
    def __init__(self, secret: bytes, min_cohort: int = MIN_COHORT, window_days: int = 30, min_items: int = 5):
        if len(secret) < 16:
            raise ValueError("secret must be at least 16 bytes")
        self._secret, self.min_cohort, self.window_days, self.min_items = secret, min_cohort, window_days, min_items
        self._records: list[PeerRecord] = []
        self._consent: dict[str, bool] = {}

    def pseudonym(self, student_id: str) -> str:
        return hmac.new(self._secret, student_id.encode(), hashlib.sha256).hexdigest()[:24]

    def has_consent(self, student_id: str) -> bool:
        return self._consent.get(self.pseudonym(student_id), False)

    def set_consent(self, student_id: str, opted_in: bool, is_minor: bool = True, guardian_ok: bool = False) -> None:
        pid = self.pseudonym(student_id)
        if opted_in:
            if is_minor and not guardian_ok:
                raise ConsentError("Students under 18 need a parent/guardian's agreement to join comparisons.")
            self._consent[pid] = True
        else:
            self.erase_student(student_id)

    def erase_student(self, student_id: str) -> None:
        pid = self.pseudonym(student_id)
        self._records = [r for r in self._records if r.pid != pid]
        self._consent.pop(pid, None)

    def add(self, student_id: str, org: str, exam: str, scope: str, ts: float, value: float, n_items: int) -> bool:
        if not self.has_consent(student_id) or n_items < self.min_items or not (0.0 <= value <= 1.0):
            return False
        self._records.append(PeerRecord(self.pseudonym(student_id), org, exam, scope, ts, value, n_items))
        return True

    def add_synthetic(self, org: str, exam: str, scope: str, now: float, n: int = 60, mean: float = 0.55,
                      sd: float = 0.18, seed: int = 0) -> None:
        rng = random.Random(seed)
        for i in range(n):
            v = min(1.0, max(0.0, rng.gauss(mean, sd)))
            self._records.append(PeerRecord(f"syn{seed}_{i}", org, exam, scope, now - rng.uniform(0, self.window_days - 1) * DAY, v, 10, "synthetic"))

    def _latest(self, recs, lo, hi) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for r in recs:
            if lo < r.ts <= hi and (r.pid not in out or r.ts > out[r.pid][0]):
                out[r.pid] = (r.ts, r.value)
        return out

    def _pool(self, org, exam, scope, pool, include_synthetic):
        return [r for r in self._records if r.exam == exam and r.scope == scope
                and (pool == "global" or r.org == org) and (include_synthetic or r.source == "real")]

    def standing(self, student_id: str, org: str, exam: str, scope: str, now: float, stress_level: str = "none",
                 pool: str = "org", include_synthetic: bool = False, next_topic: str | None = None) -> Standing:
        if stress_level != "none":
            return Standing("hidden_for_wellbeing", "Comparisons are paused for now. They'll be here whenever you want them back.")
        if not self.has_consent(student_id):
            return Standing("not_opted_in", "Comparisons are optional and off. You can turn them on any time.")
        pid = self.pseudonym(student_id)
        recs = self._pool(org, exam, scope, pool, include_synthetic)
        w = self.window_days * DAY
        cur = self._latest(recs, now - w, now)
        if pid not in cur:
            return Standing("no_data", "Finish a mock exam or some practice in this window to see where you stand.")
        mine = cur[pid][1]
        prev = self._latest([r for r in recs if r.pid == pid], now - 2 * w, now - w).get(pid)
        delta = None if prev is None else round((mine - prev[1]) * 100, 1)
        synthetic = include_synthetic and any(r.source == "synthetic" for r in recs)
        if len(cur) < self.min_cohort:
            return Standing("cohort_too_small", f"Comparisons appear once at least {self.min_cohort} students have joined this group.",
                            delta_points=delta)
        others = [v for p, (_, v) in cur.items() if p != pid]
        below, equal = sum(v < mine for v in others), sum(v == mine for v in others)
        raw = 100 * (below + 0.5 * equal) / len(others)
        pct = max(5, min(95, int(5 * round(raw / 5))))
        q = quantiles([v for _, v in cur.values()], n=4)
        s = Standing("ok", percentile=pct, band=band_for(pct), cohort_size=len(cur),
                     quartiles=(round(q[0], 2), round(q[1], 2), round(q[2], 2)), delta_points=delta, synthetic=synthetic)
        s.message = frame(s, next_topic)
        return s


def frame(s: Standing, next_topic: str | None = None) -> str:
    parts = []
    if s.delta_points is not None:
        d = s.delta_points
        parts.append(f"You're up {d:g} points on your last period." if d > 0.5 else
                     "You're holding steady compared with your last period." if d >= -0.5 else
                     "This period came in a little lower than your last one. That kind of wobble is normal, so let's look at where.")
    where = {"leading group": "You're in the leading group", "middle group": "You're in the middle of the group",
             "building group": "Many students in this group are further along on this right now, and that gap is very closable"}[s.band]
    parts.append(f"{where} (about the {s.percentile}th percentile among {s.cohort_size} students who chose to compare).")
    parts.append(f"The quickest win: {next_topic}." if next_topic else "Pick one topic and do a short set today.")
    return " ".join(parts)


SCHEMA_VERSION = "1"
MAX_DOC_BYTES = 1_000_000


class ConflictError(Exception):
    pass


class Base(DeclarativeBase):
    pass


class Meta(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[str] = mapped_column(String(200))


class OrgRow(Base):
    __tablename__ = "orgs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    parent_wellbeing: Mapped[bool] = mapped_column(Boolean, default=False)
    coach_flags: Mapped[bool] = mapped_column(Boolean, default=True)


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    role: Mapped[str] = mapped_column(String(20))
    batches: Mapped[str] = mapped_column(Text, default="[]")
    pw_hash: Mapped[str] = mapped_column(String(300))
    pw_version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[float] = mapped_column(Float, default=0.0)


class StudentRow(Base):
    __tablename__ = "students"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    batch: Mapped[str] = mapped_column(String(64), default="")
    name: Mapped[str] = mapped_column(String(200), default="")
    is_minor: Mapped[bool] = mapped_column(Boolean, default=True)
    adult_consents_parent: Mapped[bool] = mapped_column(Boolean, default=False)
    exam_date: Mapped[str | None] = mapped_column(String(10), nullable=True)


class LinkRow(Base):
    __tablename__ = "parent_links"
    parent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)


class CodeRow(Base):
    __tablename__ = "guardian_codes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    student_id: Mapped[str] = mapped_column(String(64))
    code_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires: Mapped[float] = mapped_column(Float)
    used_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))


class AttemptRow(Base):
    __tablename__ = "redeem_attempts"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[float] = mapped_column(Float, default=0.0)


class RevokedRow(Base):
    __tablename__ = "revoked_tokens"
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    exp: Mapped[float] = mapped_column(Float)


class AuditRow(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    viewer: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20))
    org: Mapped[str] = mapped_column(String(64))
    student: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(200))
    allowed: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(String(100))


class DocRow(Base):
    __tablename__ = "documents"
    org_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[float] = mapped_column(Float)


class Repo:
    def __init__(self, url: str = "sqlite+pysqlite:///:memory:", clock=time.time):
        kw = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kw["poolclass"] = StaticPool
        self.engine = create_engine(url, **kw)
        self.clock = clock
        Base.metadata.create_all(self.engine)
        self._S = sessionmaker(self.engine, expire_on_commit=False)
        with self._S.begin() as s:
            m = s.get(Meta, "schema_version")
            if m is None:
                s.add(Meta(key="schema_version", value=SCHEMA_VERSION))
            elif m.value != SCHEMA_VERSION:
                raise RuntimeError(f"Database schema {m.value} != code {SCHEMA_VERSION}: run migrations first.")

    def session(self): return self._S.begin()

    def ensure_org(self, org: str, name: str = "") -> None:
        with self.session() as s:
            if s.get(OrgRow, org) is None:
                s.add(OrgRow(id=org, name=name or org))

    def set_policy(self, org: str, p: Policy) -> None:
        self.ensure_org(org)
        with self.session() as s:
            o = s.get(OrgRow, org); o.parent_wellbeing, o.coach_flags = p.parent_sees_wellbeing_note, p.coach_sees_safeguarding_flags

    def get_policy(self, org: str) -> Policy:
        with self.session() as s:
            o = s.get(OrgRow, org)
            return Policy() if o is None else Policy(o.parent_wellbeing, o.coach_flags)

    def add_user(self, **kw) -> None:
        self.ensure_org(kw["org_id"])
        with self.session() as s:
            s.add(UserRow(**kw))

    def get_user_row(self, uid: str):
        with self.session() as s:
            return s.get(UserRow, uid)

    def get_user_by_email(self, email: str):
        with self.session() as s:
            return s.scalars(select(UserRow).where(UserRow.email == email)).first()

    def update_user(self, uid: str, **kw) -> None:
        with self.session() as s:
            s.execute(update(UserRow).where(UserRow.id == uid).values(**kw))

    def add_student(self, **kw) -> None:
        self.ensure_org(kw["org_id"])
        with self.session() as s:
            s.add(StudentRow(**kw))

    def get_student_row(self, sid: str):
        with self.session() as s:
            return s.get(StudentRow, sid)

    def update_student(self, sid: str, **kw) -> None:
        with self.session() as s:
            s.execute(update(StudentRow).where(StudentRow.id == sid).values(**kw))

    def student_ids(self) -> list[str]:
        with self.session() as s:
            return list(s.scalars(select(StudentRow.id).order_by(StudentRow.id)))

    def user_ids(self) -> list[str]:
        with self.session() as s:
            return list(s.scalars(select(UserRow.id).order_by(UserRow.id)))

    def students_in_org(self, org: str):
        with self.session() as s:
            return list(s.scalars(select(StudentRow).where(StudentRow.org_id == org).order_by(StudentRow.id)))

    def add_link(self, parent_id: str, student_id: str, org: str) -> None:
        with self.session() as s:
            if s.get(LinkRow, (parent_id, student_id)) is None:
                s.add(LinkRow(parent_id=parent_id, student_id=student_id, org_id=org))

    def has_link(self, parent_id: str, student_id: str) -> bool:
        with self.session() as s:
            return s.get(LinkRow, (parent_id, student_id)) is not None

    def get_doc(self, org: str, sid: str, kind: str):
        with self.session() as s:
            r = s.get(DocRow, (org, sid, kind))
            return None if r is None else (json.loads(r.body), r.version)

    def put_doc(self, org: str, sid: str, kind: str, data: dict, expected_version: int | None) -> int:
        body = json.dumps(data, separators=(",", ":"), sort_keys=True)
        if len(body.encode()) > MAX_DOC_BYTES:
            raise ValueError("document too large")
        try:
            with self.session() as s:
                if expected_version in (None, 0):
                    s.add(DocRow(org_id=org, student_id=sid, kind=kind, version=1, body=body, updated_at=self.clock()))
                    s.flush()
                    return 1
                res = s.execute(update(DocRow).where(DocRow.org_id == org, DocRow.student_id == sid, DocRow.kind == kind,
                                                     DocRow.version == expected_version)
                                .values(body=body, version=expected_version + 1, updated_at=self.clock()))
                if res.rowcount != 1:
                    raise ConflictError("document changed since it was loaded")
                return expected_version + 1
        except IntegrityError as e:
            raise ConflictError("document already exists") from e

    def delete_doc(self, org: str, sid: str, kind: str) -> None:
        with self.session() as s:
            s.execute(delete(DocRow).where(DocRow.org_id == org, DocRow.student_id == sid, DocRow.kind == kind))

    def find_code(self, org: str, code_hash: str, now: float):
        with self.session() as s:
            r = s.scalars(select(CodeRow).where(CodeRow.org_id == org, CodeRow.code_hash == code_hash)).first()
            return (r.id, r.student_id) if r is not None and r.used_at is None and r.expires > now else None

    def mark_code_used(self, code_id: int, now: float) -> bool:
        with self.session() as s:
            return s.execute(update(CodeRow).where(CodeRow.id == code_id, CodeRow.used_at.is_(None)).values(used_at=now)).rowcount == 1

    def unmark_code_used(self, code_id: int) -> None:
        with self.session() as s:
            s.execute(update(CodeRow).where(CodeRow.id == code_id).values(used_at=None))

    def revoke(self, jti: str, exp: float) -> None:
        with self.session() as s:
            if s.get(RevokedRow, jti) is None:
                s.add(RevokedRow(jti=jti, exp=exp))
            s.execute(delete(RevokedRow).where(RevokedRow.exp < self.clock() - 86400))

    def is_revoked(self, jti: str) -> bool:
        with self.session() as s:
            return s.get(RevokedRow, jti) is not None

    def log(self, ts, viewer, role, org, student, action, allowed, reason) -> None:
        with self.session() as s:
            s.add(AuditRow(ts=ts, viewer=viewer, role=role, org=org, student=student, action=action[:200], allowed=allowed, reason=reason))

    def audit(self, org: str | None = None, limit: int = 1000) -> list[AuditEntry]:
        with self.session() as s:
            q = select(AuditRow).order_by(AuditRow.id)
            if org is not None:
                q = q.where(AuditRow.org == org)
            return [AuditEntry(r.ts, r.viewer, r.role, r.org, r.student, r.action, r.allowed, r.reason) for r in s.scalars(q.limit(limit))]

    def export_student(self, org: str, sid: str) -> dict:
        with self.session() as s:
            st = s.get(StudentRow, sid)
            if st is None or st.org_id != org:
                raise KeyError(sid)
            docs = {r.kind: json.loads(r.body) for r in s.scalars(select(DocRow).where(DocRow.org_id == org, DocRow.student_id == sid))}
            links = [l.parent_id for l in s.scalars(select(LinkRow).where(LinkRow.org_id == org, LinkRow.student_id == sid))]
            return {"student": {"id": st.id, "name": st.name, "batch": st.batch, "is_minor": st.is_minor, "exam_date": st.exam_date},
                    "linked_parent_ids": links, "documents": docs}

    def erase_student(self, org: str, sid: str) -> None:
        with self.session() as s:
            st = s.get(StudentRow, sid)
            if st is None or st.org_id != org:
                raise KeyError(sid)
            s.execute(delete(DocRow).where(DocRow.org_id == org, DocRow.student_id == sid))
            s.execute(delete(LinkRow).where(LinkRow.org_id == org, LinkRow.student_id == sid))
            s.execute(delete(CodeRow).where(CodeRow.org_id == org, CodeRow.student_id == sid))
            s.execute(update(AuditRow).where(AuditRow.org == org, AuditRow.student == sid).values(student="erased:" + secrets.token_hex(4)))
            u = s.get(UserRow, sid)
            if u is not None and u.role == "student" and u.org_id == org:
                s.delete(u)
            s.delete(st)


class _DictLike:
    def __iter__(self): return iter(self._ids())
    def __len__(self): return len(self._ids())
    def __contains__(self, k): return self.get(k) is not None
    def keys(self): return list(self._ids())
    def values(self): return [v for v in (self.get(i) for i in self._ids()) if v is not None]
    def items(self): return [(v.id, v) for v in self.values()]


class _Users(_DictLike):
    def __init__(self, repo): self.repo = repo
    def _ids(self): return self.repo.user_ids()
    def get(self, uid):
        r = self.repo.get_user_row(uid)
        return None if r is None or not r.active else User(r.id, r.role, r.org_id, frozenset(json.loads(r.batches)))
    def __getitem__(self, uid):
        u = self.get(uid)
        if u is None: raise KeyError(uid)
        return u


class _LiveStudent(StudentRecord):
    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        repo = self.__dict__.get("_repo")
        if repo is not None and k in ("adult_consents_parent", "is_minor", "batch", "name", "exam_date"):
            repo.update_student(self.id, **{k: (v.isoformat() if k == "exam_date" and v else v)})


class _Students(_DictLike):
    def __init__(self, repo): self.repo = repo
    def _ids(self): return self.repo.student_ids()
    def get(self, sid):
        r = self.repo.get_student_row(sid)
        if r is None:
            return None
        s = _LiveStudent(r.id, r.org_id, r.batch, r.name, r.is_minor, r.adult_consents_parent,
                         date.fromisoformat(r.exam_date) if r.exam_date else None)
        object.__setattr__(s, "_repo", self.repo)
        return s
    def __getitem__(self, sid):
        s = self.get(sid)
        if s is None: raise KeyError(sid)
        return s


class _Links:
    def __init__(self, repo): self.repo = repo
    def __contains__(self, pair): return self.repo.has_link(*pair)
    def add(self, pair, org=None):
        p, s = pair
        self.repo.add_link(p, s, org or self.repo.get_student_row(s).org_id)


class DbDirectory(Directory):
    def __init__(self, policies: dict[str, Policy] | None = None, repo: Repo | None = None):
        self.repo = repo or Repo()
        self.users, self.students, self.links = _Users(self.repo), _Students(self.repo), _Links(self.repo)
        for org, p in (policies or {}).items():
            self.repo.set_policy(org, p)

    def policy(self, org: str) -> Policy: return self.repo.get_policy(org)
    def add_user(self, u: User, email: str | None = None, pw_hash: str = "!"):
        self.repo.add_user(id=u.id, org_id=u.org, email=email or f"{u.id}@local.invalid", role=u.role,
                           batches=json.dumps(sorted(u.batches)), pw_hash=pw_hash)
    def add_student(self, s: StudentRecord):
        self.repo.add_student(id=s.id, org_id=s.org, batch=s.batch, name=s.name, is_minor=s.is_minor,
                              adult_consents_parent=s.adult_consents_parent, exam_date=s.exam_date.isoformat() if s.exam_date else None)
    def link_parent(self, parent_id: str, student_id: str) -> None:
        p, s = self.users.get(parent_id), self.students.get(student_id)
        if not p or not s or p.role != "parent" or p.org != s.org:
            raise AccessError(DENIED)
        if not s.is_minor and not s.adult_consents_parent:
            raise AccessError("This student is 18+, so they need to agree to share with a parent first.")
        self.links.add((parent_id, student_id), s.org)
    def students_in_org(self, org: str):
        return [_Students(self.repo).get(r.id) for r in self.repo.students_in_org(org)]
    def _log(self, now, viewer, student, action, allowed, reason):
        u = self.users.get(viewer)
        self.repo.log(now, audit_ref(viewer, u is not None), u.role if u else "unknown", u.org if u else "unknown",
                      audit_ref(student, student is None or self.repo.get_student_row(student) is not None), action, allowed, reason)
    @property
    def audit(self): return self.repo.audit()


GENERIC = "Invalid email or password, or too many recent attempts."
ROLES = ("student", "parent", "coach", "org_admin")
ID_RE = re.compile(r"^[a-z0-9_-]{2,64}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")
COMMON = {"password", "password1", "1234567890", "qwertyuiop", "iloveyou12", "letmein123", "welcome123", "admin12345", "0123456789", "passw0rd12"}
ALPH = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LEN, CODE_TTL, REDEEM_MAX, REDEEM_WINDOW = 10, 7 * 86400, 5, 3600


class AuthError(Exception):
    pass


class Forbidden(Exception):
    pass


def _b64(b: bytes) -> str: return base64.b64encode(b).decode()


PRODUCTION_N = 2 ** 14
DEFAULT_N = PRODUCTION_N


def hash_password(pw: str, n: int | None = None, r: int = 8, p: int = 1) -> str:
    n = n or DEFAULT_N
    salt = os.urandom(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(dk)}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(dk))
    except Exception:
        return False


def check_password_policy(pw: str, email: str = "") -> None:
    if len(pw) < 10 or len(pw) > 128:
        raise ValueError("Password must be 10-128 characters.")
    if pw.lower() in COMMON or len(set(pw)) < 4:
        raise ValueError("That password is too easy to guess.")
    low = pw.lower()
    for part in re.split(r"[._+\-0-9]+", email.split("@")[0].lower()):
        if len(part) >= 4 and part in low:
            raise ValueError("Password must not contain your email name.")


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    org: str
    role: str
    batches: frozenset
    jti: str = ""
    exp: float = 0.0


class AuthService:
    def __init__(self, repo: Repo, secret: bytes, clock=time.time, token_ttl: int = 3600, max_failures: int = 5, lock_seconds: int = 900):
        if len(secret) < 32:
            raise ValueError("secret must be at least 32 bytes (use a random value from a secret manager)")
        self.repo, self._secret, self.clock = repo, secret, clock
        self.ttl, self.max_failures, self.lock_seconds = token_ttl, max_failures, lock_seconds
        self._dummy = hash_password("dummy-password-for-timing-equalisation")

    def create_user(self, org: str, user_id: str, email: str, password: str, role: str, batches=()) -> None:
        import json
        email = email.strip().lower()
        if role not in ROLES or not ID_RE.match(user_id) or not EMAIL_RE.match(email):
            raise ValueError("invalid user fields")
        check_password_policy(password, email)
        self.repo.add_user(id=user_id, org_id=org, email=email, role=role, batches=json.dumps(sorted(batches)),
                           pw_hash=hash_password(password))

    def deactivate(self, user_id: str) -> None:
        self.repo.update_user(user_id, active=False)

    def change_password(self, user_id: str, old: str, new: str) -> None:
        row = self.repo.get_user_row(user_id)
        if row is None or not verify_password(old, row.pw_hash):
            raise AuthError("Current password is incorrect.")
        check_password_policy(new, row.email)
        self.repo.update_user(user_id, pw_hash=hash_password(new), pw_version=row.pw_version + 1)

    def verify_user_password(self, user_id: str, password: str) -> bool:
        row = self.repo.get_user_row(user_id)
        return row is not None and verify_password(password, row.pw_hash)

    def login(self, email: str, password: str) -> str:
        now, row = self.clock(), self.repo.get_user_by_email(email.strip().lower())
        if row is None:
            verify_password(password, self._dummy)
            raise AuthError(GENERIC)
        locked = row.locked_until > now
        ok = verify_password(password, row.pw_hash)
        if locked or not ok or not row.active:
            if not locked:
                fails = row.failed_logins + 1
                lock = now + self.lock_seconds if fails >= self.max_failures else 0.0
                self.repo.update_user(row.id, failed_logins=0 if lock else fails, locked_until=lock or row.locked_until)
            raise AuthError(GENERIC)
        self.repo.update_user(row.id, failed_logins=0, locked_until=0.0)
        return jwt.encode({"sub": row.id, "jti": secrets.token_urlsafe(16), "iat": int(now), "exp": int(now) + self.ttl,
                           "pwv": row.pw_version}, self._secret, algorithm="HS256")

    def _decode(self, token: str) -> dict:
        try:
            p = jwt.decode(token, self._secret, algorithms=["HS256"],
                           options={"require": ["exp", "iat", "sub", "jti", "pwv"], "verify_exp": False, "verify_iat": False})
        except jwt.PyJWTError as e:
            raise AuthError("Invalid or expired token.") from e
        if p["exp"] <= self.clock():
            raise AuthError("Invalid or expired token.")
        return p

    def authenticate(self, token: str) -> AuthContext:
        import json
        p = self._decode(token)
        row = self.repo.get_user_row(p["sub"])
        if row is None or not row.active or row.pw_version != p["pwv"] or self.repo.is_revoked(p["jti"]):
            raise AuthError("Invalid or expired token.")
        return AuthContext(row.id, row.org_id, row.role, frozenset(json.loads(row.batches)), p["jti"], float(p["exp"]))

    def refresh(self, token: str) -> str:
        ctx = self.authenticate(token)
        p = self._decode(token)
        now = self.clock()
        new = jwt.encode({"sub": ctx.user_id, "jti": secrets.token_urlsafe(16), "iat": int(now), "exp": int(now) + self.ttl,
                          "pwv": p["pwv"]}, self._secret, algorithm="HS256")
        self.repo.revoke(ctx.jti, ctx.exp)
        return new

    def logout(self, token: str) -> None:
        p = self._decode(token)
        self.repo.revoke(p["jti"], float(p["exp"]))


class GuardianService:
    def __init__(self, repo: Repo, secret: bytes, clock=time.time):
        self.repo, self._secret, self.clock = repo, secret, clock

    def _hash(self, code: str) -> str:
        return hmac.new(self._secret, b"guardian:" + code.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _normalise(raw: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", raw.upper())

    def issue(self, admin: AuthContext, student_id: str) -> tuple[str, float]:
        if admin.role != "org_admin":
            raise Forbidden("Only organisation admins can issue guardian codes.")
        st = self.repo.get_student_row(student_id)
        if st is None or st.org_id != admin.org:
            raise KeyError(student_id)
        if not st.is_minor and not st.adult_consents_parent:
            raise Forbidden("This student is 18+ and has not agreed to share with a parent.")
        code = "".join(secrets.choice(ALPH) for _ in range(CODE_LEN))
        exp = self.clock() + CODE_TTL
        with self.repo.session() as s:
            s.add(CodeRow(org_id=admin.org, student_id=student_id, code_hash=self._hash(code), expires=exp, created_by=admin.user_id))
        return f"{code[:5]}-{code[5:]}", exp

    def _fail(self, user_id: str, now: float) -> None:
        with self.repo.session() as s:
            a = s.get(AttemptRow, user_id)
            if a is None or now - a.window_start >= REDEEM_WINDOW:
                s.merge(AttemptRow(user_id=user_id, count=1, window_start=now))
            else:
                a.count += 1

    def redeem(self, parent: AuthContext, raw_code: str, directory) -> str:
        if parent.role != "parent":
            raise Forbidden("Only parent accounts can redeem a guardian code.")
        now = self.clock()
        with self.repo.session() as s:
            a = s.get(AttemptRow, parent.user_id)
            if a and now - a.window_start < REDEEM_WINDOW and a.count >= REDEEM_MAX:
                raise AuthError("Too many attempts. Please try again later.")
        code = self._normalise(raw_code)
        h = self._hash(code)
        found, claimed = None, None
        if len(code) == CODE_LEN and all(c in ALPH for c in code):
            found = self.repo.find_code(parent.org, h, now)
            if found is not None and self.repo.mark_code_used(found[0], now):
                claimed = found[1]
        if claimed is None:
            self._fail(parent.user_id, now)
            raise AuthError("Invalid or expired code.")
        try:
            directory.link_parent(parent.user_id, claimed)
        except Exception:
            self.repo.unmark_code_used(found[0])
            raise
        return claimed

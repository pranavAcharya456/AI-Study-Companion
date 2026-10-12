import hashlib
import json
import os
import secrets
import time
from datetime import date, timedelta

import streamlit as st

from backend import (
    AccessError, AuthError, AuthService, ConflictError, ConsentError, DbDirectory, Directory, GuardianService,
    PeerStore, Policy, Repo, StudentRecord, User, build_snapshot, coach_student_detail, coach_view,
    difficulty_adjusted, parent_view, student_view, synthetic_snapshot
)
from engine import (
    AnthropicSolver, AnthropicVision, DEFAULT_EXAM, DiagnosticSession, DoubtSession, EngagementState,
    ExamClosed, LETTERS, MockVision, Registry, SectionedExam, StudentModel,
    TutorSession, Walkthrough, active_days, analyze, apply_feedback, assemble_exam, award_new, build_plan,
    current_stress, engagement_from_dict, engagement_to_dict, get_llm, handle_student_message, local,
    make_sample_image, minutes_on, model_kind, pick_checkin, qualifies, record, render_prompt,
    reply_to_checkin_message, streak, student_from_dict, student_to_dict, suggest_hours, ui_flags
)


st.set_page_config(page_title="AI Study Companion", page_icon="📘", layout="wide")


@st.cache_resource
def load_registry():
    return Registry.from_data()


REG = load_registry()
use_mock = os.getenv("USE_REAL_LLM", "0") != "1"
ss = st.session_state


@st.cache_resource
def services():
    from pathlib import Path
    sec = os.getenv("APP_SECRET", "").encode()
    if len(sec) < 32:
        p = Path(".dev_secret")
        if not p.exists():
            p.write_text(os.urandom(32).hex()); p.chmod(0o600)
        sec = p.read_text().encode()
    repo = Repo(os.getenv("APP_DB", "sqlite+pysqlite:///study_companion.db"))
    return repo, AuthService(repo, sec), DbDirectory(repo=repo), GuardianService(repo, sec)


def fresh_state(keep=()):
    for k in [k for k in list(ss.keys()) if k not in keep]:
        del ss[k]
    if "exam_id" not in ss:
        ss.exam_id = DEFAULT_EXAM
    ss.exam_prev = ss.exam_id
    ss.llm, ss.students = get_llm(use_mock=use_mock), {}
    ss.tutor, ss.messages, ss.recorded = None, [], False
    ss.diag, ss.diag_msgs, ss.diag_report = None, [], None
    ss.exam, ss.exam_out, ss.exam_notes, ss.exam_mastery = None, None, [], {}
    ss.doubt, ss.walk, ss.doubt_img, ss.doubt_tagged = None, None, None, None
    ss.eng, ss.support, ss.checkin_msgs, ss.milestones, ss.q_started = EngagementState(), None, [], [], time.time()
    ss.exam_hist, ss.directory, ss.syn_snaps = [], None, {}
    ss.peers, ss.syn_done = PeerStore(os.getenv("PEER_SECRET", "").encode() or os.urandom(32)), set()
    ss.versions, ss.saved, ss.save_blocked = {}, {}, False


def load_state(ctx):
    repo = services()[0]
    fresh_state(keep=("token", "ctx", "exam_id"))
    bad = []

    def get(kind):
        r = repo.get_doc(ctx.org, ctx.user_id, kind)
        if r:
            ss.versions[kind] = r[1]
            return r[0]

    loaders = [("profile", None), ("engagement", None), ("exam_hist", None)] + [(model_kind(e), e) for e in REG.ids()] + [("exam_session", None)]
    for kind, eid in loaders:
        try:
            d = get(kind)
            if d is None:
                continue
            if kind == "profile":
                if d.get("exam_id") in REG.modules:
                    ss.exam_id = ss.exam_prev = d["exam_id"]
            elif kind == "engagement": ss.eng = engagement_from_dict(d)
            elif kind == "exam_hist": ss.exam_hist = list(d["items"])
            elif eid is not None:
                m = REG.get(eid); ss.students[eid] = student_from_dict(m.graph, m.bank, d)
            else:
                sess = d["session"]
                if sess.get("kind") != "sectioned" or sess["exam_id"] not in REG.modules:
                    raise ValueError("old or unknown exam format")
                ss.exam = SectionedExam.from_dict(sess, REG.get(sess["exam_id"]).bank)
                ss.exam_mastery, ss.exam_id, ss.exam_prev = d["mastery"], sess["exam_id"], sess["exam_id"]
                ss.qsel = ss.exam.current or ss.exam.paper[0]["id"]
        except (ValueError, KeyError, TypeError):
            bad.append(kind)
    if bad:
        st.sidebar.warning("Some saved data couldn't be read, so I started fresh for: " + ", ".join(bad))


def persist():
    ctx = ss.get("ctx")
    if ctx is None or ss.get("save_blocked"):
        return
    repo = services()[0]
    docs = {"engagement": engagement_to_dict(ss.eng), "exam_hist": {"items": ss.exam_hist}, "profile": {"exam_id": ss.exam_id}}
    for eid, sm in ss.students.items():
        docs[model_kind(eid)] = student_to_dict(sm)
    if ss.exam is not None:
        docs["exam_session"] = {"session": ss.exam.to_dict(), "mastery": ss.exam_mastery}
    try:
        for kind, data in docs.items():
            h = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            if ss.saved.get(kind) != h:
                ss.versions[kind] = repo.put_doc(ctx.org, ctx.user_id, kind, data, ss.versions.get(kind)); ss.saved[kind] = h
        if ss.exam is None and "exam_session" in ss.versions:
            repo.delete_doc(ctx.org, ctx.user_id, "exam_session"); ss.versions.pop("exam_session"); ss.saved.pop("exam_session", None)
    except ConflictError:
        ss.save_blocked = True
        st.sidebar.warning("This account was changed in another window, so I've stopped saving here. Reload the page to continue.")


def sign_out(revoke=True):
    if revoke and ss.get("token"):
        try: services()[1].logout(ss.token)
        except AuthError: pass
    ss.pop("token", None); ss.pop("ctx", None)
    fresh_state()


def account_ui():
    repo, auth, directory, _ = services()
    if ss.get("token"):
        try:
            ctx = auth.authenticate(ss.token)
            if ctx.exp - time.time() < 900:
                ss.token = auth.refresh(ss.token); ctx = auth.authenticate(ss.token)
            ss.ctx = ctx
        except AuthError:
            sign_out(revoke=False); st.sidebar.warning("Your session ended. Please sign in again.")
    st.subheader("👤 Account")
    ctx = ss.get("ctx")
    if ctx is None:
        st.caption("Not signed in: your progress is NOT saved.")
        choice = st.radio("Account", ["Sign in", "Create demo account"], horizontal=True, label_visibility="collapsed")
        if choice == "Sign in":
            email = st.text_input("Email", key="si_email"); pw = st.text_input("Password", type="password", key="si_pw")
            if st.button("Sign in"):
                try:
                    tok = auth.login(email, pw); c = auth.authenticate(tok)
                    if c.role != "student":
                        auth.logout(tok); st.error("This app is for student accounts. Parents and coaches use the dashboards API.")
                    else:
                        ss.token, ss.ctx = tok, c; load_state(c); st.rerun()
                except AuthError as e:
                    st.error(str(e))
        else:
            st.caption("Demo sign-up creates your own private sandbox organisation. Real schools/coaching centres have admins create accounts.")
            email = st.text_input("Email", key="su_email"); name = st.text_input("Your name", key="su_name")
            pw = st.text_input("Password (10+ characters)", type="password", key="su_pw"); minor = st.checkbox("I'm under 18", value=True, key="su_minor")
            if st.button("Create account"):
                org, uid = "demo-" + secrets.token_hex(4), "s-" + secrets.token_hex(6)
                try:
                    auth.create_user(org, uid, email, pw, "student")
                    directory.add_student(StudentRecord(uid, org, "", (name or "Student")[:200], is_minor=minor))
                    tok = auth.login(email, pw); ss.token, ss.ctx = tok, auth.authenticate(tok); load_state(ss.ctx); st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception:
                    st.error("Couldn't create that account (the email may already be in use).")
        return
    st.caption(f"Signed in. Progress is saved automatically.")
    if st.button("Sign out"):
        sign_out(); st.rerun()
    with st.expander("My data"):
        st.download_button("Download my data (JSON)", json.dumps(repo.export_student(ctx.org, ctx.user_id), indent=1), "my_data.json", "application/json")
        pw = st.text_input("Confirm password to delete everything", type="password", key="del_pw")
        if st.button("Delete my account and all my data"):
            if auth.verify_user_password(ctx.user_id, pw):
                repo.erase_student(ctx.org, ctx.user_id); sign_out(revoke=False); st.rerun()
            else:
                st.error("Password confirmation failed.")


if "students" not in ss:
    fresh_state()
module = REG.get(ss.exam_id)
graph, bank = module.graph, module.bank
if ss.exam_id not in ss.students:
    ss.students[ss.exam_id] = StudentModel(graph, bank)
student: StudentModel = ss.students[ss.exam_id]

with st.sidebar:
    account_ui()

    def on_exam_change():
        if ss.exam is not None:
            ss.exam_id = ss.exam_prev; ss.exam_blocked = True; return
        ss.exam_prev, ss.exam_blocked = ss.exam_id, False
        ss.tutor, ss.messages, ss.recorded = None, [], False
        ss.diag, ss.diag_msgs, ss.diag_report = None, [], None
        ss.exam_out, ss.doubt, ss.walk, ss.directory, ss.syn_snaps = None, None, None, None, {}

    st.selectbox("Exam", REG.ids(), format_func=lambda i: REG.get(i).spec.name, key="exam_id", on_change=on_exam_change)
    if ss.pop("exam_blocked", False):
        st.warning("Finish or submit the running exam before switching exams.")
    src = module.spec.source
    st.caption(f"Pattern checked {src.verified_on} · confidence: {src.confidence}")
    mode = st.radio("Mode", ["Practice", "Diagnostic", "Mock exam", "Doubt solver", "Check-in", "Peer standing", "Dashboards", "Study plan"], key="mode")
    st.header("🧠 Knowledge map")
    m, unlocked = student.mastery_vector(), set(student.testable_concepts())
    for cid in graph.topological_order():
        st.caption(f"{'🔓' if cid in unlocked else '🔒'} {graph.name(cid)}")
        st.progress(min(1.0, m[cid]))
    st.info("Running on a **mock LLM**." if use_mock else "Running on the real LLM.")
    if ss.exam is not None and mode != "Mock exam":
        st.warning("⏱️ An exam is in progress. The clock keeps running: open 'Mock exam'.")
    if st.button("Reset progress"):
        if ss.get("ctx"):
            for kind in ["engagement", "exam_hist", "exam_session"] + [model_kind(e) for e in REG.ids()]:
                services()[0].delete_doc(ss.ctx.org, ss.ctx.user_id, kind)
        fresh_state(keep=("token", "ctx", "exam_id")); st.rerun()

st.title("📘 AI Study Companion")
if ss.support is not None:
    st.warning(ss.support.text)
    if st.button("Close this message"):
        ss.support = None; st.rerun()


def intercept(msg: str, sink: list) -> bool:
    _, sup = handle_student_message(ss.eng, msg, time.time())
    if sup is None:
        return False
    sink.append(("user", msg)); sink.append(("assistant", sup.text))
    if sup.level == "acute":
        ss.support = sup
    return True


def new_question():
    nxt = student.next_question()
    if nxt is None:
        ss.tutor = None; return
    q, p = nxt
    ss.tutor, ss.recorded, ss.q_started = TutorSession(q, ss.llm), False, time.time()
    flag = "⚠️ _Unreviewed question: written without expert review, so it may contain mistakes._\n\n" if q.get("trust") == "unreviewed" else ""
    ss.messages.append(("assistant", f"**{graph.name(q['concept'])}** (est. success chance {p:.0%})\n\n{flag}{ss.tutor.opening().text}"))


if mode == "Practice":
    st.caption("I'll guide you with hints instead of handing over answers. Type 'show me' twice for the solution.")
    if ss.tutor is None:
        new_question()
    tutor = ss.tutor
    for role, text in ss.messages:
        st.chat_message(role).markdown(text)
    if tutor is not None and tutor.status == "asking":
        if msg := st.chat_input("Your answer or question..."):
            if intercept(msg, ss.messages):
                st.rerun()
            ss.messages.append(("user", msg))
            reply = tutor.respond(msg)
            ss.messages.append(("assistant", reply.text))
            if reply.status != "asking" and not ss.recorded:
                student.update(tutor.q["concept"], reply.status == "correct", hints_used=tutor.hints_used)
                record(ss.eng, "attempt", time.time(), min(5.0, (time.time() - ss.q_started) / 60), reply.status == "correct")
                cs = student.state(tutor.q["concept"])
                ss.peers.add("me", "demo", ss.exam_id, tutor.q["concept"], time.time(), cs.correct / cs.attempts, cs.attempts)
                ss.recorded = True
            st.rerun()
    elif tutor is not None:
        if st.button("➡️ Next question", type="primary"):
            new_question(); st.rerun()
    else:
        st.success("You've worked through everything available. 🎉")

elif mode == "Diagnostic":
    st.caption("A short adaptive quiz (max 8 questions, no hints) to map what you already know. "
               "Each answer lets me infer related topics, so it adapts as you go.")
    if ss.diag is None and ss.diag_report is None:
        if st.button("▶️ Start diagnostic", type="primary"):
            ss.diag, ss.diag_msgs = DiagnosticSession(graph, bank), []
            q = ss.diag.next_question()
            ss.diag_msgs.append(("assistant", f"**Q1.** {render_prompt(q)}")); st.rerun()
    for role, text in ss.diag_msgs:
        st.chat_message(role).markdown(text)
    if ss.diag is not None:
        if ans := st.chat_input("Your answer (no hints in the diagnostic)..."):
            if intercept(ans, ss.diag_msgs):
                st.rerun()
            ok = ss.diag.submit(ans)
            ss.diag_msgs.append(("user", ans))
            nxt = ss.diag.next_question()
            if nxt is None:
                ss.diag.apply(student, time.time())
                ss.diag_report, ss.diag = ss.diag.report(), None
                ss.diag_msgs.append(("assistant", "Diagnostic complete - see your knowledge map below."))
            else:
                ss.diag_msgs.append(("assistant", f"**Q{len(ss.diag.tested) + 1}.** {render_prompt(nxt)}"))
            st.rerun()
    if ss.diag_report:
        r = ss.diag_report
        st.subheader("Your knowledge map")
        st.caption(f"Based on {r['questions_asked']} questions. 'Inferred' topics are estimates, not tested directly.")
        for c in graph.topological_order():
            icon = "🟢" if c in r["strong"] else "🔴" if c in r["weak"] else "⚪"
            st.write(f"{icon} {graph.name(c)} — _{r['evidence'][c]}_")
        for w in r["inconsistencies"]:
            st.warning(f"Inconsistent result (slip or guess?): {w}")


elif mode == "Mock exam":
    spec = module.spec

    def finish():
        ex = ss.exam
        if not ex.is_submitted:
            ex.submit()
        result = ex.grade()
        analysis = analyze(result, ex.by_id, ss.exam_mastery)
        drills = apply_feedback(student, analysis)
        frac = max(0.0, min(1.0, result.score / result.max_score)) if result.max_score else 0.0
        record(ss.eng, "exam", time.time(), result.time_used / 60, score_frac=frac)
        ss.exam_hist.append({"ts": time.time(), "exam": ex.exam_id, "score_frac": frac,
                             "accuracy": (result.correct / result.attempted) if result.attempted else 0.0, "counts": analysis.counts})
        ss.peers.add("me", "demo", ex.exam_id, "overall", time.time(), difficulty_adjusted(result), len(result.questions))
        ss.exam_out, ss.exam = (result, analysis, drills, ex.section_scores(result), ex.exam_id), None

    def goto_cb():
        try:
            ss.exam.goto(ss.qsel)
        except ExamClosed:
            pass

    def sec_cb():
        sec = next(x for x in ss.exam.layout if x["id"] == ss.secsel)
        ss.qsel = sec["qids"][0]
        goto_cb()

    def save_cb(qid):
        raw = ss.get(f"ans_{qid}", "")
        if ss.exam.by_id[qid].get("type") == "mcq":
            txt = raw[:1] if raw else ""
        else:
            _, sup = handle_student_message(ss.eng, raw, time.time())
            if sup is not None:
                ss.support = sup if sup.level == "acute" else ss.support
                return
            txt = raw
        try:
            ss.exam.answer(qid, txt)
        except ExamClosed:
            pass

    if ss.exam is None and ss.exam_out is None:
        src = spec.source
        st.subheader(spec.name)
        (st.success if src.confidence == "high" else st.info if src.confidence == "medium" else st.warning)(
            f"Pattern checked {src.verified_on} · confidence: **{src.confidence}**. {src.notes}")
        for c in src.conflicts:
            st.warning("Sources disagree: " + c)
        if spec.gradable_note:
            st.info(spec.gradable_note)
        st.write(f"Duration {spec.duration_min:g} min · {spec.max_marks():g} marks" + (" · **timed sections are locked in order**" if spec.sectional_timing else ""))
        st.dataframe([{"Section": x.name, "Questions": f"{x.n_questions}" + (f" (any {x.n_attempt} count)" if x.n_attempt else ""),
                       "Marking": ", ".join(f"{t}: +{m.correct:g} / -{m.wrong:g}" for t, m in x.marking.items()),
                       "Time limit": f"{x.time_limit_min:g} min" if x.time_limit_min else "-"} for x in spec.sections], hide_index=True, width="stretch")
        st.caption("How much of the real paper this question bank can fill (starter content: expand it before relying on it):")
        st.dataframe([{"Section": r["name"], "Needed": r["needed"], "Available": r["available"], "Unreviewed": r["unreviewed"], "Fill %": r["fill_pct"]}
                      for r in module.coverage()], hide_index=True, width="stretch")
        sizes = {"Full paper": 1.0, "Half": 0.5, "Quarter": 0.25, "Mini (~10%)": 0.1}
        scale = sizes[st.selectbox("Length", list(sizes), help="Shorter papers shrink the question counts and the clocks together.")]
        full = st.checkbox("Include all topics (full syllabus), even ones I haven't unlocked yet")
        unrev = st.checkbox("Include unreviewed questions (not checked by an expert)", value=True)
        if st.button("▶️ Start exam", type="primary"):
            ss.exam_mastery = student.mastery_vector()
            paper, layout, dur, ss.exam_notes = assemble_exam(module, ss.exam_mastery, seed=int(time.time()), scale=scale,
                                                              full_syllabus=full, include_unreviewed=unrev)
            if not paper:
                st.error("No questions are available with these settings. " + " ".join(ss.exam_notes))
            else:
                ex = SectionedExam(paper, layout, spec.id, spec.name, dur, spec.sectional_timing)
                ex.start(); first = layout[0]["qids"][0]; ex.goto(first)
                ss.exam, ss.qsel, ss.secsel = ex, first, layout[0]["id"]
                st.rerun()
    elif ss.exam is not None:
        ex = ss.exam
        if ex.expired or ex.is_submitted:
            finish(); st.rerun()
        for n in ss.exam_notes:
            st.info(n)
        sec = ex.current_section()
        if ex.sectional:
            st.subheader(f"{sec['name']}  (section {ex.sec_idx + 1} of {len(ex.layout)})")
            st.caption("Sections are locked in order. Leaving one early forfeits its remaining time, and you can't go back.")
        else:
            ids_all = [x["id"] for x in ex.layout]
            if ss.get("secsel") not in ids_all:
                ss.secsel = ids_all[0]
            st.selectbox("Section", ids_all, key="secsel", on_change=sec_cb, format_func=lambda i: next(
                f"{x['name']}  ({sum(q in ex.responses for q in x['qids'])}/{len(x['qids'])} answered)" for x in ex.layout if x["id"] == i))
            sec = next(x for x in ex.layout if x["id"] == ss.secsel)
        if ss.get("qsel") not in sec["qids"]:
            ss.qsel = sec["qids"][0]
            try:
                ex.goto(ss.qsel)
            except ExamClosed:
                pass
        shown = sec["id"]

        @st.fragment(run_every=1)
        def timer():
            r = ex.remaining()
            st.metric("⏱️ Section time left" if ex.sectional else "⏱️ Time left", f"{int(r // 60):02d}:{int(r % 60):02d}")
            if ex.sectional:
                st.caption(f"If you use every section's full time: {int(ex.total_remaining() // 60)} min left in total")
                now_sec = ex.current_section()
                if now_sec is None or now_sec["id"] != shown:
                    st.rerun()
            elif r <= 0:
                st.rerun()
        timer()

        if sec["n_attempt"]:
            done_n = sum(q in ex.responses for q in sec["qids"])
            (st.warning if done_n > sec["n_attempt"] else st.info)(
                f"Answer any {sec['n_attempt']}: only the first {sec['n_attempt']} you answer are marked ({done_n} answered so far)."
                + (" Extra answers won't count." if done_n > sec["n_attempt"] else ""))
        qids = sec["qids"]
        label = lambda i: f"Q{qids.index(i) + 1}  {'✅' if i in ex.responses else '⬜'}{' 🚩' if i in ex.flags else ''}"
        st.selectbox("Question", qids, format_func=label, key="qsel", on_change=goto_cb)
        q = ex.by_id[ss.qsel]
        st.markdown(f"**Q{qids.index(q['id']) + 1}.** {q['prompt']}" + ("  ⚠️ _unreviewed_" if q.get("trust") == "unreviewed" else ""))
        if q.get("type") == "mcq":
            labels = [f"{LETTERS[i]}) {o}" for i, o in enumerate(q["options"])]
            cur = (ex.responses.get(q["id"]) or "").upper()
            st.radio("Your answer", labels, index=LETTERS.index(cur) if cur and cur in LETTERS[:len(labels)] else None,
                     key=f"ans_{q['id']}", on_change=save_cb, args=(q["id"],))
            if st.button("Clear answer"):
                ex.answer(q["id"], ""); ss.pop(f"ans_{q['id']}", None); st.rerun()
        else:
            st.text_input("Your answer", value=ex.responses.get(q["id"], ""), key=f"ans_{q['id']}", on_change=save_cb, args=(q["id"],))
        if st.button("🚩 Mark / unmark for review"):
            ex.toggle_flag(q["id"]); st.rerun()
        st.divider()
        if ex.sectional and ex.sec_idx < len(ex.layout) - 1:
            sure = st.checkbox("I'm done with this section and understand I can't come back.", key=f"sure_{ex.sec_idx}")
            if st.button("➡️ Finish this section and start the next", disabled=not sure):
                ex.next_section(); st.rerun()
        if st.button("✅ Submit exam" if not ex.sectional else "✅ Finish the whole exam", type="primary"):
            finish(); st.rerun()
    else:
        result, analysis, drills, sec_scores, eid = ss.exam_out
        espec = REG.get(eid).spec
        st.subheader(f"Score: {result.score:g} / {result.max_score:g}" + ("  (auto-submitted: time ran out)" if result.auto_submitted else ""))
        qual = qualifies(espec, result.score, result.max_score)
        if qual is not None:
            (st.success if qual else st.error)(f"Qualifying bar {espec.qualifying['min_fraction']:.0%}: " + ("reached" if qual else "NOT reached"))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Correct", result.correct); k2.metric("Wrong", result.wrong); k3.metric("Skipped", result.skipped)
        k4.metric("Time used", f"{result.time_used / 60:.1f} min")
        ignored = sum(1 for r in result.questions if r.response is not None and not r.attempted)
        if ignored:
            st.warning(f"{ignored} answer(s) didn't count because of the 'answer any N' rule.")
        st.subheader("Marks by section")
        st.dataframe([{"Section": v["name"], "Marks": v["score"], "Correct": v["correct"], "Attempted": v["attempted"], "Of": v["of"]} for v in sec_scores.values()],
                     hide_index=True, width="stretch")
        st.subheader("Where the marks went")
        st.bar_chart({k: v for k, v in analysis.counts.items() if v})
        for p in analysis.patterns:
            st.write("•", p)
        st.subheader("Question by question")
        st.dataframe([{"Topic": graph.name(l.concept), "Result": l.category.replace("_", " "), "Confidence": l.confidence,
                       "Why": l.reason, "Time (s)": round(r.time_spent)} for l, r in zip(analysis.labels, result.questions)],
                     hide_index=True, width="stretch")
        st.caption("Labels are best guesses from timing, mastery and answer shape - e.g. a fast wrong answer could be a slip or a guess.")
        if analysis.recommendations:
            st.subheader("What to do next")
            for c, cat, action in analysis.recommendations:
                st.write(f"**{graph.name(c)}** ({cat.replace('_', ' ')}): {action}")
        st.success("Your knowledge map and practice queue have been updated" + (f"; drills added for {len(drills)} topic(s)." if drills else "."))
        if st.button("New exam"):
            ss.exam_out = None; st.rerun()

elif mode == "Doubt solver":
    st.caption("Photograph a textbook or exam problem. I'll read it, ask you to confirm the reading, then solve it "
               "and check the answer with a separate maths engine before walking you through it.")
    if use_mock:
        st.info("**Mock reader:** without an API key I can't actually read pixels. Type what the photo says below to "
                "simulate the reading step. Everything after that (checks, confirmation, solving, verification) is real.")
        sim_text = st.text_input("What the photo says (simulated reading)", "Solve 3x^2 - 5x - 2 = 0")
        sim_conf = st.slider("Reader confidence (simulated)", 0.30, 1.0, 0.95, 0.05)
        sim_diag = st.checkbox("Pretend the photo contains a diagram")
    up = st.file_uploader("Upload a photo (PNG/JPG, max 10 MB)", type=["png", "jpg", "jpeg"])
    if st.button("Use a sample photo instead"):
        ss.doubt_img = make_sample_image(sim_text if use_mock else "Solve x^2 - 5x + 6 = 0")
    img = up.getvalue() if up else ss.doubt_img
    if img and st.button("🔍 Read the photo", type="primary"):
        try:
            if use_mock:
                d = DoubtSession(MockVision(sim_text, sim_conf, sim_diag))
            else:
                d = DoubtSession(AnthropicVision(), AnthropicSolver())
            d.submit_image(img)
            ss.doubt, ss.walk, ss.doubt_tagged = d, None, None
        except ValueError as e:
            st.error(str(e))
    d: DoubtSession | None = ss.doubt
    if d is not None:
        for w in d.report.warnings:
            st.warning("📷 " + w)
        st.subheader("Here's what I read")
        for r in d.reasons_to_confirm():
            st.info(r)
        edited = st.text_area("Please check - fix anything I got wrong:", d.text, height=80)
        if d.parsed:
            st.caption("As equations I'll solve: " + "  |  ".join(p.text for p in d.parsed))
        mode_pick = st.radio("How should I explain?", ["Guided (think first, then reveal)", "Full solution"], horizontal=True)
        if st.button("✅ That's right - solve it"):
            d.confirm(edited if edited.strip() != d.text else None)
            sol = d.solve()
            ss.walk = Walkthrough(sol.steps, "guided" if mode_pick.startswith("Guided") else "full")
            ss.doubt_tagged = d.record(student, graph) if sol.status != "unsupported" else None
        if ss.walk is not None and d.solution is not None:
            sol, w = d.solution, ss.walk
            st.subheader(sol.trust_label)
            for note in sol.warnings:
                st.caption(note)
            if sol.status == "unsupported":
                st.error("I can solve linear equations, quadratics and pairs of linear equations so far.")
            else:
                for i, step in enumerate(w.visible(), 1):
                    st.markdown(f"**Step {i}.** _{step['ask']}_  \n{step['show']}")
                if not w.done:
                    st.markdown(f"🤔 **Think:** {w.next_prompt()}")
                    c1, c2 = st.columns(2)
                    if c1.button("Show this step"):
                        w.reveal_next(); st.rerun()
                    if c2.button("Show the whole solution"):
                        while not w.done: w.reveal_next()
                        st.rerun()
                else:
                    st.success(f"**Answer:** {sol.answer_text}")
                if ss.doubt_tagged:
                    st.caption(f"📌 Added **{ss.doubt_tagged}** to your practice queue.")


elif mode == "Check-in":
    eng, now = ss.eng, time.time()
    stress = current_stress(eng, now)
    flags = ui_flags(stress)
    st.caption("Your goals, progress, and a place to tell me how prep is really going.")
    cur, longest = streak(eng, now)
    done = minutes_on(eng, local(eng, now).date())
    n_att = sum(1 for a in eng.log if a.kind == "attempt")
    c1, c2, c3 = st.columns(3)
    if flags["show_streaks"]:
        c1.metric("🔥 Streak", f"{cur} day{'s' if cur != 1 else ''}", f"best {longest}", delta_color="off")
    else:
        c1.metric("📅 Days practised", len(active_days(eng)))
    c2.metric("Today", f"{done:.0f} / {eng.daily_goal_min} min")
    c3.metric("Questions tried", n_att)
    st.progress(min(1.0, done / eng.daily_goal_min))

    m = student.mastery_vector()
    testable = student.testable_concepts()
    topic = graph.name(min(testable, key=lambda c: m[c])) if testable else "your next topic"
    ci = pick_checkin(eng, now, plan_minutes=eng.daily_goal_min, topic=topic, push=False)
    if ci:
        st.chat_message("assistant").markdown(ci.text)
    new = award_new(eng, now, m, {c: graph.name(c) for c in graph.g})
    ss.milestones += new
    for ms in new:
        st.success(f"🏅 **{ms.title}** - {ms.message}")
    if ss.milestones:
        with st.expander(f"Your milestones ({len(ss.milestones)})"):
            for ms in ss.milestones:
                st.write(f"🏅 **{ms.title}**: {ms.message}")

    st.subheader("How's your prep going?")
    for role, text in ss.checkin_msgs:
        st.chat_message(role).markdown(text)
    if msg := st.chat_input("Tell me how you're feeling about your prep..."):
        lvl, reply = reply_to_checkin_message(eng, msg, now)
        ss.checkin_msgs += [("user", msg), ("assistant", reply)]
        if lvl == "acute":
            ss.support = None
        st.rerun()

    with st.expander("Settings and your data"):
        eng.tone = st.selectbox("Message style", ["gentle", "energetic", "direct"], index=["gentle", "energetic", "direct"].index(eng.tone))
        eng.daily_goal_min = st.slider("Daily goal (minutes)", 10, 180, eng.daily_goal_min, 5)
        eng.is_minor = st.checkbox("I'm under 18", value=eng.is_minor)
        st.caption("I keep timestamps and counts to run check-ins. I never store the words you type when you're stressed - "
                   "only that a hard moment happened, so I can go easier on you for a few days.")
        if st.button("Delete my check-in data"):
            eng.erase(); ss.milestones, ss.checkin_msgs = [], []; st.rerun()
    with st.expander("Need to talk to someone?"):
        st.write("**Tele-MANAS** (free, 24x7, many languages): **14416** or **1-800-891-4416**. "
                 "If you're in immediate danger, call **112**.")


elif mode == "Peer standing":
    now = time.time()
    st.caption("Optional and anonymous. No names, no leaderboard. I always show your own progress first, and "
               "comparisons switch off automatically if things feel heavy.")
    consent = ss.peers.has_consent("me")
    st.write("**Status:** " + ("✅ you're included in anonymous comparisons" if consent else "⚪ comparisons are off"))
    want = st.checkbox("Include me in anonymous comparisons", value=consent)
    guardian = st.checkbox("My parent/guardian agrees (required under 18)", value=False) if ss.eng.is_minor else True
    if st.button("Save my choice"):
        try:
            ss.peers.set_consent("me", want, is_minor=ss.eng.is_minor, guardian_ok=guardian)
            st.success("Saved." if want else "Done. Your comparison data has been deleted.")
            st.rerun()
        except ConsentError as e:
            st.error(str(e))

    scope_names = {"overall": "Overall (mock exams)", **{c: graph.name(c) for c in graph.topological_order()}}
    scope = st.selectbox("Compare on", list(scope_names), format_func=scope_names.get)
    demo = st.checkbox("Use a synthetic demo cohort (for trying this screen; no real students exist yet)", value=True)
    exam_key = ss.exam_id
    if demo and (exam_key, scope) not in ss.syn_done:
        ss.peers.add_synthetic("demo", exam_key, scope, now, n=60, mean=0.55, seed=sum(map(ord, scope)))
        ss.syn_done.add((exam_key, scope))
    m_, testable_ = student.mastery_vector(), student.testable_concepts()
    nxt = graph.name(min(testable_, key=lambda c: m_[c])) if testable_ else None
    res = ss.peers.standing("me", "demo", exam_key, scope, now, stress_level=current_stress(ss.eng, now),
                            include_synthetic=demo, next_topic=nxt)
    if res.status != "ok":
        st.info(res.message)
        if res.delta_points is not None:
            st.metric("Change vs your last period", f"{res.delta_points:+.1f} pts")
    else:
        st.subheader("Your progress first")
        if res.delta_points is not None:
            st.metric("Change vs your last period", f"{res.delta_points:+.1f} pts")
        else:
            st.caption("Not enough history yet to compare with your own past period.")
        st.subheader("Compared with others (anonymous)")
        st.metric("Your group", res.band.title(), f"about the {res.percentile}th percentile", delta_color="off")
        st.progress(res.percentile / 100)
        lo, med, hi = (round(x * 100) for x in res.quartiles)
        st.caption(f"Typical scores in this group: {lo}%-{hi}% (middle: {med}%), based on {res.cohort_size} students.")
        st.write(res.message)
        if res.synthetic:
            st.warning("Demo data: this comparison uses a synthetic cohort, not real students.")
    with st.expander("What's stored and who sees it"):
        st.write("Only a scrambled ID, the exam/topic, a date and a score are stored. Nobody can see your name or "
                 "your individual scores: only your own group band and percentile, shown once at least 30 students "
                 "have joined. Leaving deletes your comparison data. Under-18s need a parent/guardian's agreement.")


elif mode == "Dashboards":
    today, now = date.today(), time.time()
    if ss.directory is None:
        d = Directory({"demo": Policy(), "other": Policy()})
        d.add_student(StudentRecord("me", "demo", "morning", "You (Student me)", exam_date=today + timedelta(days=30)))
        for i in range(1, 10):
            d.add_student(StudentRecord(f"d{i}", "demo", "morning" if i % 2 else "evening", f"Demo student {i}", exam_date=today + timedelta(days=30)))
            ss.syn_snaps[f"d{i}"] = synthetic_snapshot(i, graph, today)
        for i in range(1, 5):
            d.add_student(StudentRecord(f"o{i}", "other", "morning", f"Other-academy student {i}", exam_date=today + timedelta(days=30)))
            ss.syn_snaps[f"o{i}"] = synthetic_snapshot(i + 20, graph, today)
        for u in [User("parent_me", "parent", "demo"), User("coach_demo", "coach", "demo"),
                  User("coach_morning", "coach", "demo", frozenset({"morning"})), User("coach_other", "coach", "other"),
                  User("me", "student", "demo")]:
            d.add_user(u)
        d.link_parent("parent_me", "me")
        ss.directory = d
    d: Directory = ss.directory
    snaps = {**ss.syn_snaps, "me": build_snapshot("me", student, ss.eng, ss.exam_hist, graph, now)}

    st.caption("Demo: you are 'Student me' in 'Demo Academy' with 9 synthetic classmates. Switch roles to see exactly what each person is allowed to see.")
    viewer_label = st.radio("View as", ["Parent (of you)", "Coach - Demo Academy (all batches)", "Coach - Demo Academy (morning batch only)",
                                        "Coach - Other Academy", "Student (you)"])
    viewer = {"Parent (of you)": "parent_me", "Coach - Demo Academy (all batches)": "coach_demo",
              "Coach - Demo Academy (morning batch only)": "coach_morning", "Coach - Other Academy": "coach_other",
              "Student (you)": "me"}[viewer_label]
    with st.expander("Organisation privacy settings (demo)"):
        p_note = st.checkbox("Parents may see a neutral 'heavy week' note", value=d.policies["demo"].parent_sees_wellbeing_note)
        c_flag = st.checkbox("Coaches see content-free 'please check in personally' flags", value=d.policies["demo"].coach_sees_safeguarding_flags)
        d.policies["demo"] = Policy(parent_sees_wellbeing_note=p_note, coach_sees_safeguarding_flags=c_flag)
        st.caption("Chat messages and anything typed while stressed are never shown to parents or coaches, whatever these settings are.")

    ICON = {"strong": "🟢 strong", "developing": "🟡 developing", "needs attention": "🔴 needs attention", "not started": "⚪ not started"}
    RISK = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟡 low", "none": "🟢 none"}
    heat_rows = lambda hm: [{"Topic": h["name"], "Status": ICON[h["band"]]} for h in hm]

    def show_student(v: dict, parent: bool):
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated readiness", f"{v['readiness_pct']}%", help="An estimate from practice so far, not a grade.")
        if "days_practised_this_week" in v:
            c2.metric("Days practised this week", v["days_practised_this_week"])
        if v.get("days_to_exam") is not None:
            c3.metric("Days to exam", v["days_to_exam"])
        st.subheader("Topic strengths"); st.dataframe(heat_rows(v["heatmap"]), hide_index=True, width="stretch")
        st.subheader("Minutes studied (last 7 days)")
        st.bar_chart({"minutes": {day[5:]: m for day, m in v["minutes_last_7_days"]}})
        mt = v["mock_trend"]
        st.subheader(f"Mock exams: {mt['direction']}")
        if mt["points"]:
            st.line_chart({"score %": {p["date"][5:]: p["score_pct"] for p in mt["points"]}})
        if v.get("weak_areas"):
            st.write("**Focus areas:** " + ", ".join(v["weak_areas"]))
        for line in v.get("mistake_patterns", []):
            st.write("•", line)

    try:
        if viewer == "parent_me":
            v = parent_view(d, viewer, "me", snaps, today, now)
            st.header(f"Progress report: {v['student']}")
            if "wellbeing_note" in v:
                st.info(v["wellbeing_note"])
            show_student(v, True)
            st.subheader("How you can help")
            for t in v["tips"]:
                st.write("•", t)
            with st.expander("What you can and can't see"):
                st.write("**You can see:**"); [st.write("•", x) for x in v["what_you_can_see"]]
                st.write("**You can't see:**"); [st.write("•", x) for x in v["what_you_cannot_see"]]
        elif viewer == "me":
            st.header("Your own dashboard"); show_student(student_view(d, viewer, "me", snaps, today, now), False)
        else:
            v = coach_view(d, viewer, snaps, today, now)
            st.header(f"Batch overview: {d.users[viewer].org} ({v['n_students']} students)")
            k = st.columns(4)
            k[0].metric("Active in last 7 days", f"{v['active_last_7_days_pct']}%"); k[1].metric("Avg readiness", f"{v['avg_readiness_pct']}%")
            k[2].metric("High risk", v["risk_counts"]["high"]); k[3].metric("Medium risk", v["risk_counts"]["medium"])
            st.subheader("Students (most at-risk first)")
            st.dataframe([{"Student": r["name"], "Batch": r["batch"], "Risk": RISK[r["risk"]], "Readiness %": r["readiness_pct"],
                           "Min (7d)": r["minutes_last_7_days"], "Days inactive": r["last_active_days_ago"], "Latest mock %": r["latest_mock_pct"]}
                          for r in v["students"]], hide_index=True, width="stretch")
            st.subheader("Alerts")
            for a in v["alerts"][:12]:
                (st.error if a["severity"] == 3 else st.warning if a["severity"] == 2 else st.info)(f"**{a['name']}**: {a['summary']}")
            st.subheader("Where the batch is strong")
            st.bar_chart(v["share_strong_by_topic_pct"])
            st.caption("Weakest topics across the batch: " + ", ".join(v["weakest_topics"]))
            ids = {r["name"]: r["student_id"] for r in v["students"]}
            if ids:
                pick = st.selectbox("Open a student", list(ids))
                det = coach_student_detail(d, viewer, ids[pick], snaps, today, now)
                st.subheader(det["student"] + f"  ·  risk: {RISK[det['risk']]}"); show_student(det, False)
            st.divider()
            st.write("**Isolation check:** try opening a student that is outside your access.")
            target = "o1" if d.users[viewer].org == "demo" else "me"
            if st.button(f"Try to open '{d.students[target].name}'"):
                try:
                    coach_student_detail(d, viewer, target, snaps, today, now)
                except AccessError as e:
                    st.error(f"Denied: {e}")
            if viewer == "coach_morning" and st.button("Try to open an evening-batch student (d2)"):
                try:
                    coach_student_detail(d, viewer, "d2", snaps, today, now)
                except AccessError as e:
                    st.error(f"Denied: {e}")
    except AccessError as e:
        st.error(str(e))
    with st.expander("Access log (who looked at what)"):
        st.dataframe([{"viewer": e.viewer, "role": e.role, "student": e.student, "action": e.action, "allowed": e.allowed, "reason": e.reason}
                      for e in d.audit[-15:]], hide_index=True, width="stretch")

else:
    st.caption("A schedule built from your current knowledge map: weak topics first, in prerequisite order, "
               "with spaced reviews, weekly mock tests and a daily buffer.")
    c1, c2 = st.columns(2)
    exam = c1.date_input("Exam date", date.today() + timedelta(days=30))
    lvl = current_stress(ss.eng, time.time())
    default_h = suggest_hours(1.5, lvl)
    if default_h != 1.5:
        st.info(f"You've had a heavy stretch lately, so I've set a lighter default ({default_h:g} h/day). Change it any time.")
    hours = c2.slider("Study hours per day", 0.5, 6.0, default_h, 0.25)
    if not ss.diag_report:
        st.warning("You haven't taken the diagnostic - the plan assumes you're starting from scratch.")
    try:
        plan = build_plan(graph, student.mastery_vector(), date.today(), exam, hours, drills=list(student.focus))
    except ValueError as e:
        st.error(str(e)); st.stop()
    for n in plan.notes:
        (st.error if "Not enough" in n else st.caption)(n)
    if plan.uncovered:
        st.write("Not covered:", ", ".join(graph.name(c) for c in plan.uncovered))
    rows = [{"Date": str(i.day), "Type": i.kind, "Topic": graph.name(i.concept) if i.concept else "Mock test",
             "Minutes": i.minutes} for i in plan.items]
    today_rows = [r for r in rows if r["Date"] == str(date.today())]
    st.subheader("Today"); st.dataframe(today_rows or [{"Date": "-", "Type": "-", "Topic": "Nothing scheduled", "Minutes": 0}],
                                        hide_index=True, width='stretch')
    st.subheader("Full plan"); st.dataframe(rows, hide_index=True, width='stretch')

persist()

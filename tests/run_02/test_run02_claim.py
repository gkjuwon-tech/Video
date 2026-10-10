"""pending 규칙·순서, claim(상태 형식·리스·경쟁·만료·소진)."""
import json

from run02_fakes import T0, job_yaml, spec_of, state_doc


def put_job(origin, text, stem=None):
    s = spec_of(text)
    origin.put(f"jobs/{stem or s.id}.yaml", text)
    return s


def ids(specs):
    return [s.id for s in specs]


def test_pending_order_enabled_and_invalid(origin, make):
    put_job(origin, job_yaml("b", priority=1))
    put_job(origin, job_yaml("a"))
    put_job(origin, job_yaml("c", priority=1))
    put_job(origin, job_yaml("disabled", enabled=False))
    origin.put("jobs/broken.yaml", "id: broken\nkind: nope\ncmd: [x]\n")
    origin.put("jobs/wrongstem.yaml", job_yaml("other"))
    origin.put("jobs/notes.txt", "not a job")
    r, rem, _ = make()
    assert ids(r.pending()) == ["b", "c", "a"]
    assert rem.syncs >= 1


def test_pending_state_rules(origin, make, clock):
    sd = put_job(origin, job_yaml("done1"))
    sn = put_job(origin, job_yaml("newrev"))
    sf = put_job(origin, job_yaml("failed1"))
    sr = put_job(origin, job_yaml("retry1"))
    sl = put_job(origin, job_yaml("live"))
    se = put_job(origin, job_yaml("expired"))
    so = put_job(origin, job_yaml("own"))
    origin.put("runs/jobs/done1/state.json", state_doc("done1", sd.spec_hash, "done", 1, "x", None))
    origin.put("runs/jobs/newrev/state.json", state_doc("newrev", "0" * 64, "done", 1, "x", None))
    origin.put("runs/jobs/failed1/state.json", state_doc("failed1", sf.spec_hash, "failed", 3, "x", None))
    origin.put("runs/jobs/retry1/state.json", state_doc("retry1", sr.spec_hash, "retry", 1, "x", None))
    origin.put("runs/jobs/live/state.json", state_doc("live", sl.spec_hash, "running", 1, "other", T0 + 100))
    origin.put("runs/jobs/expired/state.json", state_doc("expired", se.spec_hash, "running", 1, "other", T0))
    origin.put("runs/jobs/own/state.json", state_doc("own", so.spec_hash, "running", 1, "r1", T0 + 500))
    r, _, _ = make()
    assert ids(r.pending()) == ["expired", "newrev", "own", "retry1"]
    assert sn.id == "newrev"


def test_claim_writes_running_state(origin, make, clock):
    s = put_job(origin, job_yaml("j1"))
    r, _, _ = make(lease_s=600.0)
    assert ids(r.pending()) == ["j1"]
    assert r.claim(s) == 1
    st = origin.json("runs/jobs/j1/state.json")
    assert st["schema"] == "buc.job_state/v1"
    assert st["job_id"] == "j1" and st["spec_hash"] == s.spec_hash
    assert st["status"] == "running" and st["attempt"] == 1 and st["runner_id"] == "r1"
    assert abs(st["lease_until"] - (T0 + 600.0)) < 1e-6
    assert abs(st["updated_at"] - T0) < 1e-6
    assert r.last_activity is not None and abs(r.last_activity - T0) < 1e-6


def test_claim_attempt_numbers(origin, make):
    sr = put_job(origin, job_yaml("retry1"))
    se = put_job(origin, job_yaml("expired"))
    sn = put_job(origin, job_yaml("newrev"))
    origin.put("runs/jobs/retry1/state.json", state_doc("retry1", sr.spec_hash, "retry", 2, "x", None))
    origin.put("runs/jobs/expired/state.json", state_doc("expired", se.spec_hash, "running", 1, "o", T0 - 1))
    origin.put("runs/jobs/newrev/state.json", state_doc("newrev", "f" * 64, "retry", 2, "o", None))
    r, _, _ = make()
    r.pending()
    assert r.claim(sr) == 3
    assert r.claim(se) == 2
    assert r.claim(sn) == 1
    st = origin.json("runs/jobs/expired/state.json")
    assert st["runner_id"] == "r1" and st["attempt"] == 2 and st["status"] == "running"


def test_claim_loses_push_race(origin, make, clock):
    s = put_job(origin, job_yaml("j1"))
    r, rem, ex = make()
    r.pending()

    def rival(files):
        origin.put("runs/jobs/j1/state.json", state_doc("j1", s.spec_hash, "running", 1, "rival", T0 + 900))
    rem.before_publish = rival
    assert r.claim(s) is None
    st = origin.json("runs/jobs/j1/state.json")
    assert st["runner_id"] == "rival"
    assert rem.conflicts >= 1


def test_run_once_skips_lost_race_without_executing(origin, make):
    s = put_job(origin, job_yaml("j1"))
    r, rem, ex = make()

    def rival(files):
        origin.put("runs/jobs/j1/state.json", state_doc("j1", s.spec_hash, "running", 1, "rival", T0 + 900))
    rem.before_publish = rival
    assert r.run_once() is None
    assert ex.calls == []
    assert "runs/jobs/j1/result.json" not in origin.files


def test_conflict_from_unrelated_commit_still_claims(origin, make):
    s = put_job(origin, job_yaml("j1"))
    r, rem, _ = make()
    r.pending()
    rem.before_publish = lambda files: origin.put("README.md", "someone else's commit")
    assert r.claim(s) == 1
    assert origin.json("runs/jobs/j1/state.json")["runner_id"] == "r1"
    assert origin.files["README.md"] == b"someone else's commit"


def test_exhausted_attempts_marked_failed_without_running(origin, make):
    s = put_job(origin, job_yaml("j1", max_attempts=3))
    origin.put("runs/jobs/j1/state.json", state_doc("j1", s.spec_hash, "running", 3, "dead", T0 - 10))
    r, _, ex = make()
    assert r.run_once() is None
    assert ex.calls == []
    st = origin.json("runs/jobs/j1/state.json")
    assert st["status"] == "failed" and st["error"] == "attempts_exhausted"
    assert ids(r.pending()) == []


def test_state_json_is_sorted_utf8(origin, make):
    s = put_job(origin, job_yaml("j1"))
    r, _, _ = make()
    r.pending()
    r.claim(s)
    raw = origin.files["runs/jobs/j1/state.json"].decode("utf-8")
    d = json.loads(raw)
    assert list(d) == sorted(d)

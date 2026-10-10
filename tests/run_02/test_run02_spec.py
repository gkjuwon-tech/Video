"""잡 명세 검증·해시 오라클·저장소 jobs/*.yaml·import 순수성."""
import glob
import hashlib
import json
import os
import subprocess
import sys

import pytest
from run02_fakes import job_yaml

from buc.rt import runner as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def oracle_hash(d):
    h = {k: d[k] for k in ("id", "kind", "cmd", "env", "timeout_s", "outputs", "gpu")}
    return hashlib.sha256(json.dumps(h, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_constants_and_exceptions():
    assert R.KINDS == ("train", "render")
    assert R.JOB_SCHEMA == "buc.job/v1" and R.STATE_SCHEMA == "buc.job_state/v1"
    assert R.RESULT_SCHEMA == "buc.job_result/v1"
    assert R.JOBS_DIR == "jobs" and R.RUNS_DIR == "runs/jobs"
    assert issubclass(R.JobSpecError, R.RunnerError) and issubclass(R.JobSpecError, ValueError)
    assert issubclass(R.PushConflict, R.RunnerError) and issubclass(R.LockHeld, R.RunnerError)
    assert issubclass(R.RunnerError, RuntimeError)


def test_defaults_and_normalisation():
    s = R.parse_job_yaml("id: a1\nkind: render\ncmd: [python, r.py]\ntimeout_s: 120\n")
    assert s.id == "a1" and s.kind == "render" and s.cmd == ("python", "r.py")
    assert s.env == {} and s.outputs == () and s.gpu is True and s.priority == 0 and s.enabled is True
    assert s.max_attempts == 3
    assert isinstance(s.timeout_s, float) and s.timeout_s == 120.0
    d = s.to_dict()
    assert d["cmd"] == ["python", "r.py"] and d["outputs"] == [] and "schema" not in d
    assert set(d) == {"id", "kind", "cmd", "env", "timeout_s", "max_attempts", "outputs", "gpu", "priority", "enabled"}
    assert R.JobSpec.from_dict(d) == s
    s2 = R.parse_job_yaml("schema: buc.job/v1\nid: a1\nkind: train\ncmd: [x]\n")
    assert s2.timeout_s == 3600.0


def test_spec_hash_oracle_and_sensitivity():
    s = R.parse_job_yaml(job_yaml("h1", env={"A": "1"}, outputs=["*.png"], timeout_s=60))
    assert s.spec_hash == oracle_hash(s.to_dict())
    assert len(s.spec_hash) == 64 and all(c in "0123456789abcdef" for c in s.spec_hash)
    # int vs float 표기 차이 흡수
    assert R.parse_job_yaml(job_yaml("h1", env={"A": "1"}, outputs=["*.png"], timeout_s=60.0)).spec_hash == s.spec_hash
    # 실행 무관 필드는 해시 불변
    for kw in ({"priority": 5}, {"enabled": False}, {"max_attempts": 7}):
        t = R.parse_job_yaml(job_yaml("h1", env={"A": "1"}, outputs=["*.png"], timeout_s=60, **kw))
        assert t.spec_hash == s.spec_hash, kw
    # 실행 관련 필드는 해시 변경
    variants = [job_yaml("h2", env={"A": "1"}, outputs=["*.png"], timeout_s=60),
                job_yaml("h1", cmd=("python", "x.py"), env={"A": "1"}, outputs=["*.png"], timeout_s=60),
                job_yaml("h1", env={"A": "2"}, outputs=["*.png"], timeout_s=60),
                job_yaml("h1", env={"A": "1"}, outputs=["*.jpg"], timeout_s=60),
                job_yaml("h1", env={"A": "1"}, outputs=["*.png"], timeout_s=61),
                job_yaml("h1", env={"A": "1"}, outputs=["*.png"], timeout_s=60, gpu=False),
                job_yaml("h1", kind="render", env={"A": "1"}, outputs=["*.png"], timeout_s=60)]
    hs = {R.parse_job_yaml(v).spec_hash for v in variants}
    assert len(hs) == len(variants) and s.spec_hash not in hs


@pytest.mark.parametrize("text", [
    "id: -bad\nkind: train\ncmd: [x]\n",
    "id: 'a b'\nkind: train\ncmd: [x]\n",
    "id: " + "a" * 65 + "\nkind: train\ncmd: [x]\n",
    "id: a\nkind: eval\ncmd: [x]\n",
    "id: a\nkind: train\ncmd: 'python x.py'\n",
    "id: a\nkind: train\ncmd: []\n",
    "id: a\nkind: train\ncmd: ['']\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {HF_TOKEN: abc}\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {my_api_key: abc}\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {DB_PASSWORD: abc}\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {AWS_SECRET: abc}\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {lower: abc}\n",
    "id: a\nkind: train\ncmd: [x]\nenv: {N: 3}\n",
    "id: a\nkind: train\ncmd: [x]\noutputs: [/etc/passwd]\n",
    "id: a\nkind: train\ncmd: [x]\noutputs: ['../up.txt']\n",
    "id: a\nkind: train\ncmd: [x]\noutputs: ['a/../../b']\n",
    "id: a\nkind: train\ncmd: [x]\nmax_attempts: 0\n",
    "id: a\nkind: train\ncmd: [x]\nmax_attempts: true\n",
    "id: a\nkind: train\ncmd: [x]\ntimeout_s: 0\n",
    "id: a\nkind: train\ncmd: [x]\ntimeout_s: .inf\n",
    "id: a\nkind: train\ncmd: [x]\ntimeout_s: true\n",
    "id: a\nkind: train\ncmd: [x]\ngpu: 1\n",
    "id: a\nkind: train\ncmd: [x]\nbogus: 1\n",
    "schema: buc.job/v2\nid: a\nkind: train\ncmd: [x]\n",
    "kind: train\ncmd: [x]\n",
    "- just\n- a list\n",
])
def test_invalid_specs(text):
    with pytest.raises(R.JobSpecError):
        R.parse_job_yaml(text)


def test_stem_must_match():
    t = job_yaml("abc")
    assert R.parse_job_yaml(t, stem="abc").id == "abc"
    with pytest.raises(R.JobSpecError):
        R.parse_job_yaml(t, stem="abd")


def test_repo_jobs_yaml_are_valid():
    paths = sorted(glob.glob(os.path.join(ROOT, "jobs", "*.yaml")))
    assert paths, "jobs/*.yaml 최소 1개 필요"
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = R.parse_job_yaml(f.read(), stem=os.path.splitext(os.path.basename(p))[0])
        assert s.kind in R.KINDS
        assert not any(k.upper() in ("TOKEN",) for k in s.env)
        txt = open(p, encoding="utf-8").read()
        for bad in ("hf_", "ghp_", "github_pat_"):
            assert bad not in txt, f"{p}: 비밀처럼 보이는 문자열"


def test_import_purity():
    code = ("import sys; import buc.rt.runner as r; "
            "bad=[m for m in ('torch','numpy','yaml','requests','huggingface_hub') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""

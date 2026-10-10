"""ci.yml 구조 검사. 파싱한 YAML 과 원문 둘 다 본다."""
import re

import pytest
import yaml

from fnd05_helpers import CI_YML, ROOT, load

RULES = "E9,F63,F7,F82"


@pytest.fixture(scope="module")
def raw():
    assert CI_YML.exists(), ".github/workflows/ci.yml 이 없다"
    return CI_YML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wf(raw):
    d = yaml.safe_load(raw)
    assert isinstance(d, dict)
    return d


def triggers(wf):
    # PyYAML(YAML 1.1) 은 키 on 을 True 로 읽는다
    return wf["on"] if "on" in wf else wf[True]


def all_steps(wf):
    out = []
    for name, job in wf["jobs"].items():
        for st in job["steps"]:
            out.append((name, st))
    return out


def runs(wf):
    return [st["run"] for _, st in all_steps(wf) if "run" in st]


def test_name_and_triggers(wf):
    assert wf["name"] == "CI"
    on = triggers(wf)
    assert isinstance(on, dict) and "push" in on and "pull_request" in on
    assert "main" in (on["push"] or {}).get("branches", [])
    assert "pull_request_target" not in on


def test_permissions_least_privilege(wf):
    assert wf.get("permissions") == {"contents": "read"}


def test_jobs_shape(wf):
    assert wf["jobs"]
    for name, job in wf["jobs"].items():
        assert job["runs-on"] == "ubuntu-latest", name
        assert isinstance(job["timeout-minutes"], int) and 0 < job["timeout-minutes"] <= 30, name
        assert job["steps"], name


def test_forbidden_constructs(raw, wf):
    for bad in ("pull_request_target", "secrets.", "self-hosted", "|| true", "continue-on-error: true", "--no-verify"):
        assert bad not in raw, bad
    for _, st in all_steps(wf):
        assert st.get("continue-on-error") in (None, False)
    for name, job in wf["jobs"].items():
        assert job.get("continue-on-error") in (None, False), name
    assert not re.search(r"ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|BEGIN [A-Z ]*PRIVATE KEY", raw)


def test_actions_pinned_and_required(wf):
    uses = [st["uses"] for _, st in all_steps(wf) if "uses" in st]
    assert any(u.startswith("actions/checkout@v4") for u in uses)
    assert any(u.startswith("actions/setup-python@v5") for u in uses)
    for u in uses:
        assert re.search(r"@(v\d+(\.\d+){0,2}|[0-9a-f]{40})$", u), f"고정 안 된 액션: {u}"


def test_python_version_is_quoted_string_ge_310(wf):
    vs = [st["with"]["python-version"] for _, st in all_steps(wf)
          if str(st.get("uses", "")).startswith("actions/setup-python")]
    assert vs
    for v in vs:
        assert isinstance(v, str), "3.10 을 따옴표 없이 쓰면 YAML 이 3.1 (float)로 읽는다. 문자열로 쓸 것"
        major, minor = (int(x) for x in v.split(".")[:2])
        assert (major, minor) >= (3, 10)


def test_cpu_torch_installed_before_package(wf):
    text = "\n".join(runs(wf))
    i_torch = text.find("https://download.pytorch.org/whl/cpu")
    m = re.search(r"pip install[^\n]*-e\s+[\"']?\.\[dev\]", text)
    assert i_torch >= 0 and "torch" in text[: i_torch + 1], "CPU torch 설치 줄이 없다"
    assert m, "pip install -e .[dev] 가 없다"
    assert i_torch < m.start(), "buc 설치가 CPU torch 보다 먼저면 CUDA 휠이 끌려온다"
    low = text.lower()
    assert "pyyaml" in low and "blake3" in low


def test_ruff_step_matches_gate_rules(wf):
    text = "\n".join(runs(wf))
    m = re.search(r"ruff check[^\n]*", text)
    assert m
    line = m.group(0)
    sel = re.search(r"--select[ =]([A-Z0-9,]+)", line)
    assert sel and set(sel.group(1).split(",")) == set(RULES.split(",")), line
    targets = [t for t in line.split("--select")[1].split()[1:] if not t.startswith("-")]
    assert "buc" in targets
    for t in targets:
        assert (ROOT / t).exists(), f"ruff 대상이 없다: {t}"


def test_pytest_step_skips_gpu_and_runs_tests(wf):
    text = "\n".join(runs(wf))
    m = re.search(r"pytest[^\n]*", text)
    assert m
    line = m.group(0)
    assert re.search(r"-m\s+[\"']not gpu[\"']", line), line
    assert re.search(r"(^|\s)tests/?($|\s)", line), line
    assert not re.search(r"-m\s+[\"']?gpu", text)
    assert (ROOT / "tests").is_dir()


def test_ledger_step_guarded_by_default_ledger_path(wf):
    L = load()
    steps = [st for _, st in all_steps(wf) if "tools/ledger_check.py" in st.get("run", "")]
    assert len(steps) == 1
    cond = str(steps[0].get("if", ""))
    assert "hashFiles(" in cond and L.DEFAULT_LEDGER in cond and "!=" in cond, cond
    assert (ROOT / "tools" / "ledger_check.py").is_file()


def test_every_referenced_script_exists(wf):
    for line in runs(wf):
        for p in re.findall(r"(?:tools|ci|ops)/[\w./-]+\.py", line):
            assert (ROOT / p).is_file(), p


def test_file_hygiene(raw):
    assert raw.endswith("\n") and "\t" not in raw and "\r" not in raw

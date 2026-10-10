"""EVL-04 R-EVAL 격리: 평가 전용 LPIPS 는 별도 프로세스에서만 돌고, 부모 프로세스는 torch 를 만지지 않는다."""
import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from evl04_support import MODULE, ROOT, alive, ff, noisy, texture

BANNED = {"torch", "lpips", "torchvision", "tensorflow", "onnxruntime", "transformers", "timm", "kornia"}


def _run(code, timeout=90):
    """깨끗한 파이썬 프로세스에서 code 를 돌려 마지막 줄의 JSON 을 읽는다. cwd 는 레포 밖."""
    pre = (
        "import sys, json, importlib.util\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"spec = importlib.util.spec_from_file_location('first_frame_evl04', {str(MODULE)!r})\n"
        "ff = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ff; spec.loader.exec_module(ff)\n"
        f"BANNED = set({sorted(BANNED)!r})\n"
        "def loaded(): return sorted(m for m in BANNED if m in sys.modules)\n"
    )
    p = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True, timeout=timeout, cwd="/")
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_import_has_no_side_effects_and_no_model_libs():
    out = _run("print(json.dumps({'loaded': loaded(), 'r_eval': ff.R_EVAL}))")
    assert out["loaded"] == []
    assert out["r_eval"] is True


def test_parent_stays_torch_free_while_using_the_proxy_worker():
    code = """
import numpy as np
r = np.random.default_rng(0)
a = r.uniform(0.2, 0.8, (64, 80, 3)).astype(np.float32)
b = np.clip(a + 0.05 * r.standard_normal(a.shape), 0, 1).astype(np.float32)
with ff.LpipsClient('proxy') as c:
    d = ff.ffd(a, b, c)
    info = c.info()
    rep = ff.first_frame_report(a, b, c)
print(json.dumps({'loaded': loaded(), 'd': d, 'info': info, 'backend': rep.backend}))
"""
    out = _run(code)
    assert out["loaded"] == []  # 부모에 torch/lpips 없음
    assert out["d"] > 0
    assert out["backend"] == "proxy"
    assert out["info"]["backend"] == "proxy"
    assert out["info"]["torch_loaded"] is False  # 대역 백엔드는 워커에도 torch 가 필요 없다


def test_module_source_imports_respect_r_eval():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    def top_level_imports(body):
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                yield n
            elif isinstance(n, (ast.If, ast.Try, ast.With)):
                for sub in ("body", "orelse", "finalbody"):
                    yield from top_level_imports(getattr(n, sub, []) or [])
                for h in getattr(n, "handlers", []) or []:
                    yield from top_level_imports(h.body)

    def names(n):
        if isinstance(n, ast.Import):
            return [a.name for a in n.names]
        return [n.module or ""] if n.level == 0 else ["." * n.level + (n.module or "")]

    for n in top_level_imports(tree.body):
        for nm in names(n):
            assert nm.split(".")[0] not in BANNED, f"모듈 수준에서 평가 모델 라이브러리를 import: {nm}"
    # 어디서든(함수 안 포함) 시스템 모듈 import 금지: buc.core 만 허용
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for nm in names(n):
                if nm == "buc" or nm.startswith("buc."):
                    assert nm == "buc" or nm.startswith("buc.core"), f"buc.core 밖 import: {nm}"
                assert not nm.startswith("."), f"상대 import 금지: {nm}"


def test_worker_is_a_separate_live_process_and_close_reaps_it():
    c = ff().LpipsClient("proxy")
    a = texture(48, 64, seed=1)
    try:
        pid = c.worker_pid
        assert isinstance(pid, int) and pid != os.getpid() and alive(pid)
        info = c.info()
        assert info["pid"] == pid and info["backend"] == "proxy" and info["torch_loaded"] is False
        assert c.backend == "proxy" and c.is_perceptual is False
        assert c.distance(a, a) == 0.0
    finally:
        c.close()
    assert not alive(pid)  # 회수까지 끝났다(좀비 아님)
    c.close()  # 멱등
    with pytest.raises(ff().EvalWorkerError):
        c.distance(a, a)


def test_context_manager_closes_worker():
    with ff().LpipsClient("proxy") as c:
        pid = c.worker_pid
        assert alive(pid)
    assert not alive(pid)


def test_two_clients_have_distinct_workers():
    with ff().LpipsClient("proxy") as c1, ff().LpipsClient("proxy") as c2:
        assert c1.worker_pid != c2.worker_pid


def test_worker_matches_in_process_proxy_function():
    m = ff()
    a = texture(60, 90, seed=2)
    b = noisy(a, 0.04, seed=3)
    with m.LpipsClient("proxy") as c:
        d1 = c.distance(a, b)
        d2 = c.distance(a, b)
        assert type(d1) is float
        assert d1 == d2  # 결정론
        assert d1 == pytest.approx(m.proxy_lpips(a, b), abs=1e-12)
        # 비연속·다른 dtype 입력도 같은 값
        a64 = np.asarray(a, np.float64)
        bT = np.ascontiguousarray(b[::-1])[::-1]
        assert c.distance(a64, bT) == pytest.approx(d1, abs=1e-6)


def test_worker_crash_is_reported_not_hung():
    m = ff()
    a = texture(32, 40, seed=4)
    c = m.LpipsClient("proxy", timeout_s=20)
    try:
        assert c.distance(a, a) == 0.0
        os.kill(c.worker_pid, signal.SIGKILL)
        with pytest.raises(m.EvalWorkerError):
            c.distance(a, a)
    finally:
        c.close()  # 죽은 워커에도 예외 없이 닫힌다


def test_bad_inputs_are_rejected_in_parent_and_client_survives():
    m = ff()
    a = texture(32, 40, seed=5)
    with m.LpipsClient("proxy") as c:
        with pytest.raises(ValueError):
            c.distance(a, a[:, :-1])
        with pytest.raises(ValueError):
            c.distance(a[..., :2], a[..., :2])
        bad = a.copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            c.distance(a, bad)
        assert c.distance(a, a) == 0.0  # 여전히 쓸 수 있다


def test_unknown_backend_and_default_backend_is_strict():
    m = ff()
    with pytest.raises(ValueError):
        m.LpipsClient("nope")
    if importlib.util.find_spec("lpips") is None:
        # 기본 백엔드는 진짜 LPIPS 다. 없으면 조용히 대역으로 바꾸지 말고 실패해야 한다.
        with pytest.raises(m.EvalModelUnavailable):
            m.LpipsClient()
    assert issubclass(m.EvalModelUnavailable, m.EvalWorkerError)


def test_results_are_plain_floats_for_human_reports():
    m = ff()
    a = texture(64, 96, seed=6)
    b = noisy(a, 0.03, seed=7)
    with m.LpipsClient("proxy") as c:
        for v in (m.ffd(a, b, c), m.delta_e00(a, b), m.delta_g(a, b), m.ffg(a, b).median, m.ffg(a, b).p95):
            assert type(v) is float and np.isfinite(v)
        rep = m.first_frame_report(a, b, c)
    d = json.loads(rep.to_json())
    assert d["eval_only"] is True and d["backend"] == "proxy" and d["is_perceptual"] is False


@pytest.mark.gpu
def test_real_lpips_worker_isolated_from_parent():
    if importlib.util.find_spec("lpips") is None:
        pytest.skip("lpips 패키지 없음")
    m = ff()
    a = texture(96, 128, seed=8)
    b = noisy(a, 0.05, seed=9)
    with m.LpipsClient("lpips", device="cuda") as c:
        assert c.is_perceptual is True and c.worker_pid != os.getpid()
        assert c.info()["torch_loaded"] is True
        d = c.distance(a, b)
        assert 0.0 < d < 1.0 and c.distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert "lpips" not in sys.modules  # 부모는 오염되지 않는다

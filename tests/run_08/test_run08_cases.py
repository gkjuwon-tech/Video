"""RUN-08: 장면·케이스(실제 CPU 래스터 소형 실행)·임대 쿼리·import 위생."""
import math
import subprocess
import sys

import numpy as np
import pytest
from run08_helpers import GATE_PARAMS, MODULE_PATH, fake_cpu

torch = pytest.importorskip("torch")


# ------------------------------------------------------------------ make_scene
def test_make_scene_geometry(t4):
    s, cam = t4.make_scene(100, 64, 48)
    assert s.n == 100 and s.means.dtype == torch.float32 and s.device.type == "cpu"
    fx = 0.5 * 64 / math.tan(math.radians(30.0))
    K = cam["K"]
    assert K.shape == (3, 3) and K.dtype.name == "float64"
    assert K[0, 0] == pytest.approx(fx, abs=1e-9) and K[1, 1] == pytest.approx(fx, abs=1e-9)
    assert K[0, 2] == 32.0 and K[1, 2] == 24.0 and K[2, 2] == 1.0 and K[0, 1] == 0.0 and K[1, 0] == 0.0
    c2w = cam["c2w"]
    assert c2w.shape == (4, 4) and c2w.dtype.name == "float64"
    assert np.array_equal(c2w, np.eye(4))
    m = s.means.double()
    z = m[:, 2]
    assert float(z.min()) >= 2.0 - 1e-6 and float(z.max()) <= 8.0 + 1e-6
    u = fx * m[:, 0] / z + 32.0
    v = fx * m[:, 1] / z + 24.0
    assert float(u.min()) >= -1e-3 and float(u.max()) <= 64 + 1e-3
    assert float(v.min()) >= -1e-3 and float(v.max()) <= 48 + 1e-3
    ls = s.log_scales.double()
    assert torch.allclose(ls, torch.log(1.5 * z / fx)[:, None].expand(-1, 2), atol=1e-4)
    assert torch.allclose(s.opacity_logits, torch.ones(100))
    assert float(s.colors.min()) >= 0.0 and float(s.colors.max()) <= 1.0


def test_make_scene_is_bitwise_deterministic_and_seeded(t4):
    a, _ = t4.make_scene(50, 64, 48, seed=3)
    b, _ = t4.make_scene(50, 64, 48, seed=3)
    c, _ = t4.make_scene(50, 64, 48, seed=4)
    assert all(torch.equal(x, y) for x, y in zip(a.params(), b.params()))
    assert not torch.equal(a.means, c.means)


def test_make_scene_leaves_global_rng_alone(t4):
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    t4.make_scene(20, 32, 32)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("args", [(0, 64, 48), (-3, 64, 48), (10, 0, 48), (10, 64, 0), (10, 64.5, 48), (1.5, 64, 48)])
def test_make_scene_rejects_bad_args(t4, args):
    with pytest.raises(ValueError):
        t4.make_scene(*args)


# --------------------------------------------------------------- default_cases
def test_default_cases_names_params_and_gate(t4):
    cs = t4.default_cases()
    assert [c.name for c in cs] == [
        "splat_fwd_3840x2160_500000", "splat_fwd_3840x2160_1000000", "splat_fwd_3840x2160_3000000",
        "splat_fwdbwd_1024x1024_1000000"]
    assert [dict(c.params) for c in cs] == [
        {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 500_000},
        {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 1_000_000},
        {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 3_000_000},
        {"mode": "fwdbwd", "width": 1024, "height": 1024, "n_prims": 1_000_000}]
    assert [t4.gate_case(c.params) for c in cs] == [False, False, True, False]
    assert dict(cs[2].params) == GATE_PARAMS


def test_default_cases_sorts_and_customizes(t4):
    cs = t4.default_cases(width=64, height=48, prims=(300, 100, 200), train_size=32, train_prims=50)
    assert [c.name for c in cs] == ["splat_fwd_64x48_100", "splat_fwd_64x48_200", "splat_fwd_64x48_300",
                                    "splat_fwdbwd_32x32_50"]
    assert dict(cs[3].params) == {"mode": "fwdbwd", "width": 32, "height": 32, "n_prims": 50}


@pytest.mark.parametrize("prims", [(0,), (-1, 5), (5, 5), (1.5,), (True,), ("7",)])
def test_default_cases_validates_prims(t4, prims):
    with pytest.raises(ValueError):
        t4.default_cases(prims=prims)


@pytest.mark.parametrize("kw", [{"width": 0}, {"height": -1}, {"width": 64.0}, {"train_size": 0}, {"train_prims": 0},
                                {"train_prims": True}])
def test_default_cases_validates_dims(t4, kw):
    with pytest.raises(ValueError):
        t4.default_cases(**kw)


def test_creating_default_cases_does_no_work(t4):
    # 4K x 3M 장면을 만들면 메모리가 크다. 목록 생성만으로는 build가 돌지 않아야 한다.
    cs = t4.default_cases()
    assert len(cs) == 4 and all(callable(c.build) for c in cs)


# ---------------------------------------------------- 실제 CPU 래스터 소형 실행
def tiny_cases(t4):
    return t4.default_cases(width=64, height=48, prims=(60, 120), train_size=32, train_prims=60)


def test_fwd_case_renders_on_cpu(t4):
    c = tiny_cases(t4)[0]
    fn = c.build("cpu")
    r1 = fn()
    assert tuple(r1.rgb.shape) == (48, 64, 3)
    assert bool(torch.isfinite(r1.rgb).all())
    assert float(r1.alpha.sum()) > 1.0                      # 장면이 화면에 들어온다
    assert not r1.rgb.requires_grad
    r2 = fn()
    assert torch.equal(r1.rgb, r2.rgb)                      # 반복 호출이 같은 결과


def test_fwdbwd_case_returns_stable_float_loss(t4):
    c = tiny_cases(t4)[2]
    assert c.params["mode"] == "fwdbwd"
    fn = c.build("cpu")
    l1 = fn()
    l2 = fn()
    assert isinstance(l1, float) and math.isfinite(l1) and l1 > 0.0
    assert l1 == l2                                         # grad 누적이 손실에 새지 않는다
    # 새로 build해도 같은 손실(장면이 결정론적)
    assert c.build("cpu")() == l1


def test_run_suite_on_real_cpu_with_allow_cpu(t4):
    if torch.cuda.is_available():
        pytest.skip("CUDA 호스트에서는 CPU 시나리오가 아니다")
    dev = t4.detect_device()
    rep = t4.run_suite(tiny_cases(t4), device=dev, repeats=2, warmup=1, allow_cpu=True)
    assert [m.status for m in rep.measurements] == ["ok", "ok", "ok"]
    assert all(m.memory is None and m.timing.n == 2 and m.timing.warmup == 1 for m in rep.measurements)
    assert all(m.timing.min_s > 0.0 for m in rep.measurements)
    assert all(m.verdict == "info" for m in rep.measurements)
    assert rep.shot_on_t4 is False and rep.gate_verdict == "unmeasured"
    text = t4.render_appendix(rep)
    assert "> 미측정: 이 장치는 T4가 아니다 (cpu). T4 수치는 없다." in text.splitlines()
    row = [ln for ln in text.splitlines() if ln.startswith("| splat_fwd_64x48_60 ")][0]
    assert row == "| splat_fwd_64x48_60 | 측정 | 미측정 | 미측정 | - | - |  |"


def test_run_suite_cpu_without_allow_never_builds(t4):
    rep = t4.run_suite(tiny_cases(t4), device=t4.detect_device(fake_cpu()), torch_mod=fake_cpu())
    assert [m.status for m in rep.measurements] == ["unmeasured"] * 3


# ------------------------------------------------------------ t4_offer_query
def raw_offer(**over):
    d = {"id": 11, "machine_id": 110, "gpu_name": "Tesla T4", "num_gpus": 1, "dph_total": 0.30,
         "reliability2": 0.97, "geolocation": "Seoul, KR", "cpu_cores_effective": 8, "cpu_ram": 32 * 1024,
         "disk_space": 100, "inet_down": 500, "cuda_max_good": 12.4, "verification": "verified", "rentable": True}
    d.update(over)
    return d


def test_t4_offer_query_fields_and_filtering(t4):
    from buc.rt import vast as V
    q = t4.t4_offer_query()
    assert isinstance(q, V.OfferQuery)
    assert (q.gpu_name, q.num_gpus, q.max_dph) == ("Tesla T4", 1, 0.35)
    assert (q.min_reliability, q.min_cpu_cores, q.min_ram_gb) == (0.95, 4, 16)
    assert (q.min_disk_gb, q.min_inet_down, q.min_cuda) == (60, 100, 12.0)
    assert q.verified_only is True and tuple(q.exclude_countries) == ("CN",)
    assert q.to_api_query()["gpu_name"] == {"eq": "Tesla T4"}
    accept = lambda **o: q.accepts(V.Offer.from_api(raw_offer(**o)))  # noqa: E731
    assert accept()
    assert not accept(dph_total=0.40)
    assert not accept(gpu_name="RTX 5090")
    assert not accept(cuda_max_good=11.8)
    assert not accept(geolocation="Beijing, China")
    assert not accept(verification="unverified")
    assert t4.t4_offer_query(0.5).accepts(V.Offer.from_api(raw_offer(dph_total=0.40)))
    assert t4.t4_offer_query(max_dph=0.2).max_dph == 0.2


@pytest.mark.parametrize("bad", [0, -0.1, float("nan"), float("inf")])
def test_t4_offer_query_rejects_bad_price(t4, bad):
    with pytest.raises(ValueError):
        t4.t4_offer_query(bad)


# --------------------------------------------------------------- import 위생
def test_import_is_pure(tmp_path):
    code = (
        "import importlib.util, sys\n"
        "before = list(sys.path)\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(MODULE_PATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['m'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "bad = [n for n in ('torch', 'numpy', 'buc', 'PIL', 'scipy', 'requests') if n in sys.modules]\n"
        "print(repr(bad), sys.path == before)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=50)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[] True"
    assert list(tmp_path.iterdir()) == []                      # 파일을 만들지 않는다


# ------------------------------------------------------------------- GPU 전용
@pytest.mark.gpu
def test_gpu_tiny_cases_on_real_cuda(t4):
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    dev = t4.detect_device()
    cs = t4.default_cases(width=256, height=192, prims=(1000, 2000), train_size=128, train_prims=1000)
    rep = t4.run_suite(cs, device=dev, repeats=3, warmup=2)
    assert [m.status for m in rep.measurements] == ["ok", "ok", "ok"]
    assert all(m.memory is not None and m.memory.peak_allocated_gib > 0 for m in rep.measurements)
    assert rep.shot_on_t4 is dev.is_t4
    if dev.is_t4:
        # 진짜 T4에서만: 4K 사다리를 실제로 잰다. 통과 여부는 측정 결과이며 여기서 강제하지 않는다.
        full = t4.run_suite(t4.default_cases(), device=dev, repeats=2, warmup=1, budget_s=1800.0)
        assert {m.status for m in full.measurements} <= {"ok", "oom", "unmeasured"}
        assert full.gate_verdict in ("pass", "fail", "unmeasured")

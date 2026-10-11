"""RD-10 공개 표면·검증·순수성: 상수, 자료형 경계(CL2/CL3), 옵션, 입력 오류, torch 미로드, 전역 RNG 불변."""
import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

import rd10_helpers as H
from buc.read import fit_lights as F

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
NAMES = ["FitLightsError", "FitLightsOpts", "Light", "Flag", "DirectionPrior", "ShadowEdge", "IntensityFit", "CmaResult",
         "EnergyReport", "LightHint", "FitLightsResult", "transport", "render_shading", "solve_intensities",
         "cmaes_minimize", "find_shadow_edges", "shadow_direction_prior", "penumbra_angular_size", "classify_source",
         "direction_to_az_el", "energy_report", "cheat_shadow_index", "promote", "light_hint", "fit_lights",
         "MAX_CHEAT_LIGHTS", "MAX_FLAGS", "CHEAT_SHARE_MAX", "SUN_ANGULAR_MAX", "PRIOR_LAMBDA", "NORMAL_MIN",
         "MIN_PIXELS", "CHEAT_ANGULAR_RADIUS", "KEY_NAME"]
D = tuple(H.KEY_DIR)


def mk(**kw):
    base = dict(name="key", role="motivated", kind="sun", direction=D, intensity=0.5)
    base.update(kw)
    return F.Light(**base)


def test_surface_and_constants():
    for n in NAMES:
        assert hasattr(F, n), n
    if hasattr(F, "__all__"):
        for n in NAMES:
            assert n in F.__all__, n
    assert (F.MAX_CHEAT_LIGHTS, F.MAX_FLAGS, F.CHEAT_SHARE_MAX, F.SUN_ANGULAR_MAX) == (4, 4, 0.35, 0.02)
    assert (F.PRIOR_LAMBDA, F.NORMAL_MIN, F.MIN_PIXELS, F.CHEAT_ANGULAR_RADIUS, F.KEY_NAME) == (0.01, 0.5, 16, 0.25, "key")
    assert issubclass(F.FitLightsError, ValueError)


# ------------------------------------------------------------------ Light: CL2(물리 요소만, 음의 광원 금지) · CL3(유령)
def test_light_basic_and_ghost_property():
    a = mk()
    assert a.ghost is False and a.direction == pytest.approx(D, abs=1e-15)
    c = mk(name="cheat1", role="cheat", kind="area", angular_radius=0.25)
    assert c.ghost is True                                   # CL3: 치트 = 카메라 1차·정반사 경로에서 보이지 않는 유령
    assert c.angular_radius == 0.25
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.intensity = 1.0
    with pytest.raises(TypeError):
        F.Light("key", "motivated", "sun", D, 0.5)           # kw_only
    assert "ghost" not in {f.name for f in dataclasses.fields(F.Light)}     # 파생 속성이지 입력이 아니다


def test_light_forbids_negative_and_light_linking():
    with pytest.raises(ValueError):
        mk(intensity=-0.1)                                   # CL2: 음의 광원 금지
    names = {f.name for f in dataclasses.fields(F.Light)}
    assert not any("link" in n or "exclude" in n or "include" in n for n in names)   # CL2: light linking 금지
    mk(intensity=0.0)


@pytest.mark.parametrize("kw", [
    dict(intensity=float("nan")), dict(intensity=float("inf")), dict(name=""), dict(role="fill"), dict(kind="negative"),
    dict(direction=(0.0, 0.0, 2.0)), dict(direction=(1.0, 0.0)), dict(angular_radius=-0.1), dict(angular_radius=2.0),
    dict(role="cheat", kind="sun"),                          # 치트 태양은 없다(물리 요소 = 면광·스폿·반사판)
    dict(role="motivated", kind="bounce"),                   # 반사판은 치트 전용
])
def test_light_value_errors(kw):
    with pytest.raises(ValueError):
        mk(**kw)


def test_light_type_errors():
    with pytest.raises(TypeError):
        mk(intensity=True)
    with pytest.raises(TypeError):
        mk(intensity="1")
    with pytest.raises(TypeError):
        mk(name=3)


def test_light_to_meta_cl6_record():
    m = mk(name="cheat1", role="cheat", kind="spot", intensity=0.2).to_meta()
    json.dumps(m, allow_nan=False)
    for k in ("name", "role", "kind", "direction", "intensity", "angular_radius", "ghost", "cheat", "azimuth_deg",
              "elevation_deg"):
        assert k in m, k
    assert m["cheat"] == 1 and m["ghost"] is True            # CL6: cheat=1 기록
    m0 = mk().to_meta()
    assert m0["cheat"] == 0 and m0["ghost"] is False
    az, el = F.direction_to_az_el(D)
    assert m0["azimuth_deg"] == pytest.approx(az) and m0["elevation_deg"] == pytest.approx(el)


def test_direction_to_az_el_conventions():
    # 고도: 위(-y)가 +; 방위: 카메라 정면(+z)에서 +x(오른쪽) 쪽이 +. 카메라 쪽(-z) 광원은 |az|=180.
    az, el = F.direction_to_az_el((0.0, -1.0, 0.0))
    assert el == pytest.approx(90.0)
    az, el = F.direction_to_az_el((1.0, 0.0, 0.0))
    assert (az, el) == (pytest.approx(90.0), pytest.approx(0.0, abs=1e-9))
    az, el = F.direction_to_az_el(tuple(H.unit([0.0, 0.0, 1.0])))
    assert az == pytest.approx(0.0, abs=1e-9)
    az, el = F.direction_to_az_el((0.0, 0.0, -1.0))
    assert abs(az) == pytest.approx(180.0)
    az, el = F.direction_to_az_el(tuple(H.unit([-1.0, -1.0, 0.0])))
    assert (az, el) == (pytest.approx(-90.0), pytest.approx(45.0))
    with pytest.raises(ValueError):
        F.direction_to_az_el((0.0, 0.0, 2.0))


def test_promote():
    c = mk(name="cheat1", role="cheat", kind="area", intensity=0.3)
    p = F.promote(c)                                         # 출처가 놓인 치트 -> 동기 광원
    assert p.role == "motivated" and p.ghost is False
    assert (p.name, p.kind, p.direction, p.intensity, p.angular_radius) == (c.name, c.kind, c.direction, c.intensity,
                                                                           c.angular_radius)
    assert c.role == "cheat"                                 # 원본 불변
    with pytest.raises(ValueError):
        F.promote(mk())                                      # 이미 동기
    with pytest.raises(ValueError):
        F.promote(mk(name="b", role="cheat", kind="bounce"))   # 반사판은 동기 광원이 될 수 없다


# ------------------------------------------------------------------ Flag, DirectionPrior, Opts
def test_flag_validation_and_copy():
    cov = np.zeros((4, 5))
    f = F.Flag(name="f1", light="key", coverage=cov)
    cov[:] = 1.0
    assert float(np.max(f.coverage)) == 0.0                  # 사본
    assert f.coverage.shape == (4, 5)
    for bad in (np.full((4, 5), 1.1), np.full((4, 5), -0.1), np.full((4, 5), np.nan), np.zeros((4, 5, 1)), np.zeros(5)):
        with pytest.raises(ValueError):
            F.Flag(name="f", light="key", coverage=bad)
    with pytest.raises(TypeError):
        F.Flag(name="f", light="key", coverage=[[0.0]])
    with pytest.raises(ValueError):
        F.Flag(name="", light="key", coverage=cov)


def test_direction_prior_validation():
    p = F.DirectionPrior(direction=D, sigma_rad=0.1)
    assert p.source == "user"
    F.DirectionPrior(direction=D, sigma_rad=0.1, source="probe")
    F.DirectionPrior(direction=D, sigma_rad=0.1, source="shadow")
    for kw in (dict(sigma_rad=0.0), dict(sigma_rad=-1.0), dict(sigma_rad=float("nan")), dict(source="other"),
               dict(direction=(0.0, 0.0, 3.0))):
        base = dict(direction=D, sigma_rad=0.1)
        base.update(kw)
        with pytest.raises(ValueError):
            F.DirectionPrior(**base)


def test_opts_defaults_and_validation():
    o = F.FitLightsOpts()
    assert (o.popsize, o.generations, o.max_cheat, o.min_gain, o.resid_tau, o.r2_min, o.n_coarse, o.seed, o.loss) == (
        16, 30, 4, 0.02, 0.02, 0.5, 512, 0, "l2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.popsize = 8
    with pytest.raises(TypeError):
        F.FitLightsOpts(16)
    F.FitLightsOpts(max_cheat=0)
    F.FitLightsOpts(max_cheat=4, loss="huber", popsize=np.int64(8), r2_min=np.float32(0.25))
    for kw in (dict(popsize=3), dict(generations=0), dict(max_cheat=5), dict(max_cheat=-1), dict(min_gain=0.0),
               dict(resid_tau=-0.1), dict(r2_min=1.5), dict(r2_min=-0.1), dict(n_coarse=8), dict(seed=-1),
               dict(loss="l1"), dict(min_gain=float("nan"))):
        with pytest.raises(ValueError):
            F.FitLightsOpts(**kw)
    for kw in (dict(popsize=16.0), dict(popsize=True), dict(max_cheat=True), dict(seed=0.5), dict(loss=1),
               dict(min_gain="0.02"), dict(r2_min=True)):
        with pytest.raises(TypeError):
            F.FitLightsOpts(**kw)


# ------------------------------------------------------------------ fit_lights 입력 검증
def _ok_inputs():
    n, v = H.ball()
    s = H.shading_from(n, [(H.KEY_DIR, 0.7)], H.AMB)
    return s, n, v


def test_fit_input_type_and_value_errors():
    s, n, v = _ok_inputs()
    fast = F.FitLightsOpts(generations=2, n_coarse=32, max_cheat=0)
    with pytest.raises(TypeError):
        F.fit_lights(s.tolist(), n, opts=fast)
    with pytest.raises(TypeError):
        F.fit_lights((s * 100).astype(np.int32), n, opts=fast)
    with pytest.raises(TypeError):
        F.fit_lights(s, n, visibility="nope", opts=fast)
    with pytest.raises(ValueError):
        F.fit_lights(s[:, :-1], n, opts=fast)                # 모양 불일치
    with pytest.raises(ValueError):
        F.fit_lights(s, n[..., :2], opts=fast)
    with pytest.raises(ValueError):
        F.fit_lights(s[:3], n[:3], opts=fast)                # H,W >= 4
    bad = s.copy(); bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        F.fit_lights(bad, n, opts=fast)
    bad = s.copy(); bad[3, 3] = -0.1
    with pytest.raises(ValueError):
        F.fit_lights(bad, n, opts=fast)                      # 음영은 음수가 아니다
    bad = n.copy(); bad[2, 2, 0] = np.inf
    with pytest.raises(ValueError):
        F.fit_lights(s, bad, opts=fast)
    with pytest.raises(ValueError):
        F.fit_lights(s, n, weight=-np.ones_like(s), opts=fast)
    with pytest.raises(ValueError):
        F.fit_lights(s, n, valid=v[:-1], opts=fast)
    with pytest.raises(TypeError):
        F.fit_lights(s, n, valid=v.astype(np.float32), opts=fast)    # valid 는 bool
    many = [F.Flag(name=f"f{i}", light="key", coverage=np.zeros(s.shape)) for i in range(F.MAX_FLAGS + 1)]
    with pytest.raises(ValueError):
        F.fit_lights(s, n, flags=many, opts=fast)            # CL4: 플래그 <= 4
    with pytest.raises(TypeError):
        F.fit_lights(s, n, direction_priors=[D], opts=fast)  # DirectionPrior 여야 한다


def test_inputs_not_mutated_and_rng_untouched():
    s, n, v = _ok_inputs()
    w = v.astype(np.float64)
    snap = [a.copy() for a in (s, n, v, w)]
    st = np.random.get_state()
    F.fit_lights(s, n, valid=v, weight=w, opts=F.FitLightsOpts(generations=5, n_coarse=64, max_cheat=1))
    F.cmaes_minimize(lambda x: float(x @ x), [1.0, 1.0], 0.5, generations=3)
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1]) and st[2:] == st2[2:]      # 전역 RNG 불변
    for a, b in zip((s, n, v, w), snap):
        assert np.array_equal(a, b)


def test_import_hygiene_no_torch_no_yaml():
    code = ("import sys; import buc.read.fit_lights as m; "
            "assert 'torch' not in sys.modules, 'torch'; assert 'yaml' not in sys.modules, 'yaml'; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr


def test_torch_cpu_tensors_accepted_by_duck_typing():
    torch = pytest.importorskip("torch")
    s, n, v = _ok_inputs()
    opts = F.FitLightsOpts(generations=8, n_coarse=128, max_cheat=0)
    a = F.fit_lights(s, n, valid=v, opts=opts)
    b = F.fit_lights(torch.from_numpy(s), torch.from_numpy(n), valid=torch.from_numpy(v), opts=opts)
    assert a.status == b.status == "ok"
    assert H.angle_deg(a.lights[0].direction, b.lights[0].direction) < 1e-6


@pytest.mark.gpu
def test_cuda_tensors_accepted():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    s, n, v = _ok_inputs()
    r = F.fit_lights(torch.from_numpy(s).cuda(), torch.from_numpy(n).cuda(), valid=torch.from_numpy(v).cuda())
    assert r.status == "ok" and math.isfinite(r.r2)

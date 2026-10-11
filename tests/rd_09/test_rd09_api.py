"""RD-09 공개 표면·검증·순수성: 상수, dataclass 경계, 입력 오류, 입력 불변, torch 미로드, 전역 RNG 불변, 메타 JSON."""
import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

from buc.read import intrinsics as X
import rd09_helpers as H

NAMES = ["IntrinsicsError", "IntrinsicsOpts", "GeoLight", "LightFit", "IntrinsicsResult", "geometric_shading",
         "fit_geo_light", "decompose", "relight", "heads_from_readout", "Y_FLOOR", "S_MIN", "GEO_FLOOR",
         "NORMAL_MIN", "MIN_LIGHT_PIXELS"]


def good_light():
    return X.GeoLight(direction=tuple(H.LIGHT_DIR), intensity=H.KEY, ambient=H.AMB)


def test_surface_and_constants():
    for n in NAMES:
        assert hasattr(X, n), n
    if hasattr(X, "__all__"):
        for n in NAMES:
            assert n in X.__all__, n
    assert (X.Y_FLOOR, X.S_MIN, X.GEO_FLOOR, X.NORMAL_MIN, X.MIN_LIGHT_PIXELS) == (1e-6, 1e-4, 1e-3, 0.5, 16)
    assert issubclass(X.IntrinsicsError, ValueError)


def test_opts_defaults_and_frozen():
    o = X.IntrinsicsOpts()
    assert (o.alpha, o.beta, o.gamma, o.eta, o.delta) == (1.0, 0.1, 1.0, 0.3, 0.1)
    assert (o.tau_chroma, o.tau_normal, o.n_alt, o.max_side, o.light_poor_rel) == (0.05, 0.5, 2, 320, 0.25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.alpha = 2.0
    with pytest.raises(TypeError):
        X.IntrinsicsOpts(1.0)          # kw_only


@pytest.mark.parametrize("kw", [
    {"alpha": -1.0}, {"beta": float("nan")}, {"gamma": float("inf")}, {"eta": -1e-9}, {"delta": -0.1},
    {"tau_chroma": 0.0}, {"tau_normal": -1.0}, {"light_poor_rel": 0.0}, {"n_alt": 0}, {"n_alt": 6},
    {"max_side": 7}])
def test_opts_value_errors(kw):
    with pytest.raises(ValueError):
        X.IntrinsicsOpts(**kw)


@pytest.mark.parametrize("kw", [{"alpha": True}, {"alpha": "1"}, {"n_alt": 2.0}, {"n_alt": True}, {"max_side": 64.0}])
def test_opts_type_errors(kw):
    with pytest.raises(TypeError):
        X.IntrinsicsOpts(**kw)


def test_opts_boundaries_allowed():
    X.IntrinsicsOpts(alpha=0, beta=0, gamma=0, eta=0, delta=0, n_alt=1, max_side=8)
    X.IntrinsicsOpts(n_alt=5)
    X.IntrinsicsOpts(alpha=np.float32(0.5), n_alt=np.int64(3))      # numpy 스칼라 허용


def test_geolight_validation_and_frozen():
    g = good_light()
    assert isinstance(g.direction, tuple) and len(g.direction) == 3
    assert abs(float(np.linalg.norm(g.direction)) - 1.0) < 1e-12
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.ambient = 1.0
    for d in [(0, 0, 2.0), (0, 0, 0), (1, 0), (float("nan"), 0, 1), (0.9, 0, 0)]:
        with pytest.raises(ValueError):
            X.GeoLight(direction=d, intensity=1.0, ambient=0.1)
    for kw in [{"intensity": -1.0, "ambient": 0.1}, {"intensity": 1.0, "ambient": -1e-6},
               {"intensity": float("inf"), "ambient": 0.1}]:
        with pytest.raises(ValueError):
            X.GeoLight(direction=(0, 0, -1), **kw)
    X.GeoLight(direction=(0, 0, -1), intensity=0.0, ambient=0.0)       # 경계 허용
    m = g.to_meta()
    assert set(m) >= {"direction", "intensity", "ambient"}
    json.dumps(m, allow_nan=False)


def test_decompose_validation():
    sc = H.scene(12, 16)
    I = sc["I"].astype(np.float32)
    gs = sc["shading"]
    ok = dict(geo_shading=gs)
    X.decompose(I, **ok)                                       # 기준선 통과
    bad_img = [
        (np.zeros((12, 16)), ValueError), (np.zeros((12, 16, 4), np.float32), ValueError),
        (np.zeros((3, 16, 3), np.float32), ValueError), (np.zeros((12, 3, 3), np.float32), ValueError),
        (I.astype(np.int32), TypeError), (I.tolist(), TypeError),
        (np.where(I > 0.5, np.nan, I).astype(np.float32), ValueError),
        (np.where(I > 0.5, np.inf, I).astype(np.float32), ValueError), (-I - 1e-3, ValueError)]
    for img, exc in bad_img:
        with pytest.raises(exc):
            X.decompose(img, **ok)
    bad_kw = [
        ({"geo_shading": gs[:-1]}, ValueError), ({"geo_shading": -gs}, ValueError),
        ({"geo_shading": gs.astype(np.float16)}, TypeError),
        ({"geo_shading": gs, "geo_weight": -np.ones_like(gs)}, ValueError),
        ({"geo_shading": gs, "geo_weight": np.ones((12, 15))}, ValueError),
        ({"geo_shading": gs, "exclude": np.zeros((12, 16), np.uint8)}, TypeError),
        ({"geo_shading": gs, "exclude": np.zeros((12, 15), bool)}, ValueError),
        ({"geo_shading": gs, "status": np.zeros((12, 16), np.int32)}, TypeError),
        ({"geo_shading": gs, "status": np.zeros((12, 15), np.uint8)}, ValueError),
        ({"geo_shading": gs, "shading_head": -np.ones_like(gs)}, ValueError),
        ({"geo_shading": gs, "albedo_head": np.ones((12, 16, 2))}, ValueError),
        ({"geo_shading": gs, "albedo_head": -np.ones((12, 16, 3))}, ValueError),
        ({"geo_shading": gs, "opts": {"alpha": 1}}, TypeError),
        ({"geo_shading": gs, "normal": sc["normal"][:, :-1]}, ValueError),
        ({"geo_shading": gs, "light": good_light(), "normal": sc["normal"]}, ValueError),     # 광원·음영 동시 지정 모호
        ({"light": good_light()}, ValueError),                                                # 법선 없이 광원
        ({"normal": sc["normal"], "light": {"direction": (0, 0, -1)}}, TypeError),
    ]
    for kw, exc in bad_kw:
        with pytest.raises(exc):
            X.decompose(I, **kw)


def test_decompose_requires_some_prior_and_connected_prior_mass():
    sc = H.scene(12, 16)
    I = sc["I"].astype(np.float32)
    with pytest.raises(X.IntrinsicsError):                        # 어떤 사전도 없음
        X.decompose(I)
    with pytest.raises(X.IntrinsicsError):                        # 법선만: 광원을 맞출 음영 원천이 없다
        X.decompose(I, normal=sc["normal"])
    z = X.IntrinsicsOpts(gamma=0.0, eta=0.0, delta=0.0)
    with pytest.raises(X.IntrinsicsError):                        # 사전 가중이 전부 0 -> 음영 수준 미정
        X.decompose(I, geo_shading=sc["shading"], opts=z)
    # 연결 성분 하나에 사전 가중이 0이면 미정: 가운데 열을 제외하면 좌우가 분리되고 오른쪽 사전을 0으로
    ex = np.zeros((12, 16), bool)
    ex[:, 8] = True
    gw = np.ones((12, 16))
    gw[:, 9:] = 0.0
    with pytest.raises(X.IntrinsicsError):
        X.decompose(I, geo_shading=sc["shading"], geo_weight=gw, exclude=ex,
                    opts=X.IntrinsicsOpts(eta=0.0, delta=0.0))
    # 같은 상황이어도 델타(알베도 사전)가 있으면 풀린다
    X.decompose(I, geo_shading=sc["shading"], geo_weight=gw, exclude=ex, albedo_head=sc["albedo"].astype(np.float32),
                opts=X.IntrinsicsOpts(eta=0.0))


def test_inputs_not_mutated_and_outputs_fresh():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    kw = dict(normal=sc["normal"].astype(np.float32), shading_head=sc["shading"].astype(np.float32),
              albedo_head=sc["albedo"].astype(np.float32), geo_weight=np.ones((16, 20)),
              exclude=np.zeros((16, 20), bool), status=np.zeros((16, 20), np.uint8))
    snap = {k: v.copy() for k, v in kw.items()}
    I0 = I.copy()
    r = X.decompose(I, **kw)
    assert np.array_equal(I, I0)
    for k in kw:
        assert np.array_equal(kw[k], snap[k]), k
    for a in (r.L_d, r.albedo, r.shading):
        assert not np.shares_memory(a, I)
    r.L_d[0, 0, 0] += 1.0                       # 결과를 바꿔도 입력은 그대로
    assert np.array_equal(I, I0)


def test_result_shapes_dtypes_and_meta_json():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    r = X.decompose(I, normal=sc["normal"], shading_head=sc["shading"])
    assert r.L_d.shape == (16, 20, 3) and r.L_d.dtype == np.float32
    assert r.albedo.shape == (16, 20, 3) and r.albedo.dtype == np.float32
    assert r.shading.shape == (16, 20) and r.shading.dtype == np.float32
    assert r.valid.shape == (16, 20) and r.valid.dtype == bool and r.valid.all()
    assert r.geo_shading.shape == (16, 20) and r.geo_shading.dtype == np.float32
    assert isinstance(r.light, X.GeoLight) and r.geo_source == "light_fit"
    assert isinstance(r.warnings, tuple) and list(r.warnings) == sorted(set(r.warnings))
    assert r.reliable == (len(r.warnings) == 0)
    assert np.isfinite(r.r_intr) and r.r_intr >= 0
    assert r.n_solves == r.opts.n_alt == 2 and r.solve_stride == 1
    m = r.meta()
    json.dumps(m, allow_nan=False)
    assert set(m) >= {"light", "geo_source", "r_intr", "n_solves", "solve_stride", "n_valid", "excluded_frac",
                      "warnings", "reliable", "opts"}
    assert m["n_valid"] == 16 * 20 and m["excluded_frac"] == 0.0 and m["geo_source"] == "light_fit"
    assert set(m["light"]) >= {"direction", "intensity", "ambient"}
    # 지오메트리 없는 경우: r_intr 은 NaN -> 메타에서는 None
    r2 = X.decompose(I, shading_head=sc["shading"])
    assert np.isnan(r2.r_intr) and r2.light is None and r2.geo_shading is None and r2.geo_source == "none"
    assert r2.meta()["r_intr"] is None and r2.meta()["light"] is None
    json.dumps(r2.meta(), allow_nan=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.r_intr = 0.0


def test_no_torch_no_global_rng_and_deterministic():
    code = ("import sys, numpy as np\n"
            "np.random.seed(123); a = np.random.get_state()[1].copy(); pos = np.random.get_state()[2]\n"
            "from buc.read import intrinsics as X\n"
            "n = np.zeros((12, 16, 3)); n[..., 2] = -1.0\n"
            "I = np.full((12, 16, 3), 0.3, np.float32)\n"
            "r = X.decompose(I, normal=n, shading_head=np.full((12, 16), 0.5, np.float32))\n"
            "X.fit_geo_light(n, np.full((12, 16), 0.5))\n"
            "assert (np.random.get_state()[1] == a).all() and np.random.get_state()[2] == pos, 'global RNG touched'\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert 'yaml' not in sys.modules, 'yaml imported'\n"
            "print('OK')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout + p.stderr


def test_bitwise_deterministic():
    sc = H.scene(16, 20)
    kw = dict(normal=sc["normal"], shading_head=(sc["shading"] * np.exp(0.1 * H.smooth_field(16, 20, 3))))
    I = sc["I"].astype(np.float32)
    a, b = X.decompose(I, **kw), X.decompose(I, **kw)
    for f in ("L_d", "albedo", "shading", "valid", "geo_shading"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert a.light == b.light and a.r_intr == b.r_intr

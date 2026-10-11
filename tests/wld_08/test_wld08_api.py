"""WLD-08 API·검증·지연 import·뷰 계획 기하."""
import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

from buc.core import contracts as C
from buc.core import geom
from buc.world.depth_world import build_depth_world

import wld08_helpers as h

S = h.mod()


def test_constants_and_exports():
    assert S.NAME == "sdedit_avg" and S.VERSION == "0.1.0"
    for n in ("SDEditAvgError", "SDEditAvgConfig", "ViewSpec", "SDEditAvgResult", "auto_pivot_m", "plan_views",
              "sdedit_avg_fill", "make_world_impl"):
        assert hasattr(S, n), n
    assert issubclass(S.SDEditAvgError, C.ContractError)


def test_import_is_lazy_and_clean():
    code = ("import sys; import buc.world.sdedit_avg as m; "
            "bad=[k for k in ('diffusers','transformers') if k in sys.modules]; "
            "cuda=False\n"
            "if 'torch' in sys.modules:\n"
            "    import torch; cuda=torch.cuda.is_initialized()\n"
            "print(bad, cuda)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "[] False", out.stdout


def test_config_defaults():
    c = S.SDEditAvgConfig()
    assert (c.sigma, c.max_angle_deg, c.n_per_side, c.pivot_m) == (0.5, 20.0, 2, None)
    assert (c.erode_px, c.depth_tol_rel, c.min_votes) == (2, 0.05, 1)
    assert isinstance(c.prompt, str) and c.prompt
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.sigma = 0.3
    with pytest.raises(TypeError):
        S.SDEditAvgConfig(0.5)  # kw_only


@pytest.mark.parametrize("kw", [
    dict(sigma=0.0), dict(sigma=1.5), dict(sigma=float("nan")), dict(sigma=True),
    dict(max_angle_deg=0.0), dict(max_angle_deg=46.0), dict(max_angle_deg=float("inf")),
    dict(n_per_side=0), dict(n_per_side=9), dict(n_per_side=True), dict(n_per_side=1.5),
    dict(pivot_m=0.0), dict(pivot_m=-1.0), dict(pivot_m=float("nan")),
    dict(erode_px=-1), dict(erode_px=9), dict(erode_px=True),
    dict(depth_tol_rel=0.0), dict(depth_tol_rel=0.6),
    dict(min_votes=0), dict(min_votes=True),
    dict(prompt=3),
])
def test_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        S.SDEditAvgConfig(**kw)


def test_config_accepts_edges():
    S.SDEditAvgConfig(sigma=1.0, max_angle_deg=45.0, n_per_side=8, pivot_m=0.5, erode_px=0, depth_tol_rel=0.5,
                      min_votes=3, prompt="")


def test_fill_type_and_value_errors():
    _, dw = h.build()
    ok = h.FakeSampler(h.const_transform([0.5] * 4))
    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw.world, ok)
    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw, ok, cfg={"sigma": 0.5})
    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw, object())
    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw, ok, cfg=S.SDEditAvgConfig(), views=None, seed=0, extra=1)
    with pytest.raises(ValueError):
        S.sdedit_avg_fill(dw, ok, views=[])
    with pytest.raises(ValueError):
        S.sdedit_avg_fill(dw, ok, seed=-1)
    with pytest.raises(ValueError):
        S.sdedit_avg_fill(dw, ok, seed=True)
    v = S.plan_views(dw)[0]
    wrong = S.ViewSpec(c2w=v.c2w, K=v.K, width=v.width + 16, height=v.height, angle_deg=v.angle_deg)
    with pytest.raises(ValueError):
        S.sdedit_avg_fill(dw, ok, views=[wrong])
    assert ok.calls == [], "검증은 샘플러를 부르기 전에 실패한다"


# --------------------------------------------------------------------------- 뷰 계획

def test_auto_pivot():
    _, dw = h.build()
    assert S.auto_pivot_m(dw) == pytest.approx(h.Z_FG, abs=1e-9)
    _, flat = h.build_flat()
    assert not flat.near.any()
    assert S.auto_pivot_m(flat) == pytest.approx(h.Z_BG, abs=1e-6)


def _pose_dw():
    """회전·이동된 관측 포즈를 가진 DepthWorld(포즈 합성 검증용)."""
    R = geom.rodrigues(np.array([0.0, 0.0, np.deg2rad(12.0)])) @ geom.rodrigues(np.array([0.0, np.deg2rad(-25.0), 0.0]))
    t = np.array([0.3, -0.2, 0.5])
    obs = C.Obs(K=h.K_of(), R=R, t=t, width=h.W, height=h.H)
    img, depth = h.scene()
    return build_depth_world(h.make_read(img, depth, obs=obs))


@pytest.mark.parametrize("posed", [False, True])
def test_plan_views_geometry(posed):
    dw = _pose_dw() if posed else h.build()[1]
    obs = dw.world.obs
    cfg = S.SDEditAvgConfig(max_angle_deg=20.0, n_per_side=2)
    views = S.plan_views(dw, cfg)
    assert len(views) == 4 and all(isinstance(v, S.ViewSpec) for v in views)
    assert [v.angle_deg for v in views] == pytest.approx([-20.0, -10.0, 10.0, 20.0], abs=1e-9)
    d = S.auto_pivot_m(dw)
    c2w_obs = geom.invert_pose(geom.make_pose(obs.R, obs.t))
    pivot_w = (c2w_obs @ np.array([0.0, 0.0, d, 1.0]))[:3]
    for v in views:
        assert v.c2w.shape == (4, 4) and v.c2w.dtype == np.float64
        assert np.allclose(v.c2w[3], [0, 0, 0, 1])
        assert geom.is_rotation(v.c2w[:3, :3], atol=1e-9)
        assert (v.width, v.height) == (obs.width, obs.height)
        assert np.array_equal(v.K, obs.K) and (v.k1, v.k2) == (obs.k1, obs.k2)
        # 중심은 피벗에서 정확히 d 떨어지고, 광축은 피벗을 향한다.
        assert np.linalg.norm(v.c2w[:3, 3] - pivot_w) == pytest.approx(d, abs=1e-9)
        w2c = geom.invert_pose(v.c2w)
        uv, z = geom.project(v.K, w2c[:3, :3], w2c[:3, 3], pivot_w, v.k1, v.k2)
        assert uv == pytest.approx([obs.K[0, 2], obs.K[1, 2]], abs=1e-6)
        assert z == pytest.approx(d, abs=1e-9)
        # 관측 광축과의 각 = |angle|, 회전축은 카메라 y 축(카메라 위 방향 불변)
        assert geom.rotation_angle_deg(obs.R, w2c[:3, :3]) == pytest.approx(abs(v.angle_deg), abs=1e-6)
        rel = c2w_obs[:3, :3].T @ v.c2w[:3, :3]
        assert rel[1, 1] == pytest.approx(1.0, abs=1e-9)
        assert abs(rel[0, 1]) < 1e-9 and abs(rel[2, 1]) < 1e-9
        # 부호 규약: angle<0 이면 카메라는 관측 카메라 +x 쪽으로 간다
        cx = (np.linalg.inv(c2w_obs) @ np.append(v.c2w[:3, 3], 1.0))[0]
        assert np.sign(cx) == -np.sign(v.angle_deg)
        assert abs(v.angle_deg) <= 20.0 + 1e-9


def test_plan_views_variants_and_determinism():
    _, dw = h.build()
    a = S.plan_views(dw, S.SDEditAvgConfig(max_angle_deg=15.0, n_per_side=3))
    assert [v.angle_deg for v in a] == pytest.approx([-15, -10, -5, 5, 10, 15], abs=1e-9)
    b = S.plan_views(dw, S.SDEditAvgConfig(max_angle_deg=15.0, n_per_side=3))
    assert all(np.array_equal(x.c2w, y.c2w) for x, y in zip(a, b))
    # pivot_m 지정은 자동 값을 대체한다
    p = S.plan_views(dw, S.SDEditAvgConfig(pivot_m=4.0, n_per_side=1))
    for v in p:
        assert abs(np.linalg.norm(v.c2w[:3, 3]) - 4.0 * 2 * abs(np.sin(np.deg2rad(v.angle_deg) / 2))) < 1e-9
    assert len(S.plan_views(dw)) == 4  # 기본 cfg


def test_plan_views_type_errors():
    _, dw = h.build()
    with pytest.raises(TypeError):
        S.plan_views(dw.world)
    with pytest.raises(TypeError):
        S.plan_views(dw, cfg=1)


def test_result_meta_json_and_fields():
    r = h.oracle_run()
    res = r.res
    m = res.meta()
    s = json.dumps(m, allow_nan=False)
    assert json.loads(s) == json.loads(json.dumps(m))
    for k in ("name", "version", "cfg", "n_views", "angles_deg", "sigma_requested", "sigma_snapped", "steps",
              "hidden_px", "filled_px", "unfilled_px", "filled_frac", "view_disocc_frac", "view_hole_frac"):
        assert k in m, k
    assert m["name"] == "sdedit_avg" and m["version"] == "0.1.0"
    assert m["n_views"] == 4 and m["angles_deg"] == pytest.approx([-20, -10, 10, 20])
    assert m["sigma_requested"] == 0.5 and m["sigma_snapped"] == pytest.approx(0.5)
    assert m["steps"] == res.steps == 2
    hid = r.dw.hidden
    assert m["hidden_px"] == int(hid.sum()) == int(res.filled.sum() + res.unfilled.sum())
    assert m["filled_px"] == int(res.filled.sum()) and m["unfilled_px"] == int(res.unfilled.sum())
    assert m["filled_frac"] == pytest.approx(res.filled.sum() / hid.sum())
    assert len(m["view_disocc_frac"]) == len(m["view_hole_frac"]) == 4
    # 필드 모양
    Hh, Ww = hid.shape
    assert res.filled.shape == res.unfilled.shape == res.votes.shape == (Hh, Ww)
    assert res.filled.dtype == np.bool_ and res.unfilled.dtype == np.bool_ and res.votes.dtype == np.int16
    assert res.view_disocc_frac.shape == res.view_hole_frac.shape == (4,)
    assert (res.view_disocc_frac > 0).all(), "모든 뷰가 가림 해제를 본다"
    assert (res.view_disocc_frac <= 1).all() and (res.view_hole_frac >= 0).all()
    assert not (res.filled & res.unfilled).any()
    assert np.array_equal(res.filled | res.unfilled, hid)
    assert not res.filled[~hid].any() and not res.unfilled[~hid].any()
    assert res.votes[~hid].max() == 0

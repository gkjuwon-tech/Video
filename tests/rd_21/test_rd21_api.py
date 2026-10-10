"""RD-21 공개 API: 상수, 검증, 수치 도우미(업샘플·평면 깊이), import 위생."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from buc.read import pipeline as P
from rd21_helpers import (ALL_SIDS, fake_steps, k_from_fov, make_image, plane_oracle, up_torch)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_constants():
    assert P.PIPELINE_NAME == "rd21_read"
    assert P.PIPELINE_VERSION == "0.1.0"
    assert P.STEP_IDS == ALL_SIDS
    assert P.EXEC_ORDER == ("R0", "R1", "R2", "R4", "R5", "R6", "R3", "R7", "R8", "R9", "R10", "R11", "R12", "R13")
    assert sorted(P.EXEC_ORDER) == sorted(P.STEP_IDS)
    assert P.CORE_STEPS == ("R0", "R1", "R5", "R6")
    assert issubclass(P.StepOutputError, ValueError)


def test_import_does_not_load_torch():
    code = ("import sys; import buc.read.pipeline as p; "
            "p.default_steps(); "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": ROOT}, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("kw", [dict(z_near_m=10.0, z_far_m=5.0), dict(cam_height_m=0.0), dict(cam_height_m=-1.0),
                                dict(default_sigma_s=float("nan")), dict(depth_min_m=0.0),
                                dict(sensor_width_mm=float("inf"))])
def test_readopts_validation(kw):
    with pytest.raises(ValueError):
        P.ReadOpts(**kw)


def test_readopts_defaults():
    o = P.ReadOpts()
    assert (o.cam_height_m, o.z_near_m, o.z_far_m, o.default_sigma_s, o.depth_min_m, o.sensor_width_mm) == \
        (1.5, 0.5, 50.0, 1.5, 1e-3, 36.0)


def test_readstep_validation():
    f = lambda s, p: {"meta": {}}  # noqa: E731
    with pytest.raises(ValueError):
        P.ReadStep("R14", "x", "1", f)
    with pytest.raises(ValueError):
        P.ReadStep("R1", "", "1", f)
    with pytest.raises(ValueError):
        P.ReadStep("R1", "x", "", f)
    with pytest.raises(ValueError):
        P.ReadStep("R1", "x", "1", "notcallable")


def test_run_read_validates_before_any_call():
    calls = []
    img = make_image()
    with pytest.raises(TypeError):
        P.run_read(img.pixels, steps=fake_steps(calls))
    steps = fake_steps(calls)
    with pytest.raises(ValueError):  # 중복 sid
        P.run_read(img, steps=steps + (steps[3],))
    with pytest.raises(ValueError):  # 뒤 단계에 의존
        P.run_read(img, steps=fake_steps(calls, needs={"R1": ("R5",)}))
    with pytest.raises(ValueError):  # 자기 자신
        P.run_read(img, steps=fake_steps(calls, needs={"R1": ("R1",)}))
    with pytest.raises(ValueError):  # 모르는 sid
        P.run_read(img, steps=fake_steps(calls, needs={"R1": ("R99",)}))
    with pytest.raises(ValueError):  # R3는 R5·R6 뒤라 R3에 의존하는 R5는 불가
        P.run_read(img, steps=fake_steps(calls, needs={"R5": ("R3",)}))
    with pytest.raises(ValueError):
        P.run_read(img, steps=fake_steps(calls), params={"R77": {}})
    with pytest.raises(ValueError):
        P.run_read(img, steps=fake_steps(calls), params={"R1": 3})
    with pytest.raises(TypeError):
        P.run_read(img, steps=fake_steps(calls), params=[("R1", {})])
    assert calls == []


@pytest.mark.parametrize("hw,HW", [((3, 4), (12, 16)), ((12, 16), (48, 64)), ((5, 7), (19, 30)),
                                   ((1, 1), (4, 6)), ((6, 6), (6, 6))])
def test_upsample_grid_matches_torch(hw, HW):
    rng = np.random.default_rng(3)
    a = rng.normal(size=hw).astype(np.float32)
    out = P.upsample_grid(a, *HW)
    assert out.shape == HW and out.dtype == np.float32
    np.testing.assert_allclose(out, up_torch(a, *HW), atol=1e-5)
    if hw == HW:
        np.testing.assert_allclose(out, a, atol=1e-6)


def test_plane_depth_level_camera_matches_horizon_formula():
    H, W = 20, 30
    K = k_from_fov(55.0, W, H)
    d = P.plane_depth(K, np.array([0.0, -1.0, 0.0]), H, W, cam_height_m=1.5, z_near_m=0.5, z_far_m=50.0)
    assert d.shape == (H, W) and d.dtype == np.float32
    cy, fy = (H - 1) / 2.0, K[1, 1]
    for v in range(H):
        exp = np.clip(fy * 1.5 / (v - cy), 0.5, 50.0) if v > cy else 50.0
        np.testing.assert_allclose(d[v], exp, rtol=1e-5)


def test_plane_depth_tilted_points_lie_on_ground():
    H, W = 24, 32
    K = k_from_fov(70.0, W, H)
    pitch, roll = math.radians(-12.0), math.radians(5.0)
    up = np.array([math.cos(pitch) * math.sin(roll), -math.cos(pitch) * math.cos(roll), math.sin(pitch)])
    d = P.plane_depth(K, up, H, W, cam_height_m=2.0, z_near_m=0.3, z_far_m=80.0).astype(np.float64)
    np.testing.assert_allclose(d, plane_oracle(K, up, H, W, h=2.0, zn=0.3, zf=80.0), rtol=1e-5)
    Ki = np.linalg.inv(K)
    n_in = 0
    for v in range(H):
        for u in range(W):
            z = d[v, u]
            if 0.3 < z < 80.0:
                X = z * (Ki @ np.array([u, v, 1.0]))
                assert abs(up @ X + 2.0) < 1e-4
                n_in += 1
    assert n_in > 50
    assert np.all(np.isfinite(d)) and np.all(d > 0)

"""CAM-01: 14종 레지스트리·기본 파라미터·추종·결정론·Trajectory 보간·하류 이음매."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cam01_helpers import DT, PARAM_KEYS, RIG_NAMES, TIER1, is_rot, mass_damper_exact, rigs


def test_registry_constants():
    m = rigs()
    assert tuple(m.RIG_NAMES) == RIG_NAMES and len(set(RIG_NAMES)) == 14
    assert tuple(m.TIER1) == TIER1
    assert tuple(m.PARAM_KEYS) == PARAM_KEYS
    assert m.DT == DT and m.RATE_HZ == 1000 and m.V_EPS == 1e-3


@pytest.mark.parametrize("name", RIG_NAMES)
def test_default_params_are_complete_and_sane(name):
    m = rigs()
    rig = m.make_rig(name)
    n = rig.n_q
    assert rig.name == name and n >= 1
    qs = np.asarray(rig.q_scale)
    assert qs.shape == (n,) and np.all(qs > 0) and np.isfinite(qs).all()
    dp = m.default_params(name)
    assert set(dp) == set(PARAM_KEYS) | {"delay_s"}
    for k in PARAM_KEYS:
        assert len(dp[k]) == n and all(isinstance(x, float) for x in dp[k]), k
        assert k == "u_max" or all(np.isfinite(dp[k])), k
        np.testing.assert_allclose(np.asarray(rig.params[k], dtype=float), dp[k], atol=0)
    assert isinstance(dp["delay_s"], float) and 0.0 <= dp["delay_s"] <= 0.3
    assert min(dp["mass"]) > 0 and min(dp["ou_tau"]) > 0 and min(dp["u_max"]) > 0
    for k in ("damping", "quad_drag", "stiffness", "coulomb", "w2", "kp", "ki", "kd", "ff_gain", "ou_sigma"):
        assert min(dp[k]) >= 0, k
    dp["mass"] = (123.0,) * n                                           # 반환값을 바꿔도 내부 기본값 불변
    assert m.default_params(name)["mass"][0] != 123.0
    rig2 = m.make_rig(name, params={"damping": 7.5})
    np.testing.assert_allclose(rig2.params["damping"], 7.5)
    np.testing.assert_allclose(rig2.params["mass"], rig.params["mass"])


def test_tier1_defaults_follow_the_documented_character():
    m = rigs()
    sl, dl, cr = (m.make_rig(n) for n in TIER1)
    assert (sl.n_q, dl.n_q, cr.n_q) == (1, 1, 2)
    for r in (sl, dl, cr):
        p = r.params
        assert np.all(np.asarray(p["ou_sigma"]) == 0) and p["delay_s"] == 0.0     # 결정론 3종
        assert np.all(np.asarray(p["ff_gain"]) == 1.0)
    assert 0.5 <= sl.q_scale[0] <= 2.0                                  # 슬라이더: 길이 1 m 안팎
    assert 2.0 <= dl.q_scale[0] <= 10.0
    assert dl.params["mass"][0] >= 10.0 * sl.params["mass"][0]          # 달리는 슬라이더보다 훨씬 무겁다
    assert cr.q_scale[0] >= np.radians(20.0)                            # ±20° 오빗이 범위 안
    hh = m.make_rig("shoulder_handheld")
    assert 0.15 <= hh.params["delay_s"] <= 0.25 and hh.n_q == 6        # 반응 지연 δ = 150~250 ms
    assert np.max(hh.params["ou_sigma"]) > 0
    mc = m.make_rig("motion_control")                                   # 완벽한 재현성
    assert np.all(np.asarray(mc.params["ou_sigma"]) == 0) and mc.params["delay_s"] == 0.0
    assert np.max(m.make_rig("steadicam").params["w2"]) > 0             # 슬레드 진자
    assert np.max(m.make_rig("cablecam").params["w2"]) > 0
    assert np.max(m.make_rig("tripod_fluid").params["coulomb"]) > 0     # 정지마찰
    assert np.max(m.make_rig("cine_drone").params["quad_drag"]) > 0
    assert np.max(m.make_rig("underwater").params["quad_drag"]) > 0
    assert np.max(m.make_rig("car_mount").params["ou_sigma"]) > 0       # 노면 거칠기


def _move(rig):
    return np.zeros(rig.n_q), 0.3 * np.asarray(rig.q_scale)


@pytest.mark.parametrize("name", RIG_NAMES)
def test_every_rig_simulates_tracks_and_is_deterministic(name):
    m = rigs()
    rig = m.make_rig(name)
    a0, a1 = _move(rig)
    it = m.Intent.point_to_point(a0, a1, 1.2, tail_s=1.8)
    tr = rig.simulate(it, seed=11)
    n = 3001
    assert tr.q.shape == tr.qd.shape == tr.u.shape == tr.q_ref.shape == (n, rig.n_q)
    assert tr.pose.shape == (n, 4, 4) and tr.t.shape == (n,)
    for arr in (tr.q, tr.qd, tr.u, tr.pose):
        assert np.isfinite(arr).all()
    np.testing.assert_allclose(tr.t, np.arange(n) * DT, atol=1e-12)
    np.testing.assert_array_equal(tr.q[0], a0)                          # 기본 q0 = 첫 웨이포인트
    np.testing.assert_array_equal(tr.qd[0], 0.0)
    np.testing.assert_allclose(tr.pose[0], rig.fk(tr.q[0]), atol=1e-14)
    np.testing.assert_allclose(tr.pose[1234], rig.fk(tr.q[1234]), atol=1e-14)   # 자세 = FK(q) (홀로노믹)
    assert is_rot(tr.pose[-1][:3, :3], 1e-9) and is_rot(tr.pose[2000][:3, :3], 1e-9)
    assert tr.rig is rig or tr.rig.name == name
    err = (tr.q - tr.q_ref) / np.asarray(rig.q_scale)
    noisy = bool(np.max(rig.params["ou_sigma"]) > 0)
    if noisy:
        assert np.sqrt((err**2).mean(axis=0)).max() <= 0.1               # 외란이 있어도 RMS 오차 ≤ 0.1 스케일
    else:
        assert np.abs(err).max() <= 0.01                                 # 외란 없는 리그: 기본 ff 로 1% 이내 추종
        assert np.abs((tr.q[-1] - a1) / rig.q_scale).max() <= 0.005
    tr2 = rig.simulate(it, seed=11)
    assert np.array_equal(tr.q, tr2.q) and np.array_equal(tr.pose, tr2.pose) and np.array_equal(tr.u, tr2.u)
    if noisy:
        assert not np.array_equal(tr.q, rig.simulate(it, seed=12).q)     # 시드가 외란을 바꾼다
    elif name in TIER1:
        assert np.array_equal(tr.q, rig.simulate(it, seed=12).q)         # 결정론 리그는 시드와 무관


def test_noisy_rig_disturbance_is_bounded_and_nonzero_at_rest():
    m = rigs()
    rig = m.make_rig("shoulder_handheld")
    it = m.Intent(np.array([0.0, 1.0]), np.zeros((2, rig.n_q)), tail_s=7.0)
    tr = rig.simulate(it, seed=5)
    rms = np.sqrt((tr.q**2).mean(axis=0))
    assert np.all(rms > 0)                                               # 흔들린다
    assert np.all(rms / rig.q_scale < 0.1)                               # 그러나 폭주하지 않는다
    assert np.abs(tr.pose[:, :3, 3]).max() < 1.0


def test_trajectory_interpolation_matches_closed_form_off_grid():
    m = rigs()
    rig = m.make_rig("slider", params=dict(mass=2.0, damping=3.0))
    tr = rig.simulate_force(1.5, 2.0)
    grid = tr.t[[0, 10, 1000, 2000]]
    np.testing.assert_allclose(tr.pose_at(grid), tr.pose[[0, 10, 1000, 2000]], atol=1e-12)
    ts = np.array([0.0005, 0.0123, 0.5004, 1.2345678, 1.9995, 2.0])
    q, _ = mass_damper_exact(ts, 2.0, 3.0, 1.5)
    got = np.asarray(tr.pose_at(ts))
    np.testing.assert_allclose(got[:, 0, 3], q, atol=1e-9)               # 3차 에르미트 보간 오차 ≪ 1e-9
    one = np.asarray(tr.pose_at(np.array([0.7])))
    assert one.shape == (1, 4, 4)
    for bad in (np.array([-0.001]), np.array([2.001])):
        with pytest.raises(ValueError):
            tr.pose_at(bad)
    # 프레임 샘플: t_k = k/fps, k = 0..floor(T·fps)
    for fps in (24.0, 30.0, 60.0):
        fp = np.asarray(tr.frame_poses(fps))
        k = int(np.floor(2.0 * fps + 1e-9)) + 1
        assert fp.shape == (k, 4, 4)
        np.testing.assert_allclose(fp, tr.pose_at(np.arange(k) / fps), atol=1e-12)
    np.testing.assert_allclose(np.asarray(tr.frame_poses(24.0))[24, 0, 3], mass_damper_exact(1.0, 2.0, 3.0, 1.5)[0],
                               atol=1e-9)


def test_downstream_seam_move_recipe_usage():
    """CAM-02 가 쓰는 전형적 흐름: 의도 -> 리그 -> 24 fps 포즈 열 (c2w, OpenCV 규약, float64)."""
    m = rigs()
    rig = m.make_rig("crane_jib")
    it = m.Intent.point_to_point([0.0, 0.0], [np.radians(20.0), 0.0], 3.0, tail_s=1.0)
    tr = rig.simulate(it)
    poses = np.asarray(tr.frame_poses(24.0))
    assert poses.dtype == np.float64 and poses.shape == (97, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-12)
    assert abs(tr.q[-1, 0] - np.radians(20.0)) < 1e-3
    pivot = np.array([0.0, 0.0, 3.0])
    assert np.abs(np.linalg.norm(poses[:, :3, 3] - pivot, axis=1) - 3.0).max() < 1e-9
    # 속도 연속·정지 출발: 첫 프레임 간 이동은 2차 이하
    d = np.linalg.norm(np.diff(poses[:4, :3, 3], axis=0), axis=1)
    assert d[0] < d[1] < d[2]


def test_module_is_torch_free_and_import_has_no_side_effects():
    import buc

    root = str(Path(buc.__file__).resolve().parents[1])
    code = ("import sys, numpy as np; s0 = np.random.get_state()[1].copy();"
            "import buc.camera.rigs as r;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert (np.random.get_state()[1] == s0).all();"
            "r.make_rig('slider').simulate_force(1.0, 0.1);"
            "assert (np.random.get_state()[1] == s0).all()")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stderr[-800:]

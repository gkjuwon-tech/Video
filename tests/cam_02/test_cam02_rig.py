"""CAM-02: CAM-01 리그 실현(rig_path) — 해석 경로가 정말 CAM-01 FK 와 같은 식이고, 1 kHz ODE 가 그것을 따른다."""
import math

import numpy as np
import pytest

from cam02_helpers import DEFAULT_FRAMES, T4, is_rot, mm, odd_home, orbit_local, subject_in_camera

Z = 3.0
POS_TOL = 1e-4          # 위치 오차 / z_s
ROT_TOL = 1e-4          # 회전행렬 원소 오차


def test_cam01_fk_is_the_same_formula_as_the_ideal_paths():
    """이음매: 이상 경로 = home_c2w=C_0 인 CAM-01 리그의 FK (dolly 축 +z, slider 축 +x, crane_jib 반지름 = z_s)."""
    from buc.camera.rigs import make_rig
    c0 = odd_home()
    rng = np.random.default_rng(7)
    crane = make_rig("crane_jib", geom={"radius": Z}, home_c2w=c0)
    dolly = make_rig("dolly", home_c2w=c0)
    slider = make_rig("slider", home_c2w=c0)
    for a in rng.uniform(-math.pi, math.pi, 12):
        np.testing.assert_allclose(crane.fk([a, 0.0]), c0 @ orbit_local(a, Z), atol=1e-12)
    for x in rng.uniform(-2.0, 2.0, 6):
        np.testing.assert_allclose(dolly.fk([x]), c0 @ T4(None, [0, 0, x]), atol=1e-12)
        np.testing.assert_allclose(slider.fk([x]), c0 @ T4(None, [x, 0, 0]), atol=1e-12)


@pytest.mark.parametrize("z", [0.8, 3.0, 8.0])
@pytest.mark.parametrize("name,kw,rig", [
    ("push_in", dict(m=1.5), "dolly"), ("push_in", dict(m=4.0), "dolly"), ("push_in", dict(m=0.6), "dolly"),
    ("slider", dict(slide_zs=0.10), "slider"), ("slider", dict(slide_zs=-0.15), "slider"),
    ("orbit", dict(orbit_deg=20.0), "crane_jib"), ("orbit", dict(orbit_deg=-20.0), "crane_jib"),
    ("orbit", dict(orbit_deg=90.0), "crane_jib"), ("orbit", dict(orbit_deg=20.0, round_trip=True), "crane_jib"),
    ("push_in", dict(m=1.5, round_trip=True), "dolly"),
])
def test_rig_path_tracks_the_ideal_path(z, name, kw, rig):
    m = mm()
    c0 = odd_home()
    sp = m.recipe(name, z, **kw)
    ideal = m.ideal_path(sp, c0)
    real = m.rig_path(sp, c0)
    assert real.rig == rig and real.traj is not None and real.traj.rig.name == rig
    assert real.spec == sp and real.n_frames == DEFAULT_FRAMES
    assert real.c2w.shape == ideal.c2w.shape and np.isfinite(real.c2w).all()
    np.testing.assert_allclose(real.t, ideal.t, atol=1e-12)
    perr = np.linalg.norm(real.c2w[:, :3, 3] - ideal.c2w[:, :3, 3], axis=1).max()
    rerr = np.abs(real.c2w[:, :3, :3] - ideal.c2w[:, :3, :3]).max()
    assert perr <= POS_TOL * z and rerr <= ROT_TOL, (perr / z, rerr)
    np.testing.assert_allclose(real.c2w[0], c0, atol=1e-9)                          # 첫 프레임 = C_0
    assert all(is_rot(real.c2w[k, :3, :3], 1e-9) for k in range(0, DEFAULT_FRAMES, 12))
    np.testing.assert_allclose(real.subject, ideal.subject, atol=1e-12)
    np.testing.assert_allclose(real.focus_m, ideal.focus_m, atol=POS_TOL * z * 2)   # 초점은 실현된 경로 기준
    if kw.get("round_trip"):
        np.testing.assert_allclose(real.c2w[-1], c0, atol=POS_TOL * max(z, 1.0))   # 왕복: 시뮬이어도 거의 정확히 복귀


def test_rig_orbit_keeps_the_subject_centered():
    m = mm()
    c0 = odd_home()
    p = m.rig_path(m.recipe("orbit", Z, orbit_deg=20.0), c0)
    xc = subject_in_camera(p.c2w, p.subject)
    assert np.abs(xc - np.array([0.0, 0.0, Z])).max() <= POS_TOL * Z


def test_rig_path_is_deterministic_and_seed_independent_for_noiseless_rigs():
    m = mm()
    np.random.seed(99)
    st = np.random.get_state()[1].copy()
    sp = m.recipe("orbit", Z)
    a = m.rig_path(sp, odd_home(), seed=0)
    b = m.rig_path(sp, odd_home(), seed=0)
    c = m.rig_path(sp, odd_home(), seed=5)                                          # 3종 리그는 ou_sigma = 0
    assert a.c2w.tobytes() == b.c2w.tobytes() == c.c2w.tobytes()
    assert a.focus_m.tobytes() == b.focus_m.tobytes()
    assert (np.random.get_state()[1] == st).all()


def test_rack_focus_has_no_rig_motion():
    m = mm()
    sp = m.recipe("rack_focus", Z, focus_to_m=10.0)
    c0 = odd_home()
    ideal, real = m.ideal_path(sp, c0), m.rig_path(sp, c0)
    assert real.rig == "locked" and real.traj is None and ideal.rig == "ideal"
    np.testing.assert_allclose(real.c2w, ideal.c2w, atol=0)
    np.testing.assert_allclose(real.focus_m, ideal.focus_m, atol=1e-12)


def test_rig_path_rejects_invalid_c0():
    m = mm()
    with pytest.raises(ValueError):
        m.rig_path(m.recipe("slider", Z), np.full((4, 4), np.nan))

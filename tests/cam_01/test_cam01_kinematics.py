"""CAM-01: 홀로노믹 제약(FK) — 슬라이더·달리·오빗(crane_jib) 3종과 14종 FK 표."""
import math

import numpy as np
import pytest

from cam01_helpers import RIG_NAMES, Rx, Ry, is_rot, odd_home, ref_fk, rigs


@pytest.mark.parametrize("name", RIG_NAMES)
def test_fk_matches_contract_table_for_all_14(name):
    m = rigs()
    home = odd_home()
    rig = m.make_rig(name, home_c2w=home)
    rng = np.random.default_rng(100 + RIG_NAMES.index(name))
    qs = rng.uniform(-1.0, 1.0, (6, rig.n_q)) * rig.q_scale
    out = np.asarray(rig.fk(qs))
    assert out.shape == (6, 4, 4)
    for q, T in zip(qs, out):
        np.testing.assert_allclose(T, ref_fk(rig, q), atol=1e-12)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=0)
        assert is_rot(T[:3, :3], 1e-12)
    one = np.asarray(rig.fk(qs[0]))
    assert one.shape == (4, 4)
    np.testing.assert_allclose(one, out[0], atol=0)
    np.testing.assert_allclose(rig.fk(np.zeros(rig.n_q)), home, atol=1e-14)    # q=0 은 홈 포즈
    with pytest.raises(ValueError):
        rig.fk(np.zeros(rig.n_q + 1))


def test_home_pose_is_copied_and_validated():
    m = rigs()
    home = odd_home()
    rig = m.make_rig("slider", home_c2w=home)
    home[0, 3] = 99.0                                                # 호출자 배열을 바꿔도 리그는 불변
    assert abs(rig.home_c2w[0, 3] - 1.0) < 1e-15
    np.testing.assert_allclose(m.make_rig("slider").home_c2w, np.eye(4), atol=0)
    for bad in (np.eye(3), np.full((4, 4), np.nan), np.diag([2.0, 1.0, 1.0, 1.0])):
        with pytest.raises(ValueError):
            m.make_rig("slider", home_c2w=bad)


def test_slider_and_dolly_are_pure_translations_on_the_axis():
    m = rigs()
    home = odd_home()
    for name, default_axis in (("slider", [1.0, 0.0, 0.0]), ("dolly", [0.0, 0.0, 1.0])):
        rig = m.make_rig(name, home_c2w=home)
        np.testing.assert_allclose(rig.geom["axis"], default_axis, atol=0)    # 슬라이더 = 옆, 달리 = 앞
        tr = rig.simulate(m.Intent.point_to_point([0.0], [0.7], 1.5, tail_s=0.5))
        np.testing.assert_allclose(tr.pose[:, :3, :3], np.broadcast_to(home[:3, :3], (len(tr.t), 3, 3)), atol=1e-14)
        d = tr.pose[:, :3, 3] - home[:3, 3]
        ax_w = home[:3, :3] @ np.asarray(default_axis)
        along = d @ ax_w
        perp = d - along[:, None] * ax_w
        assert np.abs(perp).max() < 1e-12                            # 레일을 벗어나지 않는다 (홀로노믹)
        np.testing.assert_allclose(along, tr.q[:, 0], atol=1e-12)    # 이동량 = 일반화 좌표 s [m]
    rig = m.make_rig("slider", geom={"axis": (1.0, 1.0, 0.0)})
    np.testing.assert_allclose(rig.geom["axis"], np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0), atol=1e-15)
    np.testing.assert_allclose(rig.fk([2.0])[:3, 3], np.array([1.0, 1.0, 0.0]) * math.sqrt(2.0), atol=1e-12)


def test_orbit_crane_jib_keeps_radius_and_aim_exactly():
    m = rigs()
    home = odd_home()
    rig = m.make_rig("crane_jib", home_c2w=home, geom={"radius": 2.5})
    assert rig.geom["radius"] == 2.5
    pivot = (home @ np.array([0.0, 0.0, 2.5, 1.0]))[:3]
    tr = rig.simulate(m.Intent(np.array([0.0, 1.0, 2.0]), np.array([[0, 0], [0.5, 0.3], [-0.4, -0.2]]), tail_s=1.0))
    c = tr.pose[:, :3, 3]
    z = tr.pose[:, :3, 2]
    r = np.linalg.norm(c - pivot, axis=1)
    np.testing.assert_allclose(r, 2.5, atol=1e-9)                    # 반지름 상수
    dirn = (pivot - c) / r[:, None]
    np.testing.assert_allclose(np.einsum("ij,ij->i", z, dirn), 1.0, atol=1e-9)    # 광축이 항상 피벗을 본다
    # 방위각만 움직일 때 높이(홈의 y축 성분)는 변하지 않는다
    tr2 = rig.simulate(m.Intent.point_to_point([0.0, 0.0], [0.6, 0.0], 1.0, tail_s=0.2))
    y_home = home[:3, :3] @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose((tr2.pose[:, :3, 3] - home[:3, 3]) @ y_home, 0.0, atol=1e-12)


@pytest.mark.parametrize("deg", [20.0, -20.0, 5.0])
def test_orbit_closed_form_pose(deg):
    m = rigs()
    rig = m.make_rig("crane_jib")                                    # 홈 = 단위행렬, 반지름 기본 3 m
    R = rig.geom["radius"]
    assert R == 3.0
    phi = math.radians(deg)
    T = np.asarray(rig.fk([phi, 0.0]))
    np.testing.assert_allclose(T[:3, 3], [-R * math.sin(phi), 0.0, R * (1.0 - math.cos(phi))], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], Ry(phi), atol=1e-12)
    ang = math.acos(np.clip((np.trace(np.eye(3).T @ T[:3, :3]) - 1.0) / 2.0, -1, 1))
    assert abs(ang - abs(phi)) < 1e-7                                # 오빗 각 = 광축 회전각
    psi = 0.3
    T = np.asarray(rig.fk([0.0, psi]))
    np.testing.assert_allclose(T[:3, 3], [0.0, R * math.sin(psi), R * (1.0 - math.cos(psi))], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], Rx(psi), atol=1e-12)


def test_techno_crane_telescopes_along_the_optical_axis():
    m = rigs()
    rig = m.make_rig("techno_crane")
    R = rig.geom["radius"]
    for dl in (-1.0, 0.0, 1.5):
        T = np.asarray(rig.fk([0.2, 0.1, dl]))
        pivot = np.array([0.0, 0.0, R])
        assert abs(np.linalg.norm(T[:3, 3] - pivot) - (R + dl)) < 1e-12
        np.testing.assert_allclose(T[:3, 3], pivot - (R + dl) * T[:3, 2], atol=1e-12)   # 피벗은 광축 위
    np.testing.assert_allclose(rig.fk([0.2, 0.1, 0.0]), m.make_rig("crane_jib").fk([0.2, 0.1]), atol=1e-14)


def test_geom_and_param_validation():
    m = rigs()
    with pytest.raises(KeyError):
        m.make_rig("not_a_rig")
    with pytest.raises(KeyError):
        m.default_params("not_a_rig")
    for geom in ({"radius": 0.0}, {"radius": -1.0}, {"axis": (0.0, 0.0, 0.0)}, {"axis": (1.0, 0.0)}, {"nope": 1.0}):
        with pytest.raises(ValueError):
            m.make_rig("crane_jib", geom=geom)
    for params in ({"mass": 0.0}, {"mass": -1.0}, {"damping": -0.1}, {"kp": -1.0}, {"ou_tau": 0.0},
                   {"u_max": 0.0}, {"delay_s": -0.1}, {"bogus": 1.0}, {"mass": [1.0, 2.0]}, {"mass": float("nan")}):
        with pytest.raises(ValueError):
            m.make_rig("slider", params=params)                      # 슬라이더는 1 DOF: 길이 2 벡터는 오류
    rig = m.make_rig("crane_jib", params={"mass": [100.0, 200.0], "damping": 5.0})   # 스칼라 브로드캐스트·벡터 허용
    np.testing.assert_allclose(rig.params["mass"], [100.0, 200.0])
    np.testing.assert_allclose(rig.params["damping"], [5.0, 5.0])
    assert np.asarray(rig.params["kp"]).shape == (2,)

"""CAM-04: FaceSpec 데이터 형태 / 직렬화 / 검증, 상류(RD-15)·하류(CAM-03, PRD-04) 이음매, 부작용 없음."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cam04_helpers import UP, fe, frame, make_face, orbit_path

P0 = np.zeros(3)


# ---------------------------------------------------------------- FaceSpec
def test_facespec_is_frozen_keyword_only_with_normalized_float64_fields():
    m = fe()
    face = m.FaceSpec(center=[1, 2, 3], front=[0, 0, -5.0], up=[0.3, -1.0, 0.5], h_f=0.1)
    assert dataclasses.is_dataclass(face)
    for name in ("center", "front", "up"):
        a = getattr(face, name)
        assert isinstance(a, np.ndarray) and a.shape == (3,) and a.dtype == np.float64
    assert np.allclose(face.center, [1, 2, 3])
    assert np.linalg.norm(face.front) == pytest.approx(1.0, abs=1e-12) and np.allclose(face.front, [0, 0, -1])
    assert np.linalg.norm(face.up) == pytest.approx(1.0, abs=1e-12)
    assert abs(face.up @ face.front) < 1e-12                      # up 은 front 에 직교화된다
    assert (face.up @ np.array([0.3, -1.0, 0.5])) > 0.0           # 그리고 원래 방향 쪽을 유지
    r, u, f = frame(face)
    assert np.allclose(face.right, r) and np.allclose(face.right, np.cross(face.up, face.front))
    assert np.linalg.norm(face.right) == pytest.approx(1.0, abs=1e-12)
    assert face.sigma_z == 0.0 and face.head_radius_m == pytest.approx(0.10)
    assert face.h_f == pytest.approx(0.1)
    with pytest.raises(TypeError):
        m.FaceSpec([0, 0, 0], [0, 0, 1], [0, -1, 0], 0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        face.h_f = 0.5


def test_facespec_does_not_alias_inputs():
    m = fe()
    c = np.array([0.0, 0.0, 2.0])
    face = m.FaceSpec(center=c, front=[0, 0, 1], up=list(UP), h_f=0.1)
    c[2] = 99.0
    assert face.center[2] == 2.0


@pytest.mark.parametrize("kw", [
    dict(center=[0, 0]),                                  # 모양
    dict(center=[0, 0, np.nan]),
    dict(front=[0, 0, 0]),                                # 영벡터
    dict(front=[np.inf, 0, 1]),
    dict(up=[0, 0, 1]),                                   # front 와 평행
    dict(up=[0, 0, -3.0]),
    dict(h_f=0.0), dict(h_f=-0.1), dict(h_f=1.01), dict(h_f=float("nan")),
    dict(sigma_z=-0.01), dict(sigma_z=float("nan")),
    dict(head_radius_m=0.0), dict(head_radius_m=-0.1), dict(head_radius_m=float("inf")),
])
def test_facespec_validation(kw):
    base = dict(center=[0, 0, 2], front=[0, 0, 1], up=list(UP), h_f=0.1)
    base.update(kw)
    with pytest.raises(ValueError):
        fe().FaceSpec(**base)


def test_facespec_accepts_inf_sigma_and_alpha_is_zero():
    face = make_face(h_f=0.1, sigma_z=float("inf"))
    assert fe().limits("F2", face.sigma_z).yaw_deg == pytest.approx(0.0, abs=1e-9)


def test_facespec_dict_roundtrip_is_json_safe():
    m = fe()
    face = m.FaceSpec(center=[0.1, -0.2, 2.5], front=[0.2, 0.1, -1.0], up=[0, -1, 0.1], h_f=0.07, sigma_z=0.013,
                      head_radius_m=0.12)
    d = face.to_dict()
    assert set(d) == {"center", "front", "up", "h_f", "sigma_z", "head_radius_m"}
    d2 = json.loads(json.dumps(d))
    back = m.FaceSpec.from_dict(d2)
    for name in ("center", "front", "up"):
        assert np.allclose(getattr(back, name), getattr(face, name), rtol=0, atol=1e-12)   # 재정규화 오차만 허용
    assert (back.h_f, back.sigma_z, back.head_radius_m) == (face.h_f, face.sigma_z, face.head_radius_m)
    again = m.FaceSpec.from_dict(json.loads(json.dumps(back.to_dict())))
    for name in ("center", "front", "up"):
        assert np.allclose(getattr(again, name), getattr(face, name), rtol=0, atol=1e-12)


def test_facespec_from_dict_rejects_missing_and_unknown_keys():
    m = fe()
    d = make_face().to_dict()
    bad = dict(d)
    bad.pop("h_f")
    with pytest.raises(ValueError):
        m.FaceSpec.from_dict(bad)
    bad = dict(d)
    bad["extra"] = 1
    with pytest.raises(ValueError):
        m.FaceSpec.from_dict(bad)
    opt = {k: d[k] for k in ("center", "front", "up", "h_f")}       # sigma_z, head_radius_m 은 선택
    got = m.FaceSpec.from_dict(opt)
    assert got.sigma_z == 0.0 and got.head_radius_m == pytest.approx(0.10)


# ---------------------------------------------------------------- 이음매: 상류 RD-15 / FND-02 ReadPacket.faces
def test_faces_from_read_packet_faces_tuple_as_dicts():
    # RD-15 는 ReadPacket.faces 에 FaceSpec.to_dict() 형태를 넣는다 -> evaluate_faces 가 그대로 받는다.
    from buc.core.contracts import ReadPacket   # noqa: F401  (FND-02 와 함께 import 가능해야 한다)
    m = fe()
    f_near, f_far = make_face(front=(0, 0, -1), h_f=0.10), make_face(front=(0, 0, -1), d=6.0, h_f=0.02)
    faces = (f_near.to_dict(), f_far.to_dict())
    path = orbit_path(f_near, P0, 20.0)
    r_dict = m.evaluate_faces(faces, path)
    r_obj = m.evaluate_faces([f_near, f_far], path)
    assert r_dict.ok and r_dict.violations == r_obj.violations and r_dict.allowed_grades == r_obj.allowed_grades
    assert [v.category for v in r_dict.verdicts] == [v.category for v in r_obj.verdicts] == ["F2", "F1"]
    with pytest.raises(ValueError):
        m.evaluate_faces([{"center": [0, 0, 2]}], path)
    with pytest.raises((ValueError, TypeError)):
        m.evaluate_faces([42], path)


# ---------------------------------------------------------------- 이음매: 하류 CAM-03/PRD-04 쓰임새
def test_downstream_usage_dp0_hard_constraint_and_menu_grades():
    m = fe()
    face = make_face(front=(0, 0, -1), h_f=0.10, sigma_z=0.01)       # F2, alpha_sigma = 26.7 도
    # DP-0: 후보 경로를 하드 제약으로 거른다.
    cands = {d: orbit_path(face, P0, d) for d in (10.0, 20.0, 26.0, 28.0, 45.0, 90.0)}
    keep = [d for d, c in cands.items() if not m.face_viol_sigma([face], c)]
    assert keep == [10.0, 20.0, 26.0]
    # PRD-04: 메뉴는 열리는 등급과 각도 예산을 보여 준다.
    rep = m.evaluate_faces([face], cands[20.0])
    assert rep.allowed_grades == ("safe", "extended")
    assert m.orbit_budget_deg(face, P0, +1.0) == pytest.approx(26.74, abs=0.3)
    # 같은 얼굴이 뒷모습이면 90 도 오빗(확장 등급 I4)은 폐기, 얼굴이 없으면 통과.
    back = make_face(front=(0, 0, 1), h_f=0.10)
    assert m.face_viol_sigma([back], orbit_path(back, P0, 90.0)) is True
    assert m.face_viol_sigma([], orbit_path(back, P0, 90.0)) is False


def test_cam02_poses_are_consumable_when_available():
    # CAM-02 의 orbit 경로(c2w float64 (N,4,4)) 를 그대로 먹는다. CAM-02 가 없으면 건너뛴다.
    moves = pytest.importorskip("buc.camera.moves")
    m = fe()
    z_s = 3.0
    path = moves.ideal_path(moves.recipe("orbit", z_s))               # +20 도 오빗, 피사체 (0,0,3)
    face = fe().FaceSpec(center=[0, 0, z_s], front=[0, 0, -1], up=list(UP), h_f=0.10)
    v = m.evaluate_path(face, path.c2w)
    assert v.ok and v.yaw_dev_deg == pytest.approx(20.0, abs=1e-6) and v.category == "F2"
    big = moves.ideal_path(moves.bench_spec("I4", z_s))               # 90 도
    assert not m.evaluate_path(face, big.c2w).ok


# ---------------------------------------------------------------- 환경
def test_module_is_torch_free_and_import_has_no_side_effects():
    import buc

    root = str(Path(buc.__file__).resolve().parents[1])
    code = ("import sys, numpy as np, random; s0 = np.random.get_state()[1].copy(); r0 = random.getstate();"
            "import buc.camera.face_envelope as m;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "f = m.FaceSpec(center=[0,0,2], front=[0,0,1], up=[0,-1,0], h_f=0.1);"
            "p = f.center + 2*(np.sin(1.0)*f.right + np.cos(1.0)*f.front);"
            "e = m.unobserved_exposure(f, np.zeros(3), p); assert 0.0 <= e <= 1.0;"
            "m.orbit_budget_deg(f, np.zeros(3), 1.0); m.limits('F2', 0.01); m.alpha_sigma_deg(0.02);"
            "assert 'torch' not in sys.modules;"
            "assert (np.random.get_state()[1] == s0).all() and random.getstate() == r0")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=55)
    assert p.returncode == 0, p.stderr[-800:]


def test_public_names_exported():
    m = fe()
    for name in ("GRADES", "CATEGORIES", "EPS_F", "W_F_M", "BACK_YAW_DEG", "H_F1_MAX", "H_F2_MAX",
                 "YAW_TABLE_DEG", "PITCH_TABLE_DEG", "M_MAX", "UNOBS_MAX", "FACE_CAP_HALF_DEG", "HEAD_RADIUS_M",
                 "M_TOL", "ANGLE_TOL_DEG", "UNOBS_TOL", "FaceSpec", "FaceLimits", "FaceVerdict", "FaceReport",
                 "alpha_sigma_deg", "face_category", "limits", "face_angles", "unobserved_exposure",
                 "evaluate_path", "evaluate_faces", "face_viol_sigma", "orbit_budget_deg"):
        assert hasattr(m, name), name
        assert name in m.__all__, name
    # 표 상수는 계약서 값 그대로
    assert m.YAW_TABLE_DEG["F2"] == 30.0 and m.YAW_TABLE_DEG["F3"] == 15.0
    assert m.PITCH_TABLE_DEG["F2"] == 15.0 and m.PITCH_TABLE_DEG["F3"] == 10.0
    assert m.M_MAX["F1"] == 1.5 and m.M_MAX["F2"] == 1.0 and m.M_MAX["F3"] == 1.0
    assert m.UNOBS_MAX == {"F0": 0.05, "F1": 0.03, "F2": 0.05, "F3": 0.05}

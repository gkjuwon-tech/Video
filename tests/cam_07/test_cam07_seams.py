"""CAM-07: 하류 이음매(CAM-02 경로·CAM-04 규약과 함께 쓰기), 결정론, torch 미로드."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cam07_helpers import K_pinhole, cl, poses, rot

ROOT = Path(__file__).resolve().parents[2]


def test_orbit_around_subject_with_pillar_between():
    """피사체 (0,0,4) 둘레 반지름 4 의 오빗: 기둥이 오빗 경로에 있으면 충돌, 반지름 밖이면 통과.
    CAM-02 오빗과 같은 기하(피사체 중심 원호)를 해석적으로 만든다."""
    m = cl()
    P = np.array([0.0, 0.0, 4.0])
    ang = np.radians(np.linspace(0.0, 90.0, 37))
    # C_0 = 원점에서 피사체를 보는 카메라, +각은 카메라가 -x 로 이동
    cams = np.stack([P[0] - 4.0 * np.sin(ang), np.zeros_like(ang), P[2] - 4.0 * np.cos(ang)], axis=1)
    T = poses(cams)
    on_path = m.Sphere(center=[-4.0 * np.sin(np.radians(45)), 0.0, 4.0 - 4.0 * np.cos(np.radians(45))], radius=0.3)
    off_path = m.Sphere(center=[-2.0, 0.0, 2.0], radius=0.3)                       # 호의 안쪽(피사체 쪽)
    assert not m.collision_free([on_path], T, camera_radius=0.1)
    assert m.collision_free([off_path], T, camera_radius=0.1)
    ra = m.evaluate_path([on_path], T)
    rb = m.evaluate_path([off_path], T)
    assert ra.collision and rb.ok and rb.score > 0.9
    assert ra.worst_frame == pytest.approx(18.0, abs=1.0)                          # 45도 = 37프레임의 가운데
    # 피사체 가시성: 기둥이 카메라-피사체 시선을 가리는 프레임 구간만 낮아진다
    sub = P[None]
    block = m.Sphere(center=[-1.0, 0.0, 3.0], radius=0.5)
    f = m.visible_fraction([block], T, sub)
    assert f.shape == (37,) and set(np.unique(f)) <= {0.0, 1.0}
    assert 0 < (f == 0).sum() < 37 and f[0] == 1.0


def test_push_in_free_distance_matches_path_check():
    """CAM-02 푸시인 한도를 free_distance 로 얻고, 같은 경로를 collision_free 로 교차 검증."""
    m = cl()
    obs = [m.Sphere(center=[0.0, 0.0, 3.0], radius=0.5), m.HalfSpace(normal=[0, -1, 0], offset=-1.0)]
    d = m.free_distance(obs, [0, 0, 0], [0, 0, 1], radius=0.1)
    assert d == pytest.approx(3.0 - 0.5 - 0.1, abs=1e-5)
    zs = np.linspace(0.0, d - 1e-3, 30)
    assert m.collision_free(obs, np.stack([np.zeros(30), np.zeros(30), zs], axis=1), camera_radius=0.1)
    zs = np.linspace(0.0, d + 0.05, 30)
    assert not m.collision_free(obs, np.stack([np.zeros(30), np.zeros(30), zs], axis=1), camera_radius=0.1)


def test_obstacles_from_world_dicts_pipeline():
    """하류(WLD/DP-0)는 장애물을 JSON 딕셔너리로 넘긴다."""
    m = cl()
    specs = [{"type": "sphere", "center": [0, 0, 5], "radius": 1.0},
             {"type": "box", "center": [2, 0, 3], "half_extents": [0.5, 0.5, 0.5]},
             {"type": "halfspace", "normal": [0, -1, 0], "offset": -1.5}]
    obs = [m.obstacle_from_dict(d) for d in specs]
    T = poses(np.stack([np.linspace(-1, 1, 11), np.zeros(11), np.zeros(11)], axis=1))
    rep = m.evaluate_path(obs, T, subject_points=np.array([[0.0, 0.0, 3.5]]), K=K_pinhole(100.0, 64, 48), size=(64, 48))
    assert rep.ok and rep.min_visible_fraction == 1.0 and rep.min_clearance > 0.1


def test_determinism_and_global_rng_untouched():
    m = cl()
    obs = [m.Sphere(center=[0, 0, 5], radius=1.0), m.Box(center=[1, 0, 3], half_extents=[0.3, 0.3, 0.3], rotation=rot([1, 1, 0], 20.0))]
    P = np.random.default_rng(30).uniform(-3, 4, size=(10, 3))
    T = poses(P, rot([0, 1, 0], 15.0))
    pts = np.random.default_rng(31).uniform(-1, 3, size=(5, 3))
    st = np.random.get_state()
    r1 = (m.path_clearance(obs, T), m.visibility_matrix(obs, T, pts), m.evaluate_path(obs, T))
    r2 = (m.path_clearance(obs, T), m.visibility_matrix(obs, T, pts), m.evaluate_path(obs, T))
    a, b = r1[0], r2[0]
    assert (a.min_clearance, a.lower_bound, a.at_frame, tuple(a.point)) == (b.min_clearance, b.lower_bound, b.at_frame, tuple(b.point))
    assert np.array_equal(r1[1], r2[1])
    assert (r1[2].ok, r1[2].min_clearance, r1[2].score) == (r2[2].ok, r2[2].min_clearance, r2[2].score)
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1]) and st[2:] == st2[2:]


def test_import_does_not_load_torch():
    code = ("import sys; import buc.camera.clearance as c; "
            "assert 'torch' not in sys.modules, 'torch loaded'; "
            "assert hasattr(c, 'path_clearance')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr[-600:]

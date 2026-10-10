"""CAM-02: 하류 이음매 — FND-02 Shot, Obs 첫 포즈, torch 금지, 부작용 없음."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cam02_helpers import DEFAULT_FRAMES, mm, obs_from_c2w, odd_home

Z = 3.0
W, H = 320, 180


def _theta(**kw):
    from buc.core.contracts import Theta
    return Theta(**kw)


@pytest.mark.parametrize("realize", ["ideal", "rig"])
@pytest.mark.parametrize("name,kw", [("push_in", dict(m=1.5)), ("slider", {}), ("orbit", dict(orbit_deg=20.0)),
                                     ("orbit", dict(orbit_deg=20.0, round_trip=True)),
                                     ("rack_focus", dict(focus_to_m=9.0))])
def test_to_shot_builds_a_valid_fnd02_shot(name, kw, realize):
    from buc.core.contracts import Shot
    m = mm()
    th = _theta(focal_mm=35.0, fps=24.0)
    c0 = odd_home()
    sp = m.recipe(name, Z, **kw)
    path = m.ideal_path(sp, c0) if realize == "ideal" else m.rig_path(sp, c0)
    shot = m.to_shot(path, th, W, H, seed=7)
    assert isinstance(shot, Shot)                                                   # Shot 자체가 시간축·회전·초점을 검증한다
    assert shot.n_frames == DEFAULT_FRAMES and shot.fps == 24.0 and abs(shot.duration_s - 193 / 24.0) < 1e-12
    np.testing.assert_allclose(shot.c2w, path.c2w, atol=0)
    np.testing.assert_allclose(shot.t, path.t, atol=0)
    np.testing.assert_allclose(shot.focus_m, path.focus_m, atol=0)
    K = th.intrinsics(W, H)
    assert shot.K.shape == (DEFAULT_FRAMES, 3, 3)
    np.testing.assert_allclose(shot.K, np.broadcast_to(K, shot.K.shape), atol=1e-12)    # 물리 이동이므로 K 는 상수
    assert shot.move == name and shot.rig == path.rig and shot.seed == 7
    assert (shot.width, shot.height) == (W, H) and shot.theta.focal_mm == 35.0 and shot.theta.fps == 24.0
    # 첫 프레임 = 관측 카메라 C_0 (Read-Only Camera: 입력 구도에서 출발)
    obs = obs_from_c2w(c0, K, W, H)
    assert shot.first_pose_matches(obs, rot_tol_deg=1e-6, trans_tol_m=1e-9)
    # 경로 객체의 메서드도 같은 결과
    again = path.to_shot(th, W, H, seed=7)
    np.testing.assert_allclose(again.c2w, shot.c2w, atol=0)


def test_to_shot_copies_arrays_and_checks_fps():
    m = mm()
    path = m.ideal_path(m.recipe("orbit", Z))
    th = _theta(fps=24.0)
    shot = m.to_shot(path, th, W, H)
    shot.c2w[0, 0, 0] = 5.0
    shot.focus_m[0] = 99.0
    assert path.c2w[0, 0, 0] == 1.0 and path.focus_m[0] == Z                        # 경로를 건드리지 않는다
    assert shot.seed == 0
    with pytest.raises(ValueError):
        m.to_shot(path, _theta(fps=30.0), W, H)                                     # theta.fps 가 spec.fps 와 다르다
    other = m.to_shot(path, _theta(fps=24.0, focal_mm=50.0), 640, 360)
    np.testing.assert_allclose(other.K[0], _theta(focal_mm=50.0).intrinsics(640, 360), atol=1e-12)


def test_to_shot_supports_other_frame_rates():
    m = mm()
    sp = m.recipe("push_in", Z, duration_s=2.5, fps=30.0)
    shot = m.to_shot(m.ideal_path(sp), _theta(fps=30.0), W, H)
    assert shot.n_frames == 76 and shot.fps == 30.0


def test_rack_shot_pulls_focus_with_a_static_camera():
    m = mm()
    shot = m.to_shot(m.ideal_path(m.recipe("rack_focus", Z, focus_to_m=9.0)), _theta(), W, H)
    assert shot.move == "rack_focus" and np.ptp(shot.c2w, axis=0).max() == 0.0
    assert shot.focus_m[0] == pytest.approx(Z) and shot.focus_m[-1] == pytest.approx(9.0)


def test_module_is_torch_free_and_import_has_no_side_effects():
    import buc

    root = str(Path(buc.__file__).resolve().parents[1])
    code = ("import sys, numpy as np, random; s0 = np.random.get_state()[1].copy(); r0 = random.getstate();"
            "import buc.camera.moves as m;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "sp = m.bench_spec('I3', 3.0); p = m.ideal_path(sp); q = m.rig_path(sp);"
            "from buc.core.contracts import Theta; m.to_shot(q, Theta(), 64, 36);"
            "assert m.min_grade(sp) == 'safe' and m.measure(p.c2w, 3.0).alpha_deg > 19;"
            "assert 'torch' not in sys.modules;"
            "assert (np.random.get_state()[1] == s0).all() and random.getstate() == r0")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=55)
    assert p.returncode == 0, p.stderr[-800:]

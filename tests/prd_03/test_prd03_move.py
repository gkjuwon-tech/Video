"""PRD-03 fixed_move: 경로 공식 오라클(비단위 관측 포즈 포함), 시간축, params."""
import numpy as np
import pytest

from buc.core import contracts as C
from prd03_helpers import call, make_image, manual_read, tilted_obs

H, W = 18, 32


def read_default(seed=0):
    return call("S-1", {"image": make_image(H, W, seed=seed)})["read"]


def shot_of(read, params=None, seed=2024):
    world = call("S2", {"read": read})["world"]
    return call("S8", {"world": world, "read": read}, params, seed)["shot"]


def oracle(read, n=25, push=0.10, zs=None, fps=None):
    fps = float(read.theta_in.fps) if fps is None else fps
    if zs is None:
        zs = float(np.median(read.depth[np.isfinite(read.depth)].astype(np.float64)))
    c0 = read.obs.c2w()
    tau = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
    d = push * zs * tau
    c2w = np.repeat(c0[None], n, 0)
    for k in range(n):
        T = np.eye(4)
        T[2, 3] = d[k]
        c2w[k] = c0 @ T
    return c2w, d, zs, fps


def test_default_shot_matches_oracle():
    read = read_default()
    shot = shot_of(read)
    assert isinstance(shot, C.Shot)
    shot.validate()
    world = call("S2", {"read": read})["world"]
    C.check_stage_io("S8", {"world": world, "read": read}, {"shot": shot})
    c2w, d, zs, fps = oracle(read)
    assert shot.n_frames == 25 and shot.fps == fps == 24.0
    assert np.allclose(shot.c2w, c2w, atol=1e-9)
    assert np.allclose(shot.t, np.arange(25) / 24.0, atol=1e-12) and shot.t[0] == 0.0
    assert np.allclose(shot.focus_m, zs - d, atol=1e-9) and (shot.focus_m > 0).all()
    assert (shot.width, shot.height) == (W, H)
    assert np.allclose(shot.K, np.broadcast_to(read.obs.K, (25, 3, 3)), atol=1e-12)
    assert shot.move == "stub_push_in" and shot.rig == "locked"
    assert shot.theta.fps == shot.fps
    assert shot.theta.content_hash() == read.theta_in.content_hash()   # 기본 fps 이면 θ 그대로
    assert shot.duration_s == pytest.approx(25 / 24.0)


def test_first_pose_is_exactly_observation_and_advance_along_axis():
    read = read_default(1)
    shot = shot_of(read)
    assert shot.first_pose_matches(read.obs, rot_tol_deg=1e-9, trans_tol_m=1e-9)
    assert np.allclose(shot.c2w[0], read.obs.c2w(), atol=1e-12)
    # 자세 고정, 광축 전진: 전 프레임 회전 동일, 이동은 +z 방향 단조 증가
    assert np.allclose(shot.c2w[:, :3, :3], read.obs.c2w()[:3, :3][None], atol=1e-12)
    zs = float(np.median(read.depth))
    adv = shot.c2w[:, 2, 3] - shot.c2w[0, 2, 3]
    assert np.all(np.diff(adv) > 0)
    assert adv[-1] == pytest.approx(0.10 * zs, rel=1e-6)
    steps = np.diff(adv)
    assert np.allclose(steps, steps[0], rtol=1e-6)                      # 선형 프로파일


def test_nonidentity_observation_pose_equivariance():
    img_h, img_w = 18, 32
    obs = tilted_obs(img_w, img_h)
    depth = np.random.default_rng(3).uniform(2.0, 9.0, (img_h, img_w)).astype(np.float32)
    read = manual_read(depth, obs=obs, theta=C.Theta(aspect=img_w / img_h, fps=30), seed=4)
    world = call("S2", {"read": read})["world"]
    shot = call("S8", {"world": world, "read": read}, {"n_frames": 7, "push_frac": 0.2})["shot"]
    shot.validate()
    assert shot.first_pose_matches(read.obs)
    c2w, d, zs, fps = oracle(read, n=7, push=0.2)
    assert fps == 30.0 and shot.fps == 30.0 and shot.theta.fps == 30.0
    assert np.allclose(shot.c2w, c2w, atol=1e-9)
    # 이동 방향은 관측 카메라의 광축(세계에서의 c2w 3번째 열)
    ax = obs.c2w()[:3, 2]
    mv = shot.c2w[:, :3, 3] - shot.c2w[0, :3, 3]
    assert np.allclose(mv, d[:, None] * ax[None], atol=1e-9)
    assert np.allclose(np.linalg.norm(mv[-1]), 0.2 * zs, rtol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 13, 49])
def test_n_frames(n):
    read = read_default()
    shot = shot_of(read, {"n_frames": n})
    c2w, d, zs, fps = oracle(read, n=n)
    assert shot.n_frames == n and np.allclose(shot.c2w, c2w, atol=1e-9)
    assert shot.first_pose_matches(read.obs)
    if n == 1:
        assert np.allclose(shot.focus_m, zs) and shot.t.tolist() == [0.0]


def test_params_push_fps_zs():
    read = read_default()
    shot = shot_of(read, {"push_frac": -0.3, "fps": 12.0, "z_s": 4.0, "n_frames": 9})
    c2w, d, zs, fps = oracle(read, n=9, push=-0.3, zs=4.0, fps=12.0)
    assert np.allclose(shot.c2w, c2w, atol=1e-9) and zs == 4.0
    assert shot.fps == 12.0 and shot.theta.fps == 12.0
    assert np.allclose(shot.t, np.arange(9) / 12.0, atol=1e-12)
    assert np.allclose(shot.focus_m, 4.0 - d, atol=1e-9) and shot.focus_m[-1] == pytest.approx(4.0 * 1.3)
    # fps 만 θ 에서 달라지고 나머지는 θ_in 그대로
    want = C.Theta.from_dict({**read.theta_in.to_dict(), "fps": 12.0})
    assert shot.theta.content_hash() == want.content_hash()


def test_default_zs_is_median_of_finite_depth():
    d = np.array([[2.0, 4.0, 6.0], [8.0, np.inf, np.inf]], np.float32)       # 유한 중앙값 = 5
    read = manual_read(d, seed=5)
    shot = shot_of(read, {"n_frames": 3})
    assert shot.focus_m[0] == pytest.approx(5.0)
    assert shot.c2w[-1, 2, 3] == pytest.approx(0.5, rel=1e-6)


def test_seed_only_affects_shot_seed_and_inputs_untouched():
    read = read_default(2)
    world = call("S2", {"read": read})["world"]
    hr, hw = read.content_hash(), world.content_hash()
    a = call("S8", {"world": world, "read": read}, seed=11)["shot"]
    b = call("S8", {"world": world, "read": read}, seed=22)["shot"]
    assert (a.seed, b.seed) == (11, 22)
    assert read.content_hash() == hr and world.content_hash() == hw
    assert np.array_equal(a.c2w, b.c2w) and np.array_equal(a.focus_m, b.focus_m) and np.array_equal(a.t, b.t)
    # world 는 무시된다: 다른 world 를 줘도 같은 샷
    other = call("S2", {"read": read}, {"n_layers": 5})["world"]
    c = call("S8", {"world": other, "read": read}, seed=11)["shot"]
    assert c.content_hash() == a.content_hash()


@pytest.mark.parametrize("params", [
    {"n_frames": 0}, {"n_frames": 2.5}, {"n_frames": -3}, {"fps": 0.0}, {"fps": 241.0}, {"fps": float("nan")},
    {"push_frac": 0.6}, {"push_frac": -0.6}, {"push_frac": float("nan")}, {"z_s": 0.0}, {"z_s": -1.0},
    {"z_s": float("inf")}, {"nope": 1}])
def test_bad_params_rejected(params):
    read = read_default()
    world = call("S2", {"read": read})["world"]
    with pytest.raises(ValueError):
        call("S8", {"world": world, "read": read}, params)

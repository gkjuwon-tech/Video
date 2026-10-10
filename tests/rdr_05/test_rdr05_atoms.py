"""RDR-05 원자 id: 패킹, 세계 점, 지배 원자(항등·정수 이동·찢김), RDR-02와 바이트 동일."""
import numpy as np
import pytest

from buc.core import geom
from buc.render import aov as A
from buc.render.quads import RenderOpts, render_ldi
from rdr05_helpers import (FX, H, I4, W, K_default, atom_world_oracle, const_layer, layer, make_obs, make_world,
                           rand_tex, translate, world_to_cam, yaw)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
HW = H * W


def _aov(world, c2w=I4, **kw):
    return A.render_aov(world, c2w=c2w, K=K_default(), width=W, height=H, **kw)


# ---------------------------------------------------------------------------------------------------------------
def test_pack_unpack_formula_and_roundtrip():
    assert int(A.pack_atom_id(2, 5, 7, width=W, height=H)) == 2 * HW + 5 * W + 7
    got = A.pack_atom_id(0, 0, 0, width=W, height=H)
    assert np.asarray(got).dtype == np.int64 and int(got) == 0
    last = A.pack_atom_id(3, H - 1, W - 1, width=W, height=H)
    assert int(last) == 4 * HW - 1
    rng = np.random.default_rng(0)
    L = rng.integers(0, 7, 1000)
    V = rng.integers(0, H, 1000)
    U = rng.integers(0, W, 1000)
    ids = A.pack_atom_id(L, V, U, width=W, height=H)
    assert ids.dtype == np.int64 and ids.shape == (1000,)
    assert len(np.unique(ids)) == len(np.unique(L * HW + V * W + U))
    l2, v2, u2 = A.unpack_atom_id(ids, width=W, height=H)
    assert np.array_equal(l2, L) and np.array_equal(v2, V) and np.array_equal(u2, U)
    assert l2.dtype == v2.dtype == u2.dtype == np.int64
    # 브로드캐스트
    grid = A.pack_atom_id(1, np.arange(H)[:, None], np.arange(W)[None, :], width=W, height=H)
    assert grid.shape == (H, W) and int(grid[3, 4]) == HW + 3 * W + 4


@pytest.mark.parametrize("args", [(0, H, 0), (0, 0, W), (-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_pack_out_of_range(args):
    with pytest.raises(ValueError):
        A.pack_atom_id(*args, width=W, height=H)


def test_unpack_negative_rejected():
    with pytest.raises(ValueError):
        A.unpack_atom_id(np.array([3, -1]), width=W, height=H)


def test_n_atoms():
    w = make_world([const_layer((0.5, 0.5, 0.5), 1.0, d) for d in (2.0, 3.0, 4.0)])
    assert A.n_atoms(w) == 3 * HW


def test_atom_points_match_independent_oracle_with_distortion_and_pose():
    pose = yaw(translate(I4, (0.3, -0.1, 0.2)), 12.0)
    obs = make_obs(pose, k1=0.08, k2=-0.02)
    rng = np.random.default_rng(1)
    dep0 = rng.uniform(2.0, 4.0, (H, W)).astype(np.float32)
    dep1 = rng.uniform(5.0, 7.0, (H, W)).astype(np.float32)
    world = make_world([layer(rand_tex(0), dep0), layer(rand_tex(1), dep1)], obs=obs)
    ids = np.array([[0, 0, 0], [0, 5, 7], [0, H - 1, W - 1], [1, 10, 20], [1, H - 1, 0]])
    packed = A.pack_atom_id(ids[:, 0], ids[:, 1], ids[:, 2], width=W, height=H)
    X = A.atom_points(world, packed)
    assert X.shape == (5, 3) and X.dtype == np.float64
    for i, (l, v, u) in enumerate(ids):
        ref = atom_world_oracle(obs, (dep0, dep1)[l], l, v, u)
        assert np.allclose(X[i], ref, atol=1e-9)
    # 역투영: 관측 카메라로 다시 투영하면 텍셀 중심, z = 층 깊이
    uv, z = geom.project(obs.K, obs.R, obs.t, X, obs.k1, obs.k2)
    assert np.allclose(uv, ids[:, [2, 1]].astype(float), atol=1e-6)
    assert np.allclose(z, [dep0[0, 0], dep0[5, 7], dep0[H - 1, W - 1], dep1[10, 20], dep1[H - 1, 0]], atol=1e-5)
    # 없음(-1)은 NaN, 모양 보존
    Y = A.atom_points(world, np.array([[-1, packed[1]], [packed[3], -1]]))
    assert Y.shape == (2, 2, 3)
    assert np.isnan(Y[0, 0]).all() and np.isnan(Y[1, 1]).all()
    assert np.allclose(Y[0, 1], X[1], atol=1e-12) and np.allclose(Y[1, 0], X[3], atol=1e-12)
    with pytest.raises(ValueError):
        A.atom_points(world, np.array([2 * HW]))


# ---------------------------------------------------------------------------------------------------------------
def test_identity_camera_dominant_atom_is_own_texel():
    front = rand_tex(10, alpha=0.0)
    front[10:50, 30:60, 3] = 1.0
    back = rand_tex(11)
    world = make_world([layer(front, 2.0), layer(back, 5.0)])
    out = _aov(world)
    v, u = np.mgrid[0:H, 0:W]
    in_front = np.zeros((H, W), bool)
    in_front[10:50, 30:60] = True
    expect = np.where(in_front, v * W + u, HW + v * W + u)
    assert out.atom_id.dtype == np.int64 and out.atom_id.shape == (H, W)
    assert np.array_equal(out.atom_id, expect)
    assert np.array_equal(out.layer_id, np.where(in_front, 0, 1).astype(np.int16))
    # 앞 카드는 알파 1 이므로 깊이 2, 뒤는 5
    assert np.allclose(out.depth[in_front], 2.0, atol=1e-5) and np.allclose(out.depth[~in_front], 5.0, atol=1e-5)


def test_integer_pixel_shift_maps_to_shifted_texel():
    z, sx, sy = 3.0, 5, 3
    world = make_world([layer(rand_tex(2), z)])
    # 카메라 +x -> 내용은 왼쪽으로: 화소 u 는 텍셀 u+sx. 카메라 +y(아래) -> 내용은 위로: 화소 v 는 텍셀 v+sy
    pose = translate(I4, (sx * z / FX(), sy * z / FX(), 0.0))
    out = _aov(world, pose)
    v, u = np.mgrid[0:H, 0:W]
    inside = (u + sx <= W - 1 - 2) & (v + sy <= H - 1 - 2)
    assert inside.sum() > 0.8 * HW * 0.9
    assert np.array_equal(out.atom_id[inside], ((v + sy) * W + (u + sx))[inside])
    outside = (u >= W - sx + 1) | (v >= H - sy + 1)
    assert (out.atom_id[outside] == -1).all()
    assert (out.layer_id[outside] == -1).all() and (out.alpha[outside] < 1e-3).all()


def test_aov_beauty_channels_are_bytewise_render_ldi():
    # 반투명 3층 + 기울고 이동한 카메라 + 목표 왜곡 + 비기본 옵션
    la = layer(rand_tex(3, alpha=0.7), np.linspace(2.0, 2.6, W, dtype=np.float32)[None, :].repeat(H, 0))
    lb = layer(rand_tex(4, alpha=0.8), 4.0)
    lc = layer(rand_tex(5), 7.0)
    world = make_world([la, lb, lc])
    pose = yaw(translate(I4, (0.12, -0.05, 0.3)), 8.0)
    opts = RenderOpts(bg=(0.1, 0.2, 0.3), edge_px=1.5)
    kw = dict(c2w=pose, K=K_default(), width=W, height=H, k1=0.03, k2=-0.01, opts=opts)
    ref = render_ldi(world, **kw)
    out = A.render_aov(world, **kw)
    for name in ("rgb", "alpha", "depth", "layer_id"):
        a, b = getattr(out, name), getattr(ref, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name
    # 원자 id 와 층 id 의 일관성
    assert np.array_equal(out.atom_id < 0, out.layer_id < 0)
    cov = out.atom_id >= 0
    assert cov.mean() > 0.7
    assert np.array_equal(out.atom_id[cov] // HW, out.layer_id[cov].astype(np.int64))
    # 지배 원자의 투영 중심은 화소 중심 가까이(최근접 텍셀 우세): 이 장면은 배율 <= 1.3
    solid = cov & (out.alpha > 0.5)                     # 가장자리 꼬리(알파 ~0)는 제외
    assert solid.sum() > 3000
    X = A.atom_points(world, out.atom_id[solid])
    Xc = world_to_cam(pose, X)
    uv, z = geom.project(K_default(), np.eye(3), np.zeros(3), Xc, 0.03, -0.01)
    vv, uu = np.nonzero(solid)
    d = np.hypot(uv[:, 0] - uu, uv[:, 1] - vv)
    assert d.max() < 1.25
    assert (z > 0).all()


def test_depth_step_inside_layer_tears_dominant_atom_follows_near_side():
    # 한 층 안에서 왼쪽 반 z=3, 오른쪽 반 z=6. 연결 메시가 없으므로 겹침에서는 가까운 쪽, 벌어진 곳은 구멍(-1).
    dep = np.full((H, W), 6.0, np.float32)
    dep[:, : W // 2] = 3.0
    world = make_world([layer(rand_tex(6), dep)])
    dx = 0.36
    s_f, s_b = FX() * dx / 3.0, FX() * dx / 6.0
    # 카메라 -x: 장면이 오른쪽으로 밀린다. 가까운 쪽이 더 많이 밀려 먼 쪽 위에 겹친다.
    out = _aov(world, translate(I4, (-dx, 0, 0)))
    lo, hi = int(np.ceil(47.5 + s_b)) + 1, int(np.floor(47.5 + s_f)) - 1
    assert hi >= lo + 1
    dflat = dep.reshape(-1)
    for x in range(lo, hi + 1):
        col = out.atom_id[:, x]
        assert (col >= 0).all()
        assert (dflat[col % HW] == 3.0).all(), x
        assert np.allclose(out.depth[:, x], 3.0, atol=0.05), x
    # 카메라 +x: 가까운 쪽이 왼쪽으로 더 밀려 사이가 벌어진다 -> 구멍
    out2 = _aov(world, translate(I4, (+dx, 0, 0)))
    glo, ghi = int(np.ceil(48 - s_f)) + 1, int(np.floor(47 - s_b)) - 1
    assert ghi >= glo
    for x in range(glo, ghi + 1):
        assert (out2.atom_id[:, x] == -1).all(), x
        assert (out2.layer_id[:, x] == -1).all() and (out2.alpha[:, x] < 1e-3).all(), x
    # 구멍 양옆은 각자의 면 텍셀
    assert (dflat[out2.atom_id[:, glo - 3] % HW] == 3.0).all()
    assert (dflat[out2.atom_id[:, ghi + 3] % HW] == 6.0).all()


def test_dominant_layer_follows_layer_id_in_parallax_scene():
    front = rand_tex(7, alpha=0.0)
    front[:, 20:50, 3] = 1.0
    world = make_world([layer(front, 2.0), layer(rand_tex(8), 6.0)])
    out = _aov(world, translate(I4, (0.1, 0, 0)))
    cov = out.atom_id >= 0
    assert np.array_equal(out.atom_id[cov] // HW, out.layer_id[cov].astype(np.int64))
    s_f = FX() * 0.1 / 2.0
    # 앞 카드 중심부는 앞 층의 (v, u + s_f) 텍셀 근처
    row = out.atom_id[32, 25:40]
    assert (row // HW == 0).all()
    tex_u = (row % HW) % W
    assert np.all(np.abs(tex_u - (np.arange(25, 40) + s_f)) <= 0.5 + 1e-6)


def test_exact_tie_goes_to_smaller_atom_id():
    # fx = fy = 64, 주점 (47.5, 31.5), z = 2: 카메라 2^-6 m 이동 = 64*0.015625/2 = 정확히 0.5 px (모든 연산이 2진 정확).
    # 화소 하나에 두 텍셀이 정확히 같은 가중치 0.5 로 걸린다 -> atom_id 가 작은 텍셀.
    from buc.core import contracts as C
    K64 = np.array([[64.0, 0, (W - 1) / 2], [0, 64.0, (H - 1) / 2], [0, 0, 1.0]])
    obs = C.Obs(K=K64, R=np.eye(3), t=np.zeros(3), width=W, height=H)
    rgba = rand_tex(9)
    world = make_world([layer(rgba, 2.0)], obs=obs)
    v = np.arange(H)[:, None]
    p = np.arange(2, W - 4)[None, :]
    out = A.render_aov(world, c2w=translate(I4, (0.015625, 0, 0)), K=K64, width=W, height=H)
    assert np.array_equal(out.atom_id[:, 2:W - 4], np.broadcast_to(v * W + p, (H, W - 6)))       # 텍셀 p (p 와 p+1 중)
    out = A.render_aov(world, c2w=translate(I4, (-0.015625, 0, 0)), K=K64, width=W, height=H)
    assert np.array_equal(out.atom_id[:, 2:W - 4], np.broadcast_to(v * W + p - 1, (H, W - 6)))   # 텍셀 p-1 (p-1 과 p 중)

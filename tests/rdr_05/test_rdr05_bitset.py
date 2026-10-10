"""RDR-05 관측 bitset: 비트 배치, 집합 연산, 해시, observed_atoms 규칙, AOV+bitset 으로 계산한 IR_geo."""
import numpy as np
import pytest

from buc.core import det
from buc.render import aov as A
from rdr05_helpers import FX, H, I4, W, K_default, layer, make_obs, make_world, rand_tex, translate

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
HW = H * W
B = A.ObsBitset


def _bits(n, seed, p=0.4):
    return np.random.default_rng(seed).random(n) < p


# ---------------------------------------------------------------------------------------------------------------
def test_bit_layout_is_fixed():
    n = 130
    on = [0, 1, 63, 64, 65, 127, 129]
    flags = np.zeros(n, bool)
    flags[on] = True
    b = B.from_bool(flags)
    assert b.n == n
    assert b.words.dtype == np.uint64 and b.words.shape == (3,)
    w = [int(x) for x in b.words]
    assert w[0] == 1 | 2 | (1 << 63)
    assert w[1] == 1 | 2 | (1 << 63)                       # 64, 65, 127
    assert w[2] == 1 << 1                                   # 129 -> 워드 2 의 비트 1, n 이후 꼬리 비트는 0
    assert b.count() == 7
    assert np.array_equal(b.to_bool(), flags)
    assert b.to_bool().dtype == bool and b.to_bool().shape == (n,)


def test_from_bool_edge_cases():
    e = B.from_bool(np.zeros(0, bool))
    assert e.n == 0 and e.words.shape == (0,) and e.count() == 0 and e.to_bool().shape == (0,)
    exact = B.from_bool(np.ones(64, bool))
    assert exact.words.shape == (1,) and int(exact.words[0]) == (1 << 64) - 1 and exact.count() == 64
    assert B.from_bool(np.ones(65, bool)).words.shape == (2,)
    # 다차원은 C 순서로 평탄화
    f = _bits(2 * 3 * 4, 0).reshape(2, 3, 4)
    assert np.array_equal(B.from_bool(f).to_bool(), f.reshape(-1))
    for bad in (np.ones(5, np.int64), np.ones(5, np.float32), np.ones(5, np.uint8)):
        with pytest.raises(TypeError):
            B.from_bool(bad)


def test_test_method():
    flags = _bits(1000, 1)
    b = B.from_bool(flags)
    ids = np.random.default_rng(2).integers(-1, 1000, (4, 50))
    got = b.test(ids)
    assert got.dtype == bool and got.shape == ids.shape
    want = np.where(ids >= 0, flags[np.clip(ids, 0, None)], False)
    assert np.array_equal(got, want)
    assert (b.test(np.array([-1, -1])) == False).all()  # noqa: E712
    with pytest.raises(ValueError):
        b.test(np.array([3, 1000]))
    assert bool(b.test(np.array(999))) == bool(flags[999])


def test_set_operations_match_numpy_and_stay_canonical():
    n = 1000
    fa, fb = _bits(n, 3), _bits(n, 4, 0.7)
    a, b = B.from_bool(fa), B.from_bool(fb)
    for got, want in ((a | b, fa | fb), (a & b, fa & fb), (a - b, fa & ~fb)):
        ref = B.from_bool(want)
        assert got.n == n
        assert np.array_equal(got.words, ref.words)             # 꼬리 비트 0 포함 정본
        assert got.count() == int(want.sum())
        assert got.content_hash() == ref.content_hash()
    for op in (lambda x, y: x | y, lambda x, y: x & y, lambda x, y: x - y):
        with pytest.raises(ValueError):
            op(a, B.from_bool(_bits(n + 1, 5)))
    # 입력 불변
    assert np.array_equal(a.to_bool(), fa) and np.array_equal(b.to_bool(), fb)


def test_content_hash_formula_and_sensitivity():
    flags = _bits(100, 6)
    b = B.from_bool(flags)
    h = b.content_hash()
    assert isinstance(h, str) and len(h) == 64
    assert h == det.hash_parts(b"buc.obsbitset.v1", 100, det.hash_array(b.words)).hex()
    assert B.from_bool(flags.copy()).content_hash() == h
    f2 = flags.copy()
    f2[37] = ~f2[37]
    assert B.from_bool(f2).content_hash() != h
    # 같은 워드 배열이라도 n 이 다르면 해시가 다르다 (100 비트 vs 101 비트, 마지막 비트 0)
    f3 = np.concatenate([flags, [False]])
    assert B.from_bool(f3).words.tolist() == b.words.tolist()
    assert B.from_bool(f3).content_hash() != h


# ---------------------------------------------------------------------------------------------------------------
def test_observed_atoms_rule():
    # 층0: 열 0..29 알파 1, 30..59 알파 0.5, 60.. 알파 0 / 층1: 알파 1 (열 70 은 알파 0.002, 열 71 은 0.0005) / 층2: 알파 1
    a0 = np.zeros((H, W), np.float32)
    a0[:, :30] = 1.0
    a0[:, 30:60] = 0.5
    a1 = np.ones((H, W), np.float32)
    a1[:, 70] = 0.002
    a1[:, 71] = 0.0005
    a2 = np.ones((H, W), np.float32)
    mask = np.ones((H, W), bool)
    mask[20:30, 10:20] = False
    mask[20:30, 62:68] = False
    obs = make_obs(mask=mask)
    layers = []
    for i, al in enumerate((a0, a1, a2)):
        t = rand_tex(i)
        t[..., 3] = al
        layers.append(layer(t, 2.0 + 2.0 * i))
    world = make_world(layers, obs=obs)

    def oracle(vis_min, eps=1e-3):
        out = np.zeros((3, H, W), bool)
        trans = np.ones((H, W))
        for i, al in enumerate((a0, a1, a2)):
            al64 = al.astype(np.float64)
            out[i] = (al64 > eps) & (trans >= vis_min) & mask
            trans = trans * (1.0 - al64)
        return out

    for vm in (0.5, 0.4, 0.9, 1.0):
        got = A.observed_atoms(world, vis_min=vm)
        assert got.n == A.n_atoms(world) == 3 * HW
        assert np.array_equal(got.to_bool().reshape(3, H, W), oracle(vm)), vm
    # 기본값 = (vis_min 0.5, alpha_eps 1e-3)
    assert np.array_equal(A.observed_atoms(world).to_bool(), A.observed_atoms(world, vis_min=0.5,
                                                                              alpha_eps=1e-3).to_bool())
    got = A.observed_atoms(world, alpha_eps=0.0).to_bool().reshape(3, H, W)
    assert np.array_equal(got, oracle(0.5, eps=0.0))
    flags = A.observed_atoms(world).to_bool().reshape(3, H, W)
    assert flags[0, 5, 10] and flags[0, 5, 40] and not flags[0, 5, 70]            # 층0: 알파 1 / 0.5 / 0
    assert not flags[1, 5, 10] and flags[1, 5, 40] and flags[1, 5, 65]            # 층1: 가려짐 / 투과율 0.5 / 투과율 1
    assert flags[1, 5, 70] and not flags[1, 5, 71]                                 # 알파 0.002 > eps, 0.0005 는 아님
    assert flags[2, 5, 70] and flags[2, 5, 71] and not flags[2, 5, 10] and not flags[2, 5, 65]   # 층2: 층1 이 거의 투명한 열만
    assert flags[0, 5, 15] and not flags[0, 25, 15]                                 # obs.mask False 는 관측이 아니다
    assert flags[1, 5, 65] and not flags[1, 25, 65]
    assert A.observed_atoms(make_world(layers)).count() > A.observed_atoms(world).count()   # mask 없음이 더 많다


def test_observed_atoms_does_not_mutate_world():
    world = make_world([layer(rand_tex(0), 2.0), layer(rand_tex(1), 4.0)])
    h0 = world.content_hash()
    A.observed_atoms(world)
    assert world.content_hash() == h0


# ---------------------------------------------------------------------------------------------------------------
# IR_geo 이음매: AOV 의 지배 원자 + 관측 bitset 으로 계산한 값이 해석값과 맞는다
# ---------------------------------------------------------------------------------------------------------------
def _ir_geo(out, bits):
    cov = out.atom_id >= 0
    return 1.0 - float(bits.test(out.atom_id)[cov].mean()), cov


def _two_depth_world():
    front = rand_tex(20, alpha=0.0)
    front[:, :48, 3] = 1.0
    return make_world([layer(front, 3.0), layer(rand_tex(21), 6.0)])


def test_ir_geo_zero_at_observation_camera():
    world = _two_depth_world()
    out = A.render_aov(world, c2w=I4, K=K_default(), width=W, height=H)
    ir, cov = _ir_geo(out, A.observed_atoms(world))
    assert cov.all() and ir == 0.0


def test_ir_geo_matches_parallax_geometry():
    world = _two_depth_world()
    bits = A.observed_atoms(world)
    dx = 0.36
    out = A.render_aov(world, c2w=translate(I4, (dx, 0, 0)), K=K_default(), width=W, height=H)
    ir, cov = _ir_geo(out, bits)
    s_f, s_b = FX() * dx / 3.0, FX() * dx / 6.0
    expect = (s_f - s_b) / (W - s_b)                    # 드러난 뒤 층(앞 카드에 가려 있던 텍셀)의 열 수 / 덮인 열 수
    assert 0.04 < expect < 0.07
    assert ir == pytest.approx(expect, abs=0.02)
    # 미관측으로 찍힌 화소는 전부 뒤 층의 가려져 있던 텍셀(열 < 48)이고, 띠 모양으로 모여 있다
    unobs = cov & ~bits.test(out.atom_id)
    layer_i, v, u = A.unpack_atom_id(out.atom_id[unobs], width=W, height=H)
    assert (layer_i == 1).all() and (u < 48).all()
    cols = np.nonzero(unobs.any(0))[0]
    assert 36 <= cols.min() and cols.max() <= 46
    assert unobs.sum(0)[cols].min() >= H // 2           # 띠는 행 전체에 걸친다
    # 관측된 쪽은 앞 카드 또는 뒤 층의 열 >= 48
    obs_pix = cov & bits.test(out.atom_id)
    l2, v2, u2 = A.unpack_atom_id(out.atom_id[obs_pix], width=W, height=H)
    assert ((l2 == 0) | (u2 >= 48)).all()

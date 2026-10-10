"""RDR-05 안정 정렬과 결정론적 집계: sort_fragments 오라클·순열 불변, atom_pixel_counts 순서 불변."""
import numpy as np
import pytest

from buc.render import aov as A
from rdr05_helpers import H, I4, W, K_default, layer, make_world, rand_tex, translate, yaw

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
HW = H * W


def _frags(m=5000, seed=0):
    rng = np.random.default_rng(seed)
    pixel = rng.integers(0, 40, m).astype(np.int64)
    pool = np.array([0.5, 1.0, 1.0, 2.0, 2.0, 3.5, np.inf, 0.0, -0.0], np.float32)
    depth = pool[rng.integers(0, len(pool), m)]
    atom = rng.integers(0, 25, m).astype(np.int64)
    return pixel, depth, atom


def _oracle(pixel, depth, atom):
    return sorted(range(len(pixel)), key=lambda i: (int(pixel[i]), float(depth[i]), int(atom[i])))


def test_sort_fragments_matches_stable_python_oracle():
    pixel, depth, atom = _frags()
    perm = A.sort_fragments(pixel, depth, atom)
    assert perm.dtype == np.int64 and perm.shape == pixel.shape
    assert np.array_equal(np.sort(perm), np.arange(len(pixel)))
    assert perm.tolist() == _oracle(pixel, depth, atom)           # 파이썬 sorted 는 안정 -> 키가 같으면 입력 순서


def test_sort_fragments_is_permutation_invariant_in_output_keys():
    pixel, depth, atom = _frags(3000, 1)
    perm = A.sort_fragments(pixel, depth, atom)
    base = np.stack([pixel[perm].astype(np.float64), depth[perm].astype(np.float64), atom[perm].astype(np.float64)])
    rng = np.random.default_rng(7)
    for _ in range(8):
        sh = rng.permutation(len(pixel))
        p2 = A.sort_fragments(pixel[sh], depth[sh], atom[sh])
        got = np.stack([pixel[sh][p2].astype(np.float64), depth[sh][p2].astype(np.float64),
                        atom[sh][p2].astype(np.float64)])
        assert np.array_equal(got, base)


def test_sort_fragments_stability_and_ordering_properties():
    m = 100
    perm = A.sort_fragments(np.zeros(m, np.int64), np.full(m, 2.0, np.float32), np.zeros(m, np.int64))
    assert np.array_equal(perm, np.arange(m))                        # 키가 모두 같으면 항등
    # 화소 우선, 그다음 깊이(가까운 것 먼저), 그다음 원자 id
    perm = A.sort_fragments(np.array([1, 0, 1, 0]), np.array([1.0, 5.0, 1.0, 2.0]), np.array([9, 3, 4, 8]))
    assert perm.tolist() == [3, 1, 2, 0]
    perm = A.sort_fragments(np.array([0, 0, 0]), np.array([np.inf, 1.0, np.inf]), np.array([5, 7, 2]))
    assert perm.tolist() == [1, 2, 0]
    e = A.sort_fragments(np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64))
    assert e.shape == (0,) and e.dtype == np.int64
    # float32/float64 깊이 모두 허용, 정수형 폭이 달라도 허용
    p32 = A.sort_fragments(np.array([1, 0], np.int32), np.array([1.0, 1.0], np.float64), np.array([0, 0], np.int16))
    assert p32.tolist() == [1, 0]


def test_sort_fragments_errors():
    ok_p, ok_d, ok_a = np.array([0, 1]), np.array([1.0, 2.0], np.float32), np.array([0, 1])
    with pytest.raises(ValueError):
        A.sort_fragments(ok_p, np.array([1.0, np.nan], np.float32), ok_a)
    with pytest.raises(ValueError):
        A.sort_fragments(ok_p, ok_d, np.array([0, 1, 2]))
    with pytest.raises(TypeError):
        A.sort_fragments(np.array([0.0, 1.0]), ok_d, ok_a)
    with pytest.raises(TypeError):
        A.sort_fragments(ok_p, ok_d, np.array([0.5, 1.5]))
    with pytest.raises(TypeError):
        A.sort_fragments(ok_p, np.array([1, 2]), ok_a)


# ---------------------------------------------------------------------------------------------------------------
def test_atom_pixel_counts_matches_bincount_and_ignores_negative():
    rng = np.random.default_rng(3)
    n = 500
    ids = rng.integers(-1, n, (3, 20, 30)).astype(np.int64)
    got = A.atom_pixel_counts(ids, n)
    want = np.bincount(ids[ids >= 0], minlength=n)
    assert got.dtype == np.int64 and got.shape == (n,)
    assert np.array_equal(got, want)
    assert got.sum() == (ids >= 0).sum()
    assert A.atom_pixel_counts(np.full((4, 4), -1, np.int64), 10).tolist() == [0] * 10
    assert A.atom_pixel_counts(np.zeros(0, np.int64), 5).tolist() == [0] * 5
    with pytest.raises(ValueError):
        A.atom_pixel_counts(np.array([0, n]), n)


def test_atom_pixel_counts_bit_identical_under_any_pixel_or_frame_order():
    rng = np.random.default_rng(4)
    n = 800
    ids = rng.integers(-1, n, (6, 16, 16)).astype(np.int64)
    base = A.atom_pixel_counts(ids, n)
    for seed in range(6):
        r = np.random.default_rng(seed)
        flat = ids.reshape(-1)[r.permutation(ids.size)]
        assert A.atom_pixel_counts(flat, n).tobytes() == base.tobytes()
        assert A.atom_pixel_counts(ids[r.permutation(6)], n).tobytes() == base.tobytes()


def test_atom_pixel_counts_on_real_aov_frames():
    front = rand_tex(30, alpha=0.0)
    front[10:50, 20:70, 3] = 1.0
    world = make_world([layer(front, 2.0), layer(rand_tex(31), 5.0)])
    n = A.n_atoms(world)
    poses = [I4, translate(I4, (0.05, 0, 0)), yaw(translate(I4, (0.1, 0.02, 0.1)), 4.0), translate(I4, (0, 0, 0.3))]
    stack = np.stack([A.render_aov(world, c2w=p, K=K_default(), width=W, height=H).atom_id for p in poses])
    c = A.atom_pixel_counts(stack, n)
    assert np.array_equal(c, np.bincount(stack[stack >= 0], minlength=n))
    assert c.sum() == (stack >= 0).sum() > 3 * HW
    assert A.atom_pixel_counts(stack[::-1], n).tobytes() == c.tobytes()
    # 항등 카메라 한 프레임: 앞 카드 텍셀과 가려지지 않은 뒤 텍셀이 정확히 한 번씩
    one = A.atom_pixel_counts(stack[0], n).reshape(2, H, W)
    in_front = np.zeros((H, W), bool)
    in_front[10:50, 20:70] = True
    assert np.array_equal(one[0], in_front.astype(np.int64))
    assert np.array_equal(one[1], (~in_front).astype(np.int64))

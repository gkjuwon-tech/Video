"""RDR-07 secondary 광선: trace_closest(numpy 기준·warp 폴백), reflect_dirs, fresnel_schlick."""
import time

import numpy as np
import pytest

from buc.render import mirror
from buc.render.mirror import BackendUnavailable, RayHits, fresnel_schlick, reflect_dirs, trace_closest


def _mt_oracle(o, d, tris, t_min=1e-6, t_max=np.inf):
    """Möller–Trumbore 전수(양면). d 단위화. 동률은 작은 번호."""
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    n = o.shape[0]
    best_t = np.full(n, np.inf)
    best_i = np.full(n, -1)
    best_b = np.zeros((n, 2))
    for i, (v0, v1, v2) in enumerate(tris):
        e1, e2 = v1 - v0, v2 - v0
        p = np.cross(d, e2)
        det = p @ e1
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            s = o - v0
            b1 = (s * p).sum(1) * inv
            q = np.cross(s, e1)
            b2 = (d * q).sum(1) * inv
            t = (q @ e2) * inv
        ok = (np.abs(det) > 1e-12) & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1) & (t > t_min) & (t <= t_max)
        upd = ok & (t < best_t)
        best_t = np.where(upd, t, best_t)
        best_i = np.where(upd, i, best_i)
        best_b[upd] = np.c_[b1, b2][upd]
    return best_t, best_i, best_b


def _scene(seed=0, n_rays=400, n_tris=60):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-2, 2, (n_tris, 1, 3)) + np.array([0, 0, 5.0])
    tris = c + rng.normal(scale=0.6, size=(n_tris, 3, 3))
    o = rng.normal(scale=0.2, size=(n_rays, 3))
    tgt = rng.uniform(-2.5, 2.5, (n_rays, 3)) + np.array([0, 0, 5.0])
    d = (tgt - o) * rng.uniform(0.5, 3.0, (n_rays, 1))                    # 단위가 아닌 방향
    return o, d, tris


def test_trace_closest_numpy_matches_bruteforce_oracle():
    o, d, tris = _scene()
    h = trace_closest(o, d, tris, backend="numpy")
    assert isinstance(h, RayHits)
    assert h.t.shape == (400,) and h.t.dtype == np.float64
    assert h.tri.shape == (400,) and h.tri.dtype == np.int64
    assert h.bary.shape == (400, 2) and h.bary.dtype == np.float64
    ot, oi, ob = _mt_oracle(o, d, tris)
    assert (oi >= 0).mean() > 0.3 and (oi < 0).mean() > 0.05               # 적중과 미스가 섞인 장면
    same = h.tri == oi
    assert same.mean() >= 0.995
    hit = same & (oi >= 0)
    np.testing.assert_allclose(h.t[hit], ot[hit], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(h.bary[hit], ob[hit], atol=1e-7)
    miss = h.tri < 0
    assert np.isinf(h.t[miss]).all() and (h.bary[miss] == 0).all()
    du = d / np.linalg.norm(d, axis=1, keepdims=True)
    P = o[hit] + h.t[hit, None] * du[hit]                                   # 적중점 = 무게중심 보간
    v = tris[h.tri[hit]]
    b1, b2 = h.bary[hit, 0:1], h.bary[hit, 1:2]
    np.testing.assert_allclose(P, (1 - b1 - b2) * v[:, 0] + b1 * v[:, 1] + b2 * v[:, 2], atol=1e-7)


def test_trace_closest_edge_cases():
    tri = np.array([[[-1, -1, 2.0], [1, -1, 2.0], [0, 1, 2.0]],
                    [[-1, -1, 4.0], [1, -1, 4.0], [0, 1, 4.0]]])
    o = np.zeros((4, 3))
    d = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 7.0], [5.0, 0, 1.0]])
    h = trace_closest(o, d, tri, backend="numpy")
    assert h.tri.tolist() == [0, -1, 0, -1]
    assert h.t[0] == pytest.approx(2.0) and h.t[2] == pytest.approx(2.0)    # t 는 미터(방향 길이 무관)
    # 뒷면도 맞는다(양면)
    hb = trace_closest(np.array([[0, 0, 3.0]]), np.array([[0, 0, -1.0]]), tri, backend="numpy")
    assert hb.tri.tolist() == [0] and hb.t[0] == pytest.approx(1.0)
    # t_min: 표면 위 원점에서 자기 적중 제외 -> 다음 삼각형
    hs = trace_closest(np.array([[0, 0, 2.0]]), np.array([[0, 0, 1.0]]), tri, t_min=1e-6, backend="numpy")
    assert hs.tri.tolist() == [1] and hs.t[0] == pytest.approx(2.0)
    # t_max
    hm = trace_closest(o[:1], d[:1], tri, t_max=1.5, backend="numpy")
    assert hm.tri.tolist() == [-1] and np.isinf(hm.t[0])
    # 동률은 작은 번호
    dup = np.concatenate([tri[1:2], tri[0:1], tri[0:1]])
    ht = trace_closest(o[:1], d[:1], dup, backend="numpy")
    assert ht.tri.tolist() == [1]
    # 삼각형 0개
    h0 = trace_closest(o, d, np.zeros((0, 3, 3)), backend="numpy")
    assert (h0.tri == -1).all() and np.isinf(h0.t).all()


def test_trace_closest_validation_and_backends():
    o, d, tris = _scene(n_rays=10, n_tris=5)
    with pytest.raises(ValueError):
        trace_closest(o, np.zeros((10, 3)), tris, backend="numpy")
    with pytest.raises(ValueError):
        trace_closest(o, d[:5], tris, backend="numpy")
    with pytest.raises(ValueError):
        trace_closest(o, d, tris[:, :2], backend="numpy")
    with pytest.raises(ValueError):
        trace_closest(o, d, tris, backend="vulkan")
    with pytest.raises(TypeError):
        trace_closest(o, d, tris, 1e-6)
    b = mirror.available_backends()
    assert isinstance(b, tuple) and "numpy" in b and set(b) <= {"numpy", "warp", "optix"}
    if "warp" not in b:
        with pytest.raises(BackendUnavailable):
            trace_closest(o, d, tris, backend="warp")
        assert issubclass(BackendUnavailable, RuntimeError)
    ha = trace_closest(o, d, tris, backend="auto")                           # auto 는 예외 없음
    hn = trace_closest(o, d, tris, backend="numpy")
    assert (ha.tri == hn.tri).mean() >= 0.9


def test_trace_closest_numpy_is_deterministic_and_fast():
    o, d, tris = _scene(seed=3, n_rays=20000, n_tris=512)
    t0 = time.perf_counter()
    a = trace_closest(o, d, tris, backend="numpy")
    dt = time.perf_counter() - t0
    b = trace_closest(o, d, tris, backend="numpy")
    assert dt < 5.0, dt
    assert a.t.tobytes() == b.t.tobytes() and a.tri.tobytes() == b.tri.tobytes()
    assert (a.tri >= 0).mean() > 0.3


def test_reflect_dirs_properties():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(100, 3))
    n = rng.normal(size=(100, 3)) * 3.0                                     # 단위가 아닌 법선
    r = reflect_dirs(d, n)
    nu = n / np.linalg.norm(n, axis=1, keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1), rtol=1e-12)
    np.testing.assert_allclose((r * nu).sum(1), -(d * nu).sum(1), atol=1e-12)
    np.testing.assert_allclose(r - (r * nu).sum(1, keepdims=True) * nu, d - (d * nu).sum(1, keepdims=True) * nu,
                               atol=1e-12)
    np.testing.assert_allclose(reflect_dirs(r, n), d, atol=1e-12)
    np.testing.assert_allclose(reflect_dirs(np.array([1.0, -1, 0]), np.array([0, 1.0, 0])), [1, 1, 0])


def test_fresnel_schlick_values():
    f0 = 0.04
    assert fresnel_schlick(1.0, f0) == pytest.approx(0.04)
    assert fresnel_schlick(0.0, f0) == pytest.approx(1.0)
    assert fresnel_schlick(0.5, f0) == pytest.approx(0.04 + 0.96 * 0.5 ** 5)
    assert fresnel_schlick(-0.3, f0) == pytest.approx(1.0)                  # cos 는 [0,1] 로 자른다
    c = np.linspace(0, 1, 11)
    f = fresnel_schlick(c[:, None], np.array([0.04, 0.5, 0.9]))
    assert f.shape == (11, 3)
    assert (np.diff(f, axis=0) <= 1e-12).all()                              # 수직에 가까울수록 작아진다


@pytest.mark.gpu
def test_warp_backend_matches_numpy():
    if "warp" not in mirror.available_backends():
        pytest.skip("warp/CUDA 없음")
    o, d, tris = _scene(seed=5, n_rays=50000, n_tris=2000)
    hw = trace_closest(o, d, tris, backend="warp")
    hn = trace_closest(o, d, tris, backend="numpy")
    same = hw.tri == hn.tri
    assert same.mean() >= 0.999
    hit = same & (hn.tri >= 0)
    np.testing.assert_allclose(hw.t[hit], hn.t[hit], rtol=1e-4)
    assert trace_closest(o, d, tris, backend="auto").tri.tolist() == hw.tri.tolist()

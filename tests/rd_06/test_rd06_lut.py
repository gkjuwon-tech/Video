"""RD-06 룩 잔차 LUT 17³: 삼선형 적용·역, `fit_residual_lut`(복원·평활·희소 감쇠·가중), `r_look`."""
import time

import numpy as np
import pytest
from scipy import sparse

import rd06_helpers as H
from buc.core import color
from buc.lens import sensor_film as sf
from buc.read import tone_color as T

N = T.LUT_N


def warp(c):
    """검증용 매끈한 비선형 색 변환(부호화 영역): 스플릿 톤 + 그림자 채도 푸시."""
    c = np.asarray(c, np.float64)
    y = c @ H.W709
    out = c + 0.04 * np.sin(np.pi * y)[..., None] * np.array([1.0, -0.3, -1.0])
    out = out + 0.05 * (c - y[..., None]) * (1 - y[..., None])
    return np.clip(out, 0, 1)


def samples(seed, n=20000, noise=0.003):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 1, (n, 3)) ** 2.0
    cs = color.linear_to_srgb(src)
    ct = warp(cs) + noise * rng.standard_normal((n, 3))
    return src, color.srgb_to_linear(np.clip(ct, 0, 1)), cs


def de_coded(a_c, b_c):
    return H.de00(color.srgb_to_linear(np.clip(a_c, 0, 1)), color.srgb_to_linear(np.clip(b_c, 0, 1)))


def test_identity_lut_and_affine_exactness():
    ident = sf.identity_lut(N)
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 1, (500, 3))
    np.testing.assert_allclose(T.apply_residual_lut(c, ident), c, atol=2e-7)
    # 아핀 LUT 는 삼선형 보간이 정확하다
    M = np.array([[0.9, 0.05, 0.0], [0.0, 1.0, 0.1], [0.02, 0.0, 0.8]])
    b = np.array([0.03, -0.02, 0.1])
    g = ident.astype(np.float64)
    aff = (g @ M.T + b).astype(np.float32)
    got = T.apply_residual_lut(c, aff)
    np.testing.assert_allclose(got, c @ M.T + b, atol=5e-6)
    # 노드 값 그대로, 범위 밖 입력은 [0,1] 클램프, 모양 보존
    rnd = np.random.default_rng(1).uniform(0, 1, (N, N, N, 3)).astype(np.float32)
    nodes = np.linspace(0, 1, N)
    pt = np.array([[nodes[3], nodes[10], nodes[16]]])
    np.testing.assert_allclose(T.apply_residual_lut(pt, rnd)[0], rnd[3, 10, 16], atol=1e-6)
    np.testing.assert_allclose(T.apply_residual_lut(np.array([[-0.5, 2.0, 0.5]]), aff),
                               T.apply_residual_lut(np.array([[0.0, 1.0, 0.5]]), aff), atol=1e-7)
    assert T.apply_residual_lut(rng.uniform(0, 1, (4, 6, 3)), rnd).shape == (4, 6, 3)


def test_apply_matches_lns04_trilinear():
    rnd = np.random.default_rng(2).uniform(0, 1, (N, N, N, 3)).astype(np.float32)
    c = np.random.default_rng(3).uniform(0, 1, (300, 3))
    np.testing.assert_allclose(T.apply_residual_lut(c, rnd), sf.apply_lut3d(c.astype(np.float32), rnd), atol=2e-6)


def test_fit_recovers_known_warp():
    src, tgt, cs = samples(0)
    t0 = time.time()
    lut = T.fit_residual_lut(src, tgt)
    assert time.time() - t0 < 10.0
    assert lut.shape == (N, N, N, 3) and lut.dtype == np.float32 and np.isfinite(lut).all()
    assert lut.min() > -0.2 and lut.max() < 1.2                       # 자르지 않지만 폭주하지 않는다
    tst = np.random.default_rng(9).uniform(0, 1, (4000, 3)) ** 2.0
    cs_t = color.linear_to_srgb(tst)
    pred = T.apply_residual_lut(cs_t, lut)
    de = de_coded(pred, warp(cs_t))
    assert de.mean() <= 0.4 and np.quantile(de, 0.95) <= 1.0, (de.mean(), np.quantile(de, 0.95))
    # 적합 전(항등)보다 확실히 낫다
    base = de_coded(cs_t, warp(cs_t))
    assert base.mean() > 3 * de.mean()


def _design(sc):
    """삼선형 가중치 희소 행렬 A (n, 17³) — 독립 구현."""
    n = len(sc)
    p = sc * (N - 1)
    i0 = np.minimum(np.floor(p).astype(int), N - 2)
    f = p - i0
    rows, cols, vals = [], [], []
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wg = (f[:, 0] if dr else 1 - f[:, 0]) * (f[:, 1] if dg else 1 - f[:, 1]) * (f[:, 2] if db else 1 - f[:, 2])
                rows.append(np.arange(n))
                cols.append(((i0[:, 0] + dr) * N + i0[:, 1] + dg) * N + i0[:, 2] + db)
                vals.append(wg)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, N ** 3))


def _edge_op():
    idx = np.arange(N ** 3).reshape(N, N, N)
    ea, eb = [], []
    for ax in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        ea.append(idx[tuple(a)].ravel())
        eb.append(idx[tuple(b)].ravel())
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    E = len(ea)
    assert E == 13872
    return sparse.csr_matrix((np.concatenate([np.ones(E), -np.ones(E)]), (np.concatenate([np.arange(E)] * 2), np.concatenate([ea, eb]))),
                             shape=(E, N ** 3)), E


def objective_terms(R, A, d, w, smooth, ridge):
    """계약서 J(R) 의 세 항과 기울기: J=(1/N)Σw̃‖A R-d‖²+smooth·(1/E)Σ_edge‖ΔR‖²+ridge·(1/V)Σ‖R‖² (독립 구현)."""
    n, V = A.shape[0], N ** 3
    G, E = _edge_op()
    wt = w / w.mean()
    Rv = R.reshape(V, 3)
    res = A @ Rv - d
    grad = (2.0 / n) * (A.T @ (wt[:, None] * res)) + 2.0 * smooth / E * (G.T @ (G @ Rv)) + 2.0 * ridge / V * Rv
    J = np.sum(wt[:, None] * res ** 2) / n + smooth * np.sum((G @ Rv) ** 2) / E + ridge * np.sum(Rv ** 2) / V
    return J, grad, np.abs(grad).max(), max(np.abs((2.0 / n) * (A.T @ (wt[:, None] * res))).max(), np.abs(2.0 * ridge / V * Rv).max())


@pytest.mark.parametrize("smooth,ridge", [(0.01, 0.001), (0.3, 0.01), (0.002, 0.05)])
def test_fit_is_the_minimizer_of_the_contract_objective(smooth, ridge):
    """J 는 볼록이고 해는 유일하다 -> 기울기가 0 이다. 정규화 상수·정규화 방식이 다르면(예: smooth 1.2배) 기울기가 남는다."""
    src, tgt, _ = samples(0, n=5000)
    sc = np.clip(color.linear_to_srgb(src), 0, 1)
    d = np.clip(color.linear_to_srgb(tgt), 0, 1) - sc
    w = np.random.default_rng(3).uniform(0.5, 2.0, len(sc))
    A = _design(sc)
    lut = T.fit_residual_lut(src, tgt, weight=w, smooth=smooth, ridge=ridge)
    R = lut.astype(np.float64) - sf.identity_lut(N)
    J0, g, gmax, scale = objective_terms(R, A, d, w, smooth, ridge)
    assert gmax <= 1e-3 * scale, (gmax, scale)
    assert J0 < objective_terms(np.zeros_like(R), A, d, w, smooth, ridge)[0]
    for sm, rd in ((smooth * 1.2, ridge), (smooth, ridge * 1.5), (smooth / 1.2, ridge)):          # 근처의 '틀린' 목적식 해는 기울기가 남는다
        other = T.fit_residual_lut(src, tgt, weight=w, smooth=sm, ridge=rd).astype(np.float64) - sf.identity_lut(N)
        go = objective_terms(other, A, d, w, smooth, ridge)
        assert go[2] > 3e-3 * go[3]
        assert go[0] > J0


def test_identity_target_gives_identity_lut():
    src, _, _ = samples(1, n=8000)
    lut = T.fit_residual_lut(src, src)
    np.testing.assert_allclose(lut, sf.identity_lut(N), atol=2e-3)
    rng = np.random.default_rng(4)
    img = rng.uniform(0, 1, (30, 40, 3)).astype(np.float32)           # (H,W,3) 모양도 받는다
    np.testing.assert_allclose(T.fit_residual_lut(img, img), sf.identity_lut(N), atol=3e-3)


def test_sparse_coverage_decays_to_identity():
    src, tgt, cs = samples(2, n=20000, noise=0.0)
    near = np.linalg.norm(cs - 0.5, axis=1) < 0.2
    lut = T.fit_residual_lut(src[near], tgt[near])
    tst = np.random.default_rng(5).uniform(0, 1, (6000, 3))
    far = np.linalg.norm(tst - 0.5, axis=1) > 0.6
    assert far.sum() > 500
    dev = np.abs(T.apply_residual_lut(tst[far], lut) - tst[far]).max()
    assert dev < 0.01, dev                                              # 관측 없는 먼 곳은 항등으로 감쇠(외삽 폭주 없음)
    inside = np.linalg.norm(tst - 0.5, axis=1) < 0.1
    err_in = np.abs(T.apply_residual_lut(tst[inside], lut) - warp(tst[inside])).max()
    assert err_in < 0.02, err_in                                         # 관측 중심은 복원


def test_weights_scale_invariance_and_zero_weight_ignored():
    src, tgt, _ = samples(3, n=6000)
    w = np.random.default_rng(6).uniform(0.5, 2.0, len(src))
    a = T.fit_residual_lut(src, tgt, weight=w)
    b = T.fit_residual_lut(src, tgt, weight=7.0 * w)
    np.testing.assert_allclose(a, b, atol=2e-6)                         # w̃ = w/mean(w) 이라 배율 불변
    np.testing.assert_allclose(T.fit_residual_lut(src, tgt), T.fit_residual_lut(src, tgt, weight=np.full(len(src), 3.0)),
                               atol=2e-6)
    # w=0 인 점은 목적식 (1/N)Σ w̃ ... = Σ w/Σ w 에서 완전히 사라진다 -> 있어도 없어도 같은 해
    junk_src = np.random.default_rng(7).uniform(0, 1, (3000, 3))
    junk_tgt = np.random.default_rng(8).uniform(0, 1, (3000, 3))
    s2 = np.concatenate([src, junk_src])
    t2 = np.concatenate([tgt, junk_tgt])
    w0 = np.concatenate([np.ones(len(src)), np.zeros(3000)])
    np.testing.assert_allclose(T.fit_residual_lut(s2, t2, weight=w0), T.fit_residual_lut(src, tgt), atol=1e-5)
    # 한 점의 가중치를 키우면 그 점 근처 적합이 좋아진다
    tgt_bad = tgt.copy()
    tgt_bad[:50] = np.clip(tgt_bad[:50] + 0.2, 0, 1)
    ww = np.ones(len(src))
    ww[:50] = 200.0
    e_lo = np.abs(T.apply_residual_lut(color.linear_to_srgb(src[:50]), T.fit_residual_lut(src, tgt_bad))
                  - color.linear_to_srgb(tgt_bad[:50])).mean()
    e_hi = np.abs(T.apply_residual_lut(color.linear_to_srgb(src[:50]), T.fit_residual_lut(src, tgt_bad, weight=ww))
                  - color.linear_to_srgb(tgt_bad[:50])).mean()
    assert e_hi < 0.5 * e_lo


def _dirichlet(lut):
    r = lut.astype(np.float64) - sf.identity_lut(N).astype(np.float64)
    return sum(np.sum(np.diff(r, axis=ax) ** 2) for ax in range(3))


def test_smoothness_monotone_and_noise_robust():
    src, tgt, _ = samples(4, n=20000, noise=0.02)
    e = [_dirichlet(T.fit_residual_lut(src, tgt, smooth=s)) for s in (1e-4, 1e-2, 1.0, 100.0)]
    assert e[0] > e[1] > e[2] > e[3], e                                 # smooth 가 클수록 거칠기 에너지 단조 감소
    tst = np.random.default_rng(11).uniform(0, 1, (3000, 3)) ** 2.0
    cs_t = color.linear_to_srgb(tst)

    def err(**kw):
        return de_coded(T.apply_residual_lut(cs_t, T.fit_residual_lut(src, tgt, **kw)), warp(cs_t)).mean()
    e_def, e_smooth = err(), err(smooth=1.0)
    assert e_def <= 1.4, e_def                                          # 표준편차 0.02 잡음에서도 기본값으로 복원
    assert e_smooth <= 0.6 and e_smooth < e_def                         # 평활 prior 를 키우면 잡음에 더 강하다
    huge = T.fit_residual_lut(src, tgt, smooth=1e4, ridge=1e4)
    np.testing.assert_allclose(huge, sf.identity_lut(N), atol=0.02)    # 강한 prior 는 항등으로


def test_fit_validation():
    s = np.random.default_rng(0).uniform(0, 1, (50, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        T.fit_residual_lut(s, s[:40])
    with pytest.raises(ValueError):
        T.fit_residual_lut(s[:, :2], s[:, :2])
    with pytest.raises(TypeError):
        T.fit_residual_lut((s * 255).astype(np.uint8), s)
    bad = s.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        T.fit_residual_lut(bad, s)
    with pytest.raises(ValueError):
        T.fit_residual_lut(s, s, weight=np.ones(49))
    with pytest.raises(ValueError):
        T.fit_residual_lut(s, s, weight=-np.ones(50))
    for kw in ({"smooth": -1.0}, {"ridge": 0.0}, {"smooth": float("nan")}):
        with pytest.raises(ValueError):
            T.fit_residual_lut(s, s, **kw)


def test_fit_is_deterministic():
    src, tgt, _ = samples(5, n=3000)
    a = T.fit_residual_lut(src, tgt)
    b = T.fit_residual_lut(src, tgt)
    assert a.tobytes() == b.tobytes()


def test_lut_inverse_converges_for_fitted_lut():
    src, tgt, _ = samples(6)
    lut = T.fit_residual_lut(src, tgt)
    rng = np.random.default_rng(12)
    c = rng.uniform(0.03, 0.97, (3000, 3))
    out = np.clip(T.apply_residual_lut(c, lut), 0, 1)             # 표시 영역은 [0,1]
    back, ok = T.invert_residual_lut(out, lut)
    assert ok.shape == (3000,) and ok.mean() >= 0.97
    assert np.abs(T.apply_residual_lut(back[ok], lut) - out[ok]).max() < 1e-4
    inner = ok & ((out > 0.002) & (out < 0.998)).all(axis=1)           # 출력이 0/1 로 잘린 곳은 원상이 하나가 아니다
    assert inner.mean() > 0.9
    np.testing.assert_allclose(back[inner], c[inner], atol=2e-3)
    # 항등 LUT 는 정확히 되돌린다
    ident = sf.identity_lut(N)
    b2, ok2 = T.invert_residual_lut(c, ident)
    assert ok2.all()
    np.testing.assert_allclose(b2, c, atol=1e-6)


def test_lut_inverse_flags_unreachable_outputs():
    lut = sf.identity_lut(N).copy()
    lut[..., 2] = 0.5 * lut[..., 2]                                     # 파랑 출력은 [0, 0.5] 만 도달
    c, ok = T.invert_residual_lut(np.array([[0.4, 0.4, 0.3], [0.4, 0.4, 0.9]]), lut)
    assert list(ok) == [True, False]                                     # 치역 밖은 닫히지 않았다고 보고
    np.testing.assert_allclose(c[0], [0.4, 0.4, 0.6], atol=1e-4)         # 닿는 쪽은 정확히 되돌린다
    assert np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1


def test_r_look_properties():
    rng = np.random.default_rng(13)
    o = np.clip(0.5 + 0.2 * rng.standard_normal((64, 96, 3)), 0, 1)
    assert T.r_look(o, o) == 0.0
    lo = np.zeros_like(o)
    assert T.r_look(lo, lo) == 0.0                                       # 분모 0 -> 0.0
    cast = np.clip(o * np.array([1.0, 0.92, 0.8]), 0, 1)
    r1 = T.r_look(o, cast)
    r2 = T.r_look(o, np.clip(o * np.array([1.0, 0.8, 0.6]), 0, 1))
    assert 0.0 < r1 < r2                                                 # 편차가 클수록 크다
    # 저역은 보고 고역(그레인 같은 잡음)은 거의 안 본다: 같은 RMS 의 백색 잡음 vs 저주파 얼룩
    rms = 0.05
    noise = rms * rng.standard_normal(o.shape)
    yy, xx = np.mgrid[0:64, 0:96]
    blob = np.sin(2 * np.pi * (xx / 96.0 + 0.3)) * np.sin(2 * np.pi * yy / 64.0)
    blob = (rms / blob.std()) * blob
    r_noise = T.r_look(o, np.clip(o + noise, 0, 1))
    r_blob = T.r_look(o, np.clip(o + blob[..., None], 0, 1))
    assert r_blob > 5 * r_noise, (r_blob, r_noise)
    # 입력 검증
    with pytest.raises(ValueError):
        T.r_look(o, o[:-1])
    with pytest.raises((TypeError, ValueError)):
        T.r_look(o[..., :2], o[..., :2])
    assert isinstance(T.r_look(o, cast), float)

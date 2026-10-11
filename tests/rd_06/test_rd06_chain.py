"""RD-06 순방향/역방향 사슬: 왕복·소유 건전성·비트 의미·WB 규약(LNS-08)·명제 L2·bake_lut 이음매(LNS-04)."""
import numpy as np
import pytest

import rd06_helpers as H
from buc.core import color
from buc.core.contracts import Theta
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.read import tone_color as T


def rand_params(seed, *, with_cdl=True, with_lut=True):
    rng = np.random.default_rng(seed)
    kw = dict(ev=float(rng.uniform(-0.6, 0.6)), wb_kelvin=float(rng.uniform(3200, 8500)), wb_tint=float(rng.uniform(-0.15, 0.15)),
              curve=T.FilmCurve(toe=float(rng.uniform(0, 0.8)), shoulder=float(rng.uniform(0, 0.8)),
                                slope=float(rng.uniform(0.8, 1.4)), white=float(rng.uniform(0.8, 1.3))))
    if with_cdl:
        kw["cdl"] = T.CDL(slope=tuple(rng.uniform(0.9, 1.1, 3)), offset=tuple(rng.uniform(-0.02, 0.02, 3)),
                          power=tuple(rng.uniform(0.9, 1.1, 3)), sat=float(rng.uniform(0.8, 1.3)))
    if with_lut:
        kw["lut"] = H.smooth_lut(rng, amp=0.03)
    return T.ToneParams(**kw)


def log_uniform_scene(seed, n=3000, lo=0.003, hi=0.6):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), (n, 3))).astype(np.float32)


def test_identity_params_are_clip():
    L = np.concatenate([log_uniform_scene(0, 500, 0.001, 0.99), [[0, 0, 0], [1, 1, 1], [2, 0.5, 3]]]).astype(np.float32)
    D = T.forward(L, T.ToneParams())
    np.testing.assert_allclose(D, np.clip(L, 0, 1), atol=2e-7)
    inv = T.inverse(D, T.ToneParams())
    lt = (L < 0.999).all(axis=1)
    np.testing.assert_allclose(inv.linear[lt], L[lt], rtol=1e-5, atol=1e-7)
    assert (inv.status[lt] == 0).all() and inv.owned[lt].all()
    assert inv.status[-3] == 0 and (inv.linear[-3] == 0).all()                  # 검정은 정확히 닫힌다(CDL·LUT 없음)
    assert (inv.status[-2:] & T.OWN_CLIP_HI).all()                              # 백색·과노출은 포화


def test_wb_ev_only_matches_lns08_convention():
    """커브·CDL·LUT 항등이면 forward = clip(L · chain.wb_gains · 2^ev). WB 규약은 LNS-08 소유."""
    L = log_uniform_scene(1, 800, 0.01, 0.5)         # 800 = 20x40 화소
    for K, t, ev in ((3200, 0.0, 0.0), (5000, 0.1, 0.5), (9000, -0.2, -0.7), (6500, 0.0, 1.0)):
        p = T.ToneParams(ev=ev, wb_kelvin=K, wb_tint=t)
        want = np.clip(chain.apply_wb(L.reshape(20, 40, 3), Theta(wb_kelvin=K, wb_tint=t)).reshape(-1, 3).astype(np.float64)
                       * 2.0 ** ev, 0, 1)
        got = T.forward(L, p)
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(got, np.clip(L * chain.wb_gains(K, t) * 2.0 ** ev, 0, 1), rtol=2e-6, atol=2e-7)
    cool = T.forward(np.full((1, 3), 0.2, np.float32), T.ToneParams(wb_kelvin=3500))[0]
    assert cool[2] > cool[0]                                                   # 낮은 K 설정 -> 푸르다 (LNS-08 규약)
    mag = T.forward(np.full((1, 3), 0.2, np.float32), T.ToneParams(wb_tint=0.3))[0]
    assert mag[1] < mag[0] and mag[1] < mag[2]                                 # tint>0 -> 마젠타(g↓)


def test_forward_matches_independent_composition():
    p = T.ToneParams(ev=0.3, wb_kelvin=5200, wb_tint=0.05, curve=T.FilmCurve(toe=0.3, shoulder=0.4, slope=1.15, white=1.1),
                     cdl=T.CDL(slope=(1.05, 1.0, 0.95), offset=(0.01, 0.0, -0.01), power=(1.0, 1.05, 1.0), sat=1.2))
    L = log_uniform_scene(2, 600, 0.003, 0.5)
    x = L.astype(np.float64) * chain.wb_gains(5200, 0.05) * 2.0 ** 0.3
    y = np.clip(H.o_curve(x, toe=0.3, shoulder=0.4, slope=1.15, white=1.1), 0, 1)
    c = np.clip(H.o_cdl(H.srgb_enc(y), slope=(1.05, 1.0, 0.95), offset=(0.01, 0.0, -0.01), power=(1.0, 1.05, 1.0), sat=1.2), 0, 1)
    np.testing.assert_allclose(T.forward(L, p), H.srgb_dec(c), rtol=1e-5, atol=2e-6)
    rng = np.random.default_rng(4)
    lut = H.smooth_lut(rng, amp=0.04)
    p2 = T.ToneParams(ev=0.3, wb_kelvin=5200, wb_tint=0.05, curve=p.curve, cdl=p.cdl, lut=lut)
    c2 = np.clip(sf.apply_lut3d(c.astype(np.float32), lut).astype(np.float64), 0, 1)
    np.testing.assert_allclose(T.forward(L, p2), H.srgb_dec(c2), rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize("seed", range(6))
def test_roundtrip_and_ownership_soundness(seed):
    p = rand_params(seed, with_cdl=seed % 2 == 0, with_lut=seed % 3 != 0)
    L = log_uniform_scene(100 + seed)
    D = T.forward(L, p)
    inv = T.inverse(D, p)
    owned = inv.owned
    assert inv.linear.dtype == np.float32 and inv.status.dtype == np.uint8 and inv.status.shape == (len(L),)
    assert owned.mean() > 0.3                                                  # 일부는 포화되도록 장면을 밝게 잡았다
    err = np.abs(inv.linear.astype(np.float64) - L)
    tol = 2e-3 * L + 2e-4
    assert (err[owned] <= tol[owned]).all(), (err[owned] / tol[owned]).max()   # 건전성: 소유 => 정확
    assert (inv.status & ~np.uint8(7) == 0).all()                              # 알려진 비트만
    # 닫힌 화소에서 다시 찍으면 같은 표시값
    back = T.forward(inv.linear, p)
    np.testing.assert_allclose(back[owned], D[owned], atol=1e-5)


def test_completeness_in_midtones_for_mild_look():
    p = T.ToneParams(ev=0.2, wb_kelvin=5600, wb_tint=0.03, curve=T.FilmCurve(toe=0.3, shoulder=0.3, slope=1.1, white=1.0),
                     cdl=T.CDL(sat=1.1))
    L = log_uniform_scene(7, 3000, 0.02, 0.25)
    inv = T.inverse(T.forward(L, p), p)
    assert inv.owned.mean() >= 0.99                                            # 포화 없는 중간톤은 거의 전부 소유


def test_status_bits():
    # CLIP_HI: 표시 포화, 곡선 천장
    p = T.ToneParams(curve=T.FilmCurve(white=0.7))
    D = T.forward(np.array([[0.3, 0.3, 0.3], [3.0, 0.2, 0.2], [0.05, 0.05, 0.05]], np.float32), p)
    st = T.inverse(D, p).status
    assert st[0] == 0                                                           # x=0.3 <0.7 : 곡선 안
    assert st[1] & T.OWN_CLIP_HI and not st[1] & T.OWN_CLIP_LO
    assert st[2] == 0
    D1 = np.array([[1.0, 1.0, 1.0], [1.0, 0.3, 0.3], [0.5, 0.5, 0.5]], np.float32)
    s1 = T.inverse(D1, T.ToneParams()).status
    assert s1[0] & T.OWN_CLIP_HI and s1[1] & T.OWN_CLIP_HI and s1[2] == 0
    # 클립된 화소의 linear 는 하한: 다시 찍으면 표시값이 (거의) 포화 근처
    lin = T.inverse(D1, T.ToneParams()).linear
    assert (lin[0] >= 0.99).all() and np.isfinite(lin).all()
    assert (T.forward(lin[0:1], T.ToneParams()) >= 0.99).all()
    # CLIP_LO: CDL 이나 LUT 가 있으면 검정 근처는 모호
    cdl = T.ToneParams(cdl=T.CDL(offset=(-0.05, 0.0, 0.0), sat=0.8))
    Dl = T.forward(np.array([[0.002, 0.2, 0.2], [0.3, 0.3, 0.3], [0.0, 0.0, 0.0]], np.float32), cdl)
    sl = T.inverse(Dl, cdl).status
    assert sl[0] & T.OWN_CLIP_LO and sl[1] == 0 and sl[2] & T.OWN_CLIP_LO
    # CDL 도 LUT 도 없으면 검정은 정확히 x=0 으로 닫힌다
    s0 = T.inverse(np.zeros((1, 3), np.float32), T.ToneParams(curve=T.FilmCurve(toe=0.5, shoulder=0.3)))
    assert s0.status[0] == 0 and (s0.linear == 0).all()
    # LUT_LOST: 치역 밖
    lut = sf.identity_lut(17).copy()
    lut[..., 2] *= 0.5
    pl = T.ToneParams(lut=lut)
    sl = T.inverse(np.array([[0.2, 0.2, 0.9], [0.2, 0.2, 0.1]], np.float32), pl).status
    assert sl[0] & T.OWN_LUT_LOST and not sl[1] & T.OWN_LUT_LOST


def test_l2_wb_ev_misattribution_cancels_exactly():
    """명제 L2: M=Ĝ⁻¹∘G 가 채널 스케일이면 Ĝ(L̂·r) = G(L·r). 곡선·CDL·LUT 는 같고 WB·EV 만 틀린 경우."""
    rng = np.random.default_rng(5)
    shared = dict(curve=T.FilmCurve(toe=0.3, shoulder=0.4, slope=1.2, white=1.0), cdl=T.CDL(sat=1.2, offset=(0.01, 0.0, -0.01)),
                  lut=H.smooth_lut(rng, amp=0.04))
    G = T.ToneParams(ev=0.3, wb_kelvin=5200, wb_tint=0.08, **shared)
    Gh = T.ToneParams(ev=-0.4, wb_kelvin=7400, wb_tint=-0.05, **shared)         # 단단히 틀린 WB·EV
    L = log_uniform_scene(6, 2000, 0.01, 0.30)
    D = T.forward(L, G)
    inv = T.inverse(D, Gh)
    own = inv.owned
    assert own.mean() > 0.8
    M = inv.linear[own].astype(np.float64) / L[own]                              # 오귀속은 채널별 상수 스케일이다
    assert np.allclose(M, M[0], rtol=2e-3, atol=0)
    for r in (0.125, 0.5, 2.0, 8.0):
        truth = T.forward((L * r).astype(np.float32), G)
        ours = T.forward((inv.linear * r).astype(np.float32), Gh)
        assert np.abs(ours[own] - truth[own]).max() <= 2e-6, r
    # r=1 은 당연히 정확, 같은 Ĝ=G 면 전부 정확
    np.testing.assert_allclose(T.forward(inv.linear, Gh)[own], D[own], atol=2e-6)


def test_l2_curve_misattribution_grows_with_log_r():
    """비선형 성분(곡선 곡률 차)만 |log r| 에 따라 커지는 오차를 남긴다."""
    G = T.ToneParams(ev=0.2, wb_kelvin=5600, curve=T.FilmCurve(toe=0.5, shoulder=0.5, slope=1.3, white=1.0))
    Gh = T.ToneParams(ev=0.2, wb_kelvin=5600, curve=T.FilmCurve(toe=0.0, shoulder=0.0, slope=1.0, white=1.0))
    L = log_uniform_scene(8, 3000, 0.02, 0.15)
    inv = T.inverse(T.forward(L, G), Gh)
    own = inv.owned

    def err(r):
        a = T.forward((L * r).astype(np.float32), G)
        b = T.forward((inv.linear * r).astype(np.float32), Gh)
        return float(H.de00(a, b)[own].mean())
    e0 = err(1.0)
    assert e0 < 1e-3                                                             # 같은 시점에서는 Ĝ(Ĝ⁻¹(I)) = I
    ups = [err(r) for r in (2.0, 4.0, 8.0)]
    downs = [err(r) for r in (0.5, 0.25, 0.125)]
    assert e0 < ups[0] < ups[1] < ups[2], ups
    assert e0 < downs[0] < downs[1] < downs[2], downs
    assert ups[2] > 2.0 * ups[0] and downs[2] > 2.0 * downs[0]
    assert ups[0] > 0.3 and downs[0] > 0.3                                       # 곡선이 정말 다르면 오차가 실제로 난다


# ---------------------------------------------------------------------------------------------- bake_lut 이음매
def test_bake_identity_equals_neutral_lut():
    got = T.bake_lut(T.ToneParams())
    assert got.shape == (65, 65, 65, 3) and got.dtype == np.float32
    np.testing.assert_allclose(got, sf.make_lut("none"), atol=1e-6)
    np.testing.assert_allclose(got, sf.make_lut("neutral_709"), atol=1e-6)
    assert T.bake_lut(T.ToneParams(), size=9).shape == (9, 9, 9, 3)
    assert T.bake_lut(T.ToneParams(ev=1.0, wb_kelvin=3000)).tobytes() == got.tobytes()      # WB·EV 는 LUT 밖(Theta/LNS-08 소유)
    assert T.bake_lut(rand_params(3)).tobytes() == T.bake_lut(rand_params(3)).tobytes()


@pytest.mark.parametrize("seed,with_lut", [(0, False), (1, True), (2, True)])
def test_bake_lut_reproduces_forward_through_lns04(seed, with_lut):
    p = rand_params(seed, with_lut=with_lut)
    lut65 = T.bake_lut(p)
    assert lut65.min() >= 0 and lut65.max() <= 1 and np.isfinite(lut65).all()
    L = log_uniform_scene(200 + seed, 20000, 0.004, 1.0)
    g = chain.wb_gains(p.wb_kelvin, p.wb_tint)
    x = np.clip(L * g * 2.0 ** p.ev, 0, 1).astype(np.float32)                     # LNS-04 센서는 1.0 에서 자른다
    got = sf.apply_look(x, lut65)
    want = color.linear_to_srgb(T.forward((x / (g * 2.0 ** p.ev)).astype(np.float32), p).astype(np.float64))
    d = np.abs(got - want)
    ok = (x >= 0.004).all(axis=1)
    assert d[ok].mean() <= 3e-3, d[ok].mean()
    assert np.quantile(d[ok].max(axis=1), 0.99) <= 1.5e-2
    assert d[ok].max() <= 3e-2

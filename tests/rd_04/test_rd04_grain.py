"""그레인 Wiener 제거 `wiener_grain`(표시값 영역, 신호 의존 잡음 σ(u))과 explain_away 의 그레인 단계·sigma_grain."""
import numpy as np
import pytest

import rd04_helpers as H
from buc.core import color
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.read import explain_away as ea
from buc.read import screen_fx as fxm

HH, WW = H.H0, H.W0
S_GRAIN = 0.5
C_GRAIN = 0.16 * S_GRAIN          # LNS-04: σ(u) = s·0.08·2·√(u(1-u))  ->  sigma_coef = 0.16·s


def grainy(clean, s=S_GRAIN, seed=1):
    return sf.grain_field(clean.astype(np.float32), strength=s, radius_px=0.75, seed=seed)


@pytest.fixture(scope="module")
def patches():
    x, flat = H.patch_scene(HH, WW)
    clean = H.disp(x)
    return clean, flat, grainy(clean)


def test_noise_is_reduced_without_bias(patches):
    clean, flat, noisy = patches
    gf = H.grain_fit(C_GRAIN)
    out = ea.wiener_grain(noisy, gf)
    assert out.dtype == np.float32 and out.shape == noisy.shape
    assert out.min() >= 0.0 and out.max() <= 1.0 and np.all(np.isfinite(out))
    e_in, e_out = H.rms((noisy - clean)[flat]), H.rms((out - clean)[flat])
    assert e_in > 0.02                                       # 시험이 변별력이 있다(실제로 잡음이 크다)
    assert e_out <= 0.45 * e_in, (e_in, e_out)
    for q in range(4):                                       # 구역별(밝기별) 편향 없음
        hh, hw = HH // 2, WW // 2
        sl = (slice(0, hh) if q < 2 else slice(hh, HH), slice(0, hw) if q % 2 == 0 else slice(hw, WW))
        m = flat[sl]
        assert abs(float((out - clean)[sl][m].mean())) <= 0.003, q


def test_edges_and_detail_survive(patches):
    clean, flat, noisy = patches
    # (a) 강한 계단 에지(표시값 0.2 -> 0.8, 대비 ≈ 17 σ, 가장자리 σ=0.8 px): 10-90 % 상승 폭이 깨끗한 에지보다 1 px 넘게 늘지 않는다
    xs_ = np.arange(WW)[None, :] - (WW / 2 - 0.5)
    step = 0.2 + 0.6 * 0.5 * (1.0 + np.tanh(xs_ / 1.1)) * np.ones((HH, 1))
    sc = np.repeat(step[..., None], 3, 2).astype(np.float32)
    so = ea.wiener_grain(grainy(sc, seed=2), H.grain_fit(C_GRAIN))
    row = slice(8, HH - 8)
    prof_out = so[row, WW // 2 - 12:WW // 2 + 12, :].mean(axis=(0, 2))
    prof_ref = sc[row, WW // 2 - 12:WW // 2 + 12, :].mean(axis=(0, 2))

    def width(p):
        lo, hi = p[:4].mean(), p[-4:].mean()
        t = (p - lo) / (hi - lo)
        xs = np.arange(p.size)
        return float(np.interp(0.9, t, xs) - np.interp(0.1, t, xs))
    assert width(prof_out) <= width(prof_ref) + 1.0
    # (b) 강한 세부(주기 8 px 격자, 진폭 0.2)는 85 % 이상 남는다
    yy, xx = np.mgrid[0:HH, 0:WW]
    g = 0.5 + 0.2 * np.sin(2 * np.pi * xx / 8.0)
    gc = np.repeat(g[..., None], 3, 2).astype(np.float32)
    go = ea.wiener_grain(grainy(gc, seed=3), H.grain_fit(C_GRAIN))
    basis = np.sin(2 * np.pi * xx / 8.0)[8:-8, 8:-8]
    amp = float((go.mean(axis=2)[8:-8, 8:-8] * basis).mean() * 2.0)
    assert amp >= 0.85 * 0.2, amp
    # (c) 같은 격자에서 잡음은 줄었다
    assert H.rms((go - gc)) < H.rms(grainy(gc, seed=3) - gc)


def test_more_noise_means_more_smoothing_monotone(patches):
    clean, flat, noisy = patches
    steps = []
    for mult in (0.25, 1.0, 4.0):
        out = ea.wiener_grain(noisy, H.grain_fit(C_GRAIN * mult))
        steps.append(H.rms((out - noisy)[flat]))
    assert steps[0] < steps[1] < steps[2]
    steps = [H.rms((ea.wiener_grain(noisy, H.grain_fit(C_GRAIN), nsr=n) - noisy)[flat]) for n in (0.25, 1.0, 4.0)]
    assert steps[0] < steps[1] < steps[2]


def test_zero_grain_is_identity_bitwise(patches):
    clean, flat, noisy = patches
    for d in (clean, noisy):
        out = ea.wiener_grain(d, H.grain_fit(0.0, 0.0))
        assert np.array_equal(out, d) and out.dtype == np.float32
    assert np.array_equal(ea.wiener_grain(noisy, fxm.identity_screen_fx(HH, WW).grain), noisy)


def test_signal_independent_floor_is_also_removed(patches):
    """floor_sigma(신호 무관 잡음)만 있는 적합도 처리한다: 밝기와 무관하게 평탄 구역 잡음이 준다."""
    clean, flat, _ = patches
    rng = np.random.default_rng(5)
    noisy = np.clip(clean + 0.02 * rng.normal(size=clean.shape), 0, 1).astype(np.float32)
    out = ea.wiener_grain(noisy, H.grain_fit(0.0, 0.02, radius_px=0.75))
    assert H.rms((out - clean)[flat]) <= 0.5 * H.rms((noisy - clean)[flat])


def test_purity_and_validation(patches):
    clean, flat, noisy = patches
    before = noisy.copy()
    gf = H.grain_fit(C_GRAIN)
    a, b = ea.wiener_grain(noisy, gf), ea.wiener_grain(noisy, gf)
    assert np.array_equal(noisy, before) and a.tobytes() == b.tobytes()
    assert np.array_equal(ea.wiener_grain(noisy.astype(np.float64), gf), a)
    with pytest.raises(TypeError):
        ea.wiener_grain(noisy.tolist(), gf)
    with pytest.raises(TypeError):
        ea.wiener_grain((noisy * 255).astype(np.uint8), gf)
    for bad in (noisy[..., 0], noisy[:, :, :2], noisy + 0.6, noisy - 0.6, np.where(noisy > 0.5, np.nan, noisy)):
        with pytest.raises(ValueError):
            ea.wiener_grain(bad.astype(np.float32), gf)
    with pytest.raises(TypeError):
        ea.wiener_grain(noisy, {"sigma_coef": 0.1})
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            ea.wiener_grain(noisy, gf, nsr=bad)
    with pytest.raises(TypeError):
        ea.wiener_grain(noisy, gf, nsr=True)


# ------------------------------------------------------------------------------------------------ explain_away 단계
def test_sigma_grain_map_follows_the_fit_model():
    th = H.theta(lut="none")
    x, _ = H.patch_scene(HH, WW)
    im = H.look_lin(x * np.array([0.7, 1.0, 1.3]), th)                  # 채널마다 다른 밝기(평균 규칙 변별)
    gf = H.grain_fit(0.07, 0.004)
    r = ea.explain_away(im, th, screen_fx=H.make_fx(HH, WW, grain=gf))
    d = H.disp(im).astype(np.float64)
    want = np.mean(np.sqrt(0.07 ** 2 * d * (1 - d) + 0.004 ** 2), axis=2)
    assert r.sigma_grain.dtype == np.float32
    assert np.allclose(r.sigma_grain, want, atol=2e-6, rtol=1e-5)
    assert r.to_meta()["sigma_grain_mean"] == pytest.approx(float(want.mean()), rel=1e-4)
    # fx 가 없으면 0
    r0 = ea.explain_away(im, th)
    assert np.all(r0.sigma_grain == 0.0)
    # 그레인 옵션을 꺼도 sigma_grain 은 입력 그레인 통계로 보고된다
    r1 = ea.explain_away(im, th, screen_fx=H.make_fx(HH, WW, grain=gf), opts=ea.ExplainOpts(grain=False))
    assert np.array_equal(r1.sigma_grain, r.sigma_grain)


def test_grain_stage_is_wiener_then_chain_exactly():
    """계약된 단계 순서: display=선형->부호화, 그레인 Wiener(표시값), 그 다음 LNS-08 explain."""
    th = H.theta(ev=0.3, vignette=0.4, lut="neutral_709", wb_kelvin=5200.0)
    x = H.texture_scene(HH, WW, 2)
    clean_lin = H.look_lin(x, th)
    d = grainy(H.disp(clean_lin), seed=4)
    im = color.srgb_to_linear(d).astype(np.float32)
    gf = H.grain_fit(C_GRAIN)
    r = ea.explain_away(im, th, screen_fx=H.make_fx(HH, WW, grain=gf), opts=ea.ExplainOpts(grain_nsr=1.7))
    ref = chain.explain(ea.wiener_grain(H.disp(im), gf, nsr=1.7), th)
    assert np.array_equal(r.linear, ref.linear) and np.array_equal(r.status, ref.status)
    off = ea.explain_away(im, th, screen_fx=H.make_fx(HH, WW, grain=gf), opts=ea.ExplainOpts(grain=False))
    ref0 = chain.explain(H.disp(im), th)
    assert np.array_equal(off.linear, ref0.linear) and np.array_equal(off.status, ref0.status)


def test_grain_removal_reduces_world_error_with_real_rd03_fit():
    """RD-03 estimate_screen_fx 가 읽은 그레인으로 explain_away: 장면 선형 복원 오차가 크게 준다(센서 잡음 포함 look)."""
    th = H.theta(grain=0.5, lut="none")
    x, flat = H.patch_scene(144, 240, levels=(0.1, 0.22, 0.4, 0.62))
    im = H.look_lin(x, th, seed=7, noise=True)
    fx = fxm.estimate_screen_fx(im)
    assert fx.grain.sigma_coef > 0.03
    on = ea.explain_away(im, th, screen_fx=fx)
    off = ea.explain_away(im, th, screen_fx=fx, opts=ea.ExplainOpts(grain=False))
    e_on = H.rms((on.linear - x)[flat] / x[flat])
    e_off = H.rms((off.linear - x)[flat] / x[flat])
    assert e_off > 0.05
    assert e_on <= 0.45 * e_off, (e_on, e_off)

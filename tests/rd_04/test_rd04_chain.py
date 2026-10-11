"""LNS-08 이음매(효과가 없으면 `chain.explain` 과 비트 동일), 단계 순서의 분해 동치, 소유권 불변식, 전 스택 통합."""
import time

import numpy as np
import pytest
from scipy import ndimage as ndi

import rd04_helpers as H
from buc.lens import chain
from buc.read import explain_away as ea
from buc.read import screen_fx as fxm

HH, WW = H.H0, H.W0

THETAS = [
    dict(lut="none"),
    dict(lut="neutral_709", ev=0.3, vignette=0.5),
    dict(lut="bleach_bypass", ev=-0.4, wb_kelvin=4800.0, wb_tint=0.1, halation=0.2),
    dict(lut="teal_orange", vignette=0.3, halation=0.3),
    dict(lut="faded_matte", ev=0.2, wb_kelvin=8000.0),
]


def scene(seed=0):
    return H.texture_scene(HH, WW, seed, ev_std=0.4, mean=0.25)


@pytest.mark.parametrize("kw", THETAS)
def test_without_other_effects_it_is_exactly_lns08_explain(kw):
    th = H.theta(**kw)
    im = H.look_lin(scene(2), th)
    ref = chain.explain(H.disp(im), th)
    for r in (ea.explain_away(im, th), ea.explain_away(im, th, screen_fx=fxm.identity_screen_fx(HH, WW)),
              ea.explain_away(im, th, coc_px=0.0, motion_len_px=0.0, r_ratio=1.0)):
        assert np.array_equal(r.linear, ref.linear)
        assert np.array_equal(r.status, ref.status)
        assert np.array_equal(r.error_bound, ref.error_bound)
        assert r.band.mask.all() and np.all(r.band.min_octave == 0) and np.all(r.band.cutoff_nu == np.float32(0.5))
        assert np.all(r.sigma_grain == 0.0)
    assert ref.owned.mean() > 0.3


@pytest.mark.parametrize("kw", THETAS[:3])
def test_owned_pixels_recover_the_scene_within_the_bound(kw):
    """LNS-08 건전성이 image_lin 경로(선형 -> 부호화 왕복)에서도 유지된다: owned ⇒ |linear − x| <= error_bound(+float32 슬랙)."""
    th = H.theta(**kw)
    x = scene(3).astype(np.float32)
    im = H.look_lin(x, th)
    r = ea.explain_away(im, th)
    own = r.owned
    assert own.mean() > 0.5
    err = np.abs(r.linear - x)[own]
    bound = r.error_bound[own]
    assert np.all(err <= 1.02 * bound + 2e-6), float((err - bound).max())
    # 건전하되 느슨하지 않다: 오차 한계의 1/50 보다 큰 오차가 실제로 존재
    assert err.max() > bound.min() / 50.0


def test_status_never_loses_chain_bits_and_own_implies_chain_own():
    th = H.theta(lut="teal_orange", halation=0.3, vignette=0.3)
    x = scene(4).copy()
    cov = H.disc_cov(HH, WW, 60.0, 40.0, 8.0)[..., None]
    x = x * (1 - cov) + 2.4 * cov
    im = H.look_lin(x, th)
    ref = chain.explain(H.disp(im), th)
    fx = H.make_fx(HH, WW, halation=dict(blobs=(fxm.HaloBlob(x=60.0, y=40.0, area_px=200.0),), n_blobs=1, halo_gain=0.3 * 1.5,
                                         detected=True, amount=0.3, reliable=True),
                   ghost=dict(source=(60.0, 40.0), ghosts=(fxm.GhostSpot(x=90.0, y=70.0, radius_px=8.0, z=9.0, k=0.3),)))
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=2.0)
    assert np.all((r.status & ref.status) == ref.status)                 # 체인 비트 보존
    assert not np.any(r.owned & ~ref.owned)                              # 소유는 줄기만 한다
    extra = r.status & ~ref.status
    assert np.all((extra & ~np.uint8(ea.OWN_GHOST | ea.OWN_HDR)) == 0)  # 추가 비트는 GHOST/HDR 뿐
    assert (extra & ea.OWN_HDR).any() and (extra & ea.OWN_GHOST).any()


def test_defocus_stage_is_band_wiener_of_chain_linear_on_owned_pixels():
    th = H.theta(lut="neutral_709", ev=0.2, vignette=0.4)
    x = scene(5).copy()
    x[10:22, 90:110] = 3.0                                                 # 포화 구역(비소유)
    im = H.look_lin(x, th)
    ref = chain.explain(H.disp(im), th)
    coc = np.full((HH, WW), 3.5)
    coc[:, :40] = 1.5
    opts = ea.ExplainOpts(wiener_k=0.02, h_min=0.4)
    r = ea.explain_away(im, th, coc_px=coc, opts=opts)
    want = ea.wiener_band(ref.linear, coc, k=0.02, h_min=0.4, valid=(ref.status == 0))
    assert np.allclose(r.linear, want, atol=1e-6, rtol=0)
    assert np.array_equal(r.linear[ref.status != 0], ref.linear[ref.status != 0])      # 비소유 화소는 그대로
    assert np.array_equal(r.status, ref.status) and np.array_equal(r.error_bound, ref.error_bound)
    off = ea.explain_away(im, th, coc_px=coc, opts=ea.ExplainOpts(defocus=False, h_min=0.4))
    assert np.array_equal(off.linear, ref.linear)
    # 대역 지도는 보정과 무관하게 항상 보고된다
    b = ea.observed_band((HH, WW), coc_px=coc, h_min=0.4)
    assert np.array_equal(off.band.mask, b.mask) and np.array_equal(r.band.mask, b.mask)


def test_band_map_is_independent_of_options_and_matches_observed_band():
    th = H.theta(lut="none")
    im = H.look_lin(scene(6), th)
    coc = np.abs(np.random.default_rng(0).normal(3.0, 2.0, (HH, WW)))
    mot = np.full((HH, WW), 4.0)
    for opts in (None, ea.ExplainOpts(defocus=False), ea.ExplainOpts(grain=False, ghost=False, hdr=False, h_min=0.6)):
        r = ea.explain_away(im, th, coc_px=coc, motion_len_px=mot, r_ratio=0.8, opts=opts)
        hm = 0.5 if opts is None or opts.h_min == 0.5 else opts.h_min
        want = ea.observed_band((HH, WW), coc_px=coc, motion_len_px=mot, r_ratio=0.8, h_min=hm)
        assert np.array_equal(r.band.mask, want.mask) and np.array_equal(r.band.cutoff_nu, want.cutoff_nu)
        assert np.array_equal(r.band.min_octave, want.min_octave)


def test_defocus_through_the_whole_chain_restores_in_band_tone():
    """장면 코사인(흐림 c=4) -> 비네팅·EV·룩 -> image_lin -> explain_away(coc=4): 대역 안 톤 진폭이 복원되고, 대역 밖 톤은 만들어내지 않는다."""
    th = H.theta(lut="neutral_709", ev=0.2, vignette=0.5)
    c = 4.0
    nu_c = H.X_DEFOCUS_HALF / (np.pi * c)
    h, w = 128, 192
    got = {}
    for rel in (0.3, 0.6, 1.5):
        x, f = H.tone_scene(h, w, rel * nu_c, angle=0.5)
        blurred = H.blur_otf_periodic(x, c)
        im = H.look_lin(blurred, th)
        r_on = ea.explain_away(im, th, coc_px=c)
        r_off = ea.explain_away(im, th)
        a_sharp = H.tone_amplitude(x, f, 32)
        got[rel] = (H.tone_amplitude(r_on.linear, f, 32) / a_sharp, H.tone_amplitude(r_off.linear, f, 32) / a_sharp)
    assert 0.96 <= got[0.3][0] <= 1.04 and 0.96 <= got[0.6][0] <= 1.04
    assert got[0.6][1] < 0.85                                              # 보정 없이는 감쇠(변별력)
    assert got[1.5][0] == pytest.approx(got[1.5][1], rel=0.03)             # 컷오프 밖: 건드리지 않는다


def test_full_stack_with_real_rd03_estimates():
    """흐림 + 비네팅 + 그레인 + 센서 잡음 + 할레이션 + 룩: RD-03 추정치를 받아 한 번에 되돌린다."""
    h, w = 144, 240
    c = 3.0
    th = H.theta(lut="neutral_709", ev=0.2, vignette=0.6, grain=0.4, halation=0.3)
    x = H.texture_scene(h, w, 11, ev_std=0.3, mean=0.22)
    x = H.blur_otf_periodic(x, c)
    sx, sy, sr, ysrc = 190.0, 40.0, 7.0, 2.5
    cov = H.disc_cov(h, w, sx, sy, sr)[..., None]
    x = x * (1 - cov) + ysrc * cov
    im = H.look_lin(x, th, seed=3, noise=True)
    t0 = time.perf_counter()
    fx = fxm.estimate_screen_fx(im)
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=c)
    assert time.perf_counter() - t0 < 25.0
    yy, xx = np.mgrid[0:h, 0:w]
    far = np.hypot(xx - sx, yy - sy) > 50
    own = r.owned & far
    assert own.mean() > 0.5
    # (a) 비네팅이 빠졌다: 바깥 고리 대 중심의 평균 비(복원/원본 흐린 장면)가 1 ± 0.06
    rho = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / np.hypot((w - 1) / 2, (h - 1) / 2)
    lp = lambda a: ndi.gaussian_filter(a, 4.0, mode="reflect")
    ratio = lp(r.linear.mean(axis=2)) / lp(x.mean(axis=2))
    inner, outer = own & (rho < 0.35), own & (rho > 0.8)
    assert inner.sum() > 200 and outer.sum() > 200
    assert abs(ratio[inner].mean() - 1.0) <= 0.06 and abs(ratio[outer].mean() - 1.0) <= 0.06
    # (b) 대역 안에서 장면과 맞는다: 이상적 저역통과(ν <= 0.6 ν_c) 뒤의 상대 RMS 오차
    nu = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    lpmask = nu <= 0.6 * H.X_DEFOCUS_HALF / (np.pi * c)
    def band(a):
        return np.fft.ifft2(np.fft.fft2(a, axes=(0, 1)) * lpmask[..., None], axes=(0, 1)).real
    err = (band(r.linear) - band(x))[own]
    assert H.rms(err) / H.rms(band(x)[own]) <= 0.12
    # (c) 광원 휘도(HDR) 복원: 참값 2.5 의 ±0.8 EV
    hdr = (r.status & ea.OWN_HDR) != 0
    assert hdr.sum() > 10
    lum = r.linear[hdr] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert np.all(np.abs(np.log2(lum / ysrc)) <= 0.8)
    # (d) 메타·대역
    assert r.band.mask[2:].all() and not r.band.mask[:2].any()
    assert np.all(r.band.min_octave == 2)


def test_timing_budget_at_144x256():
    h, w = 144, 256
    th = H.theta(lut="teal_orange", ev=0.1, vignette=0.4, halation=0.2, grain=0.3)
    x = H.texture_scene(h, w, 21, mean=0.25)
    im = H.look_lin(x, th, seed=1, noise=True)
    fx = H.make_fx(h, w, grain=H.grain_fit(0.05),
                   ghost=dict(source=(200.0, 20.0), ghosts=(fxm.GhostSpot(x=100.0, y=60.0, radius_px=9.0, z=8.0, k=0.4),)))
    coc = np.full((h, w), 2.0)
    coc[:, w // 2:] = 5.0
    t0 = time.perf_counter()
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=coc, motion_len_px=3.0, r_ratio=0.9)
    dt = time.perf_counter() - t0
    assert dt < 10.0, dt
    assert np.all(np.isfinite(r.linear))


def test_motion_and_r_ratio_only_move_the_band_map():
    """모션블러는 디컨볼브하지 않고(마스크만), r_ratio 도 값에는 영향이 없다: 같은 linear/status, 다른 대역 지도."""
    th = H.theta(lut="neutral_709", vignette=0.3)
    im = H.look_lin(scene(7), th)
    base = ea.explain_away(im, th)
    m = ea.explain_away(im, th, motion_len_px=9.0, r_ratio=0.5)
    assert np.array_equal(m.linear, base.linear) and np.array_equal(m.status, base.status)
    assert np.array_equal(m.error_bound, base.error_bound)
    assert np.all(m.band.min_octave == 3)                                   # min(0.25, 0.067) -> 옥타브 3
    assert base.band.mask.all()

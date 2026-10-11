"""헤일로 광원 HDR 복원(H⁻¹)과 고스트 저대역 보간(F⁻¹, 부분 관측 강등)."""
import numpy as np
import pytest

import rd04_helpers as H
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.read import explain_away as ea
from buc.read import screen_fx as fxm

HH, WW = H.H0, H.W0
CX, CY, R = 64.0, 48.0, 9.0
Y_SRC = 2.2
A_HAL = 0.4


def halo_case(vignette=0.0, ev=0.0, kelvin=6500.0, y_src=Y_SRC):
    th = H.theta(halation=A_HAL, vignette=vignette, ev=ev, wb_kelvin=kelvin, lut="none")
    x = np.repeat(np.full((HH, WW), 0.06)[..., None], 3, 2)
    cov = H.disc_cov(HH, WW, CX, CY, R)[..., None]
    x = x * (1 - cov) + y_src * cov
    im = H.look_lin(x, th)
    # 광원 중심 노출(비네팅·EV·WB 이후) -> RD-03 의 halo_gain P = amount·(Y_exposure − 0.8)
    v = float(sf.vignette_gain(HH, WW, vignette)[int(CY), int(CX)])
    g = chain.wb_gains(kelvin, 0.0)
    y_exp = float(np.dot([0.2126, 0.7152, 0.0722], y_src * v * 2.0 ** ev * g))
    p = A_HAL * (y_exp - 0.8)
    return th, x, im, p, y_exp


def fx_halo(p, detected=True, amount=A_HAL, blobs=((CX, CY, 254.0),)):
    bl = tuple(fxm.HaloBlob(x=b[0], y=b[1], area_px=b[2]) for b in blobs)
    return H.make_fx(HH, WW, halation=dict(blobs=bl, n_blobs=len(bl), halo_gain=float(p), detected=detected,
                                           amount=float(amount), amount_min=0.0, reliable=detected))


def test_saturated_source_gets_hdr_luminance_back():
    th, x, im, p, y_exp = halo_case()
    base = ea.explain_away(im, th)
    r = ea.explain_away(im, th, screen_fx=fx_halo(p))
    hdr = (r.status & ea.OWN_HDR) != 0
    assert hdr.sum() >= 80                                               # 광원 코어(원판 면적 ≈ 254 px 의 상당수)
    assert np.all((base.status[hdr] & ea.OWN_SENSOR_SAT) != 0)           # 포화 화소만
    yy, xx = np.nonzero(hdr)
    assert np.hypot(xx - CX, yy - CY).max() <= R + 1.5                   # 광원 연결성분 안
    lum = r.linear[hdr] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert abs(float(np.median(lum)) - Y_SRC) <= 0.02 * Y_SRC                 # 복원 = 참 광원 휘도
    assert np.mean(np.abs(np.log2(lum / Y_SRC)) <= 0.5) >= 0.9                # 광원 휘도 ±0.5 EV (가장자리 안티앨리어싱 화소는 제외 가능)
    assert np.all(r.linear[hdr] >= base.linear[hdr])                      # 하한보다 줄지 않는다
    assert np.all(base.linear[hdr].max(axis=1) < 1.05)                   # 복원 전에는 센서 포화 하한(≈1)
    # 복원은 그 화소들만 바꾼다
    assert np.array_equal(r.linear[~hdr], base.linear[~hdr])
    assert np.array_equal(r.status[~hdr], base.status[~hdr])
    # 추정값이라 소유하지 않는다: 포화 비트 유지, error_bound +inf
    assert np.all((r.status[hdr] & ea.OWN_SENSOR_SAT) != 0)
    assert np.all(np.isinf(r.error_bound[hdr]))
    assert r.to_meta()["hdr_px"] == int(hdr.sum())


def test_hdr_luminance_with_vignette_ev_and_wb():
    """복원은 노출 영역 Y_exp = 0.8 + P/a 를 장면 선형으로 되돌린다: 비네팅·EV·WB 로 나눈다."""
    th, x, im, p, y_exp = halo_case(vignette=0.5, ev=0.4, kelvin=5000.0)
    r = ea.explain_away(im, th, screen_fx=fx_halo(p))
    hdr = (r.status & ea.OWN_HDR) != 0
    assert hdr.sum() >= 60
    v = sf.vignette_gain(HH, WW, 0.5).astype(np.float64)[..., None]
    g = chain.wb_gains(5000.0, 0.0)[None, None, :]
    expo = r.linear.astype(np.float64) * v * 2.0 ** 0.4 * g              # 다시 노출 영역으로
    lum_expo = expo[hdr] @ np.array([0.2126, 0.7152, 0.0722])
    want = 0.8 + p / A_HAL
    assert want == pytest.approx(y_exp, rel=1e-9)
    assert np.all(np.abs(lum_expo - want) <= 0.02 * want)                 # 노출 영역에서는 모든 복원 화소가 Y_exp
    # 장면 선형으로는 참값 2.2 의 중앙값 ±8 % (원판 안 비네팅 변화 때문에 화소별 오차)
    lum_scene = r.linear[hdr] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert abs(float(np.median(lum_scene)) / Y_SRC - 1.0) <= 0.08


def test_hdr_is_capped_and_never_lowers():
    th, x, im, p, _ = halo_case()
    base = ea.explain_away(im, th)
    big = ea.explain_away(im, th, screen_fx=fx_halo(1e4))
    hdr = (big.status & ea.OWN_HDR) != 0
    lum = big.linear[hdr] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert np.all(np.abs(lum - ea_cap()) <= 0.01 * ea_cap())
    for gain in (0.004, 0.02, 0.06, 0.1, 0.3):                                  # Y_exp = 0.8 + gain/a: 포화 하한(≈1)보다 작거나 비슷할 때
        small = ea.explain_away(im, th, screen_fx=fx_halo(gain))
        assert np.all(small.linear >= base.linear), gain                    # 어떤 채널도 낮추지 않는다
        if gain <= 0.02:
            assert np.array_equal(small.linear, base.linear) and not np.any(small.status & ea.OWN_HDR)
    assert np.any(ea.explain_away(im, th, screen_fx=fx_halo(0.5)).status & ea.OWN_HDR)


def ea_cap():
    return chain.HALO_SOURCE_MAX


def test_hdr_needs_detection_blobs_option_and_amount():
    th, x, im, p, _ = halo_case()
    base = ea.explain_away(im, th)
    for kw in (dict(fx=fx_halo(p, detected=False)), dict(fx=fx_halo(p, blobs=())),
               dict(fx=fx_halo(p), opts=ea.ExplainOpts(hdr=False))):
        r = ea.explain_away(im, th, screen_fx=kw["fx"], opts=kw.get("opts"))
        assert np.array_equal(r.linear, base.linear) and np.array_equal(r.status, base.status), kw
    # 블롭이 광원 밖(포화 성분이 없는 곳)을 가리키면 아무것도 안 바뀐다
    r = ea.explain_away(im, th, screen_fx=fx_halo(p, blobs=((10.0, 10.0, 80.0),)))
    assert np.array_equal(r.linear, base.linear)
    # θ.halation=0 이면 amount 는 RD-03 의 amount 를 쓴다: 둘 다 0 이면 복원 없음
    th0 = H.theta(lut="none")
    im0 = H.look_lin(np.where(H.disc_cov(HH, WW, CX, CY, R)[..., None] > 0.5, Y_SRC, 0.06) * np.ones((HH, WW, 3)), th0)
    b0 = ea.explain_away(im0, th0)
    r0 = ea.explain_away(im0, th0, screen_fx=fx_halo(p, amount=0.0))
    assert np.array_equal(r0.linear, b0.linear)
    r1 = ea.explain_away(im0, th0, screen_fx=fx_halo(p, amount=A_HAL))
    assert ((r1.status & ea.OWN_HDR) != 0).sum() > 50


def test_colored_saturated_pixels_keep_lower_bound():
    """세 채널이 모두 포화한 화소만 중립 광원으로 복원한다: 한 채널이라도 포화하지 않은 화소는 하한 그대로."""
    th, x, im, p, _ = halo_case()
    # 광원을 청색 광원으로: B 만 매우 밝고 R,G 는 중간
    xb = x.copy()
    cov = H.disc_cov(HH, WW, CX, CY, R)
    xb[..., 0] = 0.06 * (1 - cov) + 0.30 * cov
    xb[..., 1] = 0.06 * (1 - cov) + 0.30 * cov
    imb = H.look_lin(xb, th)
    base = ea.explain_away(imb, th)
    r = ea.explain_away(imb, th, screen_fx=fx_halo(p))
    assert np.array_equal(r.linear, base.linear) and not np.any(r.status & ea.OWN_HDR)


# ------------------------------------------------------------------------------------------------ 고스트
def ghost_case(spots=((70.0, 40.0, 10.0),)):
    th = H.theta(lut="none", vignette=0.2)
    yy, xx = np.mgrid[0:HH, 0:WW]
    bg = 0.25 + 0.12 * (xx / WW) + 0.06 * (yy / HH)
    x = np.repeat(bg[..., None], 3, 2)
    amp = 0.12
    for (gx, gy, gr) in spots:
        x = x + amp * np.clip((gr - np.hypot(xx - gx, yy - gy)) / 2.5 + 0.5, 0, 1)[..., None] * np.array([0.8, 1.0, 0.7])
    gh = tuple(fxm.GhostSpot(x=s[0], y=s[1], radius_px=s[2], z=9.0, k=0.3) for s in spots)
    fx = H.make_fx(HH, WW, ghost=dict(source=(110.0, 10.0), center=((WW - 1) / 2.0, (HH - 1) / 2.0), ghosts=gh))
    bgx = np.repeat(bg[..., None], 3, 2)
    return th, H.look_lin(x, th), H.look_lin(bgx, th), fx, amp


def test_ghost_pixels_are_interpolated_and_demoted_to_partial():
    th, im, im_bg, fx, amp = ghost_case()
    mask = fx.ghost.mask(HH, WW)
    assert mask.sum() > 200
    on = ea.explain_away(im, th, screen_fx=fx)
    off = ea.explain_away(im, th, screen_fx=fx, opts=ea.ExplainOpts(ghost=False))
    bg = ea.explain_away(im_bg, th)
    # (1) 마스크 안: 고스트가 걷혔다(배경과 진폭의 1/3 이내), 켜지 않았을 땐 그대로 남아 있다
    e_on = np.abs(on.linear - bg.linear)[mask].max()
    e_off = np.abs(off.linear - bg.linear)[mask].max()
    assert e_off > 0.5 * amp and e_on <= amp / 3.0, (e_on, e_off)
    # (2) 마스크 밖은 비트 동일
    assert np.array_equal(on.linear[~mask], off.linear[~mask])
    assert np.array_equal(on.status[~mask], off.status[~mask])
    # (3) 상태: GHOST 비트는 마스크와 정확히 일치, 소유 아님, error_bound +inf
    assert np.array_equal((on.status & ea.OWN_GHOST) != 0, mask)
    assert not np.any(on.owned & mask) and np.all(np.isinf(on.error_bound[mask]))
    assert on.to_meta()["ghost_px"] == int(mask.sum())
    # (4) 관측 대역 강등: 고스트 화소는 옥타브 0..GHOST_MIN_OCTAVE-1 비관측, 컷오프 <= 해당 옥타브 윗 가장자리
    assert not on.band.mask[:H.GHOST_MIN_OCTAVE, mask].any()
    assert on.band.mask[H.GHOST_MIN_OCTAVE:, mask].all()
    assert np.all(on.band.cutoff_nu[mask] <= np.float32(H.OCTAVE_HI[H.GHOST_MIN_OCTAVE]))
    assert on.band.mask[:, ~mask].all()                                   # 나머지는 온전 관측(블러 없음)
    assert np.all(on.band.min_octave[mask] == H.GHOST_MIN_OCTAVE)
    # (5) 보간 결과는 유한·비음수·float32
    assert np.all(np.isfinite(on.linear)) and on.linear.min() >= 0 and on.linear.dtype == np.float32


def test_multiple_and_border_ghosts_and_empty_fit():
    th, im, im_bg, fx, amp = ghost_case(spots=((70.0, 40.0, 9.0), (6.0, 5.0, 8.0), (118.0, 90.0, 7.0)))
    mask = fx.ghost.mask(HH, WW)
    assert mask.sum() > 300
    r = ea.explain_away(im, th, screen_fx=fx)
    assert np.array_equal((r.status & ea.OWN_GHOST) != 0, mask)
    assert np.all(np.isfinite(r.linear))
    bg = ea.explain_away(im_bg, th)
    assert np.abs(r.linear - bg.linear)[mask].max() <= amp / 2.0
    # 반점이 없으면 아무 일도 없다
    none_fx = H.make_fx(HH, WW, ghost=dict(source=None, ghosts=()))
    a = ea.explain_away(im, th, screen_fx=none_fx)
    b = ea.explain_away(im, th)
    assert np.array_equal(a.linear, b.linear) and np.array_equal(a.status, b.status)


def test_ghost_bits_combine_with_defocus_band_and_keep_chain_bits():
    th, im, im_bg, fx, amp = ghost_case()
    mask = fx.ghost.mask(HH, WW)
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=3.0, r_ratio=0.75)       # nu_c = min(0.375, 0.235) = 0.235 -> 옥타브 2 부터
    assert np.all(r.band.min_octave[~mask] == 2)
    assert np.all(r.band.min_octave[mask] == 3)                                # 강등은 더 좁게만 만든다
    assert np.all(r.band.cutoff_nu[mask] <= np.float32(0.0625))
    ref = chain.explain(H.disp(im), th)
    assert np.all((r.status & ref.status) == ref.status)                        # 체인 비트는 지워지지 않는다


def test_ghost_never_widens_the_band():
    th, im, _, fx, _ = ghost_case()
    mask = fx.ghost.mask(HH, WW)
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=12.0)                      # 어떤 옥타브도 관측 아님(min_octave=4)
    assert np.all(r.band.min_octave == 4) and not r.band.mask.any()
    assert mask.any()

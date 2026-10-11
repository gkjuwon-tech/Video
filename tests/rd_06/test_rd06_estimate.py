"""RD-06 R4 추정: 합성 장면(앨베도 prior 에 맞춘 로그정규, 알려진 G) 복원, prior·앵커·경고, L2 종단 검사."""
import json
import math
import time

import numpy as np
import pytest

import rd06_helpers as H
from buc.core import color
from buc.read import tone_color as T

CASES = [
    dict(ev=0.4, wb_kelvin=5000, wb_tint=0.05, curve=T.FilmCurve(toe=0.3, shoulder=0.4, slope=1.15, white=1.0)),
    dict(ev=-0.5, wb_kelvin=7500, wb_tint=-0.1, curve=T.FilmCurve(toe=0.0, shoulder=0.0, slope=1.0, white=1.0)),
    dict(ev=0.0, wb_kelvin=4200, wb_tint=0.0, curve=T.FilmCurve(toe=0.5, shoulder=0.6, slope=1.3, white=1.0)),
    dict(ev=0.8, wb_kelvin=6500, wb_tint=0.0, curve=T.FilmCurve(toe=0.1, shoulder=0.2, slope=0.9, white=1.2)),
]


def make(case, seed, **kw):
    p = T.ToneParams(**case)
    L = H.scene(seed, **kw)
    return p, L, T.forward(L.astype(np.float32), p)


def ramp_de(p, ph, lo=-3.0, hi=1.5):
    r = H.ramp(lo, hi)
    return H.de00(T.forward(r, p), T.forward(r, ph))


@pytest.mark.parametrize("ci", range(len(CASES)))
@pytest.mark.parametrize("seed", [11, 12])
def test_recovers_known_look(ci, seed):
    p, L, I = make(CASES[ci], seed * 10 + ci)
    r = T.estimate_tone_color(I)
    ph = r.params
    assert isinstance(r, T.ToneColor) and r.reliable and r.warnings == ()
    assert H.gains_dev_stops(p.wb_kelvin, p.wb_tint, ph.wb_kelvin, ph.wb_tint) <= 0.10      # WB 이득비 (±300 K 급)
    assert abs(ph.ev - p.ev) <= 0.3                                                          # 중간톤 ±0.3 EV
    de = ramp_de(p, ph)
    assert de.mean() <= 2.5 and de.max() <= 5.0, (de.mean(), de.max())                      # 톤 곡선 재현 ΔE00
    assert 0.0 <= r.fit_rms_stops <= 0.35
    assert ph.cdl.is_identity() and ph.lut is None                                           # CDL·LUT 는 추정 안 함
    assert 0.0 <= r.clip_hi_frac < 0.2 and r.clip_lo_frac == 0.0
    assert r.n_used == I.shape[0] * I.shape[1]


def test_estimate_beats_ignoring_the_look():
    """알려진 G 를 무시하고(항등) 재조명하는 것보다 추정 G 로 재조명하는 편이 확실히 낫다 (명제 L2 종단)."""
    p, L, I = make(CASES[0], 5)
    r = T.estimate_tone_color(I)
    inv = T.inverse(I, r.params)
    inv0 = T.inverse(I, T.ToneParams())
    assert inv.owned.mean() > 0.9
    own = inv.owned
    for s in (0.5, 2.0):
        truth = T.forward((L * s).astype(np.float32), p)
        ours = T.forward((inv.linear * s).astype(np.float32), r.params)
        naive = T.forward((inv0.linear * s).astype(np.float32), T.ToneParams())
        e = H.de00(truth, ours)[own].mean()
        e0 = H.de00(truth, naive)[own].mean()
        assert e <= 2.5, (s, e)
        assert e <= 0.7 * e0, (s, e, e0)
    own2 = inv.owned
    # r=1 에서는 역·순 합성이 입력 그대로
    np.testing.assert_allclose(T.forward(inv.linear, r.params)[own2], I[own2], atol=1e-5)


def test_recovery_is_deterministic_and_meta_is_json():
    _, _, I = make(CASES[0], 3)
    a = T.estimate_tone_color(I)
    b = T.estimate_tone_color(I.copy())
    ma, mb = a.to_meta(), b.to_meta()
    sa = json.dumps(ma, allow_nan=False, sort_keys=True)
    assert sa == json.dumps(mb, allow_nan=False, sort_keys=True)
    for k in ("ev", "wb_kelvin", "wb_tint", "curve", "cdl", "lut_present", "reliable", "warnings", "fit_rms_stops", "n_used",
              "clip_hi_frac", "clip_lo_frac"):
        assert k in ma, k
    assert set(ma["curve"]) == {"toe", "shoulder", "slope", "white"} and set(ma["cdl"]) == {"slope", "offset", "power", "sat"}
    assert ma["lut_present"] is False and ma["reliable"] is True and ma["warnings"] == []
    assert a.params.ev == b.params.ev and a.params.curve == b.params.curve


def test_priors_pull_the_solution():
    L = H.scene(21)
    p = T.ToneParams(ev=0.8, wb_kelvin=5000, wb_tint=0.05, curve=CASES[3]["curve"])
    I = T.forward(L.astype(np.float32), p)
    free = T.estimate_tone_color(I).params
    assert abs(free.ev - 0.8) <= 0.3 and H.gains_dev_stops(5000, 0.05, free.wb_kelvin, free.wb_tint) <= 0.10
    pin = T.estimate_tone_color(I, priors=T.TonePriors(ev=0.0, ev_sigma=0.01, wb_kelvin=6500.0, wb_mired_sigma=1.0,
                                                       wb_tint=0.0, wb_tint_sigma=0.005)).params
    assert abs(pin.ev) < 0.5 * abs(free.ev) and abs(pin.ev) <= 0.15
    d_free = H.gains_dev_stops(6500, 0.0, free.wb_kelvin, free.wb_tint)              # 자유 해는 prior(6500K, 0)에서 멀다
    d_pin = H.gains_dev_stops(6500, 0.0, pin.wb_kelvin, pin.wb_tint)
    assert d_free > 0.1 and d_pin < 0.3 * d_free and d_pin <= 0.03
    assert abs(pin.wb_tint) <= 0.02
    # 참값을 알려 주는 prior(MRH 헤드가 맞힌 경우)는 해를 해치지 않는다
    good = T.estimate_tone_color(I, priors=T.TonePriors(ev=0.8, wb_kelvin=5000.0, wb_tint=0.05))
    assert good.reliable
    assert abs(good.params.ev - 0.8) <= 0.3
    assert ramp_de(p, good.params).mean() <= 2.0


def test_gray_world_wb_with_neutral_scene_and_cast():
    """장면이 회색 세계이고 카메라 WB 만 틀렸다면 이득비를 정확히 찾는다."""
    for K, t in ((3800, 0.0), (9000, 0.1), (6500, -0.2)):
        p = T.ToneParams(ev=0.0, wb_kelvin=K, wb_tint=t, curve=T.FilmCurve(slope=1.1))
        I = T.forward(H.scene(30, shape=(80, 120)).astype(np.float32), p)
        ph = T.estimate_tone_color(I).params
        assert H.gains_dev_stops(K, t, ph.wb_kelvin, ph.wb_tint) <= 0.08, (K, t, ph.wb_kelvin, ph.wb_tint)


def test_uninformative_images_fall_back_to_priors():
    flat = np.full((32, 32, 3), 0.4, np.float32)
    r = T.estimate_tone_color(flat)
    assert not r.reliable and "insufficient_dynamic_range" in r.warnings
    assert r.params.is_identity() and math.isnan(r.fit_rms_stops)
    assert json.dumps(r.to_meta(), allow_nan=False) and r.to_meta()["fit_rms_stops"] is None
    pr = T.TonePriors(ev=0.5, wb_kelvin=5000.0, wb_tint=0.1)
    r2 = T.estimate_tone_color(flat, priors=pr)
    assert not r2.reliable
    assert (r2.params.ev, r2.params.wb_kelvin, r2.params.wb_tint) == (0.5, 5000.0, 0.1)
    assert r2.params.curve == T.FilmCurve() and r2.params.cdl.is_identity()
    narrow = np.clip(0.3 + 0.002 * np.random.default_rng(0).standard_normal((40, 40, 3)), 0, 1).astype(np.float32)
    assert "insufficient_dynamic_range" in T.estimate_tone_color(narrow).warnings


def test_warnings_for_clipped_black_and_few_pixels():
    _, _, I = make(CASES[0], 1)
    rng = np.random.default_rng(0)
    white = I.copy()
    white[rng.random(I.shape[:2]) < 0.6] = 1.0
    r = T.estimate_tone_color(white)
    assert "clipped_dominant" in r.warnings and not r.reliable and r.clip_hi_frac > 0.55
    p = r.params
    assert np.isfinite([p.ev, p.wb_kelvin, p.wb_tint, p.curve.toe, p.curve.shoulder, p.curve.slope, p.curve.white]).all()
    dark = I.copy()
    dark[rng.random(I.shape[:2]) < 0.6] = 0.0
    r = T.estimate_tone_color(dark)
    assert "black_dominant" in r.warnings and not r.reliable and r.clip_lo_frac > 0.55
    assert warns_sorted(r.warnings)
    # 점 수 부족: 클립 안 된 점 < 64
    tiny = I[:16, :16].copy()
    tiny.reshape(-1, 3)[:200] = 1.0
    r = T.estimate_tone_color(tiny)
    assert "insufficient_pixels" in r.warnings and not r.reliable
    # 임계값은 옵션으로 조절된다
    strict = T.estimate_tone_color(I, opts=T.ToneOpts(clip_dominant_frac=0.0))
    assert ("clipped_dominant" in strict.warnings) == (strict.clip_hi_frac > 0.0)


def warns_sorted(w):
    return list(w) == sorted(set(w))


def test_clip_fraction_definitions_and_sampling():
    rng = np.random.default_rng(2)
    base = H.scene(40, shape=(40, 50))
    img = np.clip(T.forward(base.astype(np.float32), T.ToneParams()), 0.02, 0.9).astype(np.float32)
    flat = img.reshape(-1, 3)
    flat[:500] = (1.0, 0.5, 0.5)         # 한 채널이라도 포화 -> hi
    flat[500:700] = (0.0, 0.0, 0.0)      # 모든 채널 검정 -> lo
    flat[700:760] = (0.0, 0.5, 0.5)      # 한 채널만 검정 -> lo 아님
    r = T.estimate_tone_color(img)
    assert r.n_used == 2000 and r.clip_hi_frac == pytest.approx(500 / 2000) and r.clip_lo_frac == pytest.approx(200 / 2000)
    # max_samples 로 균등 격자 솎기: k=ceil(sqrt(HW/max_samples)), n_used=ceil(H/k)*ceil(W/k)
    big = np.clip(T.forward(H.scene(41, shape=(540, 960)).astype(np.float32), T.ToneParams()), 0, 1)
    r = T.estimate_tone_color(big, opts=T.ToneOpts(max_samples=5000))
    k = math.ceil(math.sqrt(540 * 960 / 5000))
    assert r.n_used == math.ceil(540 / k) * math.ceil(960 / k)
    assert r.n_used <= 5000
    r2 = T.estimate_tone_color(big, opts=T.ToneOpts(max_samples=5000))
    assert r.to_meta() == r2.to_meta()                                           # RNG 없이 결정론


def test_sky_anchor_fixes_absolute_level():
    """어두운 장면: 전역 분포 prior 만으론 레벨이 1.5 stop 틀린다. 하늘 앵커가 위치를 고정한다."""
    L, sky = H.keyed_scene(3, mu=0.18 * 2.0 ** -1.5, sky_frac=0.3)
    p = T.ToneParams(ev=0.3, wb_kelvin=5600, curve=T.FilmCurve(shoulder=0.2))
    I = T.forward(L.astype(np.float32), p)
    Y = color.luminance(L)
    sky_y = float(np.exp(np.mean(np.log(Y[sky]))))
    sig = float(np.std(np.log2(Y)))
    base = T.TonePriors(lum_sigma_stops=sig)

    def sky_level_err(params):
        inv = T.inverse(I, params)
        Yh = color.luminance(inv.linear.astype(np.float64))
        return float(np.median(np.log2(np.maximum(Yh[sky], 1e-9))) - math.log2(sky_y))
    free = T.estimate_tone_color(I, priors=base)
    anch = T.estimate_tone_color(I, priors=T.TonePriors(lum_sigma_stops=sig, sky=T.SkyAnchor(mask=sky, luminance=sky_y)))
    assert abs(sky_level_err(free.params)) > 0.8                                  # 앵커 없이는 절대 레벨을 모른다
    assert abs(sky_level_err(anch.params)) <= 0.35
    assert abs(anch.params.ev - p.ev) <= 0.5 and abs(anch.params.ev - p.ev) < 0.5 * abs(free.params.ev - p.ev)
    assert anch.reliable
    with pytest.raises(ValueError):
        T.estimate_tone_color(I, priors=T.TonePriors(sky=T.SkyAnchor(mask=sky[:-1], luminance=0.5)))


def test_fit_sat_recovers_saturation():
    L = H.scene(3, shape=(64, 96), chroma=0.35)
    for sat in (0.7, 1.4):
        p = T.ToneParams(ev=0.2, wb_kelvin=5600, wb_tint=0.02, curve=T.FilmCurve(toe=0.2, shoulder=0.2, slope=1.1),
                         cdl=T.CDL(sat=sat))
        I = T.forward(L.astype(np.float32), p)
        r = T.estimate_tone_color(I, opts=T.ToneOpts(fit_sat=True))
        assert abs(r.params.cdl.sat - sat) <= 0.15, (sat, r.params.cdl.sat)
        assert r.params.cdl.slope == (1.0, 1.0, 1.0) and r.params.cdl.offset == (0.0, 0.0, 0.0) and r.params.cdl.power == (1.0, 1.0, 1.0)
        assert T.estimate_tone_color(I).params.cdl.is_identity()                   # 기본은 CDL 추정 안 함


def test_estimate_input_validation():
    ok = np.random.default_rng(0).uniform(0.02, 0.8, (16, 16, 3)).astype(np.float32)
    T.estimate_tone_color(ok)
    T.estimate_tone_color(ok.astype(np.float64))
    with pytest.raises(TypeError):
        T.estimate_tone_color((ok * 255).astype(np.uint8))
    for bad in (ok[..., :2], ok[0], ok[:4], ok[:, :4], np.zeros((0, 16, 3), np.float32)):
        with pytest.raises(ValueError):
            T.estimate_tone_color(bad)
    nan = ok.copy()
    nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        T.estimate_tone_color(nan)
    neg = ok.copy()
    neg[0, 0, 0] = -0.01
    with pytest.raises(ValueError):
        T.estimate_tone_color(neg)
    with pytest.raises(TypeError):
        T.estimate_tone_color(ok, priors={"ev": 0.0})
    with pytest.raises(TypeError):
        T.estimate_tone_color(ok, opts={"fit_sat": True})


def test_runtime_1080p():
    img = np.clip(T.forward(H.scene(2, shape=(1080, 1920)).astype(np.float32), T.ToneParams(ev=0.3, wb_kelvin=5400)), 0, 1)
    t0 = time.time()
    r = T.estimate_tone_color(img)
    assert time.time() - t0 < 10.0
    assert r.n_used <= 20000 and r.reliable
    assert abs(r.params.ev - 0.3) <= 0.3

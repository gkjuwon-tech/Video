"""LNS-08 핵심 성질: explain∘look ≈ id (소유 영역, 증명 가능한 오차 한계) 와 look∘explain ≈ id (치역)."""
import time

import numpy as np
import pytest

from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.lens import theta as T
from lns08_helpers import (CLEAN_LUTS, CLIPPY_LUTS, HALO_TAINT_EPS, LUTS, OWN_HALO_TAINT, OWN_LUT_LOST,
                           OWN_SENSOR_BLACK, OWN_SENSOR_SAT, bound_ref, exposure_ref, flat, gauss_blur_ref,
                           scene_hdr, scene_mid, scene_ramp, theta)

Q12 = 1.0 / 4095.0
SLACK = 1e-7  # float32 반올림 여유(절대)


def roundtrip(x, th, **kw):
    y = chain.look(x, th, noise=False, **kw)
    return y, chain.explain(y, th, **({"spec": kw["spec"]} if "spec" in kw else {}))


def assert_sound(x, r, tag=""):
    """소유 화소에서 |linear - x| <= error_bound (모든 채널). 소유 화소가 하나도 없으면 공허하므로 실패."""
    assert r.owned.any(), tag
    own3 = np.broadcast_to(r.owned[..., None], x.shape)
    err = np.abs(r.linear.astype(np.float64) - x.astype(np.float64))
    bnd = r.error_bound.astype(np.float64)
    assert np.all(np.isfinite(bnd[own3])), tag
    viol = (err > bnd + SLACK) & own3
    assert not viol.any(), (tag, int(viol.sum()), float((err / bnd)[own3].max()))


# ------------------------------------------------------------------------------------------------ 기본 왕복
def test_default_theta_roundtrip_is_tight_and_fully_owned():
    x = scene_mid()
    y, r = roundtrip(x, theta())
    assert r.owned.all()
    err = np.abs(r.linear - x)
    assert float(err.max()) <= 0.5 * Q12 + 2e-5 + 1e-7  # 12 비트 양자화 + 역변환 허용오차
    assert np.all(r.error_bound[r.owned] >= err[r.owned] - SLACK)
    assert float(r.error_bound.max()) < 1.8e-4  # 기본 θ 에서 한계가 실제로 작다(무한대로 부풀리지 않음)


@pytest.mark.parametrize("lut", LUTS)
def test_error_bound_follows_contract_formula(lut):
    th = theta(lut=lut, wb_kelvin=4300.0, wb_tint=0.2, ev=-0.4, vignette=0.5, halation=0.35)
    x = scene_mid(32, 48)
    y, r = roundtrip(x, th)
    ref = bound_ref(th, *x.shape[:2])
    own3 = np.broadcast_to(r.owned[..., None], x.shape)
    if lut != "mono_silver":
        assert r.owned.mean() > 0.5
    if r.owned.any():
        assert np.max(np.abs(r.error_bound[own3] / ref[own3] - 1.0)) < 2e-3
    assert np.all(np.isinf(r.error_bound[~own3])) and np.all(r.error_bound[~own3] > 0)


def test_error_bound_scales_with_bits_and_gains():
    x = scene_mid()
    spec8 = sf.SensorSpec(full_well_e=20000.0, read_noise_e=3.0, bits=8)
    th = theta(lut="neutral_709", vignette=0.6, ev=1.0, wb_kelvin=3000.0)
    y8, r8 = roundtrip(x * 0.5, th, spec=spec8)
    own3 = np.broadcast_to(r8.owned[..., None], x.shape)
    ref = bound_ref(th, *x.shape[:2], bits=8)
    assert r8.owned.any() and np.max(np.abs(r8.error_bound[own3] / ref[own3] - 1.0)) < 2e-3
    assert_sound(x * 0.5, r8, "8bit")
    # 비네팅이 센서 앞의 이득이라 가장자리에서 한계가 커진다
    r_center = r8.error_bound[r8.owned.shape[0] // 2, r8.owned.shape[1] // 2, 1]
    corner = r8.error_bound[0, 0, 1]
    assert (not np.isfinite(corner)) or corner > 1.5 * r_center


# ------------------------------------------------------------------------------------------------ 건전성(거짓 소유 없음)
@pytest.mark.parametrize("lut", LUTS)
@pytest.mark.parametrize("img", ["mid", "hdr"])
def test_owned_pixels_are_recovered_within_bound_per_lut(lut, img):
    x = scene_mid(32, 48) if img == "mid" else scene_hdr(32, 48)
    for kw in (dict(wb_kelvin=3300.0, wb_tint=-0.2, ev=0.6, vignette=0.7),
               dict(wb_kelvin=9500.0, wb_tint=0.4, ev=-1.2, vignette=0.2, halation=0.6),
               dict(ev=0.3, halation=1.0, vignette=1.0, wb_kelvin=2200.0)):
        th = theta(lut=lut, **kw)
        y, r = roundtrip(x, th)
        if lut == "mono_silver":
            assert not r.owned.any() and np.all((r.status & OWN_LUT_LOST) != 0)
            continue
        assert_sound(x, r, (lut, img, kw))


def test_sampled_thetas_soundness_and_nonvacuous_ownership():
    ths = T.sample_thetas(24, seed=3)
    owned_frac = []
    for i, th in enumerate(ths):
        for x in (scene_mid(32, 48), scene_hdr(32, 48)):
            y, r = roundtrip(x, th)
            if th.lut == "mono_silver":
                assert not r.owned.any()
                continue
            err = np.abs(r.linear.astype(np.float64) - x)
            own3 = np.broadcast_to(r.owned[..., None], x.shape)
            assert not ((err > r.error_bound.astype(np.float64) + SLACK) & own3).any(), (i, th.lut)
            owned_frac.append(float(r.owned.mean()))
    assert float(np.median(owned_frac)) > 0.85  # 표본 θ 대부분에서 장면 대부분을 소유한다
    assert min(owned_frac) >= 0.0 and max(owned_frac) > 0.97


@pytest.mark.parametrize("lut", LUTS)
def test_midtone_scene_ownership_completeness(lut):
    floors = {"none": 0.99, "neutral_709": 0.98, "bleach_bypass": 0.98, "faded_matte": 0.98, "cross_process": 0.95,
              "teal_orange": 0.95, "kodak_warm": 0.92, "fuji_cool": 0.95, "neon_night": 0.8, "mono_silver": 0.0}
    th = theta(lut=lut, wb_kelvin=5200.0, vignette=0.4, halation=0.3)
    x = scene_mid(32, 48)
    y, r = roundtrip(x, th)
    assert float(r.owned.mean()) >= floors[lut], (lut, float(r.owned.mean()))
    if lut == "mono_silver":
        assert float(r.owned.mean()) == 0.0


# ------------------------------------------------------------------------------------------------ 소유 상태 비트
def test_saturation_and_black_flags_on_log_ramp():
    x = scene_ramp()
    # 비네팅·할레이션·EV·WB 중립(theta(lut=...)): 센서 노출 == x
    xmax, xmin = x.max(axis=-1), x.min(axis=-1)
    for lut in CLEAN_LUTS:
        y, r = roundtrip(x, theta(lut=lut))
        st = r.status
        # 포화(어느 채널이든 노출 >= 1)는 절대 소유가 아니고 SAT 비트가 선다
        assert np.all((st[xmax >= 1.0] & OWN_SENSOR_SAT) != 0), lut
        assert not r.owned[xmax >= 1.0].any()
        # SAT 은 실제로 포화 근처일 때만 선다
        assert np.all(xmax[(st & OWN_SENSOR_SAT) != 0] >= 1.0 - 3.0 * Q12), lut
        # 소유 보장 구간: 바닥 위, 포화 한참 아래
        safe = (xmin >= 5e-3) & (xmax <= 0.8) if lut != "none" else (xmax <= 0.99)
        assert r.owned[safe].all(), (lut, float((~r.owned[safe]).mean()))
        # 비소유 화소는 한계가 무한대, 소유 화소는 유한
        assert np.all(np.isinf(r.error_bound[~r.owned])) and np.all(np.isfinite(r.error_bound[r.owned]))
        if lut != "none":
            low = xmin <= 5e-4
            assert low.any() and np.all((st[low] & OWN_SENSOR_BLACK) != 0), lut
            assert np.all(xmin[(st & OWN_SENSOR_BLACK) != 0] <= 1e-3), lut
            assert not r.owned[low].any()
        else:
            assert np.all((st & OWN_SENSOR_BLACK) == 0)  # sRGB 직부호화에는 바닥이 없다
            dark = xmax <= 1e-3
            assert dark.any() and r.owned[dark].all()
        assert_sound(x, r, lut)


def test_status_bits_are_the_documented_flags_only():
    x = scene_hdr(48, 160)
    th = theta(lut="kodak_warm", halation=0.6, vignette=0.3, ev=0.5)
    y, r = roundtrip(x, th)
    allowed = OWN_SENSOR_BLACK | OWN_SENSOR_SAT | OWN_LUT_LOST | OWN_HALO_TAINT
    assert np.all((r.status & ~np.uint8(allowed)) == 0)
    assert np.array_equal(r.owned, r.status == 0)


# ------------------------------------------------------------------------------------------------ 할레이션
def test_halation_is_inverted_exactly_where_unsaturated():
    # 포화하지 않는 밝은 영역(0.85) + 할레이션 1.0: 층이 한계보다 훨씬 크지만 역사슬이 되돌린다.
    x = np.full((48, 96, 3), 0.15, np.float32)
    x[12:36, 30:66] = 0.85
    th = theta(lut="none", halation=1.0)
    u = exposure_ref(x, th).astype(np.float32)
    layer = sf.halation_layer(u, 1.0)
    assert float(layer.max()) > 5e-3 and float((u + layer).max()) < 0.995  # 의미 있는 층, 포화는 없음
    y, r = roundtrip(x, th)
    assert r.owned.all()
    assert float(layer.max()) > 20.0 * float(r.error_bound[r.owned].max()) / 2.9
    assert_sound(x, r, "halation")
    # 할레이션 없이 되돌렸다면(=층을 무시) 한계를 넘는다: 이 테스트가 층 제거를 실제로 검증함
    r0 = chain.explain(y, theta(lut="none"))
    assert np.max(np.abs(r0.linear - x)) > 10.0 * float(r.error_bound[r.owned].max())
    # 같은 장면을 로그 LUT 룩으로 찍어도 소유로 판정된 화소는 한계 안(밝은 곳 일부는 LUT 조건 때문에 비소유일 수 있다)
    th2 = theta(lut="bleach_bypass", halation=1.0)
    y2, r2 = roundtrip(x, th2)
    assert r2.owned.mean() > 0.6
    assert_sound(x, r2, "halation-bleach")


def test_halo_taint_follows_saturated_light_support():
    h, w = 48, 160
    x = np.full((h, w, 3), 0.3, np.float32)
    x[21:27, 77:83] = (6.0, 5.5, 5.0)
    th = theta(lut="neutral_709", halation=0.8, vignette=0.2)
    y, r = roundtrip(x, th)
    u = exposure_ref(x, th).astype(np.float32)
    s_true = u + sf.halation_layer(u, 0.8)
    sat = (s_true.max(axis=-1) >= 1.0).astype(np.float64)
    assert sat.sum() > 30
    blur = gauss_blur_ref(sat, w)
    taint = (r.status & OWN_HALO_TAINT) != 0
    assert taint[blur > 10 * HALO_TAINT_EPS].all()
    assert not taint[blur < 0.1 * HALO_TAINT_EPS].any()
    assert (blur < 0.1 * HALO_TAINT_EPS).sum() > 0.2 * h * w  # 먼 곳은 실제로 소유로 남는다
    assert not r.owned[taint].any()
    assert_sound(x, r, "taint")
    # 광원 포화 화소 자신
    assert np.all((r.status[sat > 0] & OWN_SENSOR_SAT) != 0)


def test_no_halo_taint_without_halation_or_saturation():
    x = scene_hdr(48, 160)
    y, r = roundtrip(x, theta(lut="neutral_709", halation=0.0))
    assert np.all((r.status & OWN_HALO_TAINT) == 0) and (r.status & OWN_SENSOR_SAT).any()
    xm = scene_mid(48, 160)
    y, r = roundtrip(xm, theta(lut="neutral_709", halation=1.0))
    assert np.all((r.status & OWN_HALO_TAINT) == 0)


# ------------------------------------------------------------------------------------------------ look∘explain ≈ id
@pytest.mark.parametrize("lut", LUTS)
def test_look_of_explain_reproduces_display(lut):
    th = theta(lut=lut, wb_kelvin=4800.0, wb_tint=0.1, ev=0.2, vignette=0.4, halation=0.5)
    for x in (scene_mid(32, 48), scene_hdr(32, 48)):
        y = chain.look(x, th, noise=False)
        r = chain.explain(y, th)
        back = chain.look(r.linear, th, noise=False)
        sel = np.ones(y.shape[:2], bool) if lut == "mono_silver" else (r.status & OWN_LUT_LOST) == 0
        assert sel.any()
        assert float(np.max(np.abs(back - y)[sel])) <= 2e-3, (lut, float(np.max(np.abs(back - y)[sel])))
    # 포화 화소도 하한값으로 돌려져 다시 찍으면 같은 표시값(소유 여부와 무관)
    y = chain.look(scene_hdr(), theta(lut="neutral_709"), noise=False)
    r = chain.explain(y, theta(lut="neutral_709"))
    sat = (r.status & OWN_SENSOR_SAT) != 0
    assert sat.any()
    back = chain.look(r.linear, theta(lut="neutral_709"), noise=False)
    assert np.max(np.abs(back - y)[sat]) <= 2e-3
    assert np.all(r.linear[sat].max(axis=-1) > 0.5)  # 하한값이지 0 이 아니다


def test_explain_is_idempotent_through_look():
    th = theta(lut="teal_orange", wb_kelvin=3900.0, ev=0.4, vignette=0.5, halation=0.4)
    x = scene_mid()
    y = chain.look(x, th, noise=False)
    r1 = chain.explain(y, th)
    y1 = chain.look(r1.linear, th, noise=False)
    r2 = chain.explain(y1, th)
    both = r1.owned & r2.owned
    assert both.mean() > 0.9
    d = np.abs(r2.linear.astype(np.float64) - r1.linear)[both]
    assert float(d.max()) <= float(np.max(r1.error_bound[both] + r2.error_bound[both]))
    # 이미 양자화된 격자 위라 y1 == y (재양자화로 안 흔들린다)
    assert float(np.max(np.abs(y1 - y)[r1.owned])) <= 2e-4


# ------------------------------------------------------------------------------------------------ 재촬영(룩 교체)
def test_relook_through_explain_matches_direct_look():
    th1 = theta(lut="teal_orange", wb_kelvin=4500.0, ev=0.3, vignette=0.3, halation=0.3)
    th2 = theta(lut="kodak_warm", wb_kelvin=7200.0, wb_tint=0.15, ev=-0.2, vignette=0.5, halation=0.2)
    x = scene_mid()
    r = chain.explain(chain.look(x, th1, noise=False), th1)
    assert r.owned.mean() > 0.9
    got = chain.look(r.linear, th2, noise=False)
    want = chain.look(x, th2, noise=False)
    d = np.abs(got - want)[r.owned]
    # 어두운 곳에서 로그 곡선이 가파르므로 최댓값은 완화하고 평균·분위수로 본다
    assert float(d.mean()) <= 5e-4 and float(np.quantile(d, 0.999)) <= 5e-3


# ------------------------------------------------------------------------------------------------ 노이즈 경로
def test_grain_sigma_matches_formula_and_monte_carlo():
    th = theta(grain=0.5, lut="none", sensor_width_mm=36.0)
    u = np.linspace(0.0, 1.0, 33, dtype=np.float32)[None, :, None].repeat(3, axis=2).repeat(2, axis=0)
    s = chain.grain_sigma(u, th)
    assert s.dtype == np.float32 and s.shape == u.shape
    assert np.max(np.abs(s - 0.5 * sf.GRAIN_SIGMA_MAX * 2.0 * np.sqrt(u * (1.0 - u)))) < 1e-7
    assert np.all(chain.grain_sigma(u, theta(grain=0.0)) == 0.0)
    assert s[0, 0, 0] == 0.0 and s[0, -1, 0] == 0.0 and s[0, 16, 0] == pytest.approx(0.04, abs=1e-6)
    for target in (0.2, 0.46, 0.8):
        lin = float(((target + 0.055) / 1.055) ** 2.4)
        g = np.full((128, 128, 3), lin, np.float32)
        y0 = chain.look(g, th, noise=False)
        y = chain.look(g, th, seed=1)
        emp = float((y - y0).std())
        pred = float(chain.grain_sigma(y0, th).mean())
        assert emp == pytest.approx(pred, rel=0.06), (target, emp, pred)


@pytest.mark.parametrize("lut", ["none", "teal_orange", "neutral_709"])
def test_explain_of_noisy_look_is_finite_and_unbiased_on_flat_patch(lut):
    th = theta(lut=lut, grain=0.3, sensor_width_mm=36.0)
    g = flat(0.18, 48, 48)
    errs = []
    for sd in range(8):
        y = chain.look(g, th, seed=sd)
        r = chain.explain(y, th)
        assert np.all(np.isfinite(r.linear)) and np.all(np.isfinite(r.error_bound[r.owned]))
        errs.append((r.linear - g)[r.owned])
    e = np.concatenate([a.ravel() for a in errs])
    assert e.size > 8 * 48 * 48 * 3 * 0.9  # 노이즈가 있어도 중간톤은 대부분 소유
    assert abs(float(e.mean())) < 3e-3
    assert 0.005 < float(e.std()) < 0.06  # 그레인이 선형 오차로 전파됨(0 이면 그레인이 없는 것)


# ------------------------------------------------------------------------------------------------ 성능
def test_explain_speed_cpu():
    x = scene_hdr(144, 256)
    th = theta(lut="kodak_warm", halation=0.5, vignette=0.3, wb_kelvin=5000.0)
    y = chain.look(x, th, noise=False)
    chain.explain(y[:16, :16], th)  # 표 캐시 예열 허용
    t0 = time.perf_counter()
    r = chain.explain(y, th)
    dt = time.perf_counter() - t0
    assert dt < 8.0, f"{dt:.1f}s"
    assert r.linear.shape == x.shape and r.owned.any()


def test_clippy_luts_remain_sound_on_saturated_colors():
    # 클립 룩에서도 소유로 판정된 화소는 한계 안(거짓 소유 없음): 원색에 가까운 밝은 색 포함
    rng = np.random.default_rng(9)
    x = (rng.random((40, 60, 3)) ** 2.0 * 0.95 + 0.002).astype(np.float32)
    for lut in CLIPPY_LUTS:
        y, r = roundtrip(x, theta(lut=lut))
        assert (r.status != 0).any()
        assert_sound(x, r, lut)

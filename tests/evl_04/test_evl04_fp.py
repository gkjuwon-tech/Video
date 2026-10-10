"""EVL-04 FP(t): 프레임별 공동 가시 영역 LPIPS·ΔE00 표류와 기울기."""
import numpy as np
import pytest

from evl04_support import FakeClient, ff, texture


def _drift_video(n=5, step=0.01, h=64, w=96, seed=1):
    base = texture(h, w, seed=seed, amp=0.1, base=0.4)  # [0.2, 0.6] 안: 클리핑 없음
    frames = np.stack([base + step * k for k in range(n)]).astype(np.float32)
    warped = np.stack([base] * n)
    covis = np.ones((n, h, w), bool)
    return frames, warped, covis


def test_linear_drift_gives_known_lpips_curve_and_slope():
    m = ff()
    frames, warped, covis = _drift_video()
    fake = FakeClient()
    cur = m.fp_curve(frames, warped, covis, fake)
    np.testing.assert_allclose(cur.times, np.arange(5.0))
    np.testing.assert_allclose(cur.lpips, 0.01 * np.arange(5), atol=3e-5)  # 가짜 거리 = 평균 |차|
    assert cur.slope_lpips == pytest.approx(0.01, abs=1e-5)
    np.testing.assert_allclose(cur.covis, 1.0)
    assert cur.backend == "fake"
    # 입력 쪽(워프된 입력)이 앞, 프레임이 뒤
    assert len(fake.calls) == 5
    a0, b0 = fake.calls[3]
    assert float(np.mean(b0 - a0)) == pytest.approx(0.03, abs=3e-5)


def test_times_set_the_slope_unit():
    m = ff()
    frames, warped, covis = _drift_video()
    t = np.arange(5) / 24.0
    cur = m.fp_curve(frames, warped, covis, FakeClient(), times=t)
    np.testing.assert_allclose(cur.times, t)
    assert cur.slope_lpips == pytest.approx(0.01 * 24, rel=2e-3)


def test_de00_drift_increases_and_matches_polyfit_slope():
    m = ff()
    frames, warped, covis = _drift_video(n=6, step=0.02)
    cur = m.fp_curve(frames, warped, covis, FakeClient())
    assert cur.de00[0] == 0.0
    assert np.all(np.diff(cur.de00) > 0.2)
    assert cur.slope_de00 == pytest.approx(np.polyfit(cur.times, cur.de00, 1)[0], rel=1e-6)
    assert cur.slope_de00 > 0.3
    # 각 프레임의 값은 같은 입력으로 계산한 delta_e00 과 일치
    assert cur.de00[4] == pytest.approx(m.delta_e00(warped[4], frames[4]), abs=1e-6)


def test_perfect_world_has_flat_zero_curve():
    m = ff()
    base = texture(64, 96, seed=2)
    frames = np.stack([base] * 4)
    with m.LpipsClient("proxy") as c:
        cur = m.fp_curve(frames, frames.copy(), np.ones((4, 64, 96), bool), c)
    assert np.all(cur.lpips == 0.0) and np.all(cur.de00 == 0.0)
    assert cur.slope_lpips == pytest.approx(0.0, abs=1e-12) and cur.slope_de00 == pytest.approx(0.0, abs=1e-12)
    assert cur.backend == "proxy"


def test_only_co_visible_pixels_are_compared():
    m = ff()
    frames, warped, covis = _drift_video(n=3, step=0.05, h=96, w=128)
    covis = covis.copy()
    covis[:, :, 64:] = False  # 오른쪽 절반은 공동 가시 아님(워프된 입력이 거기서 틀려도 상관없음)
    warped = warped.copy()
    warped[:, :, 64:] = 0.0
    fake = FakeClient()
    cur = m.fp_curve(frames, warped, covis, fake)
    a, b = fake.calls[2]
    # 마스크 밖에서 경계(블러 반경)보다 멀리 떨어진 곳은 두 배열이 같다
    np.testing.assert_allclose(a[:, 72:], b[:, 72:], atol=1e-6)
    assert float(np.max(np.abs(a[:, :56] - b[:, :56]))) > 0.09  # 안쪽은 실제 표류(0.1)가 남아 있다
    np.testing.assert_allclose(cur.covis, 0.5)
    assert cur.de00[2] > 0.5 and cur.de00[0] == 0.0
    assert np.all(np.isfinite(cur.lpips))


def test_frames_without_covisibility_are_nan_and_excluded_from_slope():
    m = ff()
    frames, warped, covis = _drift_video(n=4, step=0.01)
    covis = covis.copy()
    covis[3] = False
    cur = m.fp_curve(frames, warped, covis, FakeClient())
    assert np.isnan(cur.lpips[3]) and np.isnan(cur.de00[3]) and cur.covis[3] == 0.0
    assert np.all(np.isfinite(cur.lpips[:3]))
    assert cur.slope_lpips == pytest.approx(0.01, abs=1e-5)
    assert m.FP_MIN_COVIS == pytest.approx(0.05)
    # 점유율 4% 프레임도 NaN
    covis2 = np.ones_like(covis)
    covis2[1] = False
    covis2[1, :8, :32] = True  # 256/6144 = 4.2% < 5%
    cur2 = m.fp_curve(frames, warped, covis2, FakeClient())
    assert np.isnan(cur2.lpips[1])
    # 유한한 점이 2 개 미만이면 기울기는 NaN
    covis3 = np.zeros_like(covis)
    covis3[0] = True
    cur3 = m.fp_curve(frames, warped, covis3, FakeClient())
    assert np.isnan(cur3.slope_lpips) and np.isnan(cur3.slope_de00)


def test_fp_validation_and_purity():
    m = ff()
    frames, warped, covis = _drift_video(n=3)
    keep = (frames.copy(), warped.copy(), covis.copy())
    m.fp_curve(frames, warped, covis, FakeClient())
    assert np.array_equal(frames, keep[0]) and np.array_equal(warped, keep[1]) and np.array_equal(covis, keep[2])
    with pytest.raises(ValueError):
        m.fp_curve(frames[:2], warped, covis, FakeClient())
    with pytest.raises(ValueError):
        m.fp_curve(frames, warped, covis[:, :-1], FakeClient())
    with pytest.raises(ValueError):
        m.fp_curve(frames, warped, covis, FakeClient(), times=[0.0, 1.0])
    with pytest.raises(ValueError):
        m.fp_curve(frames, warped, covis.astype(np.float32), FakeClient())  # 마스크는 bool
    with pytest.raises(ValueError):
        m.fp_curve(frames[:0], warped[:0], covis[:0], FakeClient())
    with pytest.raises(ValueError):
        m.fp_curve(frames, warped, covis, FakeClient(), times=[0.0, 1.0, np.nan])

"""EVL-04 FFD: Ψ(리샘플)·B(σ=0.7 블러) 전처리와 대역 거리, 대역 백엔드(proxy)의 성질."""
import io

import numpy as np
import pytest
from PIL import Image

from buc.core.pyramid import gaussian_blur, resize
from evl04_support import FakeClient, ff, noisy, texture


# ------------------------------------------------------------------------------------------- 전처리 Ψ, B
def test_prepare_small_image_is_only_blurred():
    m = ff()
    img = texture(64, 96, seed=1)
    p = m.prepare(img)
    assert p.dtype == np.float32 and p.shape == img.shape
    np.testing.assert_allclose(p, gaussian_blur(img, 0.7), atol=1e-6)  # 업샘플 없음, B 는 σ=0.7
    assert (m.GRAIN_BLUR_SIGMA, m.REF_SHORT) == (0.7, 1080)


def test_prepare_downsamples_short_side_to_target_with_bilinear_then_blur():
    m = ff()
    img = texture(300, 450, seed=2)
    p = m.prepare(img, short_side=150)
    assert p.shape == (150, 225, 3)
    ref = gaussian_blur(resize(img, (150, 225), "bilinear"), 0.7)
    np.testing.assert_allclose(p, ref, atol=1e-5)


def test_prepare_r_eff_caps_the_target_and_never_upsamples():
    m = ff()
    img = texture(300, 450, seed=3)
    assert m.prepare(img, r_eff=100, short_side=150).shape == (100, 150, 3)  # min(r_eff, short_side)
    assert m.prepare(img, r_eff=900, short_side=150).shape == (150, 225, 3)
    assert m.prepare(img, r_eff=900, short_side=1080).shape == (300, 450, 3)  # 짧은 변이 목표보다 작으면 그대로


def test_prepare_long_side_rounds_half_up():
    m = ff()
    assert m.prepare(texture(300, 453, seed=4), short_side=150).shape == (150, 227, 3)  # 226.5 -> 227
    assert m.prepare(texture(301, 450, seed=4), short_side=150).shape == (150, 224, 3)  # 224.25 -> 224
    assert m.prepare(texture(453, 300, seed=4), short_side=150).shape == (227, 150, 3)  # 세로 영상도 짧은 변 기준


def test_prepare_does_not_modify_input_and_is_deterministic():
    m = ff()
    img = texture(80, 120, seed=5)
    keep = img.copy()
    a = m.prepare(img, short_side=40)
    b = m.prepare(img, short_side=40)
    assert np.array_equal(img, keep) and np.array_equal(a, b)


# ------------------------------------------------------------------------------------------- 입력 검증
def test_as_display_accepts_uint8_float_and_rejects_garbage():
    m = ff()
    u8 = (np.random.default_rng(0).uniform(0, 255, (8, 9, 3))).astype(np.uint8)
    d = m.as_display(u8)
    assert d.dtype == np.float32 and d.shape == (8, 9, 3)
    np.testing.assert_allclose(d, u8 / 255.0, atol=1e-7)
    f64 = np.full((8, 9, 3), 0.25)
    assert m.as_display(f64).dtype == np.float32
    with pytest.raises(TypeError):
        m.as_display(np.zeros((8, 9, 3), np.int32))
    with pytest.raises(ValueError):
        m.as_display(np.zeros((8, 9), np.float32))
    with pytest.raises(ValueError):
        m.as_display(np.zeros((8, 9, 4), np.float32))
    with pytest.raises(ValueError):
        m.as_display(np.full((8, 9, 3), 1.5, np.float32))
    with pytest.raises(ValueError):
        m.as_display(np.full((8, 9, 3), np.nan, np.float32))
    with pytest.raises(ValueError):
        m.as_display(np.zeros((0, 9, 3), np.float32))


# ------------------------------------------------------------------------------------------- FFD 정의
def test_ffd_passes_prepared_pair_to_backend_input_first():
    m = ff()
    inp = texture(300, 450, seed=6)
    f0 = noisy(inp, 0.02, seed=7)
    fake = FakeClient()
    v = m.ffd(inp, f0, fake)
    assert type(v) is float and len(fake.calls) == 1
    a, b = fake.calls[0]
    np.testing.assert_allclose(a, m.prepare(inp), atol=1e-6)  # 입력이 앞, F0 가 뒤
    np.testing.assert_allclose(b, m.prepare(f0), atol=1e-6)
    assert v == pytest.approx(float(np.mean(np.abs(a.astype(np.float64) - b))), abs=1e-12)


def test_ffd_applies_psi_with_r_eff_and_same_size_for_both():
    m = ff()
    inp = texture(300, 450, seed=8)
    f0 = noisy(inp, 0.02, seed=9)
    fake = FakeClient()
    m.ffd(inp, f0, fake, r_eff=120)
    a, b = fake.calls[0]
    assert a.shape == b.shape == (120, 180, 3)


def test_ffd_validates_inputs():
    m = ff()
    a = texture(40, 60, seed=10)
    fake = FakeClient()
    with pytest.raises(ValueError):
        m.ffd(a, a[:, :-1], fake)  # 크기가 다르면 ValueError(호출자가 같은 크롭으로 맞춰야 한다)
    with pytest.raises(ValueError):
        m.ffd(a, a[..., :2], fake)
    assert fake.calls == []  # 거절은 백엔드 호출 전에


def test_ffd_accepts_uint8_same_as_float():
    m = ff()
    u8 = (texture(40, 60, seed=11) * 255).astype(np.uint8)
    u8b = np.clip(u8.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    f1, f2 = FakeClient(), FakeClient()
    v1 = m.ffd(u8, u8b, f1)
    v2 = m.ffd(u8 / np.float32(255), u8b / np.float32(255), f2)
    assert v1 == pytest.approx(v2, abs=1e-6)


def test_ffd_blur_removes_grain_realization_difference():
    m = ff()
    base = texture(96, 128, seed=12)
    a, b = noisy(base, 0.03, seed=13), noisy(base, 0.03, seed=14)  # 같은 장면, 그레인 실현만 다르다
    with m.LpipsClient("proxy") as c:
        raw = c.distance(a, b)
        blurred = m.ffd(a, b, c)
        assert c.distance(a, a) == 0.0
    assert 0.0 < blurred < 0.9 * raw


# ------------------------------------------------------------------------------------------- proxy 백엔드 성질
@pytest.fixture(scope="module")
def tex():
    return texture(96, 128, seed=20, sigma=1.5)


def test_proxy_metric_axioms(tex):
    m = ff()
    b = noisy(tex, 0.05, seed=21)
    assert m.proxy_lpips(tex, tex) == 0.0
    d_ab, d_ba = m.proxy_lpips(tex, b), m.proxy_lpips(b, tex)
    assert d_ab > 0 and np.isfinite(d_ab)
    assert d_ab == pytest.approx(d_ba, abs=1e-9)
    assert type(d_ab) is float


def test_proxy_monotone_in_blur_and_noise(tex):
    m = ff()
    from scipy import ndimage
    blur = [np.stack([ndimage.gaussian_filter(tex[..., c], s) for c in range(3)], -1).astype(np.float32)
            for s in (0.5, 1.0, 2.0, 4.0)]
    d = [m.proxy_lpips(tex, x) for x in blur]
    assert all(x < y for x, y in zip(d, d[1:])), d
    dn = [m.proxy_lpips(tex, noisy(tex, s, seed=22)) for s in (0.01, 0.03, 0.1)]
    assert all(x < y for x, y in zip(dn, dn[1:])), dn


def test_proxy_unrelated_image_is_farther_than_mild_distortion(tex):
    m = ff()
    other = texture(96, 128, seed=99, sigma=1.5)
    assert m.proxy_lpips(tex, other) > 2 * m.proxy_lpips(tex, noisy(tex, 0.03, seed=23))


def test_proxy_orders_jpeg_qualities(tex):
    m = ff()

    def jpeg(img, q):
        buf = io.BytesIO()
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=q)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())), np.float32) / 255.0

    d90, d30 = m.proxy_lpips(tex, jpeg(tex, 90)), m.proxy_lpips(tex, jpeg(tex, 30))
    assert 0 < d90 < d30


def test_proxy_rejects_bad_input(tex):
    m = ff()
    with pytest.raises(ValueError):
        m.proxy_lpips(tex, tex[:-1])
    with pytest.raises(ValueError):
        m.proxy_lpips(tex[..., 0], tex[..., 0])
    with pytest.raises(ValueError):
        m.proxy_lpips(tex * 3.0, tex)
    bad = tex.copy()
    bad[3, 3, 1] = np.inf
    with pytest.raises(ValueError):
        m.proxy_lpips(tex, bad)


def test_proxy_does_not_mutate_inputs(tex):
    m = ff()
    a, b = tex.copy(), noisy(tex, 0.05, seed=24)
    ka, kb = a.copy(), b.copy()
    m.proxy_lpips(a, b)
    assert np.array_equal(a, ka) and np.array_equal(b, kb)

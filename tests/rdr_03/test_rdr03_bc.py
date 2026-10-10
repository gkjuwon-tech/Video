"""RDR-03 BC 코덱: BC7 모드 6 디코드 비트 일치(독립 오라클), 인코더 모드·화질, BC5, 페이지 로그 코덱 오차 한계·크기,
압축 페이지 렌더."""
import numpy as np
import pytest
import torch

from buc.render import tsq as T
from rdr03_helpers import (K_of, bc5_decode_image, bc7_decode_image, make_tsq, psnr8, random_scene, smooth_image,
                           to_np)


def _rand_mode6(nblocks, rng):
    b = bytearray(rng.integers(0, 256, nblocks * 16, dtype=np.uint8).tobytes())
    for k in range(nblocks):
        b[16 * k] = 0x40 | (b[16 * k] & 0x80)
    return bytes(b)


def test_decode_bc7_mode6_bit_exact_vs_oracle():
    rng = np.random.default_rng(0)
    for w, h in ((16, 8), (8, 32), (4, 4)):
        data = _rand_mode6((w // 4) * (h // 4), rng)
        out = T.decode_bc7(data, w, h)
        assert out.dtype == np.uint8 and out.shape == (h, w, 4)
        assert np.array_equal(out, bc7_decode_image(data, w, h))


def test_decode_bc7_reserved_and_bad_length():
    rng = np.random.default_rng(1)
    data = bytearray(_rand_mode6(4, rng))
    data[16] = 0x00                      # 블록 1: 예약 모드
    out = T.decode_bc7(bytes(data), 8, 8)
    assert np.array_equal(out[0:4, 4:8], np.zeros((4, 4, 4), np.uint8))
    assert np.array_equal(out[0:4, 0:4], bc7_decode_image(bytes(data[0:16]), 4, 4))
    with pytest.raises(ValueError):
        T.decode_bc7(bytes(data[:-1]), 8, 8)
    # 모드 0 블록: 지원하면 올바른 모양, 아니면 ValueError
    m0 = bytes([0x01]) + bytes(15)
    try:
        o = T.decode_bc7(m0, 4, 4)
        assert o.shape == (4, 4, 4) and o.dtype == np.uint8
    except ValueError:
        pass


def test_encode_bc7_mode6_quality_and_oracle_consistency():
    img = np.concatenate([np.rint(smooth_image(32, 32, 3, 2)).astype(np.uint8),
                          np.full((32, 32, 1), 255, np.uint8)], -1)
    data = T.encode_bc7(img)
    assert isinstance(data, bytes) and len(data) == 64 * 16
    assert all((data[16 * k] & 0x7F) == 0x40 for k in range(64))
    dec = T.decode_bc7(data, 32, 32)
    assert np.array_equal(dec, bc7_decode_image(data, 32, 32))
    assert psnr8(dec[..., :3], img[..., :3]) >= 36.0
    assert dec[..., 3].min() >= 254
    # 상수 블록: 채널마다 오차 ≤ 1
    rng = np.random.default_rng(3)
    for _ in range(10):
        c = np.r_[rng.integers(0, 256, 3), 255].astype(np.uint8)
        blk = np.broadcast_to(c, (4, 4, 4)).copy()
        d = T.decode_bc7(T.encode_bc7(blk), 4, 4).astype(np.int64)
        assert np.abs(d - blk.astype(np.int64)).max() <= 1
    # 선형 램프 블록
    ramp = np.zeros((4, 4, 4), np.uint8)
    ramp[..., 0] = (np.arange(16) * 16).reshape(4, 4)
    ramp[..., 1] = 100
    ramp[..., 2] = 255 - (np.arange(16) * 8).reshape(4, 4)
    ramp[..., 3] = 255
    assert psnr8(T.decode_bc7(T.encode_bc7(ramp), 4, 4)[..., :3], ramp[..., :3]) >= 40.0
    for bad in (img.astype(np.float32), img[:30], img[..., :3]):
        with pytest.raises(ValueError):
            T.encode_bc7(bad)


def test_bc5_roundtrip_and_oracle():
    img = np.rint(smooth_image(32, 16, 2, 4)).astype(np.uint8)
    data = T.encode_bc5(img)
    assert len(data) == 32 * 16 // 16 * 16
    dec = T.decode_bc5(data, 16, 32)
    assert dec.dtype == np.float32 and dec.shape == (32, 16, 2)
    np.testing.assert_allclose(dec, bc5_decode_image(data, 16, 32), atol=1e-6)
    assert psnr8(dec * 255.0, img) >= 38.0
    const = np.full((4, 4, 2), [37, 201], np.uint8)
    np.testing.assert_allclose(T.decode_bc5(T.encode_bc5(const), 4, 4) * 255.0, const, atol=0.5)
    # 오라클 팔레트 두 갈래(r0>r1, r0<=r1) 모두 디코드 일치
    rng = np.random.default_rng(5)
    raw = bytearray(rng.integers(0, 256, 4 * 16, dtype=np.uint8).tobytes())
    raw[0], raw[1] = 200, 10
    raw[8], raw[9] = 10, 200
    np.testing.assert_allclose(T.decode_bc5(bytes(raw), 8, 8), bc5_decode_image(bytes(raw), 8, 8), atol=1e-6)


def _tiles(N, C, seed, lo, hi):
    """64×64 부드러운 캔버스에서 8×8 타일 N개 (N ≤ 64)."""
    can = smooth_image(64, 64, C, seed, lo, hi)
    return np.stack([can[8 * (k // 8):8 * (k // 8) + 8, 8 * (k % 8):8 * (k % 8) + 8] for k in range(N)])


def _hdr_pages(N, seed):
    rng = np.random.default_rng(seed)
    sm = _tiles(N, 3, seed, 0.0, 1.0)
    a = rng.uniform(-8, 2, (N, 1, 1, 1))
    b = rng.uniform(1, 6, (N, 1, 1, 1))
    return np.exp2(a + b * sm)


def _codes_formula(pages, lo, hi):
    y = np.log2(np.maximum(pages, T.LMIN))
    lo, hi = lo.astype(np.float64)[:, None, None, None], hi.astype(np.float64)[:, None, None, None]
    return y, lo, hi


def test_page_codec_layout_formula_and_error_bounds():
    N = 6
    pages = _hdr_pages(N, 7)
    pages[4] = 0.37                                  # 상수 페이지
    pages[5, 0, 0] = 0.0                             # 0 텍셀 (LMIN 으로)
    blob = T.compress_pages(pages)
    assert blob.n == N and blob.bc5 is None and len(blob.bc7) == 64 * N and blob.nbytes == 64 * N
    y = np.log2(np.maximum(pages, T.LMIN)).reshape(N, -1)
    lo = y.min(1).astype(np.float32)
    hi = np.maximum(y.max(1), lo.astype(np.float64) + 1e-3).astype(np.float32)
    assert blob.lo.dtype == np.float32 and np.array_equal(blob.lo, lo) and np.array_equal(blob.hi, hi)
    dec = T.decompress_pages(blob)
    assert dec.dtype == np.float32 and dec.shape == (N, 8, 8, 3)
    # 복원 = 2^(lo + c/255 (hi-lo)), c = BC7 (오라클) 디코드 코드. 페이지 p = 세로 쌓기 블록 4개
    codes = bc7_decode_image(blob.bc7, 8, 8 * N)[..., :3].reshape(N, 8, 8, 3).astype(np.float64)
    L64, H64 = lo.astype(np.float64)[:, None, None, None], hi.astype(np.float64)[:, None, None, None]
    np.testing.assert_allclose(dec, np.exp2(L64 + codes / 255.0 * (H64 - L64)), rtol=2e-6)
    # 오차 한계 (log2 영역, 코드 단위)
    yy = np.log2(np.maximum(pages, T.LMIN))
    err = np.abs(np.log2(dec.astype(np.float64)) - yy) / (H64 - L64) * 255.0
    assert err[:4].mean() <= 4.5 and err[:4].max() <= 28.0, (err[:4].mean(), err[:4].max())
    np.testing.assert_allclose(dec[4], pages[4], rtol=1e-6)
    assert np.isfinite(dec).all() and dec.min() > 0


def test_page_codec_with_normals_is_128_bytes():
    N = 4
    pages = _hdr_pages(N, 9)
    nh = _tiles(N, 2, 11, -1.0, 1.0)
    blob = T.compress_pages(pages, nh)
    assert len(blob.bc5) == 64 * N and blob.nbytes == 128 * N
    d = T.decompress_nh(blob)
    assert d.dtype == np.float32 and d.shape == (N, 8, 8, 2)
    ref = 2.0 * bc5_decode_image(blob.bc5, 8, 8 * N).reshape(N, 8, 8, 2) - 1.0
    np.testing.assert_allclose(d, ref, atol=1e-6)
    assert np.abs(d - nh).max() <= 16.0 / 255.0 and np.abs(d - nh).mean() <= 4.0 / 255.0
    assert T.decompress_nh(T.compress_pages(pages)) is None
    with pytest.raises(ValueError):
        T.compress_pages(np.full((2, 8, 8, 3), np.nan))
    with pytest.raises(ValueError):
        T.compress_pages(np.zeros((2, 8, 4, 3)))
    with pytest.raises(ValueError):
        T.compress_pages(pages, nh[:, :, :, :1])
    # torch 입력 허용
    b2 = T.compress_pages(torch.tensor(pages))
    assert b2.bc7 == T.compress_pages(pages).bc7


def test_render_from_compressed_pages():
    W, H = 48, 36
    K = K_of(42.0, W, H)
    sc = random_scene(10, 13, edge_lo=2.0, edge_hi=10.0)
    sc.pages = _tiles(10, 3, 14, 0.05, 1.0)
    t = make_tsq(sc, torch.float32)
    ref = to_np(T.render(t, c2w=np.eye(4), K=K, width=W, height=H).rgb)
    blob = T.compress_pages(t.pages)
    t2 = t.clone()
    t2.pages = torch.tensor(T.decompress_pages(blob))
    out = to_np(T.render(t2, c2w=np.eye(4), K=K, width=W, height=H).rgb)
    mse = np.mean((out - ref) ** 2)
    assert 10 * np.log10(1.0 / max(mse, 1e-20)) >= 35.0

"""RDR-04 C_0 정합 렌더 필터: 호모그래피 야코비안, 항등 루프(첫 프레임 바닥), 반화소 MTF, 왕복 오차,
유효 마스크, 축소 앨리어싱, 비등방 바닥면, RDR-02(quads) 와의 이음매, 속도."""
import math
import time

import numpy as np
import pytest

from buc.core import contracts as C
from buc.core import geom
from buc.render.filters import EwaOpts, homography_uv_jacobian, interp_mtf, warp_homography
from buc.render.quads import render_ldi
from rdr04_helpers import (N, band_image, cos_x, cos_y, fit_amp, freq, interior_rmse, rms_amp, rot_h, smooth_tex,
                           translate_h)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


# --------------------------------------------------------------------------------------------- 야코비안
def test_homography_jacobian_matches_finite_differences():
    H = np.array([[0.9, 0.2, 5.0], [-0.1, 1.1, -3.0], [4e-4, -2e-4, 1.0]])
    xy = np.stack(np.meshgrid(np.arange(0, 60, 7.0), np.arange(0, 40, 9.0)), axis=-1)
    uv, J = homography_uv_jacobian(H, xy)
    assert uv.shape == xy.shape and J.shape == xy.shape[:-1] + (2, 2)
    h = 1e-4
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    uv_x = (homography_uv_jacobian(H, xy + ex)[0] - homography_uv_jacobian(H, xy - ex)[0]) / (2 * h)
    uv_y = (homography_uv_jacobian(H, xy + ey)[0] - homography_uv_jacobian(H, xy - ey)[0]) / (2 * h)
    assert np.abs(J[..., :, 0] - uv_x).max() < 1e-6                 # J[..., i, 0] = d uv_i / d x
    assert np.abs(J[..., :, 1] - uv_y).max() < 1e-6                 # J[..., i, 1] = d uv_i / d y
    # uv 자체는 동차 좌표 정의와 같다
    p = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1) @ H.T
    assert np.abs(uv - p[..., :2] / p[..., 2:3]).max() < 1e-9


def test_homography_jacobian_identity_affine_and_behind():
    xy = np.array([[3.0, 4.0], [10.5, -2.0]])
    uv, J = homography_uv_jacobian(np.eye(3), xy)
    assert np.allclose(uv, xy) and np.allclose(J, np.eye(2))
    A = np.array([[2.0, 0.5, 1.0], [-0.3, 1.5, 2.0], [0.0, 0.0, 1.0]])
    uv, J = homography_uv_jacobian(A, xy)
    assert np.allclose(J, A[:2, :2]) and np.allclose(uv, xy @ A[:2, :2].T + A[:2, 2])
    # 분모 c <= 0 (카메라 뒤) 점은 NaN, 나머지는 영향 없음, 경고 없음
    Hp = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.02, 0, 1.0]])       # c = 1 - 0.02 x
    xy2 = np.array([[10.0, 5.0], [49.0, 5.0], [50.0, 5.0], [80.0, 5.0]])
    uv, J = homography_uv_jacobian(Hp, xy2)
    assert np.isfinite(uv[:2]).all() and np.isfinite(J[:2]).all()
    assert np.isnan(uv[2:]).all() and np.isnan(J[2:]).all()


# --------------------------------------------------------------------------------------------- 항등 루프
@pytest.mark.parametrize("mode", ["bspline", "ewa"])
def test_identity_warp_is_exact(mode):
    """X-FF0 의 소유·필터 사슬 바닥: 항등 카메라의 렌더 필터는 입력을 비트 가까이 되돌려 준다."""
    img = band_image(96)
    r = warp_homography(img, np.eye(3), (96, 96), mode=mode)
    assert r.valid.all()
    assert np.abs(r.image - img).max() < 1e-5
    # 비정사각, 3 채널 미만/초과
    img2 = np.random.default_rng(4).random((37, 53, 5)).astype(np.float32)
    r2 = warp_homography(img2, np.eye(3), (37, 53), mode=mode)
    assert np.abs(r2.image - img2).max() < 1e-5


def test_ewa_identity_is_better_than_bilinear_subpixel_chain():
    """항등이 정확하다는 것만으로는 쓸모가 없다: 부화소 이동 왕복에서 EWA/카디널이 쌍선형보다 훨씬 덜 망가진다."""
    img = band_image(96)
    err = {}
    for name, H in [("shift", translate_h(0.5, 0.5)), ("rot3", rot_h(3, 47.5, 47.5)),
                    ("rot1_s1.02", rot_h(1, 47.5, 47.5, 1.02)), ("rot10", rot_h(10, 47.5, 47.5))]:
        for mode in ("bilinear", "bspline", "ewa"):
            f = warp_homography(img, H, (96, 96), mode=mode).image
            b = warp_homography(f, np.linalg.inv(H), (96, 96), mode=mode).image
            err[(name, mode)] = interior_rmse(b, img)
        assert err[(name, "ewa")] < 0.004, (name, err)                     # 상대 2.7% 이내 (대역 제한 영상, σ=0.15)
        assert err[(name, "bspline")] < 0.004, (name, err)
        assert err[(name, "ewa")] < 0.2 * err[(name, "bilinear")], (name, err)
        assert err[(name, "bilinear")] > 0.012, (name, err)               # 대조군이 실제로 나쁘다(시험의 변별력)


# --------------------------------------------------------------------------------------------- 반화소 MTF
@pytest.mark.parametrize("m,g_min,mode", [(26, 0.995, "bspline"), (26, 0.995, "ewa"), (52, 0.98, "ewa"),
                                          (77, 0.91, "ewa"), (90, 0.82, "ewa"), (90, 0.82, "bspline")])
def test_half_pixel_shift_gain(m, g_min, mode):
    f = freq(m)
    r = warp_homography(cos_x(m), translate_h(0.5, 0.0), (8, N), mode=mode).image
    xs = np.arange(N, dtype=np.float64)[8:-9]
    amp = fit_amp(r[4, 8:-9], xs + 0.5, f)
    assert amp >= g_min
    assert amp <= 1.02


def test_half_pixel_shift_beats_bilinear_at_035_cycles():
    m = 90                                                                  # 0.3516 cy/px
    xs = np.arange(N, dtype=np.float64)[8:-9]
    f = freq(m)
    a_ewa = fit_amp(warp_homography(cos_x(m), translate_h(0.5, 0), (8, N), mode="ewa").image[4, 8:-9], xs + 0.5, f)
    a_bil = fit_amp(warp_homography(cos_x(m), translate_h(0.5, 0), (8, N), mode="bilinear").image[4, 8:-9], xs + 0.5, f)
    assert a_bil < 0.5
    assert a_ewa > 1.6 * a_bil
    # 해석 MTF 와 같은 방향: 사전필터 3차가 쌍선형보다 높다
    assert interp_mtf(f, 3) > interp_mtf(f, 1)


def test_mtf_gate_numbers_are_consistent_with_x_ff0():
    """X-FF0 게이트의 MTF 비 >= 0.9 @0.35 cy/px: 카디널 3차의 연속 응답(0.916)이 이를 만족한다."""
    assert interp_mtf(0.35, 3) >= 0.9


# --------------------------------------------------------------------------------------------- 유효 마스크 / 채움
def test_valid_mask_and_fill():
    img = band_image(40, ch=3)
    r = warp_homography(img, translate_h(10.0, 0.0), (40, 40), mode="ewa", fill=0.25)
    # dst x -> src x + 10, 원본 영역 [-0.5, 39.5): x + 10 < 39.5  <=>  x <= 29
    assert r.valid[:, :30].all() and not r.valid[:, 30:].any()
    assert np.all(r.image[:, 30:] == np.float32(0.25))
    assert np.isfinite(r.image).all()
    r2 = warp_homography(img, translate_h(-10.0, 3.0), (40, 40), mode="bspline")
    # dst (x, y) -> src (x - 10, y + 3): x - 10 >= -0.5 <=> x >= 10, y + 3 < 39.5 <=> y <= 36
    expect = np.zeros((40, 40), bool)
    expect[:37, 10:] = True
    assert np.array_equal(r2.valid, expect)
    assert np.all(r2.image[~expect] == 0.0)                                    # 기본 fill = 0


def test_perspective_behind_camera_is_invalid_not_nan():
    img = band_image(64)
    Hp = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.02, 0, 1.0]])               # x >= 50 에서 분모 <= 0
    for mode in ("bilinear", "bspline", "ewa"):
        r = warp_homography(img, Hp, (64, 64), mode=mode)
        assert not r.valid[:, 50:].any()
        assert np.isfinite(r.image).all()
        assert r.valid[:10, :10].all()


def test_constant_image_is_constant_on_valid_pixels():
    img = np.full((60, 80, 3), 0.6, np.float32)
    H = np.array([[0.6, 0.1, 3.0], [-0.1, 0.7, 4.0], [5e-4, 3e-4, 1.0]])
    r = warp_homography(img, H, (50, 70), mode="ewa")
    assert r.valid.sum() > 500
    assert np.abs(r.image[r.valid] - 0.6).max() < 1e-5


def test_warp_linear_in_image_and_channels_independent():
    a, b = band_image(48, seed=5), band_image(48, seed=6)
    H = np.array([[0.7, 0.2, 2.0], [-0.1, 0.8, 1.0], [2e-4, 1e-4, 1.0]])
    wa = warp_homography(a, H, (40, 40)).image
    wb = warp_homography(b, H, (40, 40)).image
    wab = warp_homography(0.4 * a + 0.9 * b, H, (40, 40)).image
    assert np.abs(wab - (0.4 * wa + 0.9 * wb)).max() < 2e-5
    w1 = warp_homography(a[..., 1], H, (40, 40)).image
    assert np.abs(w1 - wa[..., 1]).max() < 1e-6


# --------------------------------------------------------------------------------------------- 축소·비등방
def test_two_times_minification_passband_and_alias():
    Hd = np.array([[2.0, 0, 0.5], [0, 2.0, 0.0], [0, 0, 1.0]])               # src = 2 dst + 0.5 (화소 중심 정렬)
    g_pass = rms_amp(warp_homography(cos_x(13), Hd, (4, 60), mode="ewa").image[2, 5:-5])
    assert g_pass >= 0.93                                                      # f_dst = 0.1 cy/px: 거의 그대로
    g_ewa = rms_amp(warp_homography(cos_x(90), Hd, (4, 60), mode="ewa").image[2, 5:-5])
    g_bil = rms_amp(warp_homography(cos_x(90), Hd, (4, 60), mode="bilinear").image[2, 5:-5])
    assert g_ewa < 0.25                                                        # f_in=0.35 -> 앨리어싱 구간
    assert g_ewa < 0.55 * g_bil
    # 직류는 보존
    flat = warp_homography(np.full((120, 120), 0.7, np.float32), Hd, (50, 50), mode="ewa")
    assert np.abs(flat.image[flat.valid] - 0.7).max() < 1e-5


def test_grazing_plane_keeps_horizontal_detail_and_kills_vertical_alias():
    """y 로만 8배 축소(바닥면 먼 쪽): 가로 줄무늬 해상도를 유지하고(비등방), 세로 앨리어싱은 제거한다."""
    Hg = np.array([[1.0, 0, 0], [0, 8.0, 3.5], [0, 0, 1.0]])
    kept = warp_homography(cos_x(64, rows=N), Hg, (14, N), mode="ewa").image
    assert 0.97 <= rms_amp(kept[3:11, 10:-10]) <= 1.03
    iso = warp_homography(cos_x(64, rows=N), Hg, (14, N), mode="ewa", opts=EwaOpts(max_aniso=1.0)).image
    assert rms_amp(iso[3:11, 10:-10]) < 0.05
    alias_ewa = warp_homography(cos_y(115), Hg, (14, N), mode="ewa").image
    alias_bil = warp_homography(cos_y(115), Hg, (14, N), mode="bilinear").image
    assert np.abs(alias_ewa[3:11, 10:-10]).max() < 0.02
    assert np.abs(alias_bil[3:11, 10:-10]).max() > 0.08                       # 쌍선형은 앨리어싱이 남는다(변별력)


# --------------------------------------------------------------------------------------------- RDR-02 이음매
def _card_world(tex_rgb, depth=3.0):
    h, w = tex_rgb.shape[:2]
    K = geom.intrinsics_from_fov(60.0, w, h)
    obs = C.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=w, height=h, k1=0.0, k2=0.0)
    rgba = np.concatenate([tex_rgb, np.ones((h, w, 1), np.float32)], axis=-1)
    layer = C.LDILayer(rgba=rgba, depth=np.full((h, w), depth, np.float32))
    return C.World(obs=obs, layers=(layer,), theta_in=C.Theta(), level="L1", read_hash="r" * 8, prov=C.Prov()), K


def _dst2src(K, c2w_t, depth):
    rel = geom.invert_pose(c2w_t)                                              # 원본 카메라 = 세계 원점
    Hm = geom.plane_homography(K, K, rel[:3, :3], rel[:3, 3], np.array([0.0, 0.0, 1.0]), depth)
    Hi = np.linalg.inv(Hm)
    return Hi / Hi[2, 2]


@pytest.mark.parametrize("name,dx,dz,yaw_deg", [("identity", 0.0, 0.0, 0.0), ("yaw_shift", 0.1, -0.3, 8.0),
                                                ("pull_back", 0.0, -1.5, 0.0)])
def test_matches_rdr02_plane_render_and_plane_oracle(name, dx, dz, yaw_deg):
    tex = smooth_tex(96, 64)
    world, K = _card_world(tex)
    a = math.radians(yaw_deg)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    c2w[:3, 3] = [dx, 0.0, dz]
    q = render_ldi(world, c2w=c2w, K=K, width=96, height=64)
    Hi = _dst2src(K, c2w, 3.0)
    r = warp_homography(tex, Hi, (64, 96), mode="ewa")
    # 오라클: 같은 호모그래피를 쌍선형으로 (텍스처가 저주파라 필터 차이가 거의 없다)
    ob = warp_homography(tex, Hi, (64, 96), mode="bilinear")
    m = r.valid & ob.valid & (q.alpha > 0.98)
    # 안쪽 3 px 이상만 비교 (RDR-02 의 소프트 에지·거울 경계 영향 제외)
    uv, _ = homography_uv_jacobian(Hi, np.stack(np.meshgrid(np.arange(96.0), np.arange(64.0)), axis=-1))
    inner = (uv[..., 0] >= 3) & (uv[..., 0] <= 92) & (uv[..., 1] >= 3) & (uv[..., 1] <= 60)
    m = m & inner
    assert m.sum() > 1500
    assert np.abs(r.image - ob.image)[m].mean() < 0.005
    assert np.abs(r.image - q.rgb)[m].mean() < 0.01
    if name == "identity":
        assert np.abs(r.image - q.rgb)[m].max() < 1e-5


# --------------------------------------------------------------------------------------------- 속도
def test_speed_budgets():
    img = np.random.default_rng(0).random((120, 160, 3)).astype(np.float32)
    t0 = time.perf_counter()
    r = warp_homography(img, np.diag([2.0, 2.0, 1.0]), (60, 80), mode="ewa")
    dt_min = time.perf_counter() - t0
    assert r.valid.all() and dt_min < 15.0
    big = np.random.default_rng(1).random((256, 256, 3)).astype(np.float32)
    t0 = time.perf_counter()
    warp_homography(big, np.eye(3), (256, 256), mode="ewa")
    assert time.perf_counter() - t0 < 5.0                                       # 항등/확대(탭 1개)는 빨라야 한다

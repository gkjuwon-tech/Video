"""FND-08 기하 정답: 장면별 독립 해석 렌더(깊이·법선·인스턴스·색)와의 대조와 알려진 값."""
import math

import numpy as np
import pytest

from buc.core import geom

import fnd08_support as S

NAMES = S.NAMES
VIEWS = [(n, i) for n in NAMES for i in range(2 if n == "two_view_parallax" else 1)]


@pytest.mark.parametrize("name,vi", VIEWS)
def test_depth_normals_instances_match_analytic_render(name, vi):
    v = S.built()[name].views[vi]
    e = S.expected(name, vi)
    bad_inst = int((v.instance != e["inst"]).sum())
    assert bad_inst <= 2, f"인스턴스 불일치 {bad_inst}"
    same = v.instance == e["inst"]
    hit = e["hit"] & same
    assert hit.sum() > 0.3 * hit.size
    # 깊이: float32 반올림 한도
    np.testing.assert_allclose(v.depth[hit].astype(np.float64), e["depth"][hit], rtol=2e-6, atol=0)
    assert np.isinf(v.depth[~e["hit"] & same]).all()
    # 법선(카메라 좌표) = R n_w
    nc = e["nw"] @ v.obs.R.T
    np.testing.assert_allclose(v.normals[hit].astype(np.float64), nc[hit], rtol=0, atol=2e-6)


@pytest.mark.parametrize("name,vi", VIEWS)
def test_image_matches_analytic_shading(name, vi):
    v = S.built()[name].views[vi]
    e = S.expected(name, vi)
    same = v.instance == e["inst"]
    err = np.abs(v.image_lin.astype(np.float64) - e["img"]).max(axis=-1)
    bad = (err > 1e-5) | ~same
    # 체커 경계에서 부동소수 한 번 차이로 뒤집히는 화소만 허용(0.5 %)
    assert bad.mean() <= 0.005, f"{name}: 색 불일치 {bad.sum()} 화소 (최대 {err.max():.3g})"
    # 하늘은 정확히 SKY_LIN
    sky = e["inst"] == 0
    if sky.any():
        np.testing.assert_allclose(v.image_lin[sky & same].astype(np.float64), np.broadcast_to(S.SKY, (int((sky & same).sum()), 3)), atol=1e-7)


def test_plane_frontal_known_values():
    v = S.built()["plane_frontal"].views[0]
    assert (v.depth == np.float32(4.0)).all()
    assert (v.instance == 1).all()
    assert (v.normals == np.float32([0, 0, -1])).all()
    # 체커: 서로 다른 값 정확히 2개, 세계 0.25 m 간격 -> 화소 폭 fx*0.25/4
    cols = {tuple(c) for c in v.image_lin.reshape(-1, 3).tolist()}
    assert cols == {(np.float32(0.8).item(),) * 3, (np.float32(0.1).item(),) * 3}
    fx = S.intr(60.0)[0, 0]
    row = v.image_lin[32, :, 0] > 0.5
    edges = np.flatnonzero(np.diff(row.astype(int)) != 0)
    widths = np.diff(edges)
    np.testing.assert_allclose(widths, fx * 0.25 / 4.0, atol=1.01)
    # 격자는 세계 원점(영상 중심)에 맞춰져 있다: 중심에서 가장 가까운 경계는 u=47.5 (x=0)
    assert np.abs(edges + 0.5 - 47.5).min() <= 0.5


def test_plane_tilted_known_values():
    sc = S.built()["plane_tilted"]
    v = sc.views[0]
    n = np.array([0.3, -0.2, -1.0])
    n /= np.linalg.norm(n)
    np.testing.assert_allclose(sc.gt["plane_n"], [n], atol=1e-15)
    np.testing.assert_allclose(sc.gt["plane_d"], [n @ [0, 0, 4.0]], atol=1e-15)
    # 법선은 장면 전체에서 상수 = R n = n (항등 카메라)
    np.testing.assert_allclose(v.normals.reshape(-1, 3), np.broadcast_to(n, (96 * 64, 3)), atol=2e-7)
    # 중심 화소 근처 깊이 ~ 4/(1 - 0.3x + 0.2y)
    K = S.intr(60.0)
    for (u, vv) in [(0, 0), (95, 63), (47, 31), (10, 50)]:
        x, y = (u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]
        assert float(v.depth[vv, u]) == pytest.approx(4.0 / (1 - 0.3 * x + 0.2 * y), rel=2e-6)
    assert 3.2 < v.depth.min() < 3.3 and 5.3 < v.depth.max() < 5.4


def test_ground_horizon_known_values():
    sc = S.built()["ground_horizon"]
    v = sc.views[0]
    fx = S.intr(70.0)[0, 0]
    # 주점 행 31.5 -> 행 0..31 하늘, 32..63 바닥
    assert np.isinf(v.depth[:32]).all() and np.isfinite(v.depth[32:]).all()
    assert (v.instance[:32] == 0).all() and (v.instance[32:] == 1).all()
    for r in (32, 40, 63):
        np.testing.assert_allclose(v.depth[r].astype(np.float64), 1.5 * fx / (r - 31.5), rtol=2e-6)
    # 깊이는 행만의 함수이고 아래로 갈수록 감소
    assert (v.depth[32:] == v.depth[32:, :1]).all() and (np.diff(v.depth[32:, 0]) < 0).all()
    np.testing.assert_allclose(sc.gt["plane_n"], [[0, -1, 0]], atol=0)
    np.testing.assert_allclose(sc.gt["plane_d"], [-1.5], atol=0)


def test_sphere_known_values():
    sc = S.built()["sphere_lit"]
    v = sc.views[0]
    np.testing.assert_allclose(sc.gt["sphere_c"], [0.1, -0.05, 3.0], atol=0)
    assert sc.gt["sphere_r"] == 0.9
    m = v.instance == 1
    assert 2.1 <= float(v.depth[m].min()) <= 2.105
    # 닫힌 형태: 법선(카메라)·점이 구면 위. 점 = 역투영, ||X - c|| = r
    uv = geom.pixel_grid(64, 96)[m]
    X = geom.unproject(v.obs.K, uv, v.depth[m].astype(np.float64))
    np.testing.assert_allclose(np.linalg.norm(X - np.array([0.1, -0.05, 3.0]), axis=1), 0.9, atol=2e-5)
    np.testing.assert_allclose(v.normals[m].astype(np.float64), (X - [0.1, -0.05, 3.0]) / 0.9, atol=3e-5)
    # 실루엣: 정확한 판별식 경계 -> 면적은 해석적 투영 타원 근처
    assert 3200 < m.sum() < 3320


def test_box_pair_known_values():
    sc = S.built()["box_pair"]
    v = sc.views[0]
    np.testing.assert_allclose(sc.gt["box_min"], [[-0.9, -0.4, 2.5], [0.3, 0.0, 3.5]], atol=0)
    np.testing.assert_allclose(sc.gt["box_max"], [[-0.1, 0.6, 3.3], [1.3, 0.8, 4.5]], atol=0)
    np.testing.assert_allclose(sc.gt["plane_n"], [[0, 0, -1]], atol=0)
    np.testing.assert_allclose(sc.gt["plane_d"], [-6.0], atol=0)
    ids = {i: int((v.instance == i).sum()) for i in (1, 2, 3)}
    assert ids[2] > 400 and ids[3] > 250 and (v.instance == 0).sum() == 0
    # 상자 면 법선은 축 정렬: 세계 법선 = R^T n_c 의 한 성분이 +-1
    for i in (2, 3):
        nw = v.normals[v.instance == i].astype(np.float64) @ v.obs.R
        assert (np.abs(nw).max(axis=1) > 1 - 2e-6).all()
    # 상자 1이 상자 2보다 가깝다, 가림 경계가 존재(깊이 점프)
    assert float(v.depth[v.instance == 2].max()) < 4.0
    assert (np.abs(np.diff(v.depth, axis=1)) > 1.0).any()


def test_corridor_known_values():
    sc = S.built()["corridor"]
    v = sc.views[0]
    np.testing.assert_allclose(sc.gt["plane_n"], [[0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, -1]], atol=0)
    np.testing.assert_allclose(sc.gt["plane_d"], [-1, -1, -1.5, -1.5, -10], atol=0)
    assert set(np.unique(v.instance).tolist()) == {1, 2, 3, 4, 5}
    # 끝벽의 z-깊이는 어느 화소에서나 정확히 10
    assert (v.depth[v.instance == 5] == np.float32(10.0)).all()
    # 좌우·상하 대칭(주점이 영상 중심): 좌우 반전하면 id 3<->4, 상하 반전하면 1<->2
    lr = v.instance[:, ::-1]
    swap = np.where(lr == 3, 4, np.where(lr == 4, 3, lr))
    assert (swap != v.instance).sum() <= 2
    ud = v.instance[::-1]
    swap = np.where(ud == 1, 2, np.where(ud == 2, 1, ud))
    assert (swap != v.instance).sum() <= 2
    np.testing.assert_allclose(v.depth, v.depth[:, ::-1], rtol=1e-6)
    # 소실점: 바닥 깊이는 중심 행으로 갈수록 커진다
    col = v.depth[:, 48].astype(np.float64)
    fl = v.instance[:, 48] == 1
    assert (np.diff(col[fl]) < 0).all()


def test_two_layer_card_known_values_and_ldi():
    from buc.core import contracts
    sc = S.built()["two_layer_card"]
    v = sc.views[0]
    np.testing.assert_allclose(sc.gt["card_min"], [-0.6, -0.4, 2.0], atol=0)
    np.testing.assert_allclose(sc.gt["card_max"], [0.6, 0.4, 2.0], atol=0)
    K = S.intr(60.0)
    # 카드 화소 = 중심 화소가 투영된 사각형 안(해석적): u in [fx*-0.6/2+cx, ...]
    u0, u1 = K[0, 0] * -0.6 / 2.0 + K[0, 2], K[0, 0] * 0.6 / 2.0 + K[0, 2]
    v0, v1 = K[1, 1] * -0.4 / 2.0 + K[1, 2], K[1, 1] * 0.4 / 2.0 + K[1, 2]
    vv, uu = np.mgrid[0:64, 0:96]
    card = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
    assert int((card != (v.instance == 2)).sum()) <= 1
    assert (v.depth[v.instance == 2] == np.float32(2.0)).all()
    # LDI 정답
    rgba, dep = sc.gt["ldi_rgba"], sc.gt["ldi_depth"]
    assert rgba.dtype == np.float32 and rgba.shape == (2, 64, 96, 4) and dep.dtype == np.float32 and dep.shape == (2, 64, 96)
    c = v.instance == 2
    assert (rgba[0][..., 3] == c.astype(np.float32)).all() and (rgba[1][..., 3] == 1.0).all()
    assert (rgba[0][~c] == 0).all()
    np.testing.assert_array_equal(rgba[0][c][:, :3], v.image_lin[c])
    np.testing.assert_array_equal(dep[0][c], v.depth[c])
    np.testing.assert_array_equal(dep[0][~c], dep[1][~c])
    assert (dep[1] == np.float32(5.0)).all()          # 가려진 배경까지 z=5 평면 전체
    # 층1은 가려진 곳까지 칠한 체커: 보이는 배경 화소와 정확히 같고, 카드 뒤에도 값이 있다
    np.testing.assert_array_equal(rgba[1][~c][:, :3], v.image_lin[~c])
    _, dwd, _ = S.rays("two_layer_card")
    P = dwd * 5.0
    par = (np.floor(P[..., 0] / 0.25) + np.floor(P[..., 1] / 0.25)) % 2
    exp1 = np.where((par == 0)[..., None], S.CHECK_A, S.CHECK_B) * (0.2 + 0.8 * -S.LIGHT[2])
    bad = (np.abs(rgba[1][..., :3].astype(np.float64) - exp1).max(-1) > 1e-5)
    assert bad[c].mean() <= 0.01 and bad.mean() <= 0.005
    # 합성(over) = 장면 영상
    comp = rgba[0][..., :3] * rgba[0][..., 3:4] + rgba[1][..., :3] * (1 - rgba[0][..., 3:4])
    np.testing.assert_allclose(comp, v.image_lin, atol=1e-6)
    # FND-02 계약으로 구성된다(층 정렬: 둘 다 불투명인 화소에서 앞 <= 뒤)
    L = tuple(contracts.LDILayer(rgba=rgba[i], depth=dep[i]) for i in range(2))
    rp = S.tool().to_read_packet(sc)
    w = contracts.World(obs=v.obs, layers=L, theta_in=rp.theta_in, level="L1", read_hash=rp.content_hash(), prov=rp.prov)
    w.validate()


def test_two_view_geometry_and_plane_transfer():
    from buc.core import pyramid
    sc = S.built()["two_view_parallax"]
    v0, v1 = sc.views
    o0, o1 = v0.obs, v1.obs
    np.testing.assert_allclose(sc.gt["T_01"], o1.w2c() @ o0.c2w(), atol=1e-12)
    T = sc.gt["T_01"]
    assert T.dtype == np.float64 and T.shape == (4, 4)
    assert geom.rotation_angle_deg(o0.R, o1.R) == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_allclose(o1.center(), [0.3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(o0.w2c(), np.eye(4), atol=1e-15)
    np.testing.assert_allclose(sc.gt["plane_n"], [[0, 0, -1]], atol=0)
    np.testing.assert_allclose(sc.gt["plane_d"], [-6.0], atol=0)
    # 벽 화소 전이: 뷰0 깊이 -> 세계점 -> 뷰1 투영. 평면의 역깊이는 화소에서 아핀이므로 뷰1 역깊이를 쌍선형 표본해도 정확
    wall0 = v0.instance == 1
    vv, uu = np.mgrid[0:64, 0:96]
    uv0 = np.stack([uu, vv], -1).astype(np.float64)
    X = geom.unproject_world(o0.K, o0.R, o0.t, uv0, v0.depth.astype(np.float64))
    uv1, z1 = o1.project(X.reshape(-1, 3))
    uv1, z1 = uv1.reshape(64, 96, 2), z1.reshape(64, 96)
    inside = (uv1[..., 0] >= 1) & (uv1[..., 0] <= 94) & (uv1[..., 1] >= 1) & (uv1[..., 1] <= 62)
    # 뷰1에서 2x2 이웃이 모두 벽인 곳만
    ok = np.zeros((64, 96), bool)
    idx = np.argwhere(wall0 & inside)
    inv1 = (1.0 / v1.depth.astype(np.float64))
    cnt = 0
    for (r, c) in idx:
        u, w = uv1[r, c]
        u0_, w0_ = int(np.floor(u)), int(np.floor(w))
        if (v1.instance[w0_:w0_ + 2, u0_:u0_ + 2] != 1).any():
            continue
        s = pyramid.sample_bilinear(inv1, np.array([[u, w]]))[0]
        assert s == pytest.approx(1.0 / z1[r, c], rel=1e-5)
        cnt += 1
    assert cnt > 1500
    # 평면 호모그래피(FND-06)와 깊이 전이가 같은 화소를 준다
    R = T[:3, :3]
    t = T[:3, 3]
    H = geom.plane_homography(o0.K, o1.K, R, t, np.array([0.0, 0.0, -1.0]), -6.0)
    p = geom.apply_homography(H, uv0[wall0])
    np.testing.assert_allclose(p, uv1[wall0], atol=1e-3)


def test_distortion_dots_ground_truth():
    from scipy import ndimage
    sc = S.built()["distortion_dots"]
    v = sc.views[0]
    o = v.obs
    assert (o.k1, o.k2) == (-0.2, 0.05)
    dw = S.dots_world()
    np.testing.assert_allclose(sc.gt["dots_world"], dw, atol=1e-15)
    uv, z = o.project(dw)
    np.testing.assert_allclose(sc.gt["dots_uv"], uv, atol=1e-9)
    assert sc.gt["dot_radius"] == 0.1
    assert (v.depth == np.float32(3.0)).all()
    # 왜곡이 실제로 의미 있다: 핀홀 투영과 가장자리에서 1 px 넘게 다르다
    uv_lin, _ = geom.project(o.K, o.R, o.t, dw, 0.0, 0.0)
    assert np.abs(uv - uv_lin).max() > 1.0
    # 렌더된 점의 무게중심이 정답 투영과 0.35 px 안에 있다 (왜곡된 광선 역투영의 일관성)
    lab, n = ndimage.label(v.image_lin[..., 0] > 0.5)
    assert n == 35
    cent = np.array(ndimage.center_of_mass(v.image_lin[..., 0] > 0.5, lab, range(1, n + 1)))[:, ::-1]   # (u,v)
    d = np.linalg.norm(cent[:, None, :] - uv[None, :, :], axis=-1)
    assert (d.argmin(axis=0) == np.arange(35)).all() or sorted(d.argmin(axis=1)) == list(range(35))
    assert d.min(axis=0).max() <= 0.35 and d.min(axis=1).max() <= 0.35
    # 인스턴스: 배경 1, 점 k = 2+k
    assert set(np.unique(v.instance).tolist()) == set([1] + [2 + k for k in range(35)])
    for k in (0, 17, 34):
        m = v.instance == 2 + k
        c = np.array([uu for uu in np.argwhere(m).mean(axis=0)[::-1]])
        assert np.linalg.norm(c - uv[k]) <= 0.35


def test_color_chart_ground_truth():
    from buc.core import color
    sc = S.built()["color_chart"]
    v = sc.views[0]
    np.testing.assert_array_equal(sc.gt["patch_codes"], S.CHART_CODES)
    assert sc.gt["patch_codes"].dtype == np.uint8 and sc.gt["chart_bg_code"] == 40
    lin = color.srgb8_to_linear(S.CHART_CODES)
    np.testing.assert_array_equal(sc.gt["patch_lin"], lin)
    assert sc.gt["patch_lin"].dtype == np.float32
    K = S.intr(60.0)
    Z = 2.5
    rects = []
    for r in range(4):
        for c in range(6):
            x0, x1, y0, y1 = -1.2 + 0.4 * c, -0.8 + 0.4 * c, -0.8 + 0.4 * r, -0.4 + 0.4 * r
            rects.append([K[0, 0] * x0 / Z + K[0, 2], K[0, 0] * x1 / Z + K[0, 2], K[1, 1] * y0 / Z + K[1, 2], K[1, 1] * y1 / Z + K[1, 2]])
    np.testing.assert_allclose(sc.gt["patch_rects"], np.array(rects), atol=1e-9)
    vv, uu = np.mgrid[0:64, 0:96]
    n_checked = 0
    for k, (u0, u1, v0, v1) in enumerate(sc.gt["patch_rects"]):
        inner = (uu > u0 + 1) & (uu < u1 - 1) & (vv > v0 + 1) & (vv < v1 - 1)
        assert inner.sum() > 100
        px = v.image_lin[inner]
        assert (px == lin[k]).all(), f"패치 {k}"
        assert (v.instance[inner] == k + 1).all()
        n_checked += int(inner.sum())
    assert n_checked > 2500
    # 차트 밖은 배경 코드 40
    out = (uu < 7) & (vv > 25)
    assert (v.image_lin[out] == color.srgb8_to_linear(np.array([40, 40, 40], np.uint8))).all() and (v.instance[out] == 25).all()
    # sRGB8 왕복: 입력 영상의 패치 내부 코드가 정확히 정의 코드이고, decode_to_linear가 정답 선형값을 되돌린다
    ii = S.tool().to_input_image(sc)
    for k, (u0, u1, v0, v1) in enumerate(sc.gt["patch_rects"]):
        inner = (uu > u0 + 1) & (uu < u1 - 1) & (vv > v0 + 1) & (vv < v1 - 1)
        assert (ii.pixels[inner] == S.CHART_CODES[k]).all()
        np.testing.assert_allclose(color.decode_to_linear(ii.pixels, "srgb")[inner], np.broadcast_to(lin[k], (int(inner.sum()), 3)), atol=1e-7)

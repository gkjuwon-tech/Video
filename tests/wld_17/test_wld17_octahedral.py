"""WLD-17: 팔면체 코덱·텍셀 입체각·이음매 없는 쌍선형 표본. 뒤통수(-z, 네 모서리)와 네 변에서 값이 이어져야 한다."""
import math

import numpy as np
import pytest

from buc.core import contracts as C
from buc.world import farfield as F
from wld17_helpers import dirs_near, rand_dirs

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
UV = {"+x": (1.0, 0.5), "-x": (0.0, 0.5), "+y": (0.5, 1.0), "-y": (0.5, 0.0), "+z": (0.5, 0.5)}


def lin_field(d):
    """방향의 선형 함수(매끄러운 구 위 함수): 맵 값 = f(texel 방향)."""
    d = np.asarray(d, dtype=np.float64)
    return np.stack([0.5 + 0.4 * d[..., 0], 0.5 + 0.4 * d[..., 1], 0.5 + 0.4 * d[..., 2]], axis=-1)


def lin_map(S):
    return lin_field(F.texel_dirs(S)).astype(np.float32)


def test_axes_land_on_documented_texture_coords():
    for k, uv in UV.items():
        got = np.asarray(F.oct_encode(np.array(AXES[k], dtype=np.float64)))
        assert got.shape == (2,)
        np.testing.assert_allclose(got, uv, atol=1e-12, err_msg=k)
    # -z 는 네 모서리 중 하나
    got = np.asarray(F.oct_encode(np.array([0.0, 0.0, -1.0])))
    assert any(np.allclose(got, c, atol=1e-12) for c in [(0, 0), (0, 1), (1, 0), (1, 1)])


def test_encode_decode_roundtrip_whole_sphere():
    d = np.concatenate([rand_dirs(20000, 1), np.array(list(AXES.values()), dtype=np.float64),
                        dirs_near((0, 0, -1), 5, 2000, 2), dirs_near((1, 0, -0.01), 3, 500, 3)])
    uv = F.oct_encode(d)
    assert uv.shape == (len(d), 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    back = F.oct_decode(uv)
    np.testing.assert_allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(back, d, atol=1e-9)


def test_decode_encode_roundtrip_interior_uv_and_shapes():
    rng = np.random.default_rng(7)
    uv = 0.02 + 0.96 * rng.random((5000, 2))
    d = F.oct_decode(uv)
    np.testing.assert_allclose(F.oct_encode(d), uv, atol=1e-9)
    # 임의 선행 차원
    d3 = rand_dirs(24, 5).reshape(2, 3, 4, 3)
    assert F.oct_encode(d3).shape == (2, 3, 4, 2)
    assert F.oct_decode(F.oct_encode(d3)).shape == (2, 3, 4, 3)


def test_encode_normalizes_and_rejects_bad_input():
    d = rand_dirs(50, 3)
    np.testing.assert_allclose(F.oct_encode(d * 7.5), F.oct_encode(d), atol=1e-12)
    for bad in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 0.0]):
        with pytest.raises(F.FarFieldError):
            F.oct_encode(np.array(bad))
    with pytest.raises(F.FarFieldError):
        F.oct_encode(np.zeros((4, 2)))


def test_texel_grid_conventions():
    S = 8
    c = F.texel_centers(S)
    assert c.shape == (S, S, 2)
    # [행 i, 열 j] -> (u, v) = ((j+.5)/S, (i+.5)/S)
    np.testing.assert_allclose(c[0, 0], (0.5 / S, 0.5 / S), atol=1e-15)
    np.testing.assert_allclose(c[2, 5], (5.5 / S, 2.5 / S), atol=1e-15)
    d = F.texel_dirs(S)
    assert d.shape == (S, S, 3)
    np.testing.assert_allclose(d, F.oct_decode(c), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
    with pytest.raises(F.FarFieldError):
        F.texel_dirs(4)
    with pytest.raises(F.FarFieldError):
        F.texel_dirs(16.5)


def test_texel_solid_angle_sums_to_sphere():
    om = F.texel_solid_angle(64)
    assert om.shape == (64, 64) and (om > 0).all()
    assert om.sum() == pytest.approx(4 * math.pi, rel=1e-3)
    d = F.texel_dirs(128)
    om2 = F.texel_solid_angle(128)
    assert om2[d[..., 2] > 0].sum() == pytest.approx(2 * math.pi, rel=3e-2)  # 위 반구
    # 가장 큰/작은 텍셀 비가 팔면체 이론(l1^3: 1 .. 3sqrt3 = 5.2) 안
    assert 1.0 < om2.max() / om2.min() < 5.4


def test_texel_solid_angle_matches_monte_carlo_density():
    """균일 구 표본이 4x4 블록에 떨어지는 비율 x 4pi = 블록 입체각의 합(균일 가중 구현은 걸러진다)."""
    S, B = 16, 4
    om = F.texel_solid_angle(S)
    d = rand_dirs(400000, 11)
    uv = F.oct_encode(d)
    ij = np.minimum((uv[:, ::-1] * S).astype(int), S - 1)  # (행, 열)
    cnt = np.zeros((S // B, S // B))
    np.add.at(cnt, (ij[:, 0] // B, ij[:, 1] // B), 1)
    est = cnt / len(d) * 4 * math.pi
    ref = om.reshape(S // B, B, S // B, B).sum((1, 3))
    np.testing.assert_allclose(est, ref, rtol=0.04)


def test_sample_returns_texels_at_texel_centers():
    rng = np.random.default_rng(1)
    S = 32
    rgb = rng.random((S, S, 3)).astype(np.float32) * 3
    got = F.sample(rgb, F.texel_dirs(S))
    assert got.shape == (S, S, 3)
    np.testing.assert_allclose(got, rgb, atol=1e-5)
    ff = C.FarField(rgb=rgb)
    np.testing.assert_allclose(F.sample(ff, F.texel_dirs(S)), rgb, atol=1e-5)  # FarField 도 받는다


def test_sample_constant_map_and_shapes():
    rgb = np.empty((16, 16, 3), dtype=np.float32)
    rgb[:] = (0.25, 1.5, 3.0)
    d = rand_dirs(3000, 4)
    np.testing.assert_allclose(F.sample(rgb, d), np.broadcast_to([0.25, 1.5, 3.0], (3000, 3)), atol=1e-6)
    assert F.sample(rgb, np.array([0.0, 0.0, 1.0])).shape == (3,)
    assert F.sample(rgb, d[:24].reshape(4, 6, 3)).shape == (4, 6, 3)
    np.testing.assert_allclose(F.sample(rgb, d[:5] * 9.0), F.sample(rgb, d[:5]), atol=1e-12)  # 정규화


def test_sample_rejects_bad_input():
    rgb = np.ones((16, 16, 3), dtype=np.float32)
    with pytest.raises(F.FarFieldError):
        F.sample(rgb, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(F.FarFieldError):
        F.sample(rgb, np.array([np.nan, 0.0, 1.0]))
    with pytest.raises(F.FarFieldError):
        F.sample(np.ones((16, 15, 3), dtype=np.float32), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(F.FarFieldError):
        F.sample(np.ones((16, 16, 4), dtype=np.float32), np.array([0.0, 0.0, 1.0]))


# ---- 이음매: 뒤통수(-z) 모서리와 네 변 ------------------------------------------------------------------------------
def _seam_dirs():
    out = []
    # -z 주변 6도 이내(네 모서리가 모이는 곳)
    out.append(dirs_near((0, 0, -1), 6, 4000, 21))
    # 네 변 = 적도점(+-x, +-y)에서 -z 로 가는 자오선 4개. 각 자오선 위와 +-1.5도 양옆
    rng = np.random.default_rng(22)
    for ax in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        t = np.radians(rng.uniform(2, 88, 600))  # 적도로부터 아래로 내려간 각
        base = np.stack([ax[0] * np.cos(t), ax[1] * np.cos(t), -np.sin(t)], axis=1)
        side = np.cross(base, [0, 0, 1.0])
        side /= np.linalg.norm(side, axis=1, keepdims=True)
        off = np.radians(rng.uniform(-1.5, 1.5, 600))[:, None]
        d = base * np.cos(off) + side * np.sin(off)
        out.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    return np.concatenate(out)


@pytest.mark.parametrize("S", [32, 64])
def test_linear_field_is_reproduced_across_seams(S):
    """선형 함수는 맵 전체에서 거의 정확히 재현돼야 한다. 가장자리를 clamp 로 처리하면 모서리·변에서 크게 틀린다."""
    rgb = lin_map(S)
    d_seam = _seam_dirs()
    d_all = rand_dirs(6000, 9)
    err_seam = np.abs(F.sample(rgb, d_seam) - lin_field(d_seam)).max()
    err_all = np.abs(F.sample(rgb, d_all) - lin_field(d_all)).max()
    # 쌍선형 + 접힘선(꺾임)의 보간 오차는 ~0.45/S. 여유 2배.
    assert err_all < 1.0 / S, (err_all, S)
    assert err_seam < 1.0 / S, (err_seam, S)
    assert err_seam < 1.5 * err_all + 1e-3  # 이음매가 다른 곳보다 특별히 나쁘지 않다


@pytest.mark.parametrize("S", [32, 64])
def test_values_continuous_across_each_fold_edge_and_the_back_pole(S):
    """수치 연속성: 서로 0.2도 떨어진 두 방향(접은 변 양쪽)의 표본 차이는 필드의 기울기 x 거리 수준이다."""
    rgb = lin_map(S)
    d = _seam_dirs()
    side = np.cross(d, rand_dirs(len(d), 31))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    e = np.radians(0.2)
    d2 = d * math.cos(e) + side * math.sin(e)
    diff = np.abs(F.sample(rgb, d) - F.sample(rgb, d2)).max()
    assert diff < 0.4 * e * 1.3 + 2e-3, diff  # 기울기 <= 0.4/rad


def test_value_at_back_pole_is_single_valued():
    """-z 를 사이에 둔 네 모서리 텍셀 값이 서로 이어진다: -z 표본이 주변 표본의 평균과 같다."""
    S = 64
    rgb = lin_map(S)
    p = np.array([0.0, 0.0, -1.0])
    v = F.sample(rgb, p)
    ring = dirs_near((0, 0, -1), 0.8, 400, 5)
    np.testing.assert_allclose(F.sample(rgb, ring).mean(0), v, atol=3e-3)
    np.testing.assert_allclose(v, lin_field(p), atol=2e-2)

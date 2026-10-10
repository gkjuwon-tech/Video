"""RD-07 순수 수치 1: 전역 크기, 타일 계획, 타일 가중치(계약서 수식 오라클)."""
import math

import numpy as np
import pytest

import rd07_helpers as H
from buc.read import mrh_call as M


# ----------------------------------------------------------------------------------------- 상수

def test_constants_and_error_hierarchy():
    assert (M.STRIDE, M.UNIT, M.TILE_PX, M.GLOBAL_PIXELS, M.READ_LONG_MAX, M.FEATHER_PX) == \
        (4, 16, 1024, 589_824, 3840, 64)
    assert tuple(M.GLOBAL_SIGMAS) == (0.20, 0.35) and M.TILE_SIGMA == 0.25
    assert M.DEPTH_FLOOR == 0.05 and M.UNC_TAU == 0.10
    assert (M.LANE_ENGINE, M.LANE_CLEANROOM) == ("engine", "cleanroom")
    assert issubclass(M.MrhCallSpecError, M.MrhCallError) and issubclass(M.MrhCallSpecError, ValueError)
    assert issubclass(M.CleanroomError, M.MrhCallError) and issubclass(M.MrhCallError, RuntimeError)


# ----------------------------------------------------------------------------------------- 전역 크기

@pytest.mark.parametrize("hw,expect", [
    ((576, 1024), (576, 1024)),          # 1K: 그대로
    ((1152, 2048), (576, 1024)),         # 2K
    ((2160, 3840), (576, 1024)),         # 4K
    ((1000, 1000), (768, 768)),          # 정방은 면적 예산 0.59 MP
    ((480, 640), (480, 640)),            # 예산 안이면 업스케일 없음
    ((16, 4096), (16, 4096)),
])
def test_global_size_values(hw, expect):
    assert M.global_size(*hw) == expect


def test_global_size_custom_budget_and_properties():
    assert M.global_size(96, 128, pixels=3072) == (48, 64)
    for h, w in [(720, 1280), (1080, 1920), (3000, 2000), (512, 512), (2000, 640), (4320, 7680)]:
        gh, gw = M.global_size(h, w)
        assert gh % 16 == 0 and gw % 16 == 0 and gh >= 16 and gw >= 16
        assert gh <= h and gw <= w                                         # 업스케일 없음
        assert gh * gw <= 1.06 * M.GLOBAL_PIXELS                           # 반올림 여유 6 %
        assert abs(math.log((gw / gh) / (w / h))) < 0.08                   # 종횡비 보존(16 배수 반올림 오차 안)


# ----------------------------------------------------------------------------------------- 타일 계획

def _axis_check(starts_ends, L, tile):
    starts = [a for a, _ in starts_ends]
    ends = [b for _, b in starts_ends]
    assert starts[0] == 0 and ends[-1] == L
    assert all(s % 16 == 0 for s in starts)
    assert all(s2 > s1 for s1, s2 in zip(starts, starts[1:])), "시작은 엄격 증가"
    assert all(s2 <= e1 for e1, s2 in zip(ends, starts[1:])), "틈 없음"
    assert len(starts) == -(-L // tile)


def test_plan_tiles_known_counts_and_order():
    t4k = M.plan_tiles(2160, 3840)
    assert len(t4k) == 12
    assert len(M.plan_tiles(1152, 2048)) == 4
    assert M.plan_tiles(576, 1024) == () and M.plan_tiles(1024, 1024) == ()
    assert len(M.plan_tiles(1040, 1024)) == 2                                # 한 변만 넘어도 타일
    t = M.plan_tiles(576, 1536)
    assert len(t) == 2 and all((b.y1 - b.y0, b.x1 - b.x0) == (576, 1024) for b in t)
    # 행 우선: y가 같은 것끼리 x 증가, 다음 행으로
    ys = [b.y0 for b in t4k]
    assert ys == sorted(ys) and len(set(ys)) == 3
    for row in range(3):
        xs = [b.x0 for b in t4k[row * 4:(row + 1) * 4]]
        assert xs == sorted(xs) and len(set(xs)) == 4
    assert isinstance(t4k[0], M.TileBox) and (t4k[0].y0, t4k[0].x0) == (0, 0)


def test_plan_tiles_axis_properties_sweep_1024():
    for L in range(1040, 8209, 16):                                          # 열 방향(행은 한 줄)
        boxes = M.plan_tiles(64, L)
        _axis_check([(b.x0, b.x1) for b in boxes], L, 1024)
        assert all(b.y0 == 0 and b.y1 == 64 and b.x1 - b.x0 == 1024 for b in boxes)
    for L in range(1040, 4097, 16):                                          # 행 방향
        boxes = M.plan_tiles(L, 64)
        _axis_check([(b.y0, b.y1) for b in boxes], L, 1024)


def test_plan_tiles_axis_properties_sweep_small_tile():
    for tile in (64, 128):
        for L in range(tile + 16, 20 * tile, 16):
            boxes = M.plan_tiles(32, L, tile=tile)
            _axis_check([(b.x0, b.x1) for b in boxes], L, tile)


def test_plan_tiles_covers_canvas_2d():
    for hw, tile in [((96, 128), 64), ((2160, 3840), 1024), ((1152, 2048), 1024), ((200 - 8, 400), 64)]:
        cov = np.zeros(hw, np.int32)
        for b in M.plan_tiles(*hw, tile=tile):
            assert 0 <= b.y0 < b.y1 <= hw[0] and 0 <= b.x0 < b.x1 <= hw[1]
            assert all(v % 16 == 0 for v in (b.y0, b.x0, b.y1, b.x1))
            cov[b.y0:b.y1, b.x0:b.x1] += 1
        assert cov.min() >= 1, "모든 캔버스 화소가 어떤 타일에든 덮여야 한다"


# ----------------------------------------------------------------------------------------- 타일 가중치

@pytest.mark.parametrize("hw,tile,feather", [((96, 128), 64, 16), ((96, 128), 64, 64), ((96, 128), 64, 0),
                                              ((144, 240), 64, 32)])
def test_tile_weights_match_formula_oracle(hw, tile, feather):
    boxes = M.plan_tiles(*hw, tile=tile)
    w = M.tile_weights(boxes, hw, feather_px=feather)
    ref = H.oracle_tile_weights(boxes, hw, feather)
    assert tuple(w.shape) == (len(boxes), hw[0] // 4, hw[1] // 4)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0 + 1e-6
    np.testing.assert_allclose(w.numpy().astype(np.float64), ref, atol=2e-6)


def test_tile_weights_structure():
    hw, tile = (96, 128), 64
    boxes = M.plan_tiles(*hw, tile=tile)
    w = M.tile_weights(boxes, hw, feather_px=16).numpy()
    # 캔버스 가장자리 셀은 페더가 없다: 타일 0의 (0,0) 모서리 셀은 가중치 1
    assert w[0, 0, 0] == pytest.approx(1.0)
    # 타일 안쪽 깊은 셀은 1, 타일 밖은 0
    assert w[0, 3, 3] == pytest.approx(1.0) and w[0, 3, 24] == 0.0
    # 타일 1(x0=64): 안쪽 경계(셀 16)에서 0.5/4 만큼만 들어간 셀은 작고 단조 증가
    row = w[1, 3, 16:24]
    assert row[0] < 0.1 and np.all(np.diff(row) >= -1e-9) and row[-1] == pytest.approx(1.0)
    # 가중치 0 페더 = 하드 엣지: 타일 안은 모두 1
    w0 = M.tile_weights(boxes, hw, feather_px=0).numpy()
    for t, b in enumerate(boxes):
        inside = w0[t, b.y0 // 4:b.y1 // 4, b.x0 // 4:b.x1 // 4]
        assert np.all(inside == 1.0)
    assert M.tile_weights((), hw).shape == (0, 24, 32)

"""RDR-09 IR_geo (납품 리포트에 싣는 미관측 비율): 합성 atom_id 해석값 + 실제 렌더 해석값."""
import numpy as np
import pytest
from rdr09_helpers import (
    FX,
    I4,
    H,
    K_default,
    W,
    forward_shot,
    layer,
    make_obs,
    make_world,
    rand_tex,
    translate,
    two_layer_world,
)

from buc.render import aov as A
from buc.render import pos

HW = H * W


def _bits(n, observed):
    flags = np.zeros(n, bool)
    flags[list(observed)] = True
    return A.ObsBitset.from_bool(flags)


def test_ir_geo_synthetic_closed_form():
    bits = _bits(20, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])              # 원자 0..9 관측, 10..19 미관측
    a = np.full((3, 2, 4), -1, np.int64)
    a[0] = [[0, 1, 2, 3], [10, 11, 12, 13]]                        # 프레임 0: 8 화소 중 4 관측 -> 0.5
    a[1, 0] = [0, 1, 2, 10]                                        # 프레임 1: 유효 4(나머지 -1), 3 관측 -> 0.25
    a[2, 1, :2] = [15, 16]                                         # 프레임 2: 유효 2, 0 관측 -> 1.0
    r = pos.ir_geo(a, bits)
    assert r.per_frame == pytest.approx((0.5, 0.25, 1.0), abs=1e-15)
    assert r.max == pytest.approx(1.0, abs=1e-15)
    assert r.mean == pytest.approx(1.0 - 7 / 14, abs=1e-15)         # 화소 풀링: 관측 4+3+0, 유효 8+4+2
    assert r.mean != pytest.approx(np.mean(r.per_frame), abs=1e-3)  # 프레임 평균의 평균이 아니다
    assert isinstance(r.per_frame, tuple) and all(isinstance(x, float) for x in r.per_frame)
    assert isinstance(r.mean, float) and isinstance(r.max, float)


def test_ir_geo_edge_cases():
    bits = _bits(8, range(8))
    none = np.full((2, 3, 3), -1, np.int64)
    r = pos.ir_geo(none, bits)
    assert r.per_frame == (0.0, 0.0) and r.mean == 0.0 and r.max == 0.0       # 유효 화소 0 이면 0.0 (발명할 것도 없다)
    allobs = np.zeros((1, 2, 2), np.int64)
    assert pos.ir_geo(allobs, bits).mean == 0.0
    with pytest.raises(ValueError):
        pos.ir_geo(np.zeros((3, 3), np.int64), bits)                           # 3 차원이어야 한다
    with pytest.raises(ValueError):
        pos.ir_geo(np.full((1, 2, 2), 8, np.int64), bits)                      # id >= n 은 ObsBitset.test 가 거부


def test_ir_geo_real_render_identity_camera_half_masked():
    mask = np.ones((H, W), bool)
    mask[:, : W // 2] = False                                                  # 입력이 본 영역 = 오른쪽 절반
    world = two_layer_world(mask=mask)
    fr = A.render_shot_aov(world, _still_shot(1))
    bits = A.observed_atoms(world)
    r = pos.ir_geo(fr.aov["atom_id"], bits)
    # 항등 카메라에서 화소 = 텍셀이므로 지배 원자가 앞 층이든 뒤 층이든 마스크 열로 해석값이 정해진다.
    a = fr.aov["atom_id"][0]
    assert (a >= 0).all()
    _lay, vv, uu = A.unpack_atom_id(a.reshape(-1), width=W, height=H)
    expect_unobs = float((~mask[vv, uu]).mean())
    assert r.per_frame[0] == pytest.approx(expect_unobs, abs=1e-12)
    assert r.mean == r.max == r.per_frame[0]
    assert r.mean == pytest.approx(0.5, abs=1e-12)                             # 마스크가 화소 위치별이고 카메라가 항등이라 정확히 절반


def _still_shot(n):
    from buc.core import contracts as C
    return C.Shot(c2w=np.stack([I4] * n), t=np.arange(n) / 24.0, K=np.stack([K_default()] * n), focus_m=np.full(n, 3.0),
                  width=W, height=H, fps=24.0, theta=C.Theta())


def test_ir_geo_real_render_integer_shift_closed_form():
    """깊이 z 카드를 s px 옮기면 화소 u 는 텍셀 u+s 를 본다. 입력이 왼쪽 절반만 봤다면 IR = 1 - (W/2 - s)/(W - s)."""
    z, s = 3.0, 6
    mask = np.zeros((H, W), bool)
    mask[:, : W // 2] = True
    world = make_world([layer(rand_tex(3), z)], make_obs(mask=mask))
    dx = s * z / FX()
    from buc.core import contracts as C
    shot = C.Shot(c2w=np.stack([I4, translate(I4, (dx, 0, 0))]), t=np.arange(2) / 24.0, K=np.stack([K_default()] * 2),
                  focus_m=np.full(2, 3.0), width=W, height=H, fps=24.0, theta=C.Theta())
    fr = A.render_shot_aov(world, shot)
    r = pos.ir_geo(fr.aov["atom_id"], A.observed_atoms(world))
    assert r.per_frame[0] == pytest.approx(0.5, abs=1e-12)                      # 프레임 0 은 항등
    assert r.per_frame[1] == pytest.approx(1.0 - (W / 2 - s) / (W - s), abs=0.02)    # 가장자리 소프트 에지 허용
    assert r.max == r.per_frame[1] > r.per_frame[0]
    # 풀링 평균은 유효 화소 수 가중이다
    n0, n1 = int((fr.aov["atom_id"][0] >= 0).sum()), int((fr.aov["atom_id"][1] >= 0).sum())
    assert r.mean == pytest.approx((n0 * r.per_frame[0] + n1 * r.per_frame[1]) / (n0 + n1), abs=1e-12)


def test_ir_geo_zero_when_everything_observed():
    world = two_layer_world(mask=None)
    shot = forward_shot(3)
    fr = A.render_shot_aov(world, shot)
    r = pos.ir_geo(fr.aov["atom_id"], A.observed_atoms(world))
    # 앞 층 불투명 창 뒤의 뒤 층 텍셀은 transmittance 0 이라 비관측으로 센다 -> 화면에 보이지 않으므로 지배 원자가 될 수 없다
    assert r.mean < 0.05 and 0.0 <= min(r.per_frame) and r.max < 0.1

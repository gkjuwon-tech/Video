"""RDR-02 소프트 에지: 알파 질량 보존(부화소 정확도), 전이 폭, 단조성."""
import numpy as np
import pytest

from buc.render.quads import RenderOpts, render_ldi
from rdr02_helpers import H, W, FX, K_default, edge_left, edge_right, layer, make_world, translate

D = 3.0
C0, C1 = 30, 65                      # 카드 열 범위 [C0, C1] (포함)
MID = 47


def _card_world(rgb=(1.0, 1.0, 1.0)):
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[16:48, C0:C1 + 1] = (*rgb, 1.0)
    return make_world([layer(rgba, D)])


def _shifted(delta_px, **kw):
    tx = delta_px * D / FX()                      # 평면이 왼쪽으로 delta_px 이동
    return render_ldi(_card_world(), c2w=translate(np.eye(4), (tx, 0, 0)), K=K_default(), width=W, height=H, **kw)


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.5, 0.75, 1.3, -2.6])
def test_edge_position_is_subpixel_accurate_via_alpha_mass(delta):
    out = _shifted(delta)
    row = out.alpha[32].astype(np.float64)
    el = edge_left(row, MID)
    er = edge_right(row, MID)
    assert el == pytest.approx(C0 - 0.5 - delta, abs=0.2)
    assert er == pytest.approx(C1 + 0.5 - delta, abs=0.2)
    # 카드 면적(알파 질량)은 이동으로 변하지 않는다
    assert out.alpha[16:48].sum() == pytest.approx((C1 - C0 + 1) * 32, rel=2e-3)


@pytest.mark.parametrize("delta", [0.0, 0.5, 1.3])
def test_transition_is_compact_monotone_and_antialiased(delta):
    out = _shifted(delta)
    row = out.alpha[32].astype(np.float64)
    left = row[15:MID + 1]
    right = row[MID:W - 5]
    assert (np.diff(left) >= -1e-6).all(), "왼쪽 에지에서 알파가 단조 증가가 아니다(링잉)"
    assert (np.diff(right) <= 1e-6).all(), "오른쪽 에지에서 알파가 단조 감소가 아니다(링잉)"
    assert row.min() >= 0 and row.max() <= 1 + 1e-6
    width = int(((left > 0.02) & (left < 0.98)).sum())
    assert width <= 3, f"전이가 너무 넓다({width} px)"
    if abs(delta - round(delta)) == 0.5 or abs((delta % 1) - 0.5) < 1e-9:
        assert ((left > 0.2) & (left < 0.8)).any(), "0.5 px 어긋난 에지는 부분 알파 화소가 있어야 한다"


def test_edge_px_widens_the_transition_but_keeps_mass():
    sharp = _shifted(0.5, opts=RenderOpts(edge_px=1.0)).alpha[32].astype(np.float64)
    soft = _shifted(0.5, opts=RenderOpts(edge_px=2.0)).alpha[32].astype(np.float64)
    w_sharp = int(((sharp[15:MID + 1] > 0.02) & (sharp[15:MID + 1] < 0.98)).sum())
    w_soft = int(((soft[15:MID + 1] > 0.02) & (soft[15:MID + 1] < 0.98)).sum())
    assert w_soft > w_sharp
    assert edge_left(soft, MID) == pytest.approx(C0 - 0.5 - 0.5, abs=0.25)   # 질량(위치)은 보존

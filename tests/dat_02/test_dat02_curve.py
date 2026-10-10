"""결정 곡선: 검증, 조회, 대역 마스크(경계 포함), 키 공식, 잠정 곡선 수치."""
import math
import struct

import numpy as np
import pytest

from buc.core import det
from buc.data import realify as R
from buc.teacher.sample import DEFAULT_TABLE, SigmaTable


def _curve(**kw):
    base = dict(sigmas=[0.75, 0.5, 0.3], D=[[0.9, 0.5, 0.2], [0.8, 0.1, 0.05], [0.3, 0.1000001, 0.0]])
    base.update(kw)
    return R.DecisionCurve(**base)


def test_constants():
    assert R.SIGMA_RANGE == (0.2, 0.7)
    assert tuple(R.ALLOWED_STEPS) == (2, 3)
    assert R.D_THRESH == 0.1
    assert R.REALIFY_TAG == "buc.dat02.realify.v1"
    assert R.CURVE_TAG == "buc.dat02.curve.v1"
    assert issubclass(R.RealifyError, ValueError)


def test_normalized_and_lookup():
    c = _curve(D=np.array([[0.9, 0.5, 0.2], [0.8, 0.1, 0.05], [0.3, 0.1000001, 0.0]]))
    assert c.sigmas == (0.75, 0.5, 0.3) and isinstance(c.sigmas, tuple)
    assert isinstance(c.D, tuple) and all(isinstance(r, tuple) for r in c.D)
    assert all(type(v) is float for r in c.D for v in r)
    assert c.n_octaves == 3
    assert c.at(0.5) == (0.8, 0.1, 0.05)
    assert c.at(0.5 + 1e-12) == (0.8, 0.1, 0.05)
    with pytest.raises(R.RealifyError):
        c.at(0.4)
    with pytest.raises(R.RealifyError):
        c.at(0.5 + 1e-6)


def test_band_mask_threshold_inclusive():
    c = _curve()
    assert c.band_mask(0.5) == (False, True, True)          # 0.1 은 경계 포함
    assert c.band_mask(0.3) == (False, False, True)         # 0.1000001 은 밖
    assert c.band_mask(0.75) == (False, False, False)
    assert c.band_mask(0.75, thresh=0.5) == (False, True, True)
    assert all(isinstance(v, bool) for v in c.band_mask(0.5))


@pytest.mark.parametrize("kw", [
    dict(sigmas=[0.3, 0.5, 0.75]),                                   # 오름차순
    dict(sigmas=[0.75, 0.75, 0.3]),                                  # 중복
    dict(sigmas=[1.5, 0.5, 0.3]),                                    # >1
    dict(sigmas=[0.75, 0.5, 0.0]),                                   # 0
    dict(sigmas=[0.75, float("nan"), 0.3]),
    dict(D=[[0.9, 0.5, 0.2], [0.8, 0.1], [0.3, 0.1, 0.0]]),          # 들쭉날쭉
    dict(D=[[0.9, 0.5, 0.2], [0.8, 0.1, 0.05]]),                     # 행 수 불일치
    dict(D=[[1.1, 0.5, 0.2], [0.8, 0.1, 0.05], [0.3, 0.1, 0.0]]),    # >1
    dict(D=[[-0.1, 0.5, 0.2], [0.8, 0.1, 0.05], [0.3, 0.1, 0.0]]),   # <0
    dict(D=[[float("inf"), 0.5, 0.2], [0.8, 0.1, 0.05], [0.3, 0.1, 0.0]]),
    dict(sigmas=[], D=[]),
    dict(D=[[], [], []]),                                            # n_oct 0
])
def test_invalid_curves(kw):
    with pytest.raises(R.RealifyError):
        _curve(**kw)


def test_key_formula_and_sensitivity():
    c = _curve(name="x1", table_key="abc")
    vals = list(c.sigmas) + [v for r in c.D for v in r]
    want = det.hash_parts(R.CURVE_TAG, "x1", "abc", 3, 3, *[struct.pack("<d", v) for v in vals]).hex()
    assert c.key == want
    c0 = _curve(name="x1")
    want0 = det.hash_parts(R.CURVE_TAG, "x1", "", 3, 3, *[struct.pack("<d", v) for v in vals]).hex()
    assert c0.key == want0
    assert _curve(name="x2", table_key="abc").key != c.key
    d2 = [list(r) for r in c.D]
    d2[2][2] = 1e-12
    assert _curve(name="x1", table_key="abc", D=d2).key != c.key
    assert _curve(name="x1", table_key="abc").key == c.key


def test_provisional_curve_values():
    c = R.provisional_curve(DEFAULT_TABLE, 6)
    assert c.name == "provisional-v0"
    assert c.table_key == DEFAULT_TABLE.key
    assert c.sigmas == DEFAULT_TABLE.points
    assert c.n_octaves == 6
    for s, row in zip(c.sigmas, c.D):
        for o, v in enumerate(row):
            r = ((1 - s) / s) * 2.0 ** (o - 2)
            assert v == pytest.approx(1.0 / (1.0 + r * r), rel=1e-12, abs=1e-15)
    assert c.band_mask(0.5) == (False, False, False, False, True, True)
    assert c.band_mask(0.3) == (False, False, False, True, True, True)
    assert c.band_mask(9 / 14) == (False, False, False, False, False, True)
    assert c.band_mask(1.0) == (False,) * 6
    t2 = SigmaTable((0.9, 0.4), name="t2")
    c2 = R.provisional_curve(t2, 3)
    assert c2.sigmas == (0.9, 0.4) and c2.table_key == t2.key
    assert math.isclose(c2.at(0.4)[2], 1 / (1 + (1.5 * 1.0) ** 2))

"""RD-06 4파라미터 필름 곡선·CDL: 독립 오라클 일치, 항등, 단조, 피벗, 토/어깨/기울기/화이트 의미, 역 왕복."""
import warnings

import numpy as np
import pytest

import rd06_helpers as H
from buc.read import tone_color as T

PARAMS = [
    dict(), dict(toe=0.5), dict(shoulder=0.5), dict(slope=1.3), dict(slope=0.7), dict(white=0.7), dict(white=1.6),
    dict(toe=0.3, shoulder=0.4, slope=1.15, white=1.0), dict(toe=1.0, shoulder=1.0, slope=2.0, white=2.0),
    dict(toe=0.0, shoulder=0.01, slope=1.0, white=1.0), dict(toe=0.7, shoulder=0.2, slope=0.6, white=0.5),
]
XS = np.concatenate([[0.0], np.geomspace(1e-5, 50.0, 400)])


@pytest.mark.parametrize("kw", PARAMS)
def test_curve_matches_independent_oracle(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # 오버플로·0 나눗셈 경고 없이
        got = T.film_curve(XS, T.FilmCurve(**kw))
    want = H.o_curve(XS, **kw)
    assert got.dtype == np.float64 and got.shape == XS.shape
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_default_curve_is_min_x_1():
    x = np.array([0.0, 0.01, 0.18, 0.5, 0.999, 1.0, 1.5, 10.0])
    np.testing.assert_allclose(T.film_curve(x, T.FilmCurve()), np.minimum(x, 1.0), atol=1e-12)
    assert float(T.film_curve(0.0, T.FilmCurve(toe=1.0, shoulder=1.0))) == 0.0


def test_shapes_and_scalars():
    c = T.FilmCurve(toe=0.2)
    x2 = np.random.default_rng(0).uniform(0, 2, (3, 4))
    assert T.film_curve(x2, c).shape == (3, 4)
    assert np.isclose(float(T.film_curve(0.18, c)), H.o_curve(0.18, toe=0.2))
    y, clipped = T.film_curve_inv(x2 * 0.3, c)
    assert y.shape == (3, 4) and np.asarray(clipped).shape == (3, 4)


@pytest.mark.parametrize("kw", PARAMS)
def test_curve_monotone(kw):
    y = T.film_curve(XS, T.FilmCurve(**kw))
    d = np.diff(y)
    assert (d >= -1e-14).all()
    W = kw.get("white", 1.0)
    assert (y <= W + 1e-12).all()
    below = y < W * 0.999
    assert (d[below[1:] & below[:-1]] > 0).all()        # 천장 아래에서는 엄격 증가


def test_pivot_and_slope():
    for s in (0.5, 0.8, 1.0, 1.5, 2.0):
        c = T.FilmCurve(slope=s, white=2.0)
        assert abs(float(T.film_curve(0.18, c)) - 0.18) < 1e-12                     # 토·어깨 0 이면 피벗 0.18 고정
        assert abs(float(T.film_curve(0.36, c)) - 0.18 * 2 ** s) < 1e-12            # 로그-로그 기울기 = slope


def test_toe_semantics():
    x = np.geomspace(1e-4, 5.0, 200)
    base = T.film_curve(x, T.FilmCurve())
    prev = base
    for toe in (0.25, 0.5, 1.0):
        y = T.film_curve(x, T.FilmCurve(toe=toe))
        assert (y <= prev + 1e-15).all() and (y < prev).any()               # toe 는 그림자를 눌러 줄 뿐 올리지 않는다
        assert (y >= 0).all()
        prev = y
        h = 1e-7                                                              # 원점 기울기 1/(1+2 toe)
        slope0 = float(T.film_curve(h, T.FilmCurve(toe=toe))) / h
        assert abs(slope0 * (1 + 2 * toe) - 1) < 5e-3
    big = np.array([1.0, 1.8])                                                # 밝은 쪽에서는 오프셋 ~(b-a)=0.04 toe 만 남는다
    d = T.film_curve(big, T.FilmCurve(toe=1.0, white=2.0)) - T.film_curve(big, T.FilmCurve(white=2.0))
    assert ((d >= -0.0400) & (d <= -0.0370)).all()


def test_shoulder_and_white_semantics():
    x = np.geomspace(0.01, 1e4, 200)
    hard = T.film_curve(x, T.FilmCurve(white=1.0))
    soft = T.film_curve(x, T.FilmCurve(shoulder=0.5, white=1.0))
    assert (soft <= hard + 1e-12).all() and (soft < 1.0).all()                # 어깨는 천장 아래로 점근
    assert soft[-1] > 0.99 and abs(soft[-1] - 1.0) < 1e-3
    assert abs(float(T.film_curve(0.18, T.FilmCurve(shoulder=0.2))) - 0.18) < 5e-3     # 약한 어깨는 중간톤을 거의 안 건드림
    lo = T.film_curve(x, T.FilmCurve(shoulder=0.1))
    hi = T.film_curve(x, T.FilmCurve(shoulder=1.0))
    assert (hi <= lo + 1e-12).all() and (hi < lo).any()                       # shoulder 가 클수록 더 일찍·길게 눌린다
    assert float(T.film_curve(1e6, T.FilmCurve(white=0.7))) == pytest.approx(0.7, abs=1e-12)    # 하드 클립 천장 = white
    assert float(T.film_curve(1e6, T.FilmCurve(white=1.8, shoulder=0.3))) == pytest.approx(1.8, abs=1e-3)
    S1 = float(T.film_curve(1.0, T.FilmCurve(shoulder=1.0, white=1.0)))       # S=1: t/(1+t)
    assert S1 == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("kw", PARAMS)
def test_curve_inverse_roundtrip(kw):
    c = T.FilmCurve(**kw)
    x = np.geomspace(1e-4, 3.0, 300)
    y = T.film_curve(x, c)
    keep = y <= 0.9 * kw.get("white", 1.0)                                   # 천장 근처는 조건수가 나쁘다
    xr, clipped = T.film_curve_inv(y[keep], c)
    assert not np.asarray(clipped).any()
    np.testing.assert_allclose(xr, x[keep], rtol=1e-8, atol=1e-12)
    x0, c0 = T.film_curve_inv(np.array([0.0]), c)
    assert x0[0] == 0.0 and not c0[0]


@pytest.mark.parametrize("kw", [dict(white=1.0), dict(shoulder=0.4, white=1.0), dict(shoulder=0.4, white=0.7, toe=0.3),
                                dict(white=1.5)])
def test_curve_inverse_clipping(kw):
    c = T.FilmCurve(**kw)
    W = kw.get("white", 1.0)
    edge = W * (1 - 1e-3)
    ys = np.array([0.5 * W, 0.99 * W, edge * 0.9999, edge, W, W * 1.5 + 0.1])
    x, clipped = T.film_curve_inv(ys, c)
    assert list(np.asarray(clipped)) == [False, False, False, True, True, True]
    assert np.isfinite(x).all() and (x >= 0).all()
    assert x[3] == pytest.approx(x[4]) == pytest.approx(x[5])                # 하한: 모두 같은 값으로 잘린다
    assert x[3] >= x[2]                                                       # 하한이 아래 값보다 작지 않다
    back = T.film_curve(x[3], c)
    assert float(back) >= edge * (1 - 1e-6) and float(back) <= W + 1e-12


# ---------------------------------------------------------------------------------------------- CDL
CDLS = [dict(), dict(sat=1.4), dict(sat=0.3), dict(slope=(1.2, 1.0, 0.8)), dict(offset=(0.02, -0.01, 0.03)),
        dict(power=(0.9, 1.0, 1.2)), dict(slope=(1.1, 0.9, 1.05), offset=(0.01, 0.0, -0.02), power=(1.1, 0.95, 1.0), sat=1.25)]


@pytest.mark.parametrize("kw", CDLS)
def test_cdl_matches_oracle(kw):
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 1, (500, 3))
    np.testing.assert_allclose(T.apply_cdl(c, T.CDL(**kw)), H.o_cdl(c, **kw), rtol=1e-10, atol=1e-12)
    assert T.apply_cdl(c.reshape(20, 25, 3), T.CDL(**kw)).shape == (20, 25, 3)


def test_cdl_identity_and_luma_preservation():
    rng = np.random.default_rng(2)
    c = rng.uniform(0, 1, (400, 3))
    np.testing.assert_allclose(T.apply_cdl(c, T.CDL()), c, atol=1e-12)
    for sat in (0.05, 0.5, 1.5, 2.0):                                         # 채도는 휘도를 바꾸지 않는다
        out = T.apply_cdl(c, T.CDL(sat=sat))
        np.testing.assert_allclose(out @ H.W709, c @ H.W709, atol=1e-12)
    gray = np.full((5, 3), 0.4)
    np.testing.assert_allclose(T.apply_cdl(gray, T.CDL(sat=1.9)), gray, atol=1e-12)
    out = T.apply_cdl(np.array([[0.9, 0.1, 0.1]]), T.CDL(sat=2.0))
    assert out.min() < 0                                                       # 클립하지 않는다(음수 허용)
    assert (T.apply_cdl(np.array([[0.02, 0.5, 0.5]]), T.CDL(offset=(-0.2, 0, 0)))[0, 0] >= 0)


@pytest.mark.parametrize("kw", CDLS)
def test_cdl_inverse_roundtrip(kw):
    rng = np.random.default_rng(3)
    c = rng.uniform(0.15, 0.9, (600, 3))
    cdl = T.CDL(**kw)
    out = T.apply_cdl(c, cdl)
    back, lost = T.invert_cdl(out, cdl)
    ok = ~np.asarray(lost)
    assert ok.mean() > 0.9
    np.testing.assert_allclose(back[ok], c[ok], atol=1e-9)
    assert np.asarray(lost).shape == (600,)


def test_cdl_inverse_flags_clamped():
    cdl = T.CDL(slope=(1.0, 1.0, 1.0), offset=(-0.1, 0.0, 0.0))
    c = np.array([[0.05, 0.5, 0.5], [0.5, 0.5, 0.5]])
    out = T.apply_cdl(c, cdl)
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)
    back, lost = T.invert_cdl(out, cdl)
    assert list(np.asarray(lost)) == [True, False]
    np.testing.assert_allclose(back[1], c[1], atol=1e-12)

"""RD-15 순수 수치: sem 얼굴 확률(softmax 로짓), 템플릿 확률 지도, H/2 재판독 창, σ_z 결합."""
import math

import numpy as np
import pytest
import torch

import rd15_helpers as H

SEM_TEMP = 0.07


def oracle_sem_prob(sem, table, ids, temp=SEM_TEMP):
    s = np.asarray(sem, np.float64)
    if s.ndim == 4:
        s = s[0]
    t = np.asarray(table, np.float64)
    s = s / np.maximum(np.linalg.norm(s, axis=0, keepdims=True), 1e-6)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    lg = np.einsum("dhw,nd->nhw", s, t) / temp
    lg -= lg.max(0, keepdims=True)
    e = np.exp(lg)
    p = e / e.sum(0, keepdims=True)
    return p[list(ids)].sum(0) if ids else np.zeros(s.shape[1:])


# ----------------------------------------------------------------------------- sem 얼굴 확률
def test_sem_face_prob_matches_softmax_oracle():
    fm = H.fm()
    rng = np.random.default_rng(0)
    table = rng.normal(size=(9, 24)).astype(np.float32) * rng.uniform(0.5, 3, (9, 1)).astype(np.float32)
    sem = rng.normal(size=(1, 24, 7, 11)).astype(np.float32) * 2.5       # 단위가 아니어도 안쪽에서 정규화한다
    for ids in [(3,), (0, 5), (1, 2, 8), tuple(range(9))]:
        p = fm.sem_face_prob(sem, table, ids)
        assert p.shape == (7, 11) and p.dtype == np.float32
        np.testing.assert_allclose(p, oracle_sem_prob(sem, table, ids), atol=2e-5)
    np.testing.assert_allclose(fm.sem_face_prob(sem, table, tuple(range(9))), 1.0, atol=1e-5)
    # 서로소 분할의 합 = 합집합, 순서·중복 무관, torch 입력과 [D,h,w] 모양 허용
    a = fm.sem_face_prob(sem, table, (1, 4))
    b = fm.sem_face_prob(sem, table, (2,))
    np.testing.assert_allclose(a + b, fm.sem_face_prob(sem, table, (1, 2, 4)), atol=2e-5)
    np.testing.assert_allclose(fm.sem_face_prob(sem, table, (4, 1, 1)), a, atol=1e-6)
    np.testing.assert_allclose(fm.sem_face_prob(torch.from_numpy(sem), torch.from_numpy(table), (1, 4)), a, atol=1e-6)
    np.testing.assert_allclose(fm.sem_face_prob(sem[0], table, (1, 4)), a, atol=1e-6)
    # 척도 불변: sem 과 table 의 행 크기를 바꿔도 같다
    np.testing.assert_allclose(fm.sem_face_prob(sem * 7.0, table * 0.1, (1, 4)), a, atol=2e-5)


def test_sem_face_prob_class_vectors_and_temperature():
    fm = H.fm()
    sem, table = H.unit_sem(5, 6, np.array([[2] * 6, [0] * 6, [2] * 6, [3] * 6, [2] * 6]), n_cls=6, dim=16, seed=3)
    p = fm.sem_face_prob(sem, table, (2,))
    assert float(p[0].min()) > 0.99 and float(p[1].max()) < 0.01 and float(p[3].max()) < 0.01
    assert fm.sem_face_prob(sem, table, ()).max() == 0.0
    # 온도를 올리면 확률이 평평해진다(기본 0.07 은 계약 상수)
    hot = fm.sem_face_prob(sem, table, (2,), temp=1.0)
    assert float(hot[0].min()) < 0.5
    assert fm.SEM_TEMP == pytest.approx(SEM_TEMP)
    for bad in ((6,), (-1,), (True,), (1.5,)):
        with pytest.raises((ValueError, TypeError)):
            fm.sem_face_prob(sem, table, bad)
    with pytest.raises(ValueError):
        fm.sem_face_prob(sem[:, :8], table, (2,))                           # 차원 불일치


def test_sem_face_prob_engineered_probability():
    """H.mix_sem 은 얼굴 클래스 확률을 정확히 p 로 만든다 -> 구현이 같은 값을 내야 한다."""
    fm = H.fm()
    ro, table, tr = H.head_readout(sem_face_p=0.3)
    p = fm.sem_face_prob(ro.outputs["sem"], table, (H.FACE_CLS,))
    assert float(p[tr["mask"]].mean()) == pytest.approx(0.3, abs=1e-3)
    assert float(p[~tr["mask"]].max()) < 0.01


# ----------------------------------------------------------------------------- 템플릿 확률 지도
def test_template_prob_map_oracle():
    fm = H.fm()
    TH = fm.TemplateHit
    hits = (TH(box=(8.0, 6.0, 40.0, 30.0), score=0.9, skin=1.0, sym=1.0, eyes=1.0),
            TH(box=(20.0, 20.0, 52.0, 44.0), score=0.4, skin=1.0, sym=1.0, eyes=1.0))
    for cell, origin, grid in [(4.0, (0.0, 0.0), (16, 16)), (2.0, (10.0, 3.0), (30, 30)), (8.0, (0.0, 0.0), (8, 8))]:
        out = fm.template_prob_map(hits, grid, cell_px=cell, origin=origin)
        assert out.shape == grid and out.dtype == np.float32
        ii, jj = np.mgrid[0:grid[0], 0:grid[1]].astype(float)
        cy, cx = origin[0] + cell * (ii + 0.5), origin[1] + cell * (jj + 0.5)
        want = np.zeros(grid)
        for t in hits:
            y0, x0, y1, x1 = t.box
            ry, rx = (y1 - y0) / 2, (x1 - x0) / 2
            inside = ((cy - (y0 + y1) / 2) / ry) ** 2 + ((cx - (x0 + x1) / 2) / rx) ** 2 <= 1.0
            want = np.where(inside, np.maximum(want, t.score), want)
        # 타원 경계 칸 1~2 개의 해석 차이를 허용한다
        assert (np.abs(out - want) > 1e-6).mean() <= 0.03, cell
        assert float(out.max()) <= 0.9 + 1e-6
    assert fm.template_prob_map((), (5, 5)).max() == 0.0


# ----------------------------------------------------------------------------- 재판독 창
def _window_oracle(box, hw, pad=0.25):
    bh, bw = box[2] - box[0], box[3] - box[1]
    side = max(bh, bw) * (1 + 2 * pad)
    cy, cx = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    n = int(math.ceil(side))
    y0, x0 = int(math.floor(cy - side / 2)), int(math.floor(cx - side / 2))
    nh, nw = min(n, hw[0]), min(n, hw[1])
    y0, x0 = min(max(y0, 0), hw[0] - nh), min(max(x0, 0), hw[1] - nw)
    return (y0, x0, y0 + nh, x0 + nw)


def test_reread_window_oracle_and_invariants():
    fm = H.fm()
    hw = (240, 320)
    cases = [(100.0, 140.0, 140.0, 170.0), (0.0, 0.0, 30.0, 25.0), (200.0, 290.0, 240.0, 320.0),
             (10.5, 100.25, 70.5, 130.0), (0.0, 0.0, 240.0, 320.0)]
    for box in cases:
        w = fm.reread_window(box, hw)
        assert tuple(w) == _window_oracle(box, hw)
        y0, x0, y1, x1 = w
        assert 0 <= y0 < y1 <= hw[0] and 0 <= x0 < x1 <= hw[1]
        assert all(isinstance(v, int) for v in w)
    w = fm.reread_window((100.0, 140.0, 140.0, 170.0), hw)
    assert w[0] <= 100 and w[2] >= 140 and w[1] <= 140 and w[3] >= 170          # 여유 속에 상자가 들어 있다
    assert (w[2] - w[0]) == (w[3] - w[1]) == math.ceil(40 * 1.5)
    assert fm.REREAD_PAD == 0.25 and fm.REREAD_UP == 2
    assert fm.reread_window((100.0, 140.0, 140.0, 170.0), hw, pad=0.5) == _window_oracle(
        (100.0, 140.0, 140.0, 170.0), hw, 0.5)
    with pytest.raises(ValueError):
        fm.reread_window((50.0, 50.0, 40.0, 60.0), hw)                          # y1 <= y0


# ----------------------------------------------------------------------------- σ_z 결합
def test_combine_sigma_z():
    fm = H.fm()
    assert fm.combine_sigma_z(0.01) == pytest.approx(0.01)
    assert fm.combine_sigma_z(0.01, 0.02) == pytest.approx(math.hypot(0.01, 0.02))
    assert fm.combine_sigma_z(0.0, 0.0) == fm.SIGMA_Z_FLOOR_M == 0.002
    assert fm.combine_sigma_z(0.0005, 0.0005) == 0.002
    # 단조: 어느 항이 커져도 줄지 않는다
    xs = [fm.combine_sigma_z(r, 0.01) for r in (0.0, 0.005, 0.01, 0.03)]
    assert xs == sorted(xs)
    assert fm.combine_sigma_z(math.inf, 0.0) == math.inf
    for bad in ((-0.1, 0.0), (0.0, -0.1), (math.nan, 0.0)):
        with pytest.raises(ValueError):
            fm.combine_sigma_z(*bad)

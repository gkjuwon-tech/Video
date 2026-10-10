"""LSD 계열 선분 검출: 안티앨리어싱 다각형 변 복원, 오탐 거부, 정렬·결정론."""
import math

import numpy as np

import rd02_helpers as H
from buc.read import geom_cam as gc

W, HH = 256, 256


def _rect(angle_deg=17.0, a=120.0, b=80.0, c=(128.0, 124.0)):
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-math.sin(t), math.cos(t)])
    c = np.array(c)
    return np.array([c + 0.5 * a * u + 0.5 * b * v, c - 0.5 * a * u + 0.5 * b * v,
                     c - 0.5 * a * u - 0.5 * b * v, c + 0.5 * a * u - 0.5 * b * v])


def _angle_diff(d1, d2):
    c = abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return math.degrees(math.acos(min(1.0, c)))


def test_recovers_rectangle_edges():
    corners = _rect()
    img = H.render_polygon(W, HH, corners)
    segs = gc.detect_segments(img)
    assert segs.ndim == 2 and segs.shape[1] == 4 and segs.dtype == np.float64
    lens = np.hypot(segs[:, 0] - segs[:, 2], segs[:, 1] - segs[:, 3])
    assert (np.diff(lens) <= 1e-9).all()  # 길이 내림차순
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for a, b in edges:
        d = b - a
        best = 0.0
        for s in segs:
            p, q = s[:2], s[2:]
            if _angle_diff(q - p, d) > 1.0:
                continue
            if max(H.point_line_dist(p, a, b), H.point_line_dist(q, a, b)) > 1.5:
                continue
            # 변 위로의 사영 겹침 비율
            L = np.linalg.norm(d)
            tp = (p - a) @ d / L
            tq = (q - a) @ d / L
            lo, hi = max(0.0, min(tp, tq)), min(L, max(tp, tq))
            best = max(best, (hi - lo) / L)
        assert best >= 0.8, f"변 {a}->{b} 피복 {best:.2f}"
    # 길이 20 이상의 모든 선분은 어떤 변 근처에 있다(오탐 없음)
    for s in segs[lens >= 20]:
        mid = 0.5 * (s[:2] + s[2:])
        assert min(H.point_line_dist(mid, a, b) for a, b in edges) <= 2.0


def test_no_segments_on_flat_or_gentle_images():
    assert len(gc.detect_segments(np.full((HH, W), 0.5))) == 0
    ramp = np.tile(np.linspace(0.2, 0.7, W), (HH, 1))  # 기울기 0.002/px << grad_thresh
    assert len(gc.detect_segments(ramp)) == 0


def test_noise_gives_no_long_segments():
    rng = np.random.default_rng(0)
    noise = 0.5 + 0.02 * rng.standard_normal((HH, W))
    segs = gc.detect_segments(noise)
    if len(segs):
        lens = np.hypot(segs[:, 0] - segs[:, 2], segs[:, 1] - segs[:, 3])
        assert lens.max() < 0.1 * math.hypot(W, HH)


def test_min_length_and_determinism_and_input_unchanged():
    img = H.render_polygon(W, HH, _rect(angle_deg=-31.0))
    keep = img.copy()
    a = gc.detect_segments(img)
    b = gc.detect_segments(img)
    assert np.array_equal(a, b)
    assert np.array_equal(img, keep)
    lens = np.hypot(a[:, 0] - a[:, 2], a[:, 1] - a[:, 3])
    assert (lens >= max(8.0, 0.01 * math.hypot(W, HH)) - 1e-9).all()
    # 좌표는 영상 영역 안
    assert (a[:, [0, 2]] >= -0.5 - 1e-6).all() and (a[:, [0, 2]] <= W - 0.5 + 1e-6).all()
    assert (a[:, [1, 3]] >= -0.5 - 1e-6).all() and (a[:, [1, 3]] <= HH - 0.5 + 1e-6).all()

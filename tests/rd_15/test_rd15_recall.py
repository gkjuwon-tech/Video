"""RD-15 재현율 도구(X-FACE 게이트의 측정 함수)와 합성 영상 스윕.

face_recall 은 사람이 아니라 코드가 재는 척도다: h_f >3% 재현율 >=0.98, 1-3% >=0.90, 비얼굴 오검출 <=5%.
여기의 합성 스윕은 템플릿이 '합성 얼굴'을 놓치지 않는지만 본다. 실제 얼굴 재현율은 미측정이다(F-M8)."""
import math

import numpy as np
import pytest

import rd15_helpers as H


def test_gate_constants():
    fm = H.fm()
    assert (fm.GATE_RECALL_BIG, fm.GATE_RECALL_SMALL, fm.GATE_FP_RATE, fm.GATE_YAW_MAE_DEG) == (0.98, 0.90, 0.05, 12.0)
    assert (fm.H_F_BIG, fm.H_F_SMALL_MIN) == (0.03, 0.01)


def test_box_iou_oracle():
    fm = H.fm()
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = np.sort(rng.uniform(0, 100, (2, 2)), axis=0)
        b = np.sort(rng.uniform(0, 100, (2, 2)), axis=0)
        ba, bb = (a[0, 0], a[0, 1], a[1, 0], a[1, 1]), (b[0, 0], b[0, 1], b[1, 0], b[1, 1])
        assert fm.box_iou(ba, bb) == pytest.approx(H.iou(ba, bb), abs=1e-12)
    assert fm.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert fm.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert fm.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        fm.box_iou((5, 5, 5, 9), (0, 0, 1, 1))


def test_face_recall_counts_strata_and_false_alarms():
    fm = H.fm()
    H1, H2 = 1000, 2000
    # 표본 1 (H=1000): 큰 얼굴 2 (h_f 0.10, 0.05), 작은 얼굴 1 (h_f 0.02), 아주 작은 1 (h_f 0.005)
    gt1 = [(100, 100, 200, 180), (400, 300, 450, 340), (700, 700, 720, 715), (900, 100, 905, 104)]
    pred1 = [(102, 101, 198, 182),            # 큰 얼굴 1 적중
             (700, 700, 721, 716),            # 작은 얼굴 적중
             (900, 100, 905, 104),            # 아주 작은 얼굴 적중 (전체에만 센다)
             (10, 10, 60, 50)]                # 오검출
    # 표본 2 (H=2000): 큰 얼굴 1 (h_f 0.04) 놓침, 작은 얼굴 1 (h_f 0.015)은 중복 예측 둘 중 하나만 적중
    gt2 = [(500, 500, 580, 560), (1500, 800, 1530, 820)]
    pred2 = [(1500, 800, 1530, 820), (1501, 801, 1531, 821)]
    r = fm.face_recall([(pred1, gt1, (H1, 1500)), (pred2, gt2, (H2, 3000))], iou_min=0.3)
    assert (r.n_gt, r.n_hit) == (6, 4)
    assert (r.n_big, r.n_hit_big) == (3, 1)               # 0.10, 0.05, 0.04
    assert (r.n_small, r.n_hit_small) == (2, 2)           # 0.02, 0.015
    assert r.recall == pytest.approx(4 / 6) and r.recall_big == pytest.approx(1 / 3) and r.recall_small == 1.0
    assert r.n_pred == 6 and r.n_false == 2               # 오검출 1 + 중복 1
    # 경계: h_f 정확히 0.03 은 작은 쪽, 0.01 도 작은 쪽
    r2 = fm.face_recall([([], [(0, 0, 30, 10), (0, 0, 10, 10), (0, 0, 9.99, 10)], (1000, 1000))])
    assert (r2.n_big, r2.n_small, r2.n_gt, r2.n_hit) == (0, 2, 3, 0) and r2.recall == 0.0
    assert math.isnan(r2.recall_big)                       # 표본이 없는 층은 nan (공허한 통과 금지)
    # 일대일: 예측 하나가 gt 둘에 걸리지 않는다
    r3 = fm.face_recall([([(0, 0, 100, 100)], [(0, 0, 100, 60), (0, 40, 100, 100)], (1000, 1000))], iou_min=0.3)
    assert r3.n_hit == 1 and r3.n_false == 0 and r3.n_gt == 2
    # 예측 순서 불변, iou_min 이 오르면 적중이 줄기만 한다
    r4a = fm.face_recall([(pred1, gt1, (H1, 1500))])
    r4b = fm.face_recall([(pred1[::-1], gt1[::-1], (H1, 1500))])
    assert (r4a.n_hit, r4a.n_false) == (r4b.n_hit, r4b.n_false)
    assert fm.face_recall([(pred1, gt1, (H1, 1500))], iou_min=0.99).n_hit <= r4a.n_hit
    # 비얼굴 영상의 예측은 전부 오검출
    r5 = fm.face_recall([([(0, 0, 50, 50), (100, 100, 150, 150)], [], (500, 500))])
    assert (r5.n_gt, r5.n_pred, r5.n_false) == (0, 2, 2) and math.isnan(r5.recall)
    with pytest.raises(ValueError):
        fm.face_recall([([], [], (0, 100))])
    with pytest.raises(ValueError):
        fm.face_recall([([], [], (100, 100))], iou_min=0.0)


def _synthetic_sweep():
    rng = np.random.default_rng(11)
    tones = list(H.TONES)
    bgs = ["sky", "foliage", "gray"]
    sizes = [28, 36, 48, 60, 72, 84]
    pos = []
    for i in range(12):
        fh = sizes[i % 6]
        cy = float(rng.uniform(fh * 0.6 + 4, 240 - fh * 0.6 - 4))
        cx = float(rng.uniform(fh * 0.5 + 4, 320 - fh * 0.5 - 4))
        pos.append((cy, cx, fh, tones[i % 4], bgs[i % 3], i % 3 == 2))
    return pos


def test_template_recall_on_synthetic_faces_and_no_false_alarms():
    fm = H.fm()
    samples = []
    for i, (cy, cx, fh, tone, bg, hair) in enumerate(_synthetic_sweep()):
        img, gt = H.scene(240, 320, [(cy, cx, fh, dict(tone=tone, hair=hair))], bg=bg, seed=20 + i)
        pred = [t.box for t in fm.detect_template(img)]
        samples.append((pred, gt, (240, 320)))
    r = fm.face_recall(samples, iou_min=0.3)
    assert r.n_gt == 12 and r.n_big == 12 and r.n_small == 0
    assert r.recall_big >= 0.9, (r.n_hit, r.n_gt)            # 합성 12 장: 최대 1 장 놓침 허용
    assert r.n_false <= 1                                                     # 얼굴 하나에 곁다리 상자 몇 개까지만
    neg = []
    for i, bg in enumerate(["sky", "foliage", "gray", "dark"]):
        img = H.to_linear(H.background(240, 320, bg, seed=50 + i), 0.02, 50 + i)
        neg.append(([t.box for t in fm.detect_template(img)], [], (240, 320)))
    rn = fm.face_recall(neg)
    assert rn.n_pred == rn.n_false == 0

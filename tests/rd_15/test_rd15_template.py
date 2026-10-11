"""RD-15 파라미터 0 템플릿 검출: 피부 색도 타원 + 좌우 대칭 + 눈 위치 어두운 점 쌍.

합성 얼굴로만 검증한다. 실제 얼굴 재현율은 이 환경에서 잴 수 없다(계약서 '미검증', X-FACE 게이트 몫)."""
import time

import numpy as np
import pytest

import rd15_helpers as H


def hits_of(img, **kw):
    return H.fm().detect_template(img, **kw)


def best(hits, gt):
    return max(hits, key=lambda t: H.iou(t.box, gt), default=None)


# ----------------------------------------------------------------------------- 피부 확률
def test_skin_prob_matches_oracle_on_skin_tones_and_random_images():
    fm = H.fm()
    rng = np.random.default_rng(1)
    imgs = [rng.uniform(0, 1, (24, 32, 3)).astype(np.float32) ** 2.2]
    for c in list(H.TONES.values()) + list(H.NON_SKIN.values()):
        imgs.append(H.to_linear(np.broadcast_to(np.array(c, np.float64) / 255.0, (6, 6, 3)).copy()))
    for im in imgs:
        p = fm.skin_prob(im)
        assert p.shape == im.shape[:2] and p.dtype == np.float32
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
        np.testing.assert_allclose(p, H.skin_oracle(im), atol=2e-3)


def test_skin_prob_separates_skin_from_background():
    fm = H.fm()
    for name, c in H.TONES.items():
        im = H.to_linear(np.broadcast_to(np.array(c, np.float64) / 255.0, (4, 4, 3)).copy())
        assert float(fm.skin_prob(im).min()) > 0.5, name
    for kind in ("sky", "foliage", "gray", "dark"):
        bg = H.to_linear(H.background(60, 80, kind, seed=2))
        assert float(fm.skin_prob(bg).max()) < 0.05, kind
    white = np.ones((4, 4, 3), np.float32)
    assert float(fm.skin_prob(white).max()) == 0.0


def test_constants_pin_the_skin_ellipse():
    fm = H.fm()
    assert tuple(fm.SKIN_CBCR_CENTER) == H.SKIN_CENTER and tuple(fm.SKIN_CBCR_AXES) == H.SKIN_AXES


# ----------------------------------------------------------------------------- 검출: 재현
@pytest.mark.parametrize("fh,tone", [(40, "light"), (40, "deep"), (60, "medium"), (60, "tan"), (80, "light"),
                                      (80, "deep")])
def test_single_face_is_found_sure_and_localized(fh, tone):
    fm = H.fm()
    cy, cx = 118.0, 163.0
    img, gt = H.head_image(fh=fh, cy=cy, cx=cx, tone=tone, bg="foliage", seed=4)
    hs = hits_of(img)
    b = best(hs, gt)
    assert b is not None
    assert H.iou(b.box, gt) >= 0.5, (fh, tone, b.box, gt)
    assert b.score >= fm.P_SURE
    sure = [t for t in hs if t.score >= fm.P_SURE]
    assert len(sure) == 1, [(t.box, t.score) for t in hs]


@pytest.mark.parametrize("fh,tones", [(24, ("light", "medium")), (32, ("tan", "deep"))])
def test_small_face_is_found_as_at_least_uncertain(fh, tones):
    fm = H.fm()
    for tone in tones:
        img, gt = H.head_image(fh=fh, cy=100, cx=140, tone=tone, bg="sky", seed=1)
        b = best(hits_of(img), gt)
        assert b is not None and b.score >= fm.P_ACCEPT, (fh, tone)
        assert H.iou(b.box, gt) >= 0.4, (fh, tone, b.box, gt)


def test_face_with_hair_is_found_near_the_skin_ellipse():
    fm = H.fm()
    for tone in ("light", "deep"):
        img, gt = H.head_image(fh=70, cy=120, cx=150, tone=tone, bg="gray", seed=2, hair=True)
        hs = [t for t in hits_of(img) if t.score >= fm.P_ACCEPT]
        assert hs
        cy = (gt[0] + gt[2]) / 2
        cx = (gt[1] + gt[3]) / 2
        near = [t for t in hs if abs((t.box[0] + t.box[2]) / 2 - cy) < 0.25 * 70
                and abs((t.box[1] + t.box[3]) / 2 - cx) < 0.25 * 70]
        assert near, (tone, [t.box for t in hs], gt)


def test_n_faces_give_n_sure_hits_matched_one_to_one():
    fm = H.fm()
    spec = [(70, 70, 56, dict(tone="light")), (150, 160, 72, dict(tone="deep")), (80, 255, 44, dict(tone="tan"))]
    img, gts = H.scene(240, 320, spec, bg="sky", seed=6)
    sure = [t for t in hits_of(img) if t.score >= fm.P_SURE]
    assert len(sure) == 3, [(t.box, t.score) for t in sure]
    used = set()
    for g in gts:
        k = max(range(len(sure)), key=lambda i: H.iou(sure[i].box, g))
        assert H.iou(sure[k].box, g) >= 0.5 and k not in used
        used.add(k)


def test_translation_and_scale_covariance():
    img0, g0 = H.head_image(fh=60, cy=100, cx=120, tone="medium", bg="gray", seed=3)
    img1, g1 = H.head_image(fh=60, cy=100 + 17, cx=120 + 23, tone="medium", bg="gray", seed=3)
    b0, b1 = best(hits_of(img0), g0), best(hits_of(img1), g1)
    assert b0 is not None and b1 is not None
    dy = ((b1.box[0] + b1.box[2]) - (b0.box[0] + b0.box[2])) / 2
    dx = ((b1.box[1] + b1.box[3]) - (b0.box[1] + b0.box[3])) / 2
    assert abs(dy - 17) <= 5.0 and abs(dx - 23) <= 5.0
    h0, h1 = b0.box[2] - b0.box[0], b1.box[2] - b1.box[0]
    assert 0.8 <= h1 / h0 <= 1.25
    # 크기: 얼굴을 1.5 배로 키우면 상자 높이도 약 1.5 배
    img2, g2 = H.head_image(fh=90, cy=100, cx=120, tone="medium", bg="gray", seed=3)
    b2 = best(hits_of(img2), g2)
    assert 1.25 <= (b2.box[2] - b2.box[0]) / h0 <= 1.8


# ----------------------------------------------------------------------------- 검출: 오검출 억제 (세 성분 각각이 필요하다)
def _blob(kind, fh=64):
    """얼굴이 아닌데 한 성분씩 모자란 후보."""
    h, w = 240, 320
    if kind == "no_eyes":
        return H.scene(h, w, [(120, 160, fh, dict(tone="medium", eyes="none"))], bg="foliage", seed=5)[0]
    if kind == "one_eye":
        return H.scene(h, w, [(120, 160, fh, dict(tone="medium", eyes="one"))], bg="foliage", seed=5)[0]
    if kind == "not_skin_blue":
        return H.scene(h, w, [(120, 160, fh, dict(tone="blue", eyes="pair"))], bg="gray", seed=5)[0]
    if kind == "not_skin_gray":
        return H.scene(h, w, [(120, 160, fh, dict(tone="gray", eyes="pair"))], bg="sky", seed=5)[0]
    if kind == "skin_rect":
        img = H.background(h, w, "foliage", 5)
        img[60:190, 90:230] = np.array(H.TONES["tan"]) / 255.0
        return H.to_linear(img, 0.01, 6)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["no_eyes", "one_eye", "not_skin_blue", "not_skin_gray", "skin_rect"])
def test_incomplete_templates_are_not_faces(kind):
    fm = H.fm()
    hs = hits_of(_blob(kind))
    assert all(t.score < fm.P_ACCEPT for t in hs), [(t.box, round(t.score, 3)) for t in hs]


def test_backgrounds_and_noise_give_no_hits():
    rng = np.random.default_rng(9)
    for kind in ("sky", "foliage", "gray", "dark"):
        assert hits_of(H.to_linear(H.background(180, 240, kind, seed=3), 0.02, 3)) == ()
    noise = H.to_linear(rng.uniform(0, 1, (160, 200, 3)))
    assert hits_of(noise) == ()
    assert hits_of(np.zeros((64, 64, 3), np.float32)) == ()


# ----------------------------------------------------------------------------- 검출: 출력 규약
def test_hit_fields_ordering_and_nms_rule():
    fm = H.fm()
    spec = [(60, 60, 50, dict(tone="light")), (62, 140, 50, dict(tone="medium")), (170, 90, 66, dict(tone="deep"))]
    img, _ = H.scene(240, 320, spec, bg="foliage", seed=7)
    hs = hits_of(img, accept=0.1)
    assert hs
    keys = [(-t.score, t.box[0], t.box[1]) for t in hs]
    assert keys == sorted(keys)
    for t in hs:
        assert 0.0 <= t.score <= 1.0 and 0.0 <= t.skin <= 1.0 and 0.0 <= t.sym <= 1.0 and 0.0 <= t.eyes <= 1.0
        y0, x0, y1, x1 = t.box
        assert 0 <= y0 < y1 <= 240 and 0 <= x0 < x1 <= 320
        assert y1 - y0 >= fm.FACE_H_MIN_PX - 1
    for i, a in enumerate(hs):
        for b in hs[i + 1:]:
            assert H.iou(a.box, b.box) <= fm.NMS_IOU + 1e-9
            for p, q in ((a.box, b.box), (b.box, a.box)):
                cy, cx = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                assert not (q[0] <= cy <= q[2] and q[1] <= cx <= q[3]), (a.box, b.box)


def test_accept_threshold_and_min_size_filters():
    fm = H.fm()
    img, _ = H.head_image(fh=40, cy=110, cx=150, tone="medium", bg="sky", seed=1)
    lo = hits_of(img, accept=0.2)
    hi = hits_of(img, accept=0.9)
    assert all(t.score >= 0.9 for t in hi)
    assert {t.box for t in hi} <= {t.box for t in lo}
    big_only = hits_of(img, h_min_px=64)
    assert all(t.box[2] - t.box[0] >= 63 for t in big_only)
    assert not [t for t in big_only if t.score >= fm.P_SURE]


def test_determinism_immutability_and_global_rng():
    img, _ = H.head_image(fh=50, cy=100, cx=120, tone="tan", bg="foliage", seed=8)
    keep = img.copy()
    np_state = np.random.get_state()[1].copy()
    a, b = hits_of(img), hits_of(img)
    assert a == b
    np.testing.assert_array_equal(img, keep)
    np.testing.assert_array_equal(np.random.get_state()[1], np_state)


def test_input_validation():
    fm = H.fm()
    ok = np.full((32, 32, 3), 0.3, np.float32)
    for bad in (np.zeros((32, 32), np.float32), np.zeros((32, 32, 4), np.float32), np.zeros((0, 32, 3), np.float32)):
        with pytest.raises(ValueError):
            fm.detect_template(bad)
        with pytest.raises(ValueError):
            fm.skin_prob(bad)
    nan = ok.copy()
    nan[0, 0, 0] = np.nan
    neg = ok.copy()
    neg[0, 0, 0] = -0.1
    for bad in (nan, neg):
        with pytest.raises(ValueError):
            fm.detect_template(bad)
        with pytest.raises(ValueError):
            fm.skin_prob(bad)
    for bad in ("image", None, (ok * 255).astype(np.uint8)):
        with pytest.raises(TypeError):
            fm.detect_template(bad)
    with pytest.raises(ValueError):
        fm.detect_template(ok, accept=0.0)
    with pytest.raises(ValueError):
        fm.detect_template(ok, h_min_px=0)
    assert issubclass(fm.FacesError, ValueError)


def test_runtime_is_bounded():
    img, _ = H.head_image(fh=60, cy=110, cx=150, tone="medium", bg="foliage", seed=1)
    t = time.time()
    hits_of(img)
    assert time.time() - t < 15.0

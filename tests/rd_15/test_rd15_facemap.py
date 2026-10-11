"""RD-15 얼굴 맵 종단: 템플릿 OR sem 융합, 인스턴스, 머리 모형 피팅 -> FaceSpec(CAM-04 이음매), H/2 재판독."""
import json
import math

import numpy as np
import pytest
import torch

import rd15_helpers as H

FACE_IDS = (H.FACE_CLS,)
HW = (240, 320)


def opts(**kw):
    return H.fm().FaceOpts(face_ids=kw.pop("face_ids", FACE_IDS), **kw)


def run(img, ro, table, *, ms=None, o=None, **kw):
    return H.fm().face_map(img, ro, f_px=H.F_PX, sem_table=table, ms=ms, opts=o or opts(), **kw)


MS = type("MS", (), {"s": H.S_TRUE, "reliable": True, "sigma_s": 0.1})()
MS_BAD = type("MS", (), {"s": 3.3, "reliable": False, "sigma_s": 1.0})()


def scene_pair(yaw=20.0, pitch=5.0, fh=80, cy=110, cx=150, tone="medium", bg="sky", geo=False, **ro_kw):
    """영상 + 판독. 피팅은 비싸므로 깊이·법선은 geo=True 일 때만 싣는다."""
    img, gt = H.head_image(fh=fh, cy=cy, cx=cx, tone=tone, bg=bg, seed=1)
    ro, table, tr = H.head_readout(yaw, pitch, fh, cy, cx, with_geometry=geo, **ro_kw)
    return img, gt, ro, table, tr


# ----------------------------------------------------------------------------- 종단
def test_single_face_end_to_end_and_cam04_seam():
    fm = H.fm()
    from buc.camera.face_envelope import FaceSpec, face_angles, face_category, limits, orbit_budget_deg
    img, gt, ro, table, tr = scene_pair(yaw=20.0, pitch=5.0, geo=True)
    fmap = run(img, ro, table, ms=MS)
    assert len(fmap.faces) == 1
    f = fmap.faces[0]
    assert f.source == "both" and f.p >= fm.P_SURE and not f.uncertain and not f.reread
    assert H.iou(f.box, gt) >= 0.5
    assert f.h_f == pytest.approx((f.box[2] - f.box[0]) / HW[0], abs=1e-9)
    assert f.category == face_category(f.h_f, f.yaw_deg) == "F3"
    assert f.fit_status == "ok" and f.fit is not None and f.fit.ok and f.fit.scale_source == "ms"
    assert abs(f.yaw_deg - tr["yaw"]) <= 8.0 and abs(f.pitch_deg - tr["pitch"]) <= 8.0
    assert 0.002 <= f.sigma_z_m <= 0.02
    assert f.alpha_max_deg == pytest.approx(limits(f.category, f.sigma_z_m).yaw_deg)
    # ReadPacket.faces 원소 = FaceSpec.to_dict(); CAM-04 가 그대로 먹고, 같은 yaw/pitch 를 낸다
    assert fmap.specs() == (f.spec,)
    spec = FaceSpec.from_dict(f.spec)
    yaw, pitch, dist = face_angles(spec, np.zeros(3))
    assert yaw == pytest.approx(f.yaw_deg, abs=1e-6) and pitch == pytest.approx(f.pitch_deg, abs=1e-6)
    assert dist == pytest.approx(np.linalg.norm(tr["center"]), rel=0.05)
    assert spec.h_f == pytest.approx(f.h_f) and spec.sigma_z == pytest.approx(f.sigma_z_m)
    budget = orbit_budget_deg(spec, np.zeros(3), +1)
    assert 0.0 < budget <= 15.5                                                  # F3 의 yaw 표 15°
    b2 = orbit_budget_deg(spec, np.zeros(3), +1, margin_deg=5.0)
    assert b2 <= budget - 4.0


def test_or_fusion_sources_and_probability_map():
    fm = H.fm()
    img, gt, ro, table, tr = scene_pair()
    both = run(img, ro, table, ms=MS)
    # sem 만: 영상에는 얼굴이 없다
    sem_only = run(H.blank_image(), ro, table, ms=MS)
    # 템플릿 만: sem 은 전부 배경
    sem_bg, _ = H.unit_sem(*tr["grid_hw"], np.full(tr["grid_hw"], H.BG_CLS))
    ro_bg = H.make_readout(HW, sem=sem_bg)
    tpl_only = run(img, ro_bg, table, ms=MS)
    assert [len(m.faces) for m in (both, sem_only, tpl_only)] == [1, 1, 1]
    a, b, c = both.faces[0], sem_only.faces[0], tpl_only.faces[0]
    assert (a.source, b.source, c.source) == ("both", "sem", "template")
    assert b.p_template == 0.0 and b.p_sem >= 0.99 and b.p == pytest.approx(b.p_sem)
    assert c.p_sem < 0.1 and c.p_template >= fm.P_SURE and c.p == pytest.approx(c.p_template)
    hit = max(fm.detect_template(img), key=lambda t: t.score)
    assert c.box == hit.box and c.p_template == hit.score and a.box == hit.box
    assert a.p == pytest.approx(max(a.p_sem, a.p_template)) and a.p_sem >= 0.99
    for m in (both, sem_only, tpl_only):
        np.testing.assert_array_equal(m.prob, np.maximum(m.p_sem, m.p_template))
        assert m.prob.shape == m.grid_hw == tr["grid_hw"] and m.prob.dtype == np.float32
    assert float(tpl_only.p_sem.max()) < 0.1 and float(sem_only.p_template.max()) == 0.0
    assert sem_only.cell_px == 4.0


@pytest.mark.parametrize("p,n_faces,uncertain,n_cand", [(0.55, 1, False, 0), (0.30, 1, True, 0), (0.21, 1, True, 0),
                                                         (0.15, 0, None, 1), (0.05, 0, None, 0)])
def test_uncertain_probabilities_still_count_as_faces(p, n_faces, uncertain, n_cand):
    """0.2-0.5 는 얼굴로 취급한다(오검출의 대가는 카메라 제한뿐). 0.1-0.2 는 재판독 후보일 뿐 얼굴이 아니다."""
    img = H.blank_image()
    ro, table, tr = H.head_readout(sem_face_p=p, with_geometry=False)
    m = run(img, ro, table, ms=MS)
    assert len(m.faces) == n_faces and m.n_candidates == n_cand and m.n_reread == 0
    if n_faces:
        f = m.faces[0]
        assert f.uncertain is uncertain and f.p == pytest.approx(p, abs=0.01) and f.source == "sem"


def test_thresholds_come_from_opts():
    img = H.blank_image()
    ro, table, tr = H.head_readout(sem_face_p=0.3, with_geometry=False)
    assert len(run(img, ro, table, o=opts(accept=0.4)).faces) == 0
    assert len(run(img, ro, table, o=opts(accept=0.25)).faces) == 1
    assert run(img, ro, table, o=opts(accept=0.25, sure=0.28)).faces[0].uncertain is False
    assert run(img, ro, table, o=opts(accept=0.25, sure=0.28, cand=0.05)).n_candidates == 0


def test_instances_are_ordered_deduplicated_and_labelled():
    fm = H.fm()
    spec = [(60, 70, 50, dict(tone="light")), (62, 220, 50, dict(tone="medium"))]
    img, gts = H.scene(240, 320, spec, bg="foliage", seed=6)
    sem_boxes = list(gts) + [(170.0, 100.0, 220.0, 140.0)]                       # 영상에는 없는 sem 만의 얼굴
    ro, table = H.sem_readout_from_boxes(sem_boxes)
    m = run(img, ro, table)
    assert len(m.faces) == 3
    keys = [((f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2) for f in m.faces]
    assert keys == sorted(keys)
    assert [f.source for f in m.faces] == ["both", "both", "sem"]
    for i, a in enumerate(m.faces):                                              # 서로 겹치지 않는다
        for b in m.faces[i + 1:]:
            assert H.iou(a.box, b.box) <= fm.NMS_IOU
    sem_face = m.faces[2]
    assert H.iou(sem_face.box, sem_boxes[2]) >= 0.6                              # 칸 단위(4 px) 상자
    assert all(f.fit_status == "no_geometry" and f.yaw_deg is None and math.isinf(f.sigma_z_m) for f in m.faces)
    assert m.n_candidates == 0


def test_boxes_are_in_source_pixels_when_read_resolution_is_halved():
    img, gt, ro, table, tr = scene_pair(read_div=2, geo=True)
    assert ro.read_hw == (120, 160) and ro.src_hw == HW
    m = run(H.blank_image(), ro, table, ms=MS)
    assert m.cell_px == 8.0 and m.grid_hw == (30, 40)
    f = m.faces[0]
    ys, xs = np.nonzero(tr["mask"])
    want = (8.0 * ys.min(), 8.0 * xs.min(), 8.0 * (ys.max() + 1), 8.0 * (xs.max() + 1))
    assert f.box == pytest.approx(want, abs=0.5)
    assert f.fit_status in ("ok", "too_small")                                   # 격자가 성기면 칸이 모자랄 수 있다
    ro_sem = H.make_readout(HW, read_hw=ro.read_hw, sem=ro.outputs["sem"])
    m2 = run(img, ro_sem, table, ms=MS)
    assert len(m2.faces) == 1 and m2.faces[0].source == "both"


def test_empty_scene_gives_empty_map():
    img = H.blank_image()
    sem, table = H.unit_sem(60, 80, np.full((60, 80), H.BG_CLS))
    ro = H.make_readout(HW, sem=sem)
    m = run(img, ro, table)
    assert m.faces == () and m.specs() == () and m.n_candidates == 0 and m.n_reread == 0
    assert float(m.prob.max()) < 1e-3
    meta = m.meta()
    assert meta["n_faces"] == 0 and meta["faces"] == []
    json.dumps(meta, allow_nan=False)


# ----------------------------------------------------------------------------- 입력이 모자랄 때: 보수적으로
def test_missing_inputs_warn_and_close_the_envelope():
    fm = H.fm()
    from buc.camera.face_envelope import FaceSpec, limits, orbit_budget_deg
    img, gt, ro, table, tr = scene_pair()
    # sem_table 없음 -> 템플릿 만
    m = fm.face_map(img, ro, f_px=H.F_PX, ms=MS, opts=opts())
    assert "no_sem_table" in m.warnings and len(m.faces) == 1 and m.faces[0].source == "template"
    # face_ids 비어 있음
    m = fm.face_map(img, ro, f_px=H.F_PX, sem_table=table, ms=MS, opts=fm.FaceOpts())
    assert "no_face_ids" in m.warnings and m.faces[0].source == "template"
    # sem 헤드 없음
    ro_nosem = H.make_readout(HW, log_depth=np.zeros((1,) + tr["grid_hw"]))
    m = run(img, ro_nosem, table, ms=MS)
    assert "no_sem_head" in m.warnings and m.faces[0].source == "template"
    # use_sem=False 는 경고 없이 sem 을 끈다
    m = run(img, ro, table, ms=MS, o=opts(use_sem=False))
    assert not {"no_sem_table", "no_face_ids", "no_sem_head"} & set(m.warnings) and float(m.p_sem.max()) == 0.0
    assert m.faces[0].source == "template"
    # use_template=False
    m = run(img, ro, table, ms=MS, o=opts(use_template=False))
    assert m.faces[0].source == "sem" and float(m.p_template.max()) == 0.0
    # 깊이·법선 헤드 없음 -> 피팅 못 함: σ_z=inf, 봉투 닫힘(α_max=0), 정면 가정 FaceSpec
    ro_sem = H.make_readout(HW, sem=ro.outputs["sem"])
    m = run(img, ro_sem, table, ms=MS)
    f = m.faces[0]
    assert "no_geometry" in m.warnings and f.fit_status == "no_geometry" and f.fit is None
    assert f.yaw_deg is None and f.pitch_deg is None and math.isinf(f.sigma_z_m)
    assert f.category == "F3" and f.alpha_max_deg == 0.0 == limits("F3", math.inf).yaw_deg
    spec = FaceSpec.from_dict(f.spec)                                           # 유효한 명세
    assert math.isinf(spec.sigma_z) and spec.h_f == pytest.approx(f.h_f)
    ray = np.array([(0.5 * (f.box[1] + f.box[3]) - H.pp_of(*HW)[1]) / H.F_PX,
                    (0.5 * (f.box[0] + f.box[2]) - H.pp_of(*HW)[0]) / H.F_PX, 1.0])
    assert float(spec.front @ (-ray / np.linalg.norm(ray))) > 0.99              # 카메라를 정면으로 본다고 가정
    assert float(np.linalg.norm(spec.center - ray * spec.center[2])) < 1e-6      # 상자 중심 광선 위
    assert spec.center[2] == pytest.approx(H.F_PX * 0.15 / (f.box[3] - f.box[1]), rel=0.02)   # 머리 폭 15 cm 사전
    assert orbit_budget_deg(spec, np.zeros(3), +1) < 0.5


def test_garbage_geometry_fails_the_fit_conservatively():
    fm = H.fm()
    img, gt, ro, table, tr = scene_pair(geo=True)
    rng = np.random.default_rng(5)
    v = rng.normal(size=(3,) + tr["grid_hw"])
    v[2] = -np.abs(v[2]) - 0.1
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    ro_bad = H.make_readout(HW, sem=ro.outputs["sem"], log_depth=ro.outputs["log_depth"], normal=v)
    m = run(img, ro_bad, table, ms=MS)
    f = m.faces[0]
    assert f.fit_status == "bad_fit" and "fit_failed" in m.warnings
    assert f.yaw_deg is None and math.isinf(f.sigma_z_m) and f.alpha_max_deg == 0.0
    from buc.camera.face_envelope import FaceSpec, orbit_budget_deg
    assert orbit_budget_deg(FaceSpec.from_dict(f.spec), np.zeros(3), +1) < 0.5


def test_scale_source_and_sigma_hat_term():
    fm = H.fm()
    assert fm.FaceOpts().use_sigma_hat is False                                  # v0 MRH σ̂ 는 미학습
    img, gt, ro, table, tr = scene_pair(with_sigma_hat=0.01, geo=True)
    off = run(img, ro, table, ms=MS).faces[0]
    bad = run(img, ro, table, ms=MS_BAD).faces[0]                              # 신뢰 못 하는 스케일은 쓰지 않는다
    assert off.fit.scale_source == "ms" and bad.fit.scale_source == "head_prior"
    assert abs(off.yaw_deg - tr["yaw"]) <= 8.0 and abs(bad.yaw_deg - tr["yaw"]) <= 8.0
    on = run(img, ro, table, ms=MS, o=opts(use_sigma_hat=True)).faces[0]
    assert off.sigma_z_m < 0.006
    z = float(tr["center"][2])
    assert on.sigma_z_m == pytest.approx(math.hypot(off.fit.rms_depth_m, z * 0.01), rel=0.15)
    assert on.sigma_z_m > 2.5 * off.sigma_z_m
    assert on.alpha_max_deg <= off.alpha_max_deg


def test_noisy_geometry_widens_sigma_z_and_shrinks_the_envelope():
    img, gt, ro0, table, tr = scene_pair(yaw=10.0, pitch=0.0, geo=True)
    _, _, ro1, _, _ = scene_pair(yaw=10.0, pitch=0.0, geo=True, noise_d=0.02, noise_n=6.0, seed=3)
    a = run(img, ro0, table, ms=MS).faces[0]
    b = run(img, ro1, table, ms=MS).faces[0]
    assert b.sigma_z_m > 3 * a.sigma_z_m and 0.012 <= b.sigma_z_m <= 0.035
    assert b.alpha_max_deg <= a.alpha_max_deg
    from buc.camera.face_envelope import alpha_sigma_deg
    assert b.alpha_max_deg <= alpha_sigma_deg(b.sigma_z_m) + 1e-9


# ----------------------------------------------------------------------------- 출력 규약
def test_meta_is_json_and_run_is_deterministic_and_pure():
    img, gt, ro, table, tr = scene_pair()
    img0 = img.copy()
    outs0 = {k: v.clone() for k, v in ro.outputs.items()}
    st = np.random.get_state()[1].copy()
    tst = torch.get_rng_state().clone()
    a = run(img, ro, table, ms=MS)
    b = run(img, ro, table, ms=MS)
    assert a.meta() == b.meta() and a.specs() == b.specs()
    np.testing.assert_array_equal(a.prob, b.prob)
    np.testing.assert_array_equal(img, img0)
    for k, v in ro.outputs.items():
        assert torch.equal(v, outs0[k])
    np.testing.assert_array_equal(np.random.get_state()[1], st)
    assert torch.equal(torch.get_rng_state(), tst)
    meta = a.meta()
    s = json.dumps(meta, allow_nan=False)
    assert json.loads(s) == meta
    assert {"n_faces", "n_reread", "n_candidates", "cell_px", "grid_hw", "warnings", "faces"} <= set(meta)
    fm_ = meta["faces"][0]
    assert {"box", "h_f", "p", "p_sem", "p_template", "source", "uncertain", "category", "yaw_deg", "pitch_deg",
            "sigma_z_m", "alpha_max_deg", "fit_status", "reread", "scale_source"} <= set(fm_)
    assert meta["n_faces"] == 1 and meta["grid_hw"] == list(a.grid_hw) and fm_["source"] == "both"
    # 실패한 피팅 · 무제한 봉투의 inf 는 JSON 에서 null
    m2 = run(img, ro, table, ms=MS)                                          # geo=False: 피팅 없음
    j = m2.meta()["faces"][0]
    assert j["sigma_z_m"] is None and j["yaw_deg"] is None and j["scale_source"] is None
    json.dumps(m2.meta(), allow_nan=False)
    # specs() 는 raw 사전(FaceSpec.to_dict)이라 sigma_z=inf 를 그대로 둔다
    assert math.isinf(m2.specs()[0]["sigma_z"])


def test_validation_happens_before_any_work():
    fm = H.fm()
    img, gt, ro, table, tr = scene_pair()
    ok = dict(f_px=H.F_PX, sem_table=table, ms=MS, opts=opts())
    stub = H.StubReader()
    with pytest.raises(ValueError):
        fm.face_map(img[:, :, :2], ro, **ok)
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fm.face_map(bad, ro, **ok)
    with pytest.raises(TypeError):
        fm.face_map("img", ro, **ok)
    for f_px in (0.0, -5.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            fm.face_map(img, ro, **{**ok, "f_px": f_px})
    small = H.make_readout((200, 300), sem=np.zeros((1, 16, 50, 75), np.float32))        # 원본 크기 불일치
    with pytest.raises(ValueError):
        fm.face_map(img, small, **ok)
    with pytest.raises(ValueError):                                                       # reader 가 있는데 prov 없음
        fm.face_map(img, ro, reader=stub, **ok)
    for o in (dict(accept=0.6, sure=0.5), dict(cand=0.3, accept=0.2), dict(accept=0.0), dict(h_min_px=0),
              dict(face_ids=(-1,)), dict(face_ids=(True,)), dict(face_ids=(1, 1)), dict(reread_sigma=0.0),
              dict(reread_sigma=1.5), dict(sure=1.2)):
        with pytest.raises(ValueError):
            fm.FaceOpts(**o)
    with pytest.raises(ValueError):                                                       # 얼굴 id 가 표 밖
        fm.face_map(img, ro, **{**ok, "opts": fm.FaceOpts(face_ids=(99,))})
    assert stub.calls == []
    d = fm.FaceOpts()
    assert (d.accept, d.sure, d.cand, d.h_min_px, d.face_ids, d.use_template, d.use_sem, d.reread_sigma) == (
        0.2, 0.5, 0.1, 16, (), True, True, 0.25)
    with pytest.raises(TypeError):
        fm.FaceOpts(0.2)                                                                  # kw_only


# ----------------------------------------------------------------------------- H/2 재판독
def _expected_canvas(win, up):
    return tuple(16 * math.ceil(up * (b - a) / 16) for a, b in ((win[0], win[2]), (win[1], win[3])))


def test_reread_runs_once_per_face_and_feeds_the_sem_vote():
    fm = H.fm()
    img, gt, ro, table, tr = scene_pair()
    sem_bg, _ = H.unit_sem(*tr["grid_hw"], np.full(tr["grid_hw"], H.BG_CLS))
    ro_bg = H.make_readout(HW, sem=sem_bg)
    no_reader = run(img, ro_bg, table, ms=MS)
    assert no_reader.n_reread == 0 and no_reader.faces[0].p_sem < 0.1 and not no_reader.faces[0].reread
    stub = H.StubReader(sem_fn=H.face_sem_fn(table))
    m = run(img, ro_bg, table, ms=MS, reader=stub, prov=H.USER, key="faces")
    f = m.faces[0]
    assert m.n_reread == len(stub.calls) == 1
    assert f.reread and f.source == "both" and f.p_sem >= 0.99 and f.p_template >= fm.P_SURE
    win = fm.reread_window(f.box, HW)
    hc, wc = _expected_canvas(win, fm.REREAD_UP)
    assert stub.calls[0].shape == (1, 3, hc, wc)                                  # 창을 2 배로 키워 16 배수 캔버스
    assert stub.calls[0].key.startswith("faces/") and stub.calls[0].sigma == pytest.approx(0.25)
    assert f.box == no_reader.faces[0].box                                        # 상자는 그대로(템플릿)
    stub2 = H.StubReader(sem_fn=H.face_sem_fn(table))                              # 결정론: 같은 입력 -> 같은 결과
    m2 = run(img, ro_bg, table, ms=MS, reader=stub2, prov=H.USER, key="faces")
    assert m2.meta() == m.meta() and [c.shape for c in stub2.calls] == [c.shape for c in stub.calls]
    assert m.meta()["n_reread"] == 1 and m.meta()["faces"][0]["reread"] is True


def test_reread_promotes_a_candidate_that_the_h4_read_missed():
    fm = H.fm()
    img = H.blank_image()
    ro, table, tr = H.head_readout(sem_face_p=0.15)
    base = run(img, ro, table, ms=MS)
    assert base.faces == () and base.n_candidates == 1
    stub = H.StubReader(sem_fn=H.face_sem_fn(table))
    m = run(img, ro, table, ms=MS, reader=stub, prov=H.USER)
    assert m.n_candidates == 1 and m.n_reread == len(stub.calls) == 1
    assert len(m.faces) == 1
    f = m.faces[0]
    assert f.reread and f.source == "sem" and f.p_sem >= 0.99 and f.p_template == 0.0
    ys, xs = np.nonzero(tr["mask"])
    cand = (4.0 * ys.min(), 4.0 * xs.min(), 4.0 * (ys.max() + 1), 4.0 * (xs.max() + 1))
    win = fm.reread_window(cand, HW)
    assert stub.calls[0].shape == (1, 3) + _expected_canvas(win, 2)
    cy, cx = (cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2
    assert f.box[0] <= cy <= f.box[2] and f.box[1] <= cx <= f.box[3]              # 후보 중심을 품는다
    assert f.box[0] >= win[0] - 2 and f.box[1] >= win[1] - 2 and f.box[2] <= win[2] + 2 and f.box[3] <= win[3] + 2


def test_reread_failures_do_not_leave_partial_results():
    img, gt, ro, table, tr = scene_pair()

    class Boom(H.StubReader):
        def __call__(self, image, sigma, *, prov, key):
            raise RuntimeError("교사 실패")
    with pytest.raises(RuntimeError):
        run(img, ro, table, ms=MS, reader=Boom(), prov=H.USER)

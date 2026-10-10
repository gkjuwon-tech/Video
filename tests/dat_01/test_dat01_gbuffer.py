"""DAT-01 G-buffer 12ch 의미: 해석 상자 방 오라클과 대조 (깊이·법선·알베도·sem·inst·미스·레벨 마스크)."""
import numpy as np
import pytest

from buc.data import synth_render as S
from buc.invent.grammar import core as G

import dat01_helpers as H

W, HT = 64, 48


@pytest.fixture(scope="module")
def box_sample():
    prog = H.box_program(lights=H.two_lights(), matter={"floor": "tile", "wall": "brick", "ceiling": "plaster"})
    cam = H.cam_center(W, HT, fov=90.0, eye=(2.0, 1.0, 1.5))  # 뒷벽까지 3 m, 바닥·천장이 모두 보인다
    return prog, cam, S.render_gbuffer(prog, cam)


def test_shapes_dtypes(box_sample):
    _, cam, s = box_sample
    assert s.gbuf.shape == (12, HT, W) and s.gbuf.dtype == np.float32
    assert s.rgb.shape == (HT, W, 3) and s.rgb.dtype == np.float32
    assert s.depth.shape == (HT, W) and s.depth.dtype == np.float32
    assert s.sem.shape == (HT, W) and s.sem.dtype == np.int16
    assert s.inst.shape == (HT, W) and s.inst.dtype == np.int32
    assert s.flow.shape == (HT, W, 2) and s.flow.dtype == np.float32
    assert s.flow_valid.shape == (HT, W) and s.flow_valid.dtype == np.bool_
    assert np.isfinite(s.gbuf).all() and np.isfinite(s.rgb).all() and np.isfinite(s.flow).all()
    assert s.camera is cam or np.array_equal(s.camera.c2w, cam.c2w)


def test_views_are_channel_slices(box_sample):
    _, _, s = box_sample
    assert np.array_equal(s.log_depth, s.gbuf[0])
    assert np.array_equal(s.normal, s.gbuf[1:4])
    assert np.array_equal(s.albedo, s.gbuf[4:7])
    assert np.array_equal(s.sem4, s.gbuf[7:11])
    assert np.array_equal(s.level_mask, s.gbuf[11])
    assert s.normal.shape == (3, HT, W) and s.sem4.shape == (4, HT, W)


def test_depth_matches_analytic_box(box_sample):
    prog, cam, s = box_sample
    depth, ax, side, _ = H.box_oracle(cam)
    assert np.isfinite(depth).all()
    np.testing.assert_allclose(s.depth, depth, rtol=2e-5, atol=1e-6)
    # 광축 방향 뒷벽(y=4)까지 z-깊이 = 3.0
    assert s.depth[HT // 2, W // 2] == pytest.approx(3.0, abs=0.05)


def test_log_depth_channel(box_sample):
    _, _, s = box_sample
    np.testing.assert_allclose(s.log_depth, np.log(s.depth.astype(np.float64)), rtol=0, atol=3e-6)


def test_normals_match_oracle_unit_and_face_camera(box_sample):
    _, cam, s = box_sample
    depth, ax, side, _ = H.box_oracle(cam)
    sem, nw = H.face_arrays(ax, side)
    nc = H.normal_cam(nw, cam)  # (H,W,3)
    got = np.moveaxis(s.normal, 0, -1)
    np.testing.assert_allclose(got, nc, atol=2e-5)
    assert np.abs(np.linalg.norm(got, axis=-1) - 1.0).max() < 1e-5
    from buc.core import geom
    Xc = geom.unproject(cam.K, geom.pixel_grid(HT, W), s.depth.astype(np.float64), cam.k1, cam.k2)
    assert (np.sum(got * Xc, axis=-1) < 0).all(), "법선은 카메라를 향해야 한다"


def test_sem_and_one_hot_groups(box_sample):
    _, cam, s = box_sample
    _, ax, side, _ = H.box_oracle(cam)
    sem, _ = H.face_arrays(ax, side)
    assert (s.sem == sem).mean() > 0.998  # 모서리 화소의 동점 처리만 허용
    assert (s.inst == -1).all()  # 슬롯이 없다
    # 상자 방에는 구조(바닥·벽·천장)만 있다: sem4 = (1,0,0,0)
    np.testing.assert_array_equal(s.sem4[0], np.ones((HT, W), np.float32))
    np.testing.assert_array_equal(s.sem4[1:], np.zeros((3, HT, W), np.float32))
    np.testing.assert_array_equal(s.level_mask, np.ones((HT, W), np.float32))


def test_albedo_follows_matter(box_sample):
    _, cam, s = box_sample
    _, ax, side, _ = H.box_oracle(cam)
    sem, _ = H.face_arrays(ax, side)
    want = {G.SEM_FLOOR: "tile", G.SEM_WALL: "brick", G.SEM_CEILING: "plaster"}
    for sv, mat in want.items():
        m = (s.sem == sv)
        assert m.sum() > 20, (sv, mat)
        for c in range(3):
            np.testing.assert_allclose(s.albedo[c][m], S.ALBEDO[mat][c], atol=1e-6)


def test_open_wall_gives_clean_misses():
    walls = (True, True, True, False)  # 변 3 (x=0) 이 없다
    prog = H.box_program(walls=walls, lights=H.two_lights())
    cam = H.cam_center(W, HT, eye=(2.0, 2.0, 1.5), target=(0.0, 2.0, 1.5), fov=120.0)  # 열린 벽을 정면으로
    s = S.render_gbuffer(prog, cam, opts=S.SynthOpts(bg=(0.1, 0.2, 0.3)))
    depth, _, _, _ = H.box_oracle(cam, walls)
    miss = ~np.isfinite(depth)
    assert 0.2 < miss.mean() < 0.8
    assert (np.isinf(s.depth) == miss).mean() > 0.998
    m = np.isinf(s.depth)
    assert (s.sem[m] == G.SEM_VOID).all() and (s.inst[m] == -1).all()
    # 미스 화소: G-buffer 12채널 전부 0, 레벨 마스크 0, 광류 무효, rgb = bg
    assert (s.gbuf[:, m] == 0).all()
    assert not s.flow_valid[m].any() and (s.flow[m] == 0).all()
    np.testing.assert_allclose(s.rgb[m], np.broadcast_to([0.1, 0.2, 0.3], (m.sum(), 3)), atol=1e-6)
    # 맞은 화소: 레벨 마스크 1 이고 sem4 가 정확히 one-hot
    h = ~m
    assert (s.level_mask[h] == 1).all()
    np.testing.assert_array_equal(s.sem4[:, h].sum(axis=0), np.ones(h.sum(), np.float32))
    assert np.isfinite(s.gbuf).all()


@pytest.fixture(scope="module")
def furnished():
    prog = H.furnished_program()
    cam = H.cam_center(W, HT, fov=110.0)
    return prog, cam, S.render_gbuffer(prog, cam)


def test_slot_materials_and_groups(furnished):
    prog, _, s = furnished
    by_id = {sl.id: sl for sl in prog.slots}
    want_mat = {3: "marble", 5: S.DEFAULT_MATERIAL_BY_CLASS["plant"], 8: "leather"}
    for sid, mat in want_mat.items():
        m = s.inst == sid
        assert m.sum() >= 15, f"slot {sid} 가 보이지 않는다"
        assert (s.sem[m] == G.sem_id(by_id[sid].cls)).all()
        for c in range(3):
            np.testing.assert_allclose(s.albedo[c][m], S.ALBEDO[mat][c], atol=1e-6)
        grp = S.sem_group(G.sem_id(by_id[sid].cls))
        assert (s.sem4[grp][m] == 1).all() and s.sem4[:, m].sum() == m.sum()
        assert (s.depth[m] > 0.7).all() and (s.depth[m] < 1.4).all()  # 가구는 카메라와 뒷벽 사이
    # table 은 가구(2), plant 는 부속(3)
    assert (s.sem4[2][s.inst == 3] == 1).all() and (s.sem4[3][s.inst == 5] == 1).all()


def test_openings_are_painted_with_opening_albedo(furnished):
    _, _, s = furnished
    door = s.sem == G.SEM_DOOR
    win = s.sem == G.SEM_WINDOW
    assert door.sum() > 40 and win.sum() > 5
    for m, key in ((door, "door"), (win, "window")):
        for c in range(3):
            np.testing.assert_allclose(s.albedo[c][m], S.ALBEDO[key][c], atol=1e-6)
        assert (s.sem4[1][m] == 1).all() and s.sem4[:, m].sum() == m.sum()
    # 개구부는 벽 평면에 칠해진다: 깊이는 뒷벽과 같다
    assert np.allclose(s.depth[door], 2.0, atol=0.02)


def test_light_and_slot_order_do_not_change_bytes():
    import dataclasses
    prog = H.furnished_program()
    cam = H.cam_center(32, 24)
    prog2 = dataclasses.replace(prog, slots=tuple(reversed(prog.slots)), lights=tuple(reversed(prog.lights)))
    a, b = S.render_gbuffer(prog, cam), S.render_gbuffer(prog2, cam)
    for k, v in a.to_arrays().items():
        assert v.tobytes() == b.to_arrays()[k].tobytes(), k


def test_rgb_display_is_srgb_clamped(furnished):
    from buc.core import color
    _, _, s = furnished
    d = s.rgb_display()
    assert d.dtype == np.float32 and d.shape == s.rgb.shape
    assert d.min() >= 0.0 and d.max() <= 1.0
    np.testing.assert_allclose(d, color.linear_to_display(s.rgb), atol=1e-7)


def test_to_arrays_contract(furnished):
    _, cam, s = furnished
    a = s.to_arrays()
    assert set(a) == {"rgb", "gbuf", "depth", "sem", "inst", "flow", "flow_valid", "c2w", "K"}
    assert a["gbuf"].shape == (12, HT, W) and a["c2w"].shape == (4, 4) and a["K"].shape == (3, 3)
    assert a["c2w"].dtype == np.float64 and a["K"].dtype == np.float64
    np.testing.assert_allclose(a["c2w"], cam.c2w)
    np.testing.assert_allclose(a["K"], cam.K)

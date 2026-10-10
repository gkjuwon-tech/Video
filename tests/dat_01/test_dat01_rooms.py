"""DAT-01 RoomGen 방 파이프: default_camera 제약, render_room 불변식, 계보·직렬화 이음매, 속도."""
import re
import time

import numpy as np
import pytest

from buc.core import contracts as C
from buc.core import geom
from buc.data import synth_render as S
from buc.invent.grammar import core as G
from buc.invent.grammar.room import RoomGenConfig, generate_room

import dat01_helpers as H

SEEDS = (0, 1, 2, 3, 4, 5)
W, HT = 96, 72


@pytest.fixture(scope="module")
def samples():
    return {sd: S.render_room(sd, width=W, height=HT) for sd in SEEDS}


# --------------------------------------------------------------------------- default_camera
@pytest.mark.parametrize("seed", [0, 3, 7, 11])
def test_default_camera_constraints(seed):
    prog = generate_room(seed)
    cam = S.default_camera(prog, seed, width=80, height=60, fov_x_deg=70.0)
    assert (cam.width, cam.height) == (80, 60)
    np.testing.assert_allclose(cam.K, geom.intrinsics_from_fov(70.0, 80, 60), atol=1e-9)
    assert cam.k1 == 0.0 and cam.k2 == 0.0
    eye = cam.c2w[:3, 3]
    poly = np.asarray(prog.shell.corners)
    assert H.point_in_poly(eye[:2], poly)
    assert H.dist_to_boundary(eye[:2], poly) >= 0.2 - 1e-6  # 여유 사다리 0.5/0.4/0.3/0.2 m 중 하나
    for sl in prog.slots:
        if sl.cls == "rug":  # 러그 위는 허용
            continue
        fp = sl.footprint()
        assert not H.point_in_poly(eye[:2], fp), sl.cls
        assert H.dist_to_boundary(eye[:2], fp) >= 0.2 - 1e-6, sl.cls
    assert 1.0 - 1e-9 <= eye[2] <= min(1.8, prog.shell.height - 0.5) + 1e-9
    fwd, right = cam.c2w[:3, 2], cam.c2w[:3, 0]
    assert abs(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))) <= 15.0 + 1e-6  # 피치 +-15도 이내
    assert abs(right[2]) < 1e-9  # 롤 없음(수평선이 수평)
    assert geom.is_rotation(cam.c2w[:3, :3])


def test_default_camera_is_deterministic_and_varies_with_seed():
    prog = generate_room(5)
    a, b = S.default_camera(prog, 1), S.default_camera(prog, 1)
    assert np.array_equal(a.c2w, b.c2w) and np.array_equal(a.K, b.K)
    poses = {S.default_camera(prog, k).c2w.tobytes() for k in range(8)}
    assert len(poses) == 8
    assert S.default_camera(generate_room(6), 1).c2w.tobytes() != a.c2w.tobytes()
    np.random.seed(3)
    st = np.random.get_state()
    S.default_camera(prog, 2)
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1])


def test_default_camera_defaults():
    cam = S.default_camera(generate_room(2), 0)
    assert (cam.width, cam.height) == (256, 256)
    np.testing.assert_allclose(cam.K, geom.intrinsics_from_fov(75.0, 256, 256), atol=1e-9)
    with pytest.raises(ValueError):
        S.default_camera(generate_room(2), 0, width=0)
    with pytest.raises(ValueError):
        S.default_camera(generate_room(2), 0, fov_x_deg=0.0)
    with pytest.raises(TypeError):
        S.default_camera("room", 0)


# --------------------------------------------------------------------------- render_room 불변식
def test_closed_rooms_have_no_misses_and_valid_depth(samples):
    for sd, s in samples.items():
        assert np.isfinite(s.depth).all(), sd
        assert s.depth.min() >= 0.1, sd  # 카메라는 벽·가구 안쪽에 파묻히지 않는다
        assert s.depth.max() < 40.0, sd
        assert (s.level_mask == 1).all() and np.isfinite(s.gbuf).all(), sd
        assert (s.sem > 0).all() and (s.sem != G.SEM_VOID).all(), sd
        np.testing.assert_array_equal(s.sem4.sum(axis=0), np.ones((HT, W), np.float32))


def test_gbuffer_internal_consistency(samples):
    for sd, s in samples.items():
        np.testing.assert_allclose(s.log_depth, np.log(s.depth.astype(np.float64)), atol=3e-6)
        n = np.moveaxis(s.normal, 0, -1).astype(np.float64)
        assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-5, sd
        Xc = geom.unproject(s.camera.K, geom.pixel_grid(HT, W), s.depth.astype(np.float64))
        assert (np.sum(n * Xc, axis=-1) < 0).all(), sd
        # inst 와 sem 의 일치: 슬롯 화소는 sem >= 8, 셸 화소는 inst == -1
        assert ((s.inst >= 0) == (s.sem >= G.SEM_SLOT_BASE)).all(), sd
        a = np.moveaxis(s.albedo, 0, -1)
        assert (a > 0.0).all() and (a < 1.0).all(), sd


def test_inst_ids_exist_in_program_and_match_classes(samples):
    seen = 0
    for sd, s in samples.items():
        prog = generate_room(sd)
        by_id = {sl.id: sl for sl in prog.slots}
        ids = np.unique(s.inst[s.inst >= 0])
        seen += int((s.inst >= 0).mean() >= 0.02)
        for i in ids:
            assert int(i) in by_id
            assert (s.sem[s.inst == i] == G.sem_id(by_id[int(i)].cls)).all()
    # default_camera 는 가구가 보이는 시점을 우선한다(16 후보 중 슬롯 화소 >= 5% 인 첫 후보)
    assert seen >= len(samples) - 1


def test_albedo_matches_slot_materials_in_generated_rooms(samples):
    for sd, s in samples.items():
        prog = generate_room(sd)
        by_id = {sl.id: sl for sl in prog.slots}
        for i in np.unique(s.inst[s.inst >= 0]):
            sl = by_id[int(i)]
            mat = sl.material if sl.material != "default" else S.DEFAULT_MATERIAL_BY_CLASS[sl.cls]
            m = s.inst == i
            for c in range(3):
                np.testing.assert_allclose(s.albedo[c][m], S.ALBEDO[mat][c], atol=1e-6)


def test_luminance_range_is_reasonable(samples):
    for sd, s in samples.items():
        lum = float(s.rgb.mean())
        assert 0.005 < lum < 3.0, (sd, lum)  # 고정 노출의 HDR 선형값: 광원 거리 하한 0.5 m 로 폭주 방지
        assert float(s.rgb.max()) < 100.0, sd
        assert float(s.rgb.std()) > 0.005, sd  # 평평한 단색이 아니다


def test_default_flow_is_present_and_plausible(samples):
    for sd, s in samples.items():
        assert s.flow_camera is not None
        np.testing.assert_allclose(s.flow_camera.c2w[:3, 3],
                                   s.camera.c2w[:3, 3] + 0.05 * s.camera.c2w[:3, 0], atol=1e-9)
        assert s.flow_valid.mean() > 0.99, sd
        f, zb = H.flow_oracle(s.depth.astype(np.float64), s.camera, s.flow_camera)
        np.testing.assert_allclose(s.flow[s.flow_valid], f[s.flow_valid], atol=3e-3, rtol=1e-4)
        assert 0.05 < np.abs(s.flow).max() < 60.0, sd
    s = S.render_room(1, width=32, height=24, flow_dx_m=None)
    assert s.flow_camera is None and not s.flow_valid.any()


def test_render_room_is_reproducible_and_seed_sensitive():
    a = S.render_room(4, width=48, height=36)
    b = S.render_room(4, width=48, height=36)
    c = S.render_room(5, width=48, height=36)
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", a.content_hash())
    d = S.render_room(4, width=48, height=36, opts=S.SynthOpts(spp=4))
    assert d.content_hash() != a.content_hash()
    assert np.array_equal(d.gbuf, a.gbuf)  # AOV 는 spp 와 무관


def test_render_room_respects_room_cfg():
    cfg = RoomGenConfig(area_range=(30.0, 31.0), furniture_range=(6, 6), accessory_range=(0, 0))
    s = S.render_room(9, width=48, height=36, room_cfg=cfg)
    assert s.meta["program_hash"] == generate_room(9, cfg).content_hash()
    assert s.meta["program_hash"] != generate_room(9).content_hash()


def test_scene_program_lineage_in_meta_and_prov(samples):
    for sd, s in samples.items():
        prog = generate_room(sd)
        assert s.meta["program_hash"] == prog.content_hash()
        assert s.meta["seed"] == sd and s.meta["grammar"] == "room"
        assert s.meta["denoiser"] == "off" and s.meta["spp"] == 1
        assert isinstance(s.prov, C.Prov)
        assert not s.prov.taint and s.prov.ns == "engine" and not s.prov.ext
        assert all(isinstance(v, (str, int, float, bool)) for v in s.meta.values())


def test_render_room_matches_manual_pipeline():
    seed = 6
    prog = generate_room(seed)
    cam = S.default_camera(prog, seed, width=48, height=36)
    cam_b = S.translate_camera(cam, (0.05, 0.0, 0.0))
    manual = S.render_gbuffer(prog, cam, flow_to=cam_b)
    auto = S.render_room(seed, width=48, height=36)
    for k, v in manual.to_arrays().items():
        assert v.tobytes() == auto.to_arrays()[k].tobytes(), k


def test_raycast_oracle_agrees_on_generated_room(samples):
    """INV-01 raycast 를 교차 오라클로 직접 호출한 결과와 G-buffer 깊이·sem·inst·법선이 일치한다."""
    s = samples[2]
    prog = generate_room(2)
    cam = s.camera
    rc = G.raycast(prog, c2w=cam.c2w, K=cam.K, width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)
    np.testing.assert_array_equal(s.depth, rc.depth)
    np.testing.assert_array_equal(s.sem, rc.sem)
    np.testing.assert_array_equal(s.inst, rc.inst)
    nc = rc.normal.astype(np.float64) @ cam.c2w[:3, :3]
    np.testing.assert_allclose(np.moveaxis(s.normal, 0, -1), nc, atol=2e-6)


def test_speed_budget():
    S.render_room(0, width=32, height=24)  # 워밍업
    t = time.perf_counter()
    s = S.render_room(1, width=256, height=256)
    dt1 = time.perf_counter() - t
    assert s.gbuf.shape == (12, 256, 256)
    assert dt1 < 12.0, dt1
    t = time.perf_counter()
    S.render_room(2, width=128, height=96, opts=S.SynthOpts(spp=4))
    assert time.perf_counter() - t < 12.0

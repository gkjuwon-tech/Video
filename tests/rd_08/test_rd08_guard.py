"""RD-08 검증 오류, 기하 이음매(핀홀·RD-02), 앵커 헬퍼, 실제 DAT-16 표 이음매, import 위생."""
import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import rd08_helpers as H
from buc.core import geom
from buc.read import geom_cam as gc

ms = H.ms
ROOT = str(H.ROOT)
NAN, INF = float("nan"), float("inf")


# ------------------------------------------------------------------ 구성 시 검증
@pytest.mark.parametrize("kw", [
    dict(h_rel=0.0), dict(h_rel=-1.0), dict(h_rel=NAN), dict(h_rel=INF),
    dict(sigma_h=-0.1), dict(sigma_h=NAN), dict(sigma_h=INF),
    dict(q=0.0), dict(q=-0.5), dict(q=1.01), dict(q=NAN),
    dict(saliency=-0.01), dict(saliency=1.01), dict(saliency=NAN),
])
def test_sizeobs_rejects_bad_numbers(kw):
    base = dict(cls="door", h_rel=1.0)
    base.update(kw)
    with pytest.raises(ValueError):
        ms.SizeObs(**base)


@pytest.mark.parametrize("kw", [dict(cls=5), dict(cls=None), dict(h_rel="1.0"), dict(h_rel=None)])
def test_sizeobs_rejects_bad_types(kw):
    base = dict(cls="door", h_rel=1.0)
    base.update(kw)
    with pytest.raises((TypeError, ValueError)):
        ms.SizeObs(**base)


def test_sizeobs_empty_class_and_frozen_kwonly():
    with pytest.raises(ValueError):
        ms.SizeObs(cls="", h_rel=1.0)
    o = ms.SizeObs(cls="door", h_rel=1.0)
    assert (o.sigma_h, o.q, o.saliency, o.id) == (0.1, 1.0, 0.0, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.h_rel = 2.0
    with pytest.raises(TypeError):
        ms.SizeObs("door", 1.0)


@pytest.mark.parametrize("kw", [dict(sigma=0.0), dict(sigma=-1.0), dict(sigma=NAN), dict(sigma=INF),
                                dict(log_s=NAN), dict(log_s=INF), dict(name="")])
def test_anchor_rejects_bad_values(kw):
    base = dict(name="a", log_s=0.0, sigma=0.1)
    base.update(kw)
    with pytest.raises(ValueError):
        ms.Anchor(**base)


@pytest.mark.parametrize("kw", [
    dict(z_author=2.0, z_inlier=2.5), dict(z_author=2.5, z_inlier=2.5), dict(z_inlier=0.0),
    dict(min_obs_author=2), dict(sigma_reliable=0.0), dict(chi2_warn=0.0), dict(default_sigma=0.0),
    dict(default_log_s=NAN), dict(z_author=NAN),
])
def test_opts_validation(kw):
    with pytest.raises(ValueError):
        ms.ScaleOpts(**kw)
    d = ms.ScaleOpts()
    assert (d.z_author, d.z_inlier, d.min_obs_author) == (3.0, 2.5, 3)
    assert (d.sigma_reliable, d.chi2_warn, d.default_log_s, d.default_sigma) == (0.35, 4.0, 0.0, 1.5)


def test_estimate_argument_validation():
    t = H.std_table()
    good = [ms.SizeObs(cls="door", h_rel=1.0)]
    for bad in ("abc", [1], [None], 5, [good[0], "x"]):
        with pytest.raises(TypeError):
            ms.estimate_metric_scale(bad, table=t)
    with pytest.raises(TypeError):
        ms.estimate_metric_scale(good, table=t, anchors=[object()])
    with pytest.raises(TypeError):
        ms.estimate_metric_scale(good, table=t, opts={"z_author": 3})
    for slf in (-0.1, NAN, INF):
        with pytest.raises(ValueError):
            ms.estimate_metric_scale(good, table=t, sigma_log_f=slf)
    assert issubclass(ms.MetricScaleError, ValueError)


def test_result_is_frozen():
    r = ms.estimate_metric_scale([ms.SizeObs(cls="door", h_rel=1.0)], table=H.std_table())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.s = 2.0
    assert isinstance(r.fits, tuple) and isinstance(r.anchors, tuple) and isinstance(r.authored, tuple)


# ------------------------------------------------------------------ 헬퍼 수식
def test_f_px_and_rel_size_formulas():
    assert ms.f_px_from_fov(90.0, 1000) == pytest.approx(500.0, rel=1e-12)
    assert ms.f_px_from_fov(60.0, 1280) == pytest.approx(640.0 / math.tan(math.radians(30)), rel=1e-12)
    assert ms.rel_size(100.0, 6.0, 800.0) == pytest.approx(0.75, rel=1e-12)
    for args in ((0, 90), (-5, 90), (180, 90), (NAN, 90), (60, 0), (60, -5), (60, NAN)):
        with pytest.raises(ValueError):
            ms.f_px_from_fov(*args)
    for args in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1), (NAN, 1, 1), (1, INF, 1)):
        with pytest.raises(ValueError):
            ms.rel_size(*args)


def test_make_obs_fields():
    o = ms.make_obs("door", extent_px=400.0, depth_rel=5.0, f_px=1000.0, sigma_extent=0.03, sigma_depth=0.04,
                    q=0.7, saliency=0.4, id="x")
    assert o.cls == "door" and o.id == "x" and o.q == 0.7 and o.saliency == 0.4
    assert o.h_rel == pytest.approx(2.0, rel=1e-12)
    assert o.sigma_h == pytest.approx(0.05, rel=1e-12)
    d = ms.make_obs("door", extent_px=400.0, depth_rel=5.0, f_px=1000.0)
    assert (d.sigma_h, d.q, d.saliency) == (pytest.approx(math.hypot(0.05, 0.05)), 1.0, 0.0)
    with pytest.raises(ValueError):
        ms.make_obs("door", extent_px=-1.0, depth_rel=5.0, f_px=1000.0)


def test_anchor_helpers():
    a = ms.anchor_subject_distance(3.0, 6.0)
    assert isinstance(a, ms.Anchor) and a.log_s == pytest.approx(math.log(0.5), rel=1e-12)
    assert a.sigma == pytest.approx(0.10)
    assert ms.anchor_subject_distance(3.0, 6.0, sigma_log=0.2).sigma == pytest.approx(0.2)
    b = ms.anchor_camera_height(3.0)
    assert b.log_s == pytest.approx(math.log(1.5 / 3.0), rel=1e-12) and b.sigma == pytest.approx(0.30)
    c = ms.anchor_camera_height(2.0, median_m=1.2, sigma_log=0.1)
    assert c.log_s == pytest.approx(math.log(0.6), rel=1e-12) and c.sigma == pytest.approx(0.1)
    assert a.name and b.name and a.name != b.name  # 출처를 알아볼 수 있어야 한다
    for fn, args in ((ms.anchor_subject_distance, (0.0, 1.0)), (ms.anchor_subject_distance, (1.0, -1.0)),
                     (ms.anchor_camera_height, (0.0,)), (ms.anchor_camera_height, (NAN,))):
        with pytest.raises(ValueError):
            fn(*args)
    with pytest.raises(ValueError):
        ms.anchor_camera_height(1.0, sigma_log=0.0)
    # 앵커가 실제로 스케일을 정한다: 카메라 높이 1.5 m 가 상대 3.0 → s=0.5
    r = ms.estimate_metric_scale([], table=H.std_table(), anchors=[b])
    assert r.s == pytest.approx(0.5, rel=1e-12)


# ------------------------------------------------------------------ 기하 이음매
def test_pinhole_roundtrip_through_make_obs():
    """진짜 핀홀 K 로 세운 물체를 투영 -> 픽셀 범위 -> make_obs -> 스케일 복원."""
    W, Hh, fov, s_true = 1280, 720, 63.0, 0.42
    K = geom.intrinsics_from_fov(fov, W, Hh)
    f_px = ms.f_px_from_fov(fov, W)
    assert f_px == pytest.approx(K[0, 0], rel=1e-9)
    t = H.std_table()
    obs = []
    for name, z_m in (("door_interior", 3.5), ("person_adult", 6.0), ("car", 12.0), ("chair_seat", 2.2)):
        h_m = t.resolve(name).median_m
        top = K @ np.array([0.7, -h_m, z_m])
        bot = K @ np.array([0.7, 0.0, z_m])
        extent = abs(top[1] / top[2] - bot[1] / bot[2])
        obs.append(ms.make_obs(name, extent_px=float(extent), depth_rel=z_m / s_true, f_px=f_px, sigma_extent=0.02,
                               sigma_depth=0.02))
    r = ms.estimate_metric_scale(obs, table=t)
    assert r.s == pytest.approx(s_true, rel=1e-9)
    # 미터 깊이 복원: 상대 깊이 x s = 미터
    assert r.to_meters(3.5 / s_true) == pytest.approx(3.5, rel=1e-9)


def test_rd02_f_source_vocabulary_and_fov_seam():
    assert set(ms.SIGMA_LOG_F) == {"vp3", "vp2", "prior", "default"}
    assert ms.SIGMA_LOG_F["vp3"] < ms.SIGMA_LOG_F["vp2"] < ms.SIGMA_LOG_F["prior"] < ms.SIGMA_LOG_F["default"]
    assert ms.SIGMA_LOG_F == {"vp3": 0.03, "vp2": 0.05, "prior": 0.10, "default": 0.30}
    for kw, src in (({}, "default"), ({"fov_prior_deg": 40.0}, "prior")):
        g = gc.estimate_from_segments(np.zeros((0, 4)), 640, 480, **kw)
        assert g.f_source == src
        assert g.f_source in ms.SIGMA_LOG_F
        assert ms.f_px_from_fov(g.fov_x_deg, g.W) == pytest.approx(g.K[0, 0], rel=1e-9)
        r = ms.estimate_metric_scale([ms.SizeObs(cls="door", h_rel=1.0)], table=H.std_table(),
                                     sigma_log_f=ms.SIGMA_LOG_F[g.f_source])
        assert r.sigma_s >= ms.SIGMA_LOG_F[g.f_source]


# ------------------------------------------------------------------ 실제 DAT-16 표 이음매
def test_default_table_is_real_dat16_and_cached():
    t = ms.default_table()
    assert len(t) >= 480
    assert ms.default_table() is t
    for n in ("door_interior", "person_adult", "chair_seat", "stair_riser", "dining_table", "car", "bicycle"):
        assert n in t.classes
    assert t.resolve("door").name == "door_interior" and t.resolve("person").name == "person_adult"
    with pytest.raises(KeyError):
        t.resolve("definitely_not_a_class_xyz")


def test_estimate_uses_default_table_when_none_given():
    t = H.real_table()
    s_true = 1.0 / 3.0
    obs = [H.obs_at(t, n, s_true, sigma_h=0.04) for n in ("door", "person", "chair_seat", "stair_riser")]
    r = ms.estimate_metric_scale(obs)  # table=None
    assert r.s == pytest.approx(s_true, rel=1e-9)
    assert [f.name for f in r.fits] == ["door_interior", "person_adult", "chair_seat", "stair_riser"]
    assert r.reliable and r.warnings == ()
    r2 = ms.estimate_metric_scale(obs, table=t)
    assert r2.log_s == r.log_s
    # 실제 표의 앵커 클래스는 서로 다른 단위의 물체여도 같은 스케일로 일관된다(문 2.03 m 대 계단 0.17 m)
    assert max(abs(f.z) for f in r.fits) < 1e-6


def test_real_table_scale_illusion_widens_sigma():
    """기준물 없는 스케일 착각(가변 크기 클래스뿐): 확신하지 않는다."""
    t = H.real_table()
    loose = [n for n in t.names() if t.classes[n].sigma_log >= 0.3][:2]
    assert loose, "σ≥0.3 클래스가 있어야 한다"
    obs = [H.obs_at(t, n, 0.05, sigma_h=0.1) for n in loose]
    r = ms.estimate_metric_scale(obs, table=t)
    assert r.s == pytest.approx(0.05, rel=1e-9)
    assert r.sigma_s > 0.2
    assert r.reliable == (r.sigma_s <= 0.35)


# ------------------------------------------------------------------ import 위생
def test_import_is_light_and_side_effect_free():
    code = ("import sys, numpy as np; st = np.random.get_state()[1].copy(); "
            "import buc.read.metric_scale as m; "
            "bad=[x for x in ('torch','cv2','skimage','yaml','scipy') if x in sys.modules]; "
            "same = bool((np.random.get_state()[1] == st).all()); "
            "print(bad, same); sys.exit(0 if (not bad and same) else 1)")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr


def test_meta_roundtrips_json_for_real_scene():
    t = H.real_table()
    obs = [H.obs_at(t, "door", 0.7, dev=0.03, id="d"), H.obs_at(t, "person", 0.7, dev=-0.02, id="p"),
           ms.SizeObs(cls="zzz_unknown", h_rel=1.0, id="u")]
    r = ms.estimate_metric_scale(obs, table=t, sigma_log_f=0.03)
    m = json.loads(json.dumps(r.to_meta(), allow_nan=False))
    assert m["n_used"] == 2 and m["n_unknown"] == 1 and "unknown_class" in m["warnings"]

"""RD-11 상류 이음매: RD-02 GeomCamera·RD-07 MrhReadout·RD-08 MetricScale·RD-10 FitLightsResult 필드를 실제 클래스와 대조하고,
실제 estimate_metric_scale / fit_lights 출력으로 어댑터를 돌린다."""
import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import rd11_helpers as H
from buc.read import escher_score as E
from buc.read import fit_lights as F
from buc.read import geom_cam as G
from buc.read import metric_scale as MS
from buc.read import mrh_call as M


def fields(cls):
    return {f.name for f in dataclasses.fields(cls)}


# ------------------------------------------------------------------ 상류 필드 계약 (이름이 바뀌면 여기서 먼저 깨진다)
def test_upstream_fields_exist():
    assert {"r_vp_deg", "orphan_frac"} <= fields(G.GeomCamera)
    assert {"r_dis", "outputs"} <= fields(M.MrhReadout)
    assert {"fits"} <= fields(MS.MetricScale)
    assert {"status", "name", "cls", "log_size_resid"} <= fields(MS.ObsFit)
    assert {"status", "lights", "priors", "cheat_share"} <= fields(F.FitLightsResult)
    assert {"direction", "source"} <= fields(F.DirectionPrior) and {"direction", "role"} <= fields(F.Light)


def geom(r_vp=1.0, orphan=0.05):
    return SimpleNamespace(r_vp_deg=r_vp, orphan_frac=orphan)


def mrh(r_dis=0.03, log_depth=None):
    out = {} if log_depth is None else {"log_depth": log_depth}
    return SimpleNamespace(r_dis=r_dis, outputs=out)


# ------------------------------------------------------------------ geom / mrh
def test_geom_fields_pass_through_and_nan_vp_is_unmeasured():
    i = E.inputs_from_reads(geom=geom(1.5, 0.12))
    assert (i.r_vp_deg, i.orphan_frac) == (1.5, 0.12) and i.r_dis is None
    j = E.inputs_from_reads(geom=geom(float("nan"), 0.0))
    assert j.r_vp_deg is None and j.orphan_frac == 0.0
    s = E.escher_from_reads(geom=geom(float("nan"), 0.0))
    assert "vp" in s.pending and "multi_vp" not in s.pending


def test_mrh_r_dis_passes_through():
    assert E.inputs_from_reads(mrh=mrh(0.07)).r_dis == 0.07
    assert E.escher_from_reads(mrh=mrh(0.05)).component("r_dis").x == 1.0


def test_plane_from_mrh_log_depth_numpy_and_torch():
    ld = H.plane_log_depth() + np.random.default_rng(3).normal(0.0, 0.02, (64, 64))
    lab = np.zeros((64, 64), np.int64)
    want = E.plane_residual(ld, lab).rel
    i = E.inputs_from_reads(mrh=mrh(0.0, ld[None, None]), plane_labels=lab)
    assert i.plane_rel == pytest.approx(want, rel=1e-12)
    torch = pytest.importorskip("torch")
    j = E.inputs_from_reads(mrh=mrh(0.0, torch.from_numpy(ld[None, None].copy()).float()), plane_labels=lab)
    assert j.plane_rel == pytest.approx(want, rel=1e-3)


def test_plane_labels_without_mrh_or_log_depth_is_error_and_no_labels_means_unmeasured():
    lab = np.zeros((64, 64), np.int64)
    with pytest.raises(E.EscherError):
        E.inputs_from_reads(plane_labels=lab)
    with pytest.raises(E.EscherError):
        E.inputs_from_reads(mrh=mrh(0.0), plane_labels=lab)        # outputs 에 log_depth 없음
    assert E.inputs_from_reads(mrh=mrh(0.0, np.zeros((1, 1, 64, 64)))).plane_rel is None
    all_neg = -np.ones((64, 64), np.int64)
    assert E.inputs_from_reads(mrh=mrh(0.0, np.zeros((1, 1, 64, 64))), plane_labels=all_neg).plane_rel is None


# ------------------------------------------------------------------ RD-08: 실제 estimate_metric_scale
def windows(factors, cls="window_residential", base=0.35):
    return [MS.SizeObs(cls=cls, h_rel=base * f, sigma_h=0.03, q=1.0, saliency=0.0, id=f"w{i}")
            for i, f in enumerate(factors)]


def test_size_from_real_metric_scale_consistent_vs_drifting_floors():
    table = MS.default_table()
    ok = MS.estimate_metric_scale(windows([1.0, 1.0, 1.0, 1.0]), table=table)
    bad_f = [1.0, 1.2, 1.45, 1.75]                                  # 층마다 커지는 창
    bad = MS.estimate_metric_scale(windows(bad_f), table=table)
    a = E.inputs_from_reads(scale=ok).size_sd
    b = E.inputs_from_reads(scale=bad).size_sd
    assert a == pytest.approx(0.0, abs=1e-9)
    assert b == pytest.approx(H.mad_sd(np.log(bad_f)), rel=1e-9) and b > E.TAU["size"]
    s = E.escher_from_reads(scale=bad)
    assert s.component("size").x == 1.0 and s.esc == pytest.approx(0.15)


def test_size_ignores_real_authored_hero():
    """RD-08 이 저술로 표시한 주 피사체(×3)는 같은 클래스 일관성 검사에 들어가지 않는다."""
    table = MS.default_table()
    crowd = [MS.SizeObs(cls=c, h_rel=h, sigma_h=0.03, saliency=s, id=c)
             for c, h, s in (("door_interior", 2.03 * 0.8, 0.1), ("dining_table", 0.75 * 0.8, 0.12),
                             ("car", 1.5 * 0.8, 0.15), ("chair_seat", 0.45 * 0.8, 0.14))]
    hero = MS.SizeObs(cls="person_adult", h_rel=1.7 * 0.8 * 3.0, sigma_h=0.03, saliency=0.9, id="hero")
    other = MS.SizeObs(cls="person_adult", h_rel=1.7 * 0.8, sigma_h=0.03, saliency=0.2, id="bystander")
    ms = MS.estimate_metric_scale(crowd + [other, hero], table=table)
    assert [a.id for a in ms.authored] == ["hero"]
    r = E.size_residual(ms.fits)
    assert r.sd is None and "person_adult" not in r.per_class      # 남은 사람은 1명: 반복 객체 아님
    assert E.inputs_from_reads(scale=ms).size_sd is None


def test_size_unknown_classes_do_not_crash():
    table = MS.default_table()
    obs = windows([1.0, 1.1]) + [MS.SizeObs(cls="zzz_unknown_thing", h_rel=1.0, id="u")]
    ms = MS.estimate_metric_scale(obs, table=table)
    assert E.inputs_from_reads(scale=ms).size_sd is not None


# ------------------------------------------------------------------ RD-10: 빛
KEY = H.unit([-0.5, -0.5, -0.7])


def prior(d, src="shadow", sigma=0.1):
    return F.DirectionPrior(direction=tuple(H.unit(d)), sigma_rad=sigma, source=src)


def key_light(d=KEY):
    return F.Light(name="key", role="motivated", kind="sun", direction=tuple(H.unit(d)), intensity=0.6)


def cheat_light(d):
    return F.Light(name="cheat1", role="cheat", kind="area", direction=tuple(H.unit(d)), intensity=0.3,
                   angular_radius=0.25)


def fit_stub(status="ok", lights=(), priors=(), cheat_share=0.0):
    return SimpleNamespace(status=status, lights=tuple(lights), priors=tuple(priors), cheat_share=cheat_share)


def test_light_cue_dirs_key_plus_shadow_priors_only():
    pr = [prior([0.2, -0.6, -0.7]), prior([1, 0, 0], src="user"), prior([0, 1, 0], src="probe")]
    fit = fit_stub(lights=[key_light(), cheat_light([0.75, -0.2, 0.63])], priors=pr)
    cues = E.light_cue_dirs(fit)
    assert len(cues) == 2
    assert np.allclose(cues[0], KEY, atol=1e-12) and np.allclose(cues[1], H.unit([0.2, -0.6, -0.7]), atol=1e-12)


def test_light_cue_dirs_fallback_bake_uses_priors_only_and_no_lights_is_empty():
    fb = fit_stub(status="fallback_bake", lights=(), priors=[prior([0.2, -0.6, -0.7])])
    assert len(E.light_cue_dirs(fb)) == 1
    assert E.light_cue_dirs(fit_stub()) == []
    fb2 = fit_stub(status="fallback_bake", lights=[key_light()], priors=())          # 폴백이면 광원은 믿지 않는다
    assert E.light_cue_dirs(fb2) == []


def test_light_cue_dirs_uses_first_motivated_light_even_if_not_listed_first():
    fit = fit_stub(lights=[cheat_light([0, 0, 1]), key_light()])
    cues = E.light_cue_dirs(fit)
    assert len(cues) == 1 and np.allclose(cues[0], KEY, atol=1e-12)


def test_light_cue_dirs_takes_only_the_first_motivated_light():
    fill = F.Light(name="fill", role="motivated", kind="area", direction=tuple(H.unit([0.9, -0.1, -0.4])), intensity=0.2)
    cues = E.light_cue_dirs(fit_stub(lights=[key_light(), fill]))
    assert len(cues) == 1 and np.allclose(cues[0], KEY, atol=1e-12)


def test_lights_adapter_sd_cheat_and_extra_dirs():
    sh = H.rot_about([0, 0, 1], 25.0, KEY)
    fit = fit_stub(lights=[key_light()], priors=[prior(sh)], cheat_share=0.2)
    i = E.inputs_from_reads(lights=fit)
    assert i.cheat_share == 0.2 and i.light_sd_deg == pytest.approx(H.csd([KEY, H.unit(sh)]), rel=1e-9)
    extra = [H.rot_about([0, 0, 1], -20.0, KEY)]
    j = E.inputs_from_reads(lights=fit, light_dirs=extra)
    assert j.light_sd_deg == pytest.approx(H.csd([KEY, H.unit(sh), extra[0]]), rel=1e-9)


def test_lights_adapter_single_cue_means_sd_zero_but_measured():
    i = E.inputs_from_reads(lights=fit_stub(lights=[key_light()], cheat_share=0.1))
    assert i.light_sd_deg == 0.0 and i.cheat_share == 0.1
    assert "light" not in E.escher_from_reads(lights=fit_stub(lights=[key_light()])).pending


def test_extra_light_dirs_without_fit_and_neither_given():
    d = [KEY, H.rot_about([0, 0, 1], 30.0, KEY)]
    i = E.inputs_from_reads(light_dirs=d)
    assert i.light_sd_deg == pytest.approx(H.csd(d), rel=1e-9) and i.cheat_share is None
    assert E.inputs_from_reads(light_dirs=[KEY]).light_sd_deg is None               # 단서 1개, 적합도 없음 -> 모른다
    k = E.inputs_from_reads()
    assert k.light_sd_deg is None and k.cheat_share is None


def test_cheat_excess_from_fit_raises_light_residual():
    s = E.escher_from_reads(lights=fit_stub(lights=[key_light()], cheat_share=0.7))
    assert s.component("light").r == pytest.approx(15.0) and s.component("light").x == 1.0


def test_real_fit_lights_end_to_end_consistent_and_contradicting_shadow():
    n, valid = H.ball()
    shading = 0.15 + 0.6 * np.maximum(n @ KEY, 0.0)
    fit = F.fit_lights(shading, n, valid=valid)
    assert fit.status == "ok"
    consistent = E.escher_from_reads(lights=fit)
    assert consistent.component("light").measured and consistent.component("light").r == pytest.approx(0.0, abs=1e-6)
    # 그림자 단서가 음영과 다른 방향(방위 60° 회전)을 가리킨다: 약한 사전분포는 자료에 진다 -> 단서 둘이 어긋난다
    off = prior(H.rot_about([0, 1, 0], 60.0, KEY), sigma=0.5)
    fit2 = F.fit_lights(shading, n, valid=valid, direction_priors=[off])
    cues = E.light_cue_dirs(fit2)
    assert len(cues) == 2
    bad = E.escher_from_reads(lights=fit2)
    assert bad.component("light").r == pytest.approx(H.csd(cues), rel=1e-9) and bad.component("light").r > 10.0
    assert bad.esc > consistent.esc


# ------------------------------------------------------------------ 전부 합쳐서
def test_escher_from_reads_equals_manual_inputs_and_meta_is_json():
    kw = dict(geom=geom(2.5, 0.3), mrh=mrh(0.08), scale=MS.estimate_metric_scale(windows([1.0, 1.0, 1.0]),
                                                                                table=MS.default_table()),
              lights=fit_stub(lights=[key_light()], cheat_share=0.0), topo_count=3, ffg_px1080=None, refl_px1080=2.0)
    s = E.escher_from_reads(**kw)
    i = E.inputs_from_reads(**kw)
    assert s.inputs == i and s == E.escher_score(i)
    want = H.oracle_esc({"vp": 2.5, "multi_vp": 0.3, "size": 0.0, "plane": None, "light": 0.0, "topo": 3.0,
                         "reconcile": None, "r_dis": 0.08, "r_refl": 2.0})[0]
    assert s.esc == pytest.approx(want, abs=1e-12)
    assert s.pending == ("plane", "reconcile")
    json.dumps(s.to_meta(), allow_nan=False)
    assert math.isfinite(s.esc)


def test_escher_from_reads_all_empty_is_zero_everything_pending():
    s = E.escher_from_reads()
    assert s.esc == 0.0 and s.pending == H.ORDER


def test_escher_from_reads_forwards_tau_scale():
    kw = dict(geom=geom(1.0, 0.05), mrh=mrh(0.025))
    assert E.escher_from_reads(**kw, tau_scale=2.0).esc < E.escher_from_reads(**kw).esc

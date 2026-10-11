"""WLD-08 채움 품질·불변식: 해석적 가림 오라클, 정답 오라클 교사, 시각 대리 지표, 관측 불변."""
import json
import warnings

import numpy as np
import torch
from scipy import ndimage as ndi

from buc.core import color
from buc.core import contracts as C
from buc.core import geom
from buc.render.quads import render_ldi
from buc.world.strata import OWNERS

import wld08_helpers as h

S = h.mod()


# --------------------------------------------------------------------------- 정확도 (오라클 교사)

def test_fill_matches_ground_truth_and_beats_nearest_copy():
    r = h.oracle_run()
    dw, res = r.dw, r.res
    gt = h.hidden_gt_linear(dw)
    old = dw.world.layers[1].rgba[..., :3].astype(np.float64)
    new = res.world.layers[1].rgba[..., :3].astype(np.float64)
    f = res.filled
    assert f.sum() >= 0.6 * dw.hidden.sum(), "±20° 뷰로 숨은 텍셀의 60% 이상이 채워진다"
    e_new = np.abs(new[f] - gt[f])
    e_old = np.abs(old[f] - gt[f])
    assert e_old.mean() >= 0.03, "임시 최근접 복사는 확실히 틀려야 시험이 의미가 있다"
    # 쌍선형 표본(계약)은 평균 4e-4 수준; 최근접 표본(평균 4e-3)이나 반화소 어긋남(1e-2)은 여기서 걸린다
    assert e_new.mean() <= 0.002
    assert np.percentile(e_new, 95) <= 0.005
    assert e_new.max() <= 0.01
    assert e_new.mean() <= 0.05 * e_old.mean()


def test_black_box_color_does_not_leak_into_fill():
    """교사 출력의 상자를 검정으로 칠해도(입력은 빨강) 숨은 텍셀 색은 가림 해제 영역에서만 읽으므로 번지지 않는다."""
    r = h.oracle_run()
    dw, res = r.dw, r.res
    gt2 = h.gt_views_display(dw.world.obs, r.views, box_rgb=(0.0, 0.0, 0.0))
    res2 = S.sdedit_avg_fill(dw, h.FakeSampler(h.oracle_transform(gt2)), cfg=r.cfg, views=r.views, seed=11)
    gt = h.hidden_gt_linear(dw)
    f = res2.filled
    assert f.sum() == res.filled.sum()
    e = np.abs(res2.world.layers[1].rgba[..., :3][f].astype(np.float64) - gt[f])
    assert e.max() <= 0.01, e.max()


def test_unplanned_view_visual_proxy():
    """계획 뷰 사이/바깥(±20° 이내)의 카메라로 찍었을 때 화면에 보이는 가림 해제 띠가 정답에 가깝다."""
    r = h.oracle_run()
    dw, res = r.dw, r.res
    gw = h.gt_world(dw.world.obs)
    for ang in (15.0, -17.0, 6.0):
        v = next(x for x in S.plan_views(dw, S.SDEditAvgConfig(max_angle_deg=abs(ang), n_per_side=1))
                 if abs(x.angle_deg - ang) < 1e-9)
        kw = dict(c2w=v.c2w, K=v.K, width=v.width, height=v.height)
        before = render_ldi(dw.world, **kw)
        after = render_ldi(res.world, **kw)
        truth = render_ldi(gw, **kw)
        core = ndi.binary_erosion(before.layer_id == 1, structure=np.ones((3, 3), bool), iterations=2,
                                  border_value=0)
        assert core.sum() >= 40, (ang, core.sum())
        e0 = np.abs(before.rgb[core] - truth.rgb[core]).mean()
        e1 = np.abs(after.rgb[core] - truth.rgb[core]).mean()
        assert e0 >= 0.025, (ang, e0)
        assert e1 <= 0.004, (ang, e1)
        assert e1 <= 0.1 * e0
        assert np.array_equal(after.layer_id, before.layer_id), "채움은 어느 화소가 어느 층인지를 바꾸지 않는다"
        assert np.array_equal(after.depth, before.depth) and np.array_equal(after.alpha, before.alpha)


# --------------------------------------------------------------------------- 가림 오라클 (구현과 독립)

def test_votes_agree_with_analytic_occlusion_oracle():
    r = h.oracle_run()
    dw, res = r.dw, r.res
    (vv, uu), vis, occ = h.box_visibility(dw, r.views)
    votes = res.votes[vv, uu].astype(int)
    n_vis, n_notocc = vis.sum(0), (~occ).sum(0)
    assert (votes >= n_vis).all(), "분명히 보이는 뷰마다 투표해야 한다"
    assert (votes <= n_notocc).all(), "분명히 가려진 뷰는 투표할 수 없다"
    clear = ~(~(vis | occ)).any(0)
    assert clear.sum() >= 20, "오라클 시험의 검정력"
    assert np.array_equal(votes[clear], n_vis[clear]), "모호한 뷰가 없는 텍셀은 투표 수가 정확히 같다"
    assert occ.any(), "상자에 가려지는 (뷰, 텍셀) 쌍이 실제로 있어야 시험이 의미가 있다"
    # 투표가 0인 텍셀은 분명히 보이는 뷰가 없는 텍셀뿐이다
    assert (votes[vis.any(0)] >= 1).all()
    assert (res.filled[vv, uu] == (votes >= 1)).all()


def test_unfilled_texels_keep_their_color_exactly():
    r = h.oracle_run()
    old = r.dw.world.layers[1].rgba
    new = r.res.world.layers[1].rgba
    keep = ~r.res.filled
    assert np.array_equal(old[keep], new[keep])
    assert r.res.unfilled.sum() >= 1, "중앙/가장자리 미관측 텍셀이 있다(1뷰도 못 본 곳을 지어내지 않는다)"
    changed = np.any(old != new, axis=-1)
    assert changed.any() and not (changed & ~r.res.filled).any()


# --------------------------------------------------------------------------- 불변식

def test_observed_regions_are_bit_identical_and_obs_render_unchanged():
    r = h.oracle_run()
    a, b = r.dw.world, r.res.world
    assert h.layer_bytes_equal(a.layers[0], b.layers[0])
    assert np.array_equal(a.layers[1].rgba[..., 3], b.layers[1].rgba[..., 3])
    assert np.array_equal(a.layers[1].depth, b.layers[1].depth)
    nh = ~r.dw.hidden
    assert np.array_equal(a.layers[1].rgba[nh], b.layers[1].rgba[nh])
    assert len(b.layers) == 2
    for ly in b.layers:
        assert ly.rgba.dtype == np.float32 and ly.depth.dtype == np.float32
        assert ly.rgba.min() >= 0 and ly.rgba.max() <= 1 and np.isfinite(ly.rgba).all()
    # 관측 포즈로 찍으면 채움 전과 같다
    obs = a.obs
    c2w = geom.invert_pose(geom.make_pose(obs.R, obs.t))
    kw = dict(c2w=c2w, K=obs.K, width=obs.width, height=obs.height)
    ra, rb = render_ldi(a, **kw), render_ldi(b, **kw)
    assert np.abs(ra.rgb - rb.rgb).max() <= 1e-6
    assert np.array_equal(ra.alpha, rb.alpha) and np.array_equal(ra.depth, rb.depth)
    assert b.obs.content_hash() == a.obs.content_hash() and b.read_hash == a.read_hash
    assert b.theta_in.content_hash() == a.theta_in.content_hash() and b.level == a.level
    assert b.prov.taint >= a.prov.taint


def test_garbage_teacher_cannot_touch_observed_pixels():
    """교사가 영상 전체를 회색으로 망쳐도 관측 영역은 그대로이고, 숨은 텍셀만 (선형광 회색으로) 바뀐다."""
    _, dw = h.build()
    res = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.5] * 4)), seed=3)
    a, b = dw.world, res.world
    assert h.layer_bytes_equal(a.layers[0], b.layers[0])
    assert np.array_equal(a.layers[1].rgba[~dw.hidden], b.layers[1].rgba[~dw.hidden])
    gray = float(color.srgb_to_linear(0.5))
    assert res.filled.any()
    got = b.layers[1].rgba[res.filled][:, :3]
    assert np.abs(got - gray).max() <= 1e-5
    assert np.array_equal(a.layers[1].rgba[res.unfilled], b.layers[1].rgba[res.unfilled])


def test_average_is_linear_light_arithmetic_mean_of_voting_views():
    r = h.oracle_run()
    dw = r.dw
    vals = [0.2, 0.4, 0.6, 0.8]
    res = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform(vals)), cfg=r.cfg, views=r.views, seed=1)
    (vv, uu), vis, occ = h.box_visibility(dw, r.views)
    clear = ~(~(vis | occ)).any(0)
    assert clear.sum() >= 20
    lin = np.asarray([float(color.srgb_to_linear(v)) for v in vals])
    got = res.world.layers[1].rgba[vv, uu, :3].astype(np.float64)
    exp = np.array([lin[vis[:, j]].mean() for j in np.nonzero(clear)[0]])
    for c in range(3):
        assert np.abs(got[clear, c] - exp).max() <= 2e-5
    # 표시값 평균(잘못된 방식)과는 구별된다
    wrong = np.array([float(color.srgb_to_linear(np.mean(np.asarray(vals)[vis[:, j]]))) for j in np.nonzero(clear)[0]])
    multi = vis[:, clear].sum(0) >= 2
    assert multi.any() and np.abs(wrong[multi] - exp[multi]).max() > 1e-3


def test_single_view_vote_is_that_view_sample():
    """n_per_side=1 에서 한 뷰만 본 텍셀의 색은 그 뷰 상수값이다."""
    _, dw = h.build()
    cfg = S.SDEditAvgConfig(n_per_side=1)
    views = S.plan_views(dw, cfg)
    res = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.3, 0.7])), cfg=cfg, views=views, seed=0)
    one = res.filled & (res.votes == 1)
    two = res.filled & (res.votes == 2)
    assert one.any() and two.any()
    lo, hi = float(color.srgb_to_linear(0.3)), float(color.srgb_to_linear(0.7))
    px = res.world.layers[1].rgba[..., :3]
    ok1 = np.isclose(px[one][:, 0], lo, atol=2e-5) | np.isclose(px[one][:, 0], hi, atol=2e-5)
    assert ok1.all()
    assert np.allclose(px[two], (lo + hi) / 2, atol=2e-5)


def test_depth_consistency_tolerance_is_applied():
    r = h.oracle_run()
    dw, base = r.dw, r.res
    mk = lambda tol: S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.5] * 4)),  # noqa: E731
                                       cfg=S.SDEditAvgConfig(depth_tol_rel=tol), views=r.views, seed=0)
    tight, loose = mk(1e-6), mk(0.5)
    assert tight.votes.sum() <= 0.2 * base.votes.sum(), "렌더 깊이(필터링 오차 ~1e-3)와 1e-6 이내로 맞는 텍셀은 거의 없다"
    assert loose.votes.sum() >= base.votes.sum()
    assert (loose.votes >= base.votes).all() and (base.votes >= tight.votes).all(), "허용오차에 대해 단조"


def test_min_votes_and_small_angle_and_erode():
    r = h.oracle_run()
    dw, base = r.dw, r.res
    res3 = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.5] * 4)), cfg=S.SDEditAvgConfig(min_votes=3),
                             views=r.views, seed=0)
    assert np.array_equal(res3.votes, base.votes), "votes 는 min_votes 와 무관한 원 투표 수"
    assert np.array_equal(res3.filled, (base.votes >= 3) & dw.hidden)
    assert res3.filled.sum() < base.filled.sum()
    assert np.array_equal(res3.world.layers[1].rgba[res3.unfilled], dw.world.layers[1].rgba[res3.unfilled])
    # 2° 오빗: 노출 띠가 침식 반경보다 얇아 채울 것이 거의 없다. 죽지 않고 보존한다.
    small = S.SDEditAvgConfig(max_angle_deg=2.0, n_per_side=1)
    res_s = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.5, 0.5])), cfg=small, seed=0)
    assert res_s.filled.sum() <= 0.05 * dw.hidden.sum()
    assert np.array_equal(res_s.world.layers[1].rgba[res_s.unfilled], dw.world.layers[1].rgba[res_s.unfilled])
    assert res_s.meta()["filled_frac"] <= 0.05
    # 침식 0 은 더 많이 채운다(가장자리 띠 포함)
    r0 = S.sdedit_avg_fill(dw, h.FakeSampler(h.const_transform([0.5] * 4)), cfg=S.SDEditAvgConfig(erode_px=0),
                           views=r.views, seed=0)
    assert r0.filled.sum() > base.filled.sum()


def test_flat_scene_has_nothing_to_fill_and_never_calls_the_teacher():
    _, flat = h.build_flat()
    sam = h.FakeSampler(h.const_transform([0.5] * 4))
    res = S.sdedit_avg_fill(flat, sam, seed=0)
    assert sam.calls == []
    assert not res.filled.any() and not res.unfilled.any() and res.votes.max() == 0
    for a, b in zip(flat.world.layers, res.world.layers):
        assert h.layer_bytes_equal(a, b)
    m = res.meta()
    assert m["hidden_px"] == 0 and m["filled_px"] == 0 and m["filled_frac"] == 0.0
    json.dumps(m, allow_nan=False)
    assert json.dumps(res.world.meta["sdedit_avg"], sort_keys=True) == json.dumps(m, sort_keys=True)


# --------------------------------------------------------------------------- Strata / World 일관성

def test_strata_world_and_tsq_consistency():
    r = h.oracle_run()
    dw, res = r.dw, r.res
    assert res.world.content_hash() == res.strata.to_world().content_hash()
    assert res.world.meta["strata_root"] == res.strata.merkle_root().hex()
    assert json.dumps(res.world.meta["sdedit_avg"], sort_keys=True) == json.dumps(res.meta(), sort_keys=True)
    assert res.world.meta["depth_world"] == dw.world.meta["depth_world"], "WLD-03 메타는 보존된다"
    assert res.strata.merkle_root() != dw.strata.merkle_root()
    # 입력 strata 는 변하지 않았다
    assert dw.strata.merkle_root().hex() == dw.world.meta["strata_root"]
    old, new = dw.strata.tsq.columns, res.strata.tsq.columns
    assert list(old) == list(new)
    eng = old["own"] == OWNERS["ENGINE"]
    assert eng.sum() >= 50
    for k in old:
        if k == "color":
            continue
        assert np.array_equal(old[k], new[k]), k
    assert np.array_equal(old["color"][~eng], new["color"][~eng])
    obs = dw.world.obs
    pr, pc = h.project_row_pixels(obs, new["mu"][eng])
    assert dw.hidden[pr, pc].all()
    layer1 = res.world.layers[1].rgba[..., :3]
    assert np.abs(new["color"][eng] - layer1[pr, pc]).max() <= 1e-6
    changed = np.any(old["color"][eng] != new["color"][eng], axis=1)
    assert changed.any() and (res.filled[pr, pc] | ~changed).all()


def test_determinism_and_input_immutability_and_no_global_state():
    r = h.oracle_run()
    dw = r.dw
    snap = (dw.world.content_hash(), dw.strata.merkle_root(), dw.hidden.copy(), dw.z.copy())
    np_state = np.random.get_state()[1].copy()
    th_state = torch.get_rng_state().clone()
    gt = h.gt_views_display(dw.world.obs, r.views)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a = S.sdedit_avg_fill(dw, h.FakeSampler(h.oracle_transform(gt)), cfg=r.cfg, seed=11)  # views=None: 기본 계획
        b = S.sdedit_avg_fill(dw, h.FakeSampler(h.oracle_transform(gt)), cfg=r.cfg, views=r.views, seed=11)
    assert a.strata.merkle_root() == b.strata.merkle_root() == r.res.strata.merkle_root()
    assert a.world.content_hash() == r.res.world.content_hash()
    assert a.strata.to_bytes() == r.res.strata.to_bytes()
    assert (dw.world.content_hash(), dw.strata.merkle_root()) == snap[:2]
    assert np.array_equal(dw.hidden, snap[2]) and np.array_equal(dw.z, snap[3])
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.get_rng_state(), th_state)
    # 시드는 교사에게 전달될 뿐이라(오라클은 무시) 결과가 같고, 교사 호출 인자에는 반영된다
    sam = h.FakeSampler(h.oracle_transform(gt))
    S.sdedit_avg_fill(dw, sam, cfg=r.cfg, seed=12)
    assert sam.calls[0].seeds == 12


def test_filled_world_is_a_valid_world_for_downstream():
    r = h.oracle_run()
    w = r.res.world
    w.validate()
    assert C.loads(C.dumps(w)).content_hash() == w.content_hash()
    # 층 0 의 하드 에지에서 숨은 텍셀 알파가 1 로 유지되어(가림 해제에 구멍이 안 생김) 깊이 6 m
    hid = r.dw.hidden
    assert (w.layers[1].rgba[..., 3][hid] == 1.0).all()
    assert np.allclose(w.layers[1].depth[hid], h.Z_BG)

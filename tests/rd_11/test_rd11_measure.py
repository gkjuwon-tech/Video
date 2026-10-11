"""RD-11 성분 잔차 계산기: 크기(MAD), 평면(역깊이 아핀 적합), 빛 방향(원형 SD)·합성, px@1080p. 독립 오라클 대조."""
import math

import numpy as np
import pytest

import rd11_helpers as H
from buc.read import escher_score as E


# ================================================================== 크기
def test_size_residual_oracle_two_classes_and_max():
    fits = [H.fit_obj(name="A", resid=0.0), H.fit_obj(name="A", resid=0.3),
            H.fit_obj(name="B", resid=0.0), H.fit_obj(name="B", resid=0.1), H.fit_obj(name="B", resid=0.5),
            H.fit_obj(name="C", resid=2.0)]                                        # C 는 1개 -> 무시
    r = E.size_residual(fits)
    assert r.per_class["A"] == pytest.approx(1.4826 * 0.15, rel=1e-12)
    assert r.per_class["B"] == pytest.approx(1.4826 * 0.1, rel=1e-12)
    assert "C" not in r.per_class
    assert r.sd == pytest.approx(max(r.per_class.values())) == pytest.approx(0.22239, abs=1e-9)
    assert (r.n_groups, r.n_objects) == (2, 5)


def test_size_residual_excludes_authored_unknown_and_none_resid():
    base = [H.fit_obj(name="A", resid=0.0), H.fit_obj(name="A", resid=0.3)]
    ref = E.size_residual(base).sd
    noisy = base + [H.fit_obj(status="authored", name="A", resid=5.0),
                    H.fit_obj(status="unknown_class", name=None, cls="blob", resid=None),
                    H.fit_obj(status="used", name="A", resid=None)]
    r = E.size_residual(noisy)
    assert r.sd == pytest.approx(ref, rel=1e-12) and r.n_objects == 2


def test_size_residual_authored_hero_does_not_create_violation():
    """저술된 스케일(야크)은 크기 위반으로 세지 않는다: 같은 클래스 둘이 일관되면 SD=0."""
    fits = [H.fit_obj(name="person_adult", resid=0.01), H.fit_obj(name="person_adult", resid=0.01),
            H.fit_obj(status="authored", name="person_adult", resid=1.1)]
    assert E.size_residual(fits).sd == pytest.approx(0.0, abs=1e-15)


def test_size_residual_robust_to_one_outlier_and_order_invariant():
    x = [0.0, 0.0, 0.0, 0.0, 5.0]
    assert E.size_residual([H.fit_obj(resid=v) for v in x]).sd == 0.0
    rng = np.random.default_rng(2)
    vals = rng.normal(0.0, 0.2, 9)
    a = E.size_residual([H.fit_obj(resid=v) for v in vals]).sd
    b = E.size_residual([H.fit_obj(resid=v) for v in vals[::-1]]).sd
    assert a == b == pytest.approx(H.mad_sd(vals), rel=1e-12)


def test_size_residual_groups_by_name_then_cls_and_min_group():
    fits = [H.fit_obj(name="door_interior", cls="door", resid=0.0), H.fit_obj(name="door_interior", cls="interior_door",
                                                                               resid=0.4),
            H.fit_obj(name=None, cls="x", resid=0.0), H.fit_obj(name=None, cls="x", resid=0.2)]
    r = E.size_residual(fits)
    assert set(r.per_class) == {"door_interior", "x"}
    assert r.per_class["door_interior"] == pytest.approx(1.4826 * 0.2)
    r3 = E.size_residual(fits, min_group=3)
    assert r3.sd is None and r3.n_groups == 0 and r3.per_class == {}


def test_size_residual_none_when_no_repeated_class_but_zero_when_identical():
    assert E.size_residual([]).sd is None
    assert E.size_residual([H.fit_obj(name="A", resid=1.0), H.fit_obj(name="B", resid=2.0)]).sd is None
    assert E.size_residual([H.fit_obj(name="A", resid=0.7)] * 3).sd == 0.0


def test_size_residual_validation():
    with pytest.raises(E.EscherError):
        E.size_residual([], min_group=1)
    with pytest.raises(TypeError):
        E.size_residual(None)


# ================================================================== 평면
ZERO = np.zeros((64, 64), np.int64)


def test_exact_plane_has_zero_residual():
    r = E.plane_residual(H.plane_log_depth(), ZERO)
    assert r.rel == pytest.approx(0.0, abs=1e-9) and r.n_regions == 1 and r.n_pixels == 64 * 64
    assert set(r.regions) == {0} and r.regions[0] == pytest.approx(0.0, abs=1e-9)
    assert r.rel_map.shape == (64, 64) and np.nanmax(r.rel_map) < 1e-8


@pytest.mark.parametrize("abc", [(0.004, -0.003, 0.5), (0.0, 0.0, 0.3), (-0.006, 0.005, 0.6), (0.01, 0.0, 0.5)])
def test_any_affine_inverse_depth_plane_is_zero(abc):
    assert E.plane_residual(H.plane_log_depth(*abc), ZERO).rel == pytest.approx(0.0, abs=1e-9)


def test_gauge_shift_of_log_depth_is_invariant():
    ld = H.plane_log_depth() + np.random.default_rng(0).normal(0, 0.03, (64, 64))
    a = E.plane_residual(ld, ZERO).rel
    for shift in (-3.0, 0.7, 11.0):
        assert E.plane_residual(ld + shift, ZERO).rel == pytest.approx(a, rel=1e-9)


def test_noisy_plane_median_matches_noise_level():
    sigma = 0.02
    ld = H.plane_log_depth() + np.random.default_rng(3).normal(0.0, sigma, (64, 64))
    r = E.plane_residual(ld, ZERO).rel
    assert 0.011 < r < 0.016                                        # 중앙값 |ε| ≈ 0.6745 σ = 0.0135
    assert r == pytest.approx(float(np.median(H.indep_plane_rel(ld, np.ones((64, 64), bool)))), rel=1e-6)


def test_curved_surface_is_not_a_plane():
    z = 1.0 + 3.0 * (H.COL / 32.0) ** 2
    ld = np.log(z)
    r = E.plane_residual(ld, ZERO).rel
    assert r == pytest.approx(float(np.median(H.indep_plane_rel(ld, np.ones((64, 64), bool)))), rel=1e-6)
    assert r > E.TAU["plane"] * 5                                   # 0.38


def test_folded_wall_one_label_is_flagged_but_split_labels_are_clean():
    w = 0.5 - 0.012 * np.abs(H.COL)
    ld = -np.log(w)
    folded = E.plane_residual(ld, ZERO)
    assert folded.rel > E.TAU["plane"] * 3
    split = np.where(H.COL < 0, 0, 1).astype(np.int64)             # 꺾임 선에서 라벨을 나누면 각각 평면
    ok = E.plane_residual(ld, split)
    assert ok.rel == pytest.approx(0.0, abs=1e-9) and ok.n_regions == 2


def test_median_over_union_ignores_small_curved_region_and_reports_it_per_region():
    labels = np.zeros((64, 64), np.int64)
    labels[:, 48:] = 1                                              # 오른쪽 25 % 가 곡면
    ld = H.plane_log_depth()
    z = 2.0 + 1.2 * np.sin(H.COL / 1.5)                              # 물결 면(역깊이가 아핀이 아니다)
    ld = np.where(labels == 1, np.log(z), ld)
    r = E.plane_residual(ld, labels)
    assert r.n_regions == 2 and r.rel == pytest.approx(0.0, abs=1e-9)
    mask1 = labels == 1
    assert r.regions[1] == pytest.approx(float(np.median(H.indep_plane_rel(ld, mask1))), rel=1e-6)
    assert r.regions[1] > 0.1 and r.regions[0] == pytest.approx(0.0, abs=1e-9)


def test_non_plane_pixels_are_ignored_whatever_their_depth():
    labels = np.zeros((64, 64), np.int64)
    labels[:16] = -1
    ld = H.plane_log_depth().copy()
    ld[:16] = np.random.default_rng(4).normal(0.0, 5.0, (16, 64))
    r = E.plane_residual(ld, labels)
    assert r.rel == pytest.approx(0.0, abs=1e-9) and r.n_pixels == 48 * 64
    assert np.isnan(r.rel_map[:16]).all() and np.isfinite(r.rel_map[16:]).all()


def test_min_pixels_and_no_region_returns_none():
    ld = H.plane_log_depth()
    labels = -np.ones((64, 64), np.int64)
    assert E.plane_residual(ld, labels).rel is None
    labels[:3, :4] = 3                                              # 12 화소 블록
    labels[2, 3] = -1                                               # -> 11 화소
    small = E.plane_residual(ld, labels)
    assert small.rel is None and small.n_regions == 0 and small.n_pixels == 0
    labels[2, 3] = 3
    ok = E.plane_residual(ld, labels)
    assert ok.n_regions == 1 and ok.n_pixels == 12 and ok.rel == pytest.approx(0.0, abs=1e-8)
    assert E.plane_residual(ld, labels, min_pixels=13).rel is None


def test_plane_residual_accepts_1x1xhxw_arrays_and_torch_tensors():
    ld = H.plane_log_depth() + np.random.default_rng(1).normal(0, 0.01, (64, 64))
    a = E.plane_residual(ld, ZERO).rel
    assert E.plane_residual(ld[None, None], ZERO).rel == pytest.approx(a, rel=1e-12)
    torch = pytest.importorskip("torch")
    t = torch.from_numpy(ld[None, None].copy())
    assert E.plane_residual(t, ZERO).rel == pytest.approx(a, rel=1e-12)
    assert E.plane_residual(t.float(), ZERO).rel == pytest.approx(a, rel=1e-4)


def test_plane_residual_inputs_untouched_and_deterministic():
    ld = H.plane_log_depth() + np.random.default_rng(1).normal(0, 0.01, (64, 64))
    lab = ZERO.copy()
    ld0, lab0 = ld.copy(), lab.copy()
    a, b = E.plane_residual(ld, lab), E.plane_residual(ld, lab)
    assert a.rel == b.rel and np.array_equal(a.rel_map, b.rel_map, equal_nan=True)
    assert np.array_equal(ld, ld0) and np.array_equal(lab, lab0)


def test_plane_residual_validation():
    ld = H.plane_log_depth()
    with pytest.raises(TypeError):
        E.plane_residual(ld, ZERO.astype(np.float64))
    with pytest.raises(TypeError):
        E.plane_residual(ld, ZERO.astype(bool))
    with pytest.raises(E.EscherError):
        E.plane_residual(ld, np.zeros((64, 63), np.int64))
    with pytest.raises(E.EscherError):
        E.plane_residual(np.stack([ld, ld]), ZERO)                  # (2,H,W)
    with pytest.raises(E.EscherError):
        E.plane_residual(ld[0], ZERO[0])                            # 1-D
    with pytest.raises(E.EscherError):
        E.plane_residual(ld, ZERO, min_pixels=2)


# ================================================================== 빛 방향 원형 SD
def test_circular_sd_identical_and_single_are_zero():
    assert E.circular_sd_deg([[0.3, -0.5, -0.8]] * 4) == pytest.approx(0.0, abs=1e-6)
    assert E.circular_sd_deg([[0.3, -0.5, -0.8]]) == 0.0
    assert E.circular_sd_deg(np.zeros((0, 3))) == 0.0


@pytest.mark.parametrize("half", [1.0, 5.0, 10.0, 30.0, 60.0])
def test_circular_sd_two_symmetric_directions_closed_form(half):
    d = [[math.sin(math.radians(half)), 0.0, -math.cos(math.radians(half))],
         [-math.sin(math.radians(half)), 0.0, -math.cos(math.radians(half))]]
    want = math.degrees(math.sqrt(-2.0 * math.log(math.cos(math.radians(half)))))
    assert E.circular_sd_deg(d) == pytest.approx(want, rel=1e-9)
    assert want == pytest.approx(half, rel=0.02 * (1 + half / 10))  # 작은 각에서 SD ≈ 반각


def test_circular_sd_matches_oracle_random_weights_and_scale_rotation_order_invariance():
    rng = np.random.default_rng(7)
    d = rng.normal(0, 1, (6, 3)) + np.array([0.0, -1.0, -3.0])
    w = rng.random(6) + 0.1
    base = E.circular_sd_deg(d, w)
    assert base == pytest.approx(H.csd(d, w), rel=1e-9)
    assert E.circular_sd_deg(d * 7.3, w) == pytest.approx(base, rel=1e-9)             # 길이 무관
    assert E.circular_sd_deg(d[::-1], w[::-1]) == pytest.approx(base, rel=1e-9)
    assert E.circular_sd_deg(d, w * 5.0) == pytest.approx(base, rel=1e-9)             # 가중치 스케일 무관
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert E.circular_sd_deg(d @ q.T, w) == pytest.approx(base, rel=1e-9)             # 회전 불변
    assert E.circular_sd_deg(list(map(list, d)), list(w)) == pytest.approx(base, rel=1e-9)


def test_circular_sd_duplicate_cues_do_not_change_it_and_weights_matter():
    a, b = [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]
    base = E.circular_sd_deg([a, b])
    assert E.circular_sd_deg([a, b, a, b]) == pytest.approx(base, rel=1e-9)
    heavy = E.circular_sd_deg([a, b], [9.0, 1.0])
    assert heavy < base and heavy == pytest.approx(H.csd([a, b], [9, 1]), rel=1e-9)


def test_circular_sd_opposite_is_capped_at_180():
    assert E.circular_sd_deg([[0, 0, -1], [0, 0, 1]]) == pytest.approx(180.0)
    assert 0.0 <= E.circular_sd_deg(np.random.default_rng(0).normal(size=(50, 3))) <= 180.0


def test_circular_sd_validation():
    with pytest.raises(E.EscherError):
        E.circular_sd_deg([[0, 0, 0], [0, 0, 1]])                    # 영벡터
    with pytest.raises(E.EscherError):
        E.circular_sd_deg([[0, 1], [1, 0]])                          # 모양
    with pytest.raises(E.EscherError):
        E.circular_sd_deg([[0, 0, 1], [0, 1, 0]], [1.0])             # 가중치 길이
    with pytest.raises(E.EscherError):
        E.circular_sd_deg([[0, 0, 1], [0, 1, 0]], [1.0, -1.0])
    with pytest.raises(E.EscherError):
        E.circular_sd_deg([[0, 0, 1], [0, float("nan"), 0]])


def test_two_suns_cue_set_is_far_above_tolerance_while_one_sun_is_not():
    sun = H.unit([-0.5, -0.5, -0.7])
    one = [sun, H.rot_about([0, 0, 1], 3.0, sun), H.rot_about([0, 0, 1], -2.0, sun)]
    two = [sun, H.rot_about([0, 1, 0], 90.0, sun)]
    assert E.circular_sd_deg(one) < 4.0 and E.circular_sd_deg(two) > E.TAU["light"]


# ================================================================== 빛 합성
@pytest.mark.parametrize("sd,share,want", [(0.0, 0.0, 0.0), (6.0, 0.35, 6.0), (6.0, 0.1, 6.0),
                                           (6.0, 0.7, 21.0), (0.0, 0.525, 7.5), (2.0, 1.0, 2.0 + 15.0 * 0.65 / 0.35)])
def test_light_residual_deg(sd, share, want):
    assert E.light_residual_deg(sd, share) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("sd,share", [(-1.0, 0.0), (1.0, -0.1), (1.0, 1.1), (float("nan"), 0.0), (1.0, float("inf"))])
def test_light_residual_validation(sd, share):
    with pytest.raises(E.EscherError):
        E.light_residual_deg(sd, share)


# ================================================================== px@1080p
def test_px_at_1080p_scaling():
    assert E.px_at_1080p(10.0, 2160) == pytest.approx(5.0)
    assert E.px_at_1080p(10.0, 1080) == pytest.approx(10.0)
    assert E.px_at_1080p(3.0, 540) == pytest.approx(6.0)
    assert E.px_at_1080p(0.0, 720) == 0.0
    for bad in (0, -1080, float("nan")):
        with pytest.raises(E.EscherError):
            E.px_at_1080p(1.0, bad)
    with pytest.raises(E.EscherError):
        E.px_at_1080p(-1.0, 1080)


def test_rms_px1080_vectors_and_scalars():
    assert E.rms_px1080([[3.0, 4.0]], 1080) == pytest.approx(5.0)
    assert E.rms_px1080([[3.0, 4.0], [0.0, 0.0]], 1080) == pytest.approx(math.sqrt(12.5))
    assert E.rms_px1080([3.0, 4.0], 1080) == pytest.approx(math.sqrt(12.5))                   # (N,) 크기
    assert E.rms_px1080(np.array([[3.0, 4.0]]), 2160) == pytest.approx(2.5)
    assert E.rms_px1080(np.zeros((5, 2)), 1080) == 0.0
    d = np.random.default_rng(1).normal(size=(40, 2)) * 3.0
    assert E.rms_px1080(d, 720) == pytest.approx(math.sqrt(np.mean(np.sum(d ** 2, 1))) * 1.5, rel=1e-12)


def test_rms_px1080_validation():
    with pytest.raises(E.EscherError):
        E.rms_px1080([], 1080)
    with pytest.raises(E.EscherError):
        E.rms_px1080(np.zeros((0, 2)), 1080)
    with pytest.raises(E.EscherError):
        E.rms_px1080([[1.0, 2.0, 3.0]], 1080)
    with pytest.raises(E.EscherError):
        E.rms_px1080([[float("nan"), 0.0]], 1080)
    with pytest.raises(E.EscherError):
        E.rms_px1080([[1.0, 0.0]], 0)


def test_four_px_at_1080p_is_the_reflection_threshold():
    """1080p 에서 RMS 4 px 변위 = τ_refl: 4K(2160) 에서는 8 px 가 같은 점수."""
    a = E.escher_score(E.EscherInputs(r_refl_px1080=E.rms_px1080([[4.0, 0.0]] * 6, 1080)))
    b = E.escher_score(E.EscherInputs(r_refl_px1080=E.rms_px1080([[8.0, 0.0]] * 6, 2160)))
    assert a.component("r_refl").x == b.component("r_refl").x == 1.0
    assert a.component("r_refl").r == pytest.approx(b.component("r_refl").r)

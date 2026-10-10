"""RD-08 수식·불변식: 손계산 리터럴, 독립 오라클, 스케일 등변성, 순열 불변, 앵커, 공통모드 FOV, 증거 없음, 알 수 없는 클래스."""
import math
import random

import numpy as np
import pytest

import rd08_helpers as H

ms = H.ms
APPROX = dict(rel=1e-10, abs=1e-12)


def est(obs, table=None, **kw):
    return ms.estimate_metric_scale(obs, table=table or H.std_table(), **kw)


def test_hand_computed_two_objects_literal():
    # door: mu=ln2.0, sigma=.04, h=4.2, sigma_h=.03, q=1  -> a=-0.741937, v=.0025, w=400
    # chair: mu=ln.45, sigma=.06, h=.88, sigma_h=.05, q=.8 -> a=-0.670675..., v=.0061,  w=.8/.0061
    obs = [ms.SizeObs(cls="door_interior", h_rel=4.2, sigma_h=0.03, q=1.0),
           ms.SizeObs(cls="chair_seat", h_rel=0.88, sigma_h=0.05, q=0.8)]
    r = est(obs)
    a1 = math.log(2.0) - math.log(4.2)
    a2 = math.log(0.45) - math.log(0.88)
    w1, w2 = 1 / 0.0025, 0.8 / 0.0061
    ls = (w1 * a1 + w2 * a2) / (w1 + w2)
    assert r.log_s == pytest.approx(ls, **APPROX)
    assert r.s == pytest.approx(math.exp(ls), **APPROX)
    assert r.sigma_stat == pytest.approx((w1 + w2) ** -0.5, **APPROX)
    assert r.sigma_s == pytest.approx(r.sigma_stat, **APPROX)
    # 독립 리터럴(손계산): a1=-0.7419373, a2=-0.6706743, ln s*=-0.7243415, s=0.4846436, sigma=0.0433903
    assert r.log_s == pytest.approx(-0.7243415, abs=2e-7)
    assert r.s == pytest.approx(0.4846436, abs=2e-7)
    assert r.sigma_stat == pytest.approx(0.0433903, abs=2e-7)
    assert r.source == "size_prior" and r.n_used == 2 and r.n_unknown == 0
    assert r.authored == () and r.anchors == ()
    # chi2: 두 항, 자유도 1
    chi = ((a1 - ls) ** 2 / 0.0025 + (a2 - ls) ** 2 / 0.0061) / 1.0
    assert r.chi2_red == pytest.approx(chi, **APPROX)


def test_fits_fields_match_oracle():
    t = H.std_table()
    obs = [ms.SizeObs(cls="door", h_rel=4.1, sigma_h=0.02, id="d"),
           ms.SizeObs(cls="mystery_blob", h_rel=1.0, id="m"),
           ms.SizeObs(cls="Person", h_rel=3.3, sigma_h=0.1, q=0.5, id="p")]
    r = est(obs, t)
    assert [f.index for f in r.fits] == [0, 1, 2]
    assert [f.id for f in r.fits] == ["d", "m", "p"]
    assert [f.status for f in r.fits] == ["used", "unknown_class", "used"]
    assert r.fits[0].name == "door_interior" and r.fits[2].name == "person_adult"
    assert r.fits[0].dim == "height"
    f1 = r.fits[1]
    assert f1.name is None and f1.z is None and f1.log_size_resid is None
    o = H.oracle([obs[0], obs[2]], t)
    assert r.log_s == pytest.approx(o["log_s"], **APPROX)
    for f, zi in zip((r.fits[0], r.fits[2]), o["z"]):
        assert f.z == pytest.approx(zi, **APPROX)
    f0 = r.fits[0]
    assert f0.log_s_k == pytest.approx(math.log(2.0) - math.log(4.1), **APPROX)
    assert f0.sigma_k == pytest.approx(math.hypot(0.04, 0.02), **APPROX)
    assert f0.weight == pytest.approx(1 / (0.04 ** 2 + 0.02 ** 2), **APPROX)
    assert r.fits[2].weight == pytest.approx(0.5 / (0.06 ** 2 + 0.1 ** 2), **APPROX)
    for f, h in ((r.fits[0], 4.1), (r.fits[2], 3.3)):
        mu = math.log(2.0) if f.name == "door_interior" else math.log(1.7)
        assert f.log_size_resid == pytest.approx(r.log_s + math.log(h) - mu, **APPROX)
    assert r.n_used == 2 and r.n_unknown == 1
    assert "unknown_class" in r.warnings


def test_median_sized_observations_recover_scale_exactly():
    t = H.std_table()
    s_true = 0.37
    names = ["door_interior", "person_adult", "chair_seat", "dining_table", "car"]
    obs = [H.obs_at(t, n, s_true, sigma_h=0.05) for n in names]
    r = est(obs, t)
    assert r.s == pytest.approx(s_true, rel=1e-9)
    sw = sum(1 / (t.resolve(n).sigma_log ** 2 + 0.05 ** 2) for n in names)
    assert r.sigma_stat == pytest.approx(sw ** -0.5, rel=1e-10)
    assert r.chi2_red == pytest.approx(0.0, abs=1e-12)
    assert all(abs(f.z) < 1e-9 for f in r.fits)
    assert r.reliable and r.warnings == ()
    lo, hi = r.interval(2.0)
    assert lo == pytest.approx(r.s * math.exp(-2 * r.sigma_s), **APPROX)
    assert hi == pytest.approx(r.s * math.exp(+2 * r.sigma_s), **APPROX)
    assert lo < s_true < hi


def test_oracle_random_mixtures():
    t = H.std_table()
    rng = np.random.default_rng(8)
    names = list(t.classes)
    for _ in range(40):
        n = int(rng.integers(1, 7))
        obs = [ms.SizeObs(cls=names[int(rng.integers(len(names)))], h_rel=float(np.exp(rng.normal(0, 1))),
                          sigma_h=float(rng.uniform(0, 0.3)), q=float(rng.uniform(0.05, 1.0))) for _ in range(n)]
        anchors = [ms.Anchor(name=f"a{j}", log_s=float(rng.normal()), sigma=float(rng.uniform(0.05, 0.6)))
                   for j in range(int(rng.integers(0, 3)))]
        slf = float(rng.uniform(0, 0.2))
        # 저술 판정이 끼지 않게 임계를 크게 한다: 순수 LS 만 검증
        r = est(obs, t, anchors=anchors, sigma_log_f=slf, opts=ms.ScaleOpts(z_author=1e6, z_inlier=1e5))
        o = H.oracle(obs, t, anchors, slf)
        assert r.log_s == pytest.approx(o["log_s"], **APPROX)
        assert r.sigma_stat == pytest.approx(o["sigma_stat"], **APPROX)
        assert r.sigma_s == pytest.approx(o["sigma_s"], **APPROX)
        if o["chi2_red"] is None:
            assert r.chi2_red is None
        else:
            assert r.chi2_red == pytest.approx(o["chi2_red"], rel=1e-9, abs=1e-12)
        assert r.n_used == n


def test_scale_equivariance_exact_shift():
    t = H.std_table()
    base = [H.obs_at(t, "door_interior", 0.5, dev=0.03, sigma_h=0.04),
            H.obs_at(t, "person_adult", 0.5, dev=-0.05, sigma_h=0.07, q=0.7),
            H.obs_at(t, "car", 0.5, dev=0.08, sigma_h=0.05)]
    r0 = est(base, t)
    c = 3.7
    scaled = [ms.SizeObs(cls=o.cls, h_rel=o.h_rel * c, sigma_h=o.sigma_h, q=o.q) for o in base]
    r1 = est(scaled, t)
    assert r1.log_s == pytest.approx(r0.log_s - math.log(c), rel=1e-12, abs=1e-12)
    assert r1.s == pytest.approx(r0.s / c, rel=1e-12)
    assert r1.sigma_s == pytest.approx(r0.sigma_s, rel=1e-12)
    assert r1.chi2_red == pytest.approx(r0.chi2_red, rel=1e-9, abs=1e-15)
    for f0, f1 in zip(r0.fits, r1.fits):
        assert f1.z == pytest.approx(f0.z, rel=1e-9, abs=1e-12)
        assert f1.log_size_resid == pytest.approx(f0.log_size_resid, rel=1e-9, abs=1e-12)


def test_permutation_invariance():
    t = H.std_table()
    obs = [H.obs_at(t, n, 0.9, dev=d, sigma_h=0.05 + 0.01 * i, q=1.0 - 0.1 * i)
           for i, (n, d) in enumerate([("door", 0.02), ("person", -0.04), ("chair_seat", 0.05),
                                       ("dining_table", -0.01), ("car", 0.03)])]
    anchors = [ms.Anchor(name="x", log_s=-0.1, sigma=0.2)]
    r0 = est(obs, t, anchors=anchors, sigma_log_f=0.05)
    rnd = random.Random(3)
    for _ in range(6):
        idx = list(range(len(obs)))
        rnd.shuffle(idx)
        r = est([obs[i] for i in idx], t, anchors=anchors, sigma_log_f=0.05)
        assert r.log_s == pytest.approx(r0.log_s, rel=1e-12, abs=1e-14)
        assert r.sigma_s == pytest.approx(r0.sigma_s, rel=1e-12)
        assert r.chi2_red == pytest.approx(r0.chi2_red, rel=1e-10)
        # fits 는 입력 순서를 따른다
        assert [f.index for f in r.fits] == list(range(len(obs)))
        assert [f.name for f in r.fits] == [r0.fits[i].name for i in idx]


def test_negligible_confidence_object_changes_nothing():
    t = H.std_table()
    base = [H.obs_at(t, "door", 0.6, dev=0.02), H.obs_at(t, "person", 0.6, dev=-0.02)]
    r0 = est(base, t)
    wild = ms.SizeObs(cls="car", h_rel=77.0, sigma_h=0.01, q=1e-12)
    r1 = est(base + [wild], t, opts=ms.ScaleOpts(z_author=1e6, z_inlier=1e5))
    assert r1.log_s == pytest.approx(r0.log_s, rel=1e-9, abs=1e-9)
    assert r1.sigma_stat == pytest.approx(r0.sigma_stat, rel=1e-9)


def test_anchor_only_and_combination():
    t = H.std_table()
    a1 = ms.Anchor(name="exif", log_s=math.log(0.5), sigma=0.1)
    r = est([], t, anchors=[a1])
    assert r.source == "anchor" and r.n_used == 0 and r.anchors == ("exif",)
    assert r.s == pytest.approx(0.5, rel=1e-12)
    assert r.sigma_stat == pytest.approx(0.1, rel=1e-12)
    assert r.chi2_red is None
    assert r.reliable and "no_scale_evidence" not in r.warnings

    obs = [H.obs_at(t, "door", 0.4, dev=0.0, sigma_h=0.05)]  # 객체는 s=0.4 를 말한다
    a2 = ms.Anchor(name="cam", log_s=math.log(0.6), sigma=0.2)
    r2 = est(obs, t, anchors=[a1, a2])
    o = H.oracle(obs, t, [a1, a2])
    assert r2.source == "size_prior+anchor"
    assert r2.log_s == pytest.approx(o["log_s"], **APPROX)
    assert r2.sigma_stat == pytest.approx(o["sigma_stat"], **APPROX)
    assert r2.chi2_red == pytest.approx(o["chi2_red"], rel=1e-9)
    assert set(r2.anchors) == {"exif", "cam"}
    # 앵커는 q 가 없고 가중이 1/sigma^2 이다
    w_obj = 1 / (0.04 ** 2 + 0.05 ** 2)
    assert 1 / r2.sigma_stat ** 2 == pytest.approx(w_obj + 1 / 0.1 ** 2 + 1 / 0.2 ** 2, rel=1e-10)


def test_common_mode_fov_error_does_not_shrink_with_n():
    t = H.std_table()
    for n in (1, 4, 16):
        obs = [H.obs_at(t, "door", 1.0, sigma_h=0.05) for _ in range(n)]
        r0 = est(obs, t)
        r1 = est(obs, t, sigma_log_f=0.10)
        assert r1.sigma_stat == pytest.approx(r0.sigma_stat, rel=1e-12)
        assert r1.sigma_s == pytest.approx(math.hypot(r0.sigma_stat, 0.10), rel=1e-12)
        assert r1.log_s == pytest.approx(r0.log_s, abs=1e-12)
    r16 = est([H.obs_at(t, "door", 1.0, sigma_h=0.05) for _ in range(16)], t, sigma_log_f=0.10)
    assert r16.sigma_s > 0.10


def test_no_evidence_default():
    t = H.std_table()
    r = est([], t)
    assert r.source == "default" and r.n_used == 0 and r.fits == ()
    assert r.log_s == 0.0 and r.s == 1.0
    assert r.sigma_s == pytest.approx(1.5) and not r.reliable
    assert "no_scale_evidence" in r.warnings and "scale_ambiguous" in r.warnings
    assert r.chi2_red is None
    r2 = est([], t, opts=ms.ScaleOpts(default_log_s=math.log(2.0), default_sigma=0.7), sigma_log_f=0.2)
    assert r2.s == pytest.approx(2.0)
    assert r2.sigma_s == pytest.approx(math.hypot(0.7, 0.2), rel=1e-12)
    # 알 수 없는 클래스뿐이면 증거 없음과 같다
    r3 = est([ms.SizeObs(cls="xyzzy", h_rel=2.0)], t)
    assert r3.source == "default" and r3.n_unknown == 1 and r3.n_used == 0
    assert {"no_scale_evidence", "unknown_class"} <= set(r3.warnings)
    assert r3.fits[0].status == "unknown_class"


def test_unknown_class_is_excluded_not_fatal_and_aliases_resolve():
    t = H.std_table()
    good = [H.obs_at(t, "door", 0.5, dev=0.01), H.obs_at(t, "person", 0.5, dev=-0.01)]
    r_good = est(good, t)
    r_mix = est(good + [ms.SizeObs(cls="unobtainium", h_rel=5.0)], t)
    assert r_mix.log_s == pytest.approx(r_good.log_s, abs=1e-14)
    assert r_mix.n_unknown == 1 and r_mix.n_used == 2
    assert "unknown_class" in r_mix.warnings
    # 별칭: 표의 resolve 를 호출자 문자열 그대로 부른다
    t2 = H.std_table()
    est([ms.SizeObs(cls="  Human ", h_rel=3.4)], t2)
    assert "  Human " in t2.calls
    r = est([ms.SizeObs(cls="  Human ", h_rel=3.4)], t2)
    assert r.fits[0].name == "person_adult" and r.fits[0].status == "used"


def test_single_weak_object_is_unreliable_single_firm_object_is_reliable():
    t = H.std_table()
    tree = est([H.obs_at(t, "tree", 2.0, sigma_h=0.1)], t)
    assert tree.sigma_s == pytest.approx(math.hypot(0.4, 0.1), rel=1e-12)
    assert tree.s == pytest.approx(2.0, rel=1e-9)
    assert not tree.reliable and "scale_ambiguous" in tree.warnings
    door = est([H.obs_at(t, "door", 2.0, sigma_h=0.1)], t)
    assert door.sigma_s == pytest.approx(math.hypot(0.04, 0.1), rel=1e-12)
    assert door.reliable and door.warnings == ()
    assert door.chi2_red is None  # 항 1개: 자유도 0


def test_scale_inconsistent_warning_threshold():
    t = H.std_table()
    ok = [H.obs_at(t, "door", 1.0, dev=0.04), H.obs_at(t, "person", 1.0, dev=-0.06)]
    assert "scale_inconsistent" not in est(ok, t).warnings
    # 서로 다른 스케일을 말하는 두 객체: 저술 판정은 불가(2객체)이므로 경고만
    bad = [H.obs_at(t, "door", 1.0), H.obs_at(t, "person", 2.5)]
    r = est(bad, t)
    assert r.chi2_red > 4.0 and "scale_inconsistent" in r.warnings and not r.reliable
    assert r.authored == () and r.fits[0].status == r.fits[1].status == "used"
    # chi2_warn 을 올리면 경고가 사라진다
    r2 = est(bad, t, opts=ms.ScaleOpts(chi2_warn=1e6))
    assert "scale_inconsistent" not in r2.warnings


def test_to_meters_and_interval_and_meta():
    import json
    t = H.std_table()
    r = est([H.obs_at(t, "door", 0.25, dev=0.01), H.obs_at(t, "person", 0.25, dev=-0.02)], t,
            anchors=[ms.Anchor(name="cam", log_s=math.log(0.25), sigma=0.3)], sigma_log_f=0.05)
    assert r.to_meters(2.0) == pytest.approx(2.0 * r.s, rel=1e-14)
    x = np.array([[1.0, 2.0], [4.0, 8.0]])
    x0 = x.copy()
    y = r.to_meters(x)
    np.testing.assert_allclose(y, x * r.s, rtol=1e-14)
    np.testing.assert_array_equal(x, x0)
    assert not np.shares_memory(y, x)
    lo, hi = r.interval()
    assert lo < r.s < hi
    meta = r.to_meta()
    s = json.dumps(meta, allow_nan=False)
    meta2 = json.loads(s)
    for k in ("s", "log_s", "sigma_s", "sigma_stat", "source", "n_used", "n_unknown", "authored", "anchors",
              "chi2_red", "reliable", "warnings"):
        assert k in meta2
    assert meta2["s"] == pytest.approx(r.s) and meta2["anchors"] == ["cam"] and meta2["source"] == "size_prior+anchor"
    # 증거 없음·항 1개: chi2 None 도 JSON 가능
    for rr in (est([], t), est([H.obs_at(t, "door", 1.0)], t)):
        m = json.loads(json.dumps(rr.to_meta(), allow_nan=False))
        assert m["chi2_red"] is None
    assert est([], t).to_meta()["warnings"] == sorted(set(est([], t).to_meta()["warnings"]))


def test_warnings_sorted_unique_tuple():
    t = H.std_table()
    r = est([ms.SizeObs(cls="a1", h_rel=1.0), ms.SizeObs(cls="a2", h_rel=1.0)], t)
    assert isinstance(r.warnings, tuple)
    assert list(r.warnings) == sorted(set(r.warnings))


def test_deterministic_and_no_global_rng_and_inputs_untouched():
    t = H.std_table()
    obs = [H.obs_at(t, "door", 0.5, dev=0.02, id="a"), H.obs_at(t, "person", 0.5, dev=-0.03, id="b"),
           H.obs_at(t, "car", 0.5, factor=3.0, saliency=0.9, id="hero"), H.obs_at(t, "chair_seat", 0.5, id="c")]
    anchors = [ms.Anchor(name="n", log_s=math.log(0.5), sigma=0.2)]
    snap = (list(obs), list(anchors))
    np_state = np.random.get_state()[1].copy()
    py_state = random.getstate()
    r1 = est(obs, t, anchors=anchors, sigma_log_f=0.03)
    r2 = est(obs, t, anchors=anchors, sigma_log_f=0.03)
    assert r1.s == r2.s and r1.sigma_s == r2.sigma_s and r1.chi2_red == r2.chi2_red
    assert r1.to_meta() == r2.to_meta()
    assert [(f.status, f.z, f.log_size_resid) for f in r1.fits] == [(f.status, f.z, f.log_size_resid) for f in r2.fits]
    assert (obs, anchors) == snap and all(a is b for a, b in zip(obs, snap[0]))
    np.testing.assert_array_equal(np.random.get_state()[1], np_state)
    assert random.getstate() == py_state


def test_obs_may_be_any_sequence_type():
    t = H.std_table()
    obs = (H.obs_at(t, "door", 0.5), H.obs_at(t, "person", 0.5))
    assert est(obs, t).log_s == pytest.approx(est(list(obs), t).log_s, abs=1e-15)
    assert est((), t).source == "default"

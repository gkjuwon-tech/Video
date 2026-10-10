"""RD-08 저술된 스케일: 한 객체에 몰린 크기 잔차(주 피사체)만 풀이에서 빼고, 모호한 경우엔 빼지 않는다."""
import math

import pytest

import rd08_helpers as H

ms = H.ms
S_TRUE = 0.8
DEVS = {"door_interior": 0.02, "dining_table": -0.03, "car": 0.04, "chair_seat": -0.05}


def crowd(t, skip=()):
    """사전 안의 일관된 객체들(진짜 스케일 S_TRUE, 작은 편차). 가장 두드러진 것은 saliency 0.3."""
    out = []
    for i, (n, d) in enumerate(DEVS.items()):
        if n in skip:
            continue
        out.append(H.obs_at(t, n, S_TRUE, dev=d, sigma_h=0.03, saliency=0.1 + 0.05 * i, id=n))
    out.append(H.obs_at(t, "person_adult", S_TRUE, dev=0.01, sigma_h=0.03, saliency=0.2, id="bystander"))
    return out


def hero(t, factor, saliency=0.9, cls="person_adult", id="hero"):
    return H.obs_at(t, cls, S_TRUE, dev=0.0, factor=factor, sigma_h=0.03, saliency=saliency, id=id)


def est(obs, t, **kw):
    return ms.estimate_metric_scale(obs, table=t, **kw)


def test_giant_hero_is_authored_and_excluded():
    t = H.std_table()
    others = crowd(t)
    r_ref = est(others, t)
    r = est(others + [hero(t, 3.0)], t)
    assert [a.id for a in r.authored] == ["hero"] and r.authored[0].index == len(others)
    assert r.authored[0].cls == "person_adult"
    assert r.authored[0].factor == pytest.approx(3.0, rel=0.05)
    assert r.fits[-1].status == "authored" and all(f.status == "used" for f in r.fits[:-1])
    assert "authored_scale" in r.warnings and "scale_inconsistent" not in r.warnings
    # 풀이는 저술 객체 없이 푼 것과 같다
    assert r.log_s == pytest.approx(r_ref.log_s, rel=1e-12, abs=1e-12)
    assert r.sigma_stat == pytest.approx(r_ref.sigma_stat, rel=1e-12)
    assert r.chi2_red == pytest.approx(r_ref.chi2_red, rel=1e-9)
    assert r.s == pytest.approx(S_TRUE, rel=0.05) and r.n_used == len(others)
    # E_size 용 잔차: 저술 객체는 최종 s 로 계산한 큰 양수, 나머지는 작다
    assert r.fits[-1].log_size_resid == pytest.approx(math.log(3.0), abs=0.08)
    assert all(abs(f.log_size_resid) < 0.12 for f in r.fits[:-1])
    m = r.to_meta()
    assert m["authored"][0]["id"] == "hero" and m["authored"][0]["factor"] == pytest.approx(r.authored[0].factor)


def test_miniature_hero_authored_with_factor_below_one():
    t = H.std_table()
    others = crowd(t)
    r = est(others + [hero(t, 0.3, cls="car", id="tiny")], t)
    assert [a.id for a in r.authored] == ["tiny"]
    assert r.authored[0].factor == pytest.approx(0.3, rel=0.06)
    assert r.fits[-1].log_size_resid < -1.0


def test_unnoticeable_deviation_not_authored_and_no_warning():
    t = H.std_table()
    r = est(crowd(t) + [hero(t, 1.1)], t)
    assert r.authored == () and "authored_scale" not in r.warnings
    assert "scale_inconsistent" not in r.warnings
    assert all(f.status == "used" for f in r.fits)


def test_not_top_saliency_is_not_authored_but_flagged_inconsistent():
    t = H.std_table()
    others = crowd(t)
    # 더 두드러진 정상 객체가 있다
    others = [o if o.id != "bystander" else ms.SizeObs(cls=o.cls, h_rel=o.h_rel, sigma_h=o.sigma_h,
                                                         saliency=0.95, id=o.id) for o in others]
    r = est(others + [hero(t, 3.0, saliency=0.5)], t)
    assert r.authored == () and all(f.status == "used" for f in r.fits)
    assert "scale_inconsistent" in r.warnings and not r.reliable
    r_ref = est(others, t)
    assert abs(r.log_s - r_ref.log_s) > 0.1  # 편향이 실제로 남는다(그래서 경고가 필요하다)


def test_saliency_tie_or_zero_means_no_candidate():
    t = H.std_table()
    others = [o if o.id != "bystander" else ms.SizeObs(cls=o.cls, h_rel=o.h_rel, sigma_h=o.sigma_h,
                                                         saliency=0.9, id=o.id) for o in crowd(t)]
    tie = est([hero(t, 3.0, saliency=0.9)] + others, t)  # 최대 saliency 가 둘: 유일하지 않다(저술 후보를 앞에 둔다)
    assert tie.authored == () and "scale_inconsistent" in tie.warnings
    zero = est([ms.SizeObs(cls=o.cls, h_rel=o.h_rel, sigma_h=o.sigma_h, id=o.id) for o in crowd(t)]
               + [ms.SizeObs(cls="person_adult", h_rel=1.7 * 3 / S_TRUE, sigma_h=0.03, id="hero")], t)
    assert zero.authored == () and "scale_inconsistent" in zero.warnings


def test_two_violators_are_not_authored():
    t = H.std_table()
    others = crowd(t)
    r = est(others + [hero(t, 3.0, saliency=0.9, id="h1"), hero(t, 3.0, saliency=0.8, id="h2", cls="car")], t)
    assert r.authored == ()
    assert "scale_inconsistent" in r.warnings


def test_min_obs_author_boundary():
    t = H.std_table()
    two_others = [H.obs_at(t, "door_interior", S_TRUE, dev=0.01, sigma_h=0.03, saliency=0.1, id="a"),
                  H.obs_at(t, "dining_table", S_TRUE, dev=-0.01, sigma_h=0.03, saliency=0.1, id="b")]
    r3 = est(two_others + [hero(t, 3.0)], t)  # 사용 객체 3 = min_obs_author
    assert [a.id for a in r3.authored] == ["hero"]
    r2 = est(two_others[:1] + [hero(t, 3.0)], t)  # 2객체: 누가 틀렸는지 알 수 없다
    assert r2.authored == () and "scale_inconsistent" in r2.warnings
    # 모르는 클래스 객체는 세지 않는다
    unk = est(two_others[:1] + [ms.SizeObs(cls="mystery", h_rel=1.0, id="u"), hero(t, 3.0)], t)
    assert unk.authored == ()
    # 임계를 올리면 3객체여도 4 이상을 요구한다
    r_hi = est(two_others + [hero(t, 3.0)], t, opts=ms.ScaleOpts(min_obs_author=4))
    assert r_hi.authored == ()


def test_z_author_threshold_is_respected():
    t = H.std_table()
    r = est(crowd(t) + [hero(t, 3.0)], t, opts=ms.ScaleOpts(z_author=60.0, z_inlier=50.0))
    assert r.authored == () and r.fits[-1].status == "used"


def test_anchors_participate_in_leave_one_out():
    t = H.std_table()
    two = [H.obs_at(t, "door_interior", S_TRUE, dev=0.01, sigma_h=0.03, saliency=0.1, id="a")]
    anchor = ms.Anchor(name="cam", log_s=math.log(S_TRUE), sigma=0.1)
    # 객체 2 + 앵커 1(풀이 항 3): 앵커가 있어도 '사용 객체' 수는 2 이므로 저술 불가
    r = est(two + [hero(t, 3.0)], t, anchors=[anchor])
    assert r.authored == ()
    # 객체 3 + 앵커: 저술, 풀이는 앵커를 포함한 나머지 항으로
    three = two + [H.obs_at(t, "dining_table", S_TRUE, dev=-0.01, sigma_h=0.03, saliency=0.1, id="b")]
    r2 = est(three + [hero(t, 3.0)], t, anchors=[anchor])
    assert [a.id for a in r2.authored] == ["hero"]
    ref = est(three, t, anchors=[anchor])
    assert r2.log_s == pytest.approx(ref.log_s, rel=1e-12, abs=1e-12)
    assert r2.source == "size_prior+anchor"


def test_authored_does_not_leak_into_uncertainty():
    t = H.std_table()
    others = crowd(t)
    r_ref = est(others, t, sigma_log_f=0.05)
    r = est(others + [hero(t, 4.0)], t, sigma_log_f=0.05)
    assert r.sigma_s == pytest.approx(r_ref.sigma_s, rel=1e-12)
    assert r.reliable == r_ref.reliable

"""Realify 패스 본체: sdedit 오라클 대조, NFE 수, 탭 오라클, 결정론, 뷰 독립, 청크."""
import numpy as np
import pytest
import torch

import dat02_fakes as F
from buc.data import realify as R
from buc.gov import prov as P

CAPS = ("a photo of a living room", "a photo of a kitchen", "an office, daylight")
KEYS = ("s000", "s001", "s002")


def items(n=3, img=F.smooth_image, prov=None):
    return [R.RealifyItem(key=KEYS[i], image=img(10 + i), prov=prov or F.ENGINE, caption=CAPS[i],
                          labels={"depth": np.full((F.IMG, F.IMG), 2.0 + i, np.float32)}) for i in range(n)]


def curve():
    return R.provisional_curve(F.make_rig()[1].table, 6)


def run(sampler, bus, its, **kw):
    kw.setdefault("curve", R.provisional_curve(sampler.table, 6))
    kw.setdefault("sigma", 0.5)
    kw.setdefault("seed", 7)
    kw.setdefault("token_offset", F.OFF)
    kw.setdefault("tau", 10.0)
    return R.realify(sampler, bus, its, **kw)


@pytest.mark.parametrize("sigma,req,steps", [(0.5, 2, 2), (9 / 14, 3, 3), (0.3, 2, 1), (0.62, 2, 2), (0.5, 3, 2),
                                             (0.25, 3, 1)])
def test_matches_sdedit_oracle(sigma, req, steps):
    """steps 는 요청 NFE, 유효 n = min(요청, 표 끝까지 남은 점 수)."""
    t, s, bus = F.make_rig()
    its = items()
    c0 = t.calls
    res = run(s, bus, its, sigma=sigma, steps=req)
    assert t.calls - c0 == steps
    assert [r.key for r in res] == list(KEYS)
    ref, _, img = F.oracle_sdedit(t, s, [i.image for i in its], CAPS, KEYS, sigma=sigma, steps=steps, seed=7)
    snap = s.table.snap(sigma)
    for i, r in enumerate(res):
        assert isinstance(r, R.RealifyResult)
        assert r.image.dtype == np.float32 and r.image.shape == (F.IMG, F.IMG, 3)
        assert float(r.image.min()) >= 0.0 and float(r.image.max()) <= 1.0
        np.testing.assert_allclose(r.image, img[i], atol=1e-6, rtol=0)
        assert r.sigma == snap.sigma
        assert r.steps == steps
        assert r.seed == F.item_seeds(7, [KEYS[i]])[0]
        assert r.table_key == s.table.key
        assert r.tap_sigma == s.table.points[snap.index + steps - 1]
        assert r.prov == P.derive("realify", [F.ENGINE])


@pytest.mark.parametrize("steps", [2, 3])
def test_nfe_is_exactly_steps(steps):
    t, s, bus = F.make_rig()
    c0, cap0 = t.calls, bus.stats["captures"]
    run(s, bus, items(), sigma=9 / 14, steps=steps)
    assert t.calls - c0 == steps                     # 추가 NFE 없음
    assert t.transformer.batch_sizes[-steps:] == [3] * steps
    assert bus.stats["captures"] - cap0 == 1         # 탭 캡처는 청크당 한 번


@pytest.mark.parametrize("tap_step", [-1, 0, 1, -3, 2])
def test_taps_match_direct_capture(tap_step):
    t, s, bus = F.make_rig()
    its = items()
    steps = 3
    res = run(s, bus, its, sigma=9 / 14, steps=steps, tap_step=tap_step)
    q_or, sj = F.oracle_taps(t, s, bus, [i.image for i in its], CAPS, KEYS, sigma=9 / 14, steps=steps, seed=7,
                             tap_step=tap_step)
    for i, r in enumerate(res):
        assert r.tap_sigma == sj
        ts = r.taps
        assert ts.key == KEYS[i]
        assert ts.ephemeral is False
        assert ts.grid == F.GRID
        assert ts.basis_fp == F.basis().fingerprint
        assert P.as_record(ts.prov) == r.prov
        assert sorted(ts.layers) == [8, 15, 23]
        for l in (8, 15, 23):
            assert tuple(ts.layers[l].shape) == (1, F.K, *F.GRID)
            F.assert_tap_close(ts.layers[l], q_or[l][i:i + 1])
        f = ts.features(F.basis())
        assert tuple(f[15].shape) == (1, F.K, *F.GRID)


def test_token_offset_is_used():
    t, s, bus = F.make_rig()
    its = items(1)
    a = run(s, bus, its, token_offset=F.OFF)[0].taps.layers[8]
    b = run(s, bus, its, token_offset=0)[0].taps.layers[8]
    assert not torch.equal(a, b)


def test_deterministic_bitwise():
    t, s, bus = F.make_rig()
    r1 = run(s, bus, items())
    t2, s2, bus2 = F.make_rig()
    r2 = run(s2, bus2, items())
    for a, b in zip(r1, r2):
        assert np.array_equal(a.image, b.image)
        assert a.lowband_err == b.lowband_err
        assert a.band_valid == b.band_valid
        for l in (8, 15, 23):
            assert torch.equal(a.taps.layers[l], b.taps.layers[l])
        assert a.taps.digest() == b.taps.digest()


def test_seed_and_sigma_change_output():
    t, s, bus = F.make_rig()
    a = run(s, bus, items(1), seed=1)[0]
    b = run(s, bus, items(1), seed=2)[0]
    assert float(np.abs(a.image - b.image).max()) > 1e-3
    c = run(s, bus, items(1), sigma=0.3)[0]
    assert c.sigma == 0.3 and c.tap_sigma == 0.3 and a.tap_sigma == 0.3
    d = run(s, bus, items(1), sigma=9 / 14)[0]
    assert d.tap_sigma == 0.5
    assert not torch.equal(c.taps.layers[23], d.taps.layers[23])


def test_view_independence_order_and_chunks():
    t, s, bus = F.make_rig()
    its = items()
    full = {r.key: r for r in run(s, bus, its)}
    rev = {r.key: r for r in run(s, bus, list(reversed(its)))}
    assert [r.key for r in run(s, bus, list(reversed(its)))] == list(reversed(KEYS))
    single = {k: run(s, bus, [it])[0] for k, it in zip(KEYS, its)}
    t3, s3, bus3 = F.make_rig(max_views_per_call=2)
    c0 = t3.calls
    chunked = {r.key: r for r in run(s3, bus3, its)}
    assert t3.calls - c0 == 2 * 2                    # 청크 2개 x steps 2
    assert bus3.stats["captures"] == 2
    for other in (rev, single, chunked):
        for k in KEYS:
            np.testing.assert_allclose(other[k].image, full[k].image, atol=1e-5, rtol=0)
            assert other[k].lowband_err == pytest.approx(full[k].lowband_err, abs=1e-5)
            for l in (8, 15, 23):
                F.assert_tap_close(other[k].taps.layers[l], full[k].taps.layers[l])


def test_inputs_not_mutated_and_labels_copied():
    t, s, bus = F.make_rig()
    its = items()
    before = [(i.image.copy(), {k: v.copy() for k, v in i.labels.items()}) for i in its]
    res = run(s, bus, its)
    for it, (im, lab), r in zip(its, before, res):
        assert np.array_equal(it.image, im)
        assert set(r.labels) == {"depth"}
        assert np.array_equal(r.labels["depth"], lab["depth"])
        r.labels["depth"][0, 0] = -5.0
        assert it.labels["depth"][0, 0] != -5.0


def test_curve_key_recorded_and_lowband_err_oracle():
    t, s, bus = F.make_rig()
    c = R.provisional_curve(s.table, 6)
    its = items()
    res = run(s, bus, its, curve=c, sigma=0.5)
    for it, r in zip(its, res):
        assert r.curve_key == c.key
        assert r.band_valid == c.band_mask(0.5)
        want = F.oracle_lowband_err(r.image, it.image, r.band_valid)
        assert r.lowband_err == pytest.approx(want, rel=1e-6, abs=1e-9)
        assert isinstance(r.lowband_err, float)

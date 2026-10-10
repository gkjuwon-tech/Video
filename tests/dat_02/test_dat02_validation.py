"""사전 검증: 모든 거부는 교사 호출(encode·forward·decode) 전에. 방화벽 카나리아."""
import numpy as np
import pytest

import dat02_fakes as F
from buc.data import realify as R
from buc.gov import prov as P
from buc.teacher import taps as T
from buc.teacher.sample import SigmaTable


def _items(n=2, prov=None, **over):
    out = []
    for i in range(n):
        kw = dict(key=f"k{i}", image=F.smooth_image(i), prov=(prov[i] if isinstance(prov, list) else prov) or F.ENGINE,
                  caption="room")
        kw.update(over)
        out.append(R.RealifyItem(**kw))
    return out


def _call(rig, its, **kw):
    t, s, bus = rig
    kw.setdefault("curve", R.provisional_curve(s.table, 6))
    kw.setdefault("sigma", 0.5)
    kw.setdefault("seed", 1)
    kw.setdefault("token_offset", F.OFF)
    return R.realify(s, bus, its, **kw)


def _assert_untouched(rig, ring=None):
    t, s, bus = rig
    assert t.total_calls() == 0
    assert t.text_calls == 0
    assert bus.stats["captures"] == 0
    if ring is not None:
        assert len(ring) == 0


@pytest.mark.parametrize("tags", [["USER"], ["EXTGEN:test"], ["HOLDOUT"], ["COMP"], ["ENGINE", "NC"], ["IP-FLAG"]])
def test_firewall_canaries(tags):
    rig = F.make_rig()
    ring = T.TapRing(F.basis(), max_entries=8)
    bad = P.as_record(tags)
    with pytest.raises(P.FirewallError):
        _call(rig, _items(2, prov=[F.ENGINE, bad]), ring=ring)
    _assert_untouched(rig, ring)


def test_firewall_accepts_engine_and_cc0_asset():
    rig = F.make_rig()
    res = _call(rig, _items(2, prov=[F.ENGINE, P.as_record(["ENGINE", "ASSET:CC0"])]), tau=10.0)
    assert res[1].prov == P.derive("realify", [P.as_record(["ENGINE", "ASSET:CC0"])])


@pytest.mark.parametrize("kw", [
    dict(sigma=0.9), dict(sigma=1.0), dict(sigma=0.8), dict(sigma=0.7),   # 스냅 결과가 [0.2,0.7] 밖
    dict(steps=1), dict(steps=4), dict(steps=0),
    dict(tap_step=2), dict(tap_step=-3),
    dict(tau=-0.1), dict(tau=float("nan")), dict(d_thresh=1.5), dict(d_thresh=-0.01),
    dict(seed=-1), dict(seed=1 << 64),
])
def test_bad_parameters(kw):
    rig = F.make_rig()
    with pytest.raises(ValueError):
        _call(rig, _items(), **kw)
    _assert_untouched(rig)


def test_curve_problems():
    rig = F.make_rig()
    s = rig[1]
    no_sigma = R.DecisionCurve(sigmas=[0.9, 0.4], D=[[1.0] * 4, [0.0] * 4])
    with pytest.raises(R.RealifyError):
        _call(rig, _items(), curve=no_sigma)
    other = R.provisional_curve(SigmaTable((1.0, 0.75, 0.5, 0.3), name="other"), 6)
    with pytest.raises(R.RealifyError):
        _call(rig, _items(), curve=other)                                 # table_key 불일치
    too_many = R.provisional_curve(s.table, 7)                           # 64x64 는 최대 6
    with pytest.raises(R.RealifyError):
        _call(rig, _items(), curve=too_many)
    keyless = R.DecisionCurve(sigmas=s.table.points, D=[[0.0] * 6] * len(s.table))   # table_key None 은 허용
    _assert_untouched(rig)
    assert len(_call(rig, _items(), curve=keyless, tau=10.0)) == 2


@pytest.mark.parametrize("bad", ["shape", "nan", "range_hi", "range_lo", "size", "mismatch"])
def test_bad_images(bad):
    rig = F.make_rig()
    its = _items()
    img = its[1].image.copy()
    if bad == "shape":
        img = np.zeros((64, 64, 4), np.float32)
    elif bad == "nan":
        img[3, 3, 0] = np.nan
    elif bad == "range_hi":
        img[0, 0, 0] = 1.01
    elif bad == "range_lo":
        img[0, 0, 0] = -0.01
    elif bad == "size":
        img = np.full((60, 64, 3), 0.5, np.float32)
    elif bad == "mismatch":
        img = np.full((32, 32, 3), 0.5, np.float32)
    its[1] = R.RealifyItem(key="k1", image=img, prov=F.ENGINE, caption="room")
    with pytest.raises(ValueError):
        _call(rig, its)
    _assert_untouched(rig)


def test_keys_and_labels():
    rig = F.make_rig()
    with pytest.raises(R.RealifyError):
        _call(rig, _items(2, key="same"))
    for bad_key in ("a.b", "", "-x", "x" * 65):
        with pytest.raises(R.RealifyError):
            _call(rig, [R.RealifyItem(key=bad_key, image=F.smooth_image(0), prov=F.ENGINE, caption="r")])
    for bad_label in ("rgb", "meta", "tap8", "a.b", "a-b"):
        with pytest.raises(R.RealifyError):
            _call(rig, [R.RealifyItem(key="ok", image=F.smooth_image(0), prov=F.ENGINE, caption="r",
                                      labels={bad_label: np.zeros(3, np.float32)})])
    with pytest.raises(R.RealifyError):
        _call(rig, [])
    _assert_untouched(rig)


def test_types():
    rig = F.make_rig()
    with pytest.raises(TypeError):
        _call(rig, _items(), seed="1")
    with pytest.raises(TypeError):
        _call(rig, ["not an item"])
    _assert_untouched(rig)


def test_bus_and_ring_preconditions():
    rig = F.make_rig()
    rig[2].detach()
    with pytest.raises((R.RealifyError, T.TapError)):
        _call(rig, _items())
    assert rig[0].total_calls() == 0
    rig = F.make_rig()
    other = T.fit_pca_basis({l: v + 1.0 for l, v in _fit_samples().items()}, prov=F.ENGINE, k=F.K)
    ring = T.TapRing(other, max_entries=4)
    with pytest.raises(T.TapBasisError):
        _call(rig, _items(), ring=ring)
    _assert_untouched(rig, ring)


def _fit_samples():
    import torch
    g = torch.Generator().manual_seed(9)
    return {l: torch.randn(40, F.D, generator=g) for l in T.TAP_LAYERS}

"""EVL-04 보고서: S11·EVL-09·WLD-14 가 받는 한 장짜리 직렬화 가능한 KPI 묶음."""
import json

import numpy as np
import pytest

from buc.core.contracts import KPI
from evl04_support import FakeClient, ff, noisy, shift_img, texture

KEYS = {"schema", "eval_only", "backend", "is_perceptual", "r_eff", "ffd", "ffg_median", "ffg_p95", "ffg_sal_p95",
        "de00", "dg", "oir", "klass", "verdict"}


@pytest.fixture(scope="module")
def pair():
    a = texture(96, 128, seed=30)
    return a, noisy(shift_img(a, 0.3, 0.5), 0.02, seed=31)


def test_report_identity_is_all_zero():
    m = ff()
    a = texture(96, 128, seed=32)
    with m.LpipsClient("proxy") as c:
        r = m.first_frame_report(a, a, c)
    assert r.ffd == 0.0 and r.de00 == 0.0 and r.dg == 0.0
    assert r.ffg_median < 0.05 and r.ffg_p95 < 0.1
    assert r.ffg_sal_p95 is None and r.oir is None and r.verdict is None and r.klass is None


def test_report_fields_match_individual_metrics(pair):
    m = ff()
    a, b = pair
    sal = m.top_fraction_mask(np.abs(np.gradient(a.mean(-1))[0]) + np.abs(np.gradient(a.mean(-1))[1]))
    with m.LpipsClient("proxy") as c:
        r = m.first_frame_report(a, b, c, sal_mask=sal, oir_responses=[True] * 11 + [False] * 9)
        assert r.ffd == pytest.approx(m.ffd(a, b, c), abs=1e-12)
    f = m.ffg(a, b)
    assert r.ffg_median == pytest.approx(f.median, abs=1e-9) and r.ffg_p95 == pytest.approx(f.p95, abs=1e-9)
    assert r.ffg_sal_p95 == pytest.approx(m.ffg_sal(a, b, sal), abs=1e-9)
    assert r.de00 == pytest.approx(m.delta_e00(a, b), abs=1e-12)
    assert r.dg == pytest.approx(m.delta_g(a, b), abs=1e-12)
    assert r.oir == pytest.approx(0.55)
    assert (r.backend, r.is_perceptual, r.eval_only, r.schema) == ("proxy", False, True, 1)


def test_report_r_eff_is_forwarded_to_ffd(pair):
    m = ff()
    a, b = pair
    fake = FakeClient()
    r = m.first_frame_report(a, b, fake, r_eff=48)
    x, y = fake.calls[0]
    assert x.shape == (48, 64, 3) and r.r_eff == 48 and r.backend == "fake"


def test_report_json_roundtrip_and_kpi_seam(pair):
    m = ff()
    a, b = pair
    r = m.first_frame_report(a, b, FakeClient(), klass="L1_photo_green")
    d = json.loads(r.to_json())
    assert set(d) == KEYS and d == r.as_dict()
    assert d["klass"] == "L1_photo_green" and set(d["verdict"]) == {"ok", "checks"}
    assert d["verdict"]["checks"].keys() == {"ffd", "ffg", "ffg_sal", "de00", "oir"}
    assert d["verdict"]["checks"]["ffg_sal"] is None and d["verdict"]["checks"]["oir"] is None  # 입력이 없으면 보고만
    assert all(np.isfinite(v) for v in (d["ffd"], d["ffg_median"], d["ffg_p95"], d["de00"], d["dg"]))
    k = r.kpi_fields()
    assert set(k) == {"ffd", "ffg", "de00"} and k["ffg"] == r.ffg_median
    kpi = KPI(**k)  # FND-02 KPI 계약에 그대로 들어간다
    assert kpi.ffd == r.ffd and kpi.ffg == r.ffg_median and kpi.de00 == r.de00


def test_report_is_immutable_and_deterministic(pair):
    m = ff()
    a, b = pair
    r1 = m.first_frame_report(a, b, FakeClient())
    r2 = m.first_frame_report(a, b, FakeClient())
    assert r1 == r2
    with pytest.raises(Exception):
        r1.ffd = 1.0  # frozen dataclass


def test_report_rejects_unknown_klass_and_mismatched_inputs(pair):
    m = ff()
    a, b = pair
    with pytest.raises(ValueError):
        m.first_frame_report(a, b, FakeClient(), klass="nope")
    with pytest.raises(ValueError):
        m.first_frame_report(a, b[:, :-1], FakeClient())

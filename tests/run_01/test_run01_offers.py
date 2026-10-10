"""오퍼 매핑·필터·정렬과 임대 경로(성공·무오퍼·한도·label)."""
import itertools

import pytest
from run01_fakes import raw_offer

from buc.rt import vast as V


@pytest.mark.parametrize("geo,code", [
    ("Seoul, KR", "KR"), ("Guangdong, CN", "CN"), ("China", "CN"), ("Beijing, china", "CN"),
    ("Quebec, ca", "CA"), ("", None), (None, None), ("Somewhere, Atlantis", None),
])
def test_country_code(geo, code):
    assert V.country_code(geo) == code


def test_offer_from_api_mapping():
    d = raw_offer(7, machine_id=143424, dph=0.508, ram_mb=31744, rel=0.991)
    d["reliability"] = 0.5  # reliability2가 우선
    o = V.Offer.from_api(d)
    assert (o.id, o.machine_id, o.gpu_name, o.num_gpus) == (7, 143424, "RTX 5090", 1)
    assert o.dph_total == pytest.approx(0.508, abs=1e-12)
    assert o.reliability == pytest.approx(0.991, abs=1e-12)
    assert o.cpu_ram_gb == pytest.approx(31.0, abs=1e-9)
    assert o.cpu_cores == 16 and o.disk_gb == 200 and o.inet_down == 900
    assert o.cuda_vers == pytest.approx(12.8) and o.verified is True and o.rentable is True
    assert o.country == "KR"
    d2 = dict(d)
    del d2["reliability2"]
    assert V.Offer.from_api(d2).reliability == pytest.approx(0.5)
    d3 = dict(d, verification="unverified")
    assert V.Offer.from_api(d3).verified is False
    assert V.Offer.from_api(dict(d3, verified=True)).verified is True


@pytest.mark.parametrize("bad", ["dph_total", "id"])
def test_offer_from_api_requires_price_and_id(bad):
    d = raw_offer(1)
    del d[bad]
    with pytest.raises(ValueError):
        V.Offer.from_api(d)
    with pytest.raises(ValueError):
        V.Offer.from_api(dict(raw_offer(1), dph_total=float("nan")))


GOOD = raw_offer(1)


@pytest.mark.parametrize("patch,ok", [
    ({}, True),
    ({"gpu_name": "RTX_5090"}, True),
    ({"gpu_name": "rtx 5090"}, True),
    ({"dph_total": 0.50}, True),
    ({"dph_total": 0.501}, False),
    ({"dph_total": 0.0}, False),
    ({"reliability2": 0.979}, False),
    ({"gpu_name": "RTX 4090"}, False),
    ({"num_gpus": 2}, False),
    ({"geolocation": "Guangdong, CN"}, False),
    ({"geolocation": "China"}, False),
    ({"verification": "unverified"}, False),
    ({"rentable": False}, False),
    ({"cpu_ram": 29 * 1024}, False),
    ({"cpu_cores_effective": 11}, False),
    ({"disk_space": 99}, False),
    ({"inet_down": 299}, False),
    ({"cuda_max_good": 12.6}, False),
])
def test_query_accepts_table(patch, ok):
    assert V.OfferQuery().accepts(V.Offer.from_api(dict(GOOD, **patch))) is ok


def test_query_excludes_machines_and_relaxes():
    o = V.Offer.from_api(GOOD)
    assert V.OfferQuery(exclude_machines=frozenset({o.machine_id})).accepts(o) is False
    pricey = V.Offer.from_api(dict(GOOD, dph_total=0.8))
    assert V.OfferQuery(max_dph=1.0).accepts(pricey) is True
    unverified = V.Offer.from_api(dict(GOOD, verification="unverified"))
    assert V.OfferQuery(verified_only=False).accepts(unverified) is True


def test_to_api_query_shape():
    q = V.OfferQuery(max_dph=0.33).to_api_query()
    assert q["gpu_name"] == {"eq": "RTX 5090"}
    assert q["num_gpus"] == {"eq": 1}
    assert q["dph_total"]["lte"] == pytest.approx(0.33)
    assert q["rentable"] == {"eq": True}
    assert q["order"] == [["dph_total", "asc"]]


def test_select_offers_order_is_deterministic():
    raws = [raw_offer(5, dph=0.30, rel=0.99), raw_offer(3, dph=0.30, rel=0.999), raw_offer(4, dph=0.30, rel=0.999),
            raw_offer(2, dph=0.25, rel=0.98), raw_offer(1, dph=0.20, geo="Sichuan, CN"),
            raw_offer(6, dph=0.60)]
    offers = [V.Offer.from_api(r) for r in raws]
    want = [2, 3, 4, 5]
    for perm in itertools.permutations(offers):
        assert [o.id for o in V.select_offers(list(perm), V.OfferQuery())] == want


def test_policy_validation():
    V.VastPolicy()
    for kw in ({"idle_timeout_s": -1}, {"boot_timeout_s": float("nan")}, {"idle_action": "nuke"},
               {"cap_margin_h": -0.1}, {"reserve_hours": float("inf")}, {"stopped_usd_per_h": -0.01}):
        with pytest.raises(ValueError):
            V.VastPolicy(**kw)
    assert V.VastPolicy(idle_action="destroy").idle_action == "destroy"


# ------------------------------------------------------------------ 임대

def test_rent_picks_cheapest_acceptable_and_records(api, ledger, make, clock):
    api.offers = [raw_offer(1, dph=0.10, geo="Shanghai, CN"), raw_offer(2, dph=0.47),
                  raw_offer(3, dph=0.31, machine_id=999), raw_offer(4, dph=0.05, verified=False)]
    orch = make()
    iid = orch.rent(onstart="bash /boot.sh", env={"HF_TOKEN": "secret-value-xyz", "MODE": "a"})
    assert isinstance(iid, int) and iid in api.instances
    assert api.n("create_instance") == 1
    _, offer_id, cfg = next(c for c in api.calls if c[0] == "create_instance")
    assert offer_id == 3
    assert cfg["image"] == V.DEFAULT_IMAGE and cfg["disk"] == 90
    assert cfg["label"].startswith("buc-") and cfg["onstart"] == "bash /boot.sh"
    assert cfg["env"] == {"HF_TOKEN": "secret-value-xyz", "MODE": "a"}
    search = next(c for c in api.calls if c[0] == "search_offers")
    assert search[1] == V.OfferQuery().to_api_query()
    evs = ledger.events(run_id=f"vast:{iid}")
    assert len(evs) >= 1 and all(e.kind in ("usd", "gpu_h") for e in evs)
    assert ledger.total("usd") == 0.0
    rent = [e for e in evs if e.meta.get("event") == "rent"]
    assert len(rent) == 1 and rent[0].meta["t1"] == pytest.approx(clock())
    blob = repr([e.meta for e in ledger.events()]) + repr([e.note for e in ledger.events()])
    assert "secret-value-xyz" not in blob


def test_rent_no_offer_does_not_create(api, make):
    api.offers = [raw_offer(1, dph=0.9), raw_offer(2, geo="China")]
    with pytest.raises(V.NoOfferError):
        make().rent()
    assert api.n("create_instance") == 0


def test_rent_label_must_carry_prefix(api, make):
    with pytest.raises(ValueError):
        make().rent(label="founder-box")
    assert api.n("create_instance") == 0
    iid = make().rent(label="buc-train")
    assert api.instances[iid]["label"] == "buc-train"


def test_rent_limit_counts_running_only(api, make, clock):
    orch = make(max_instances=1)
    iid = orch.rent()
    with pytest.raises(V.RentLimitError):
        orch.rent()
    assert api.n("create_instance") == 1
    orch.stop(iid)
    assert api.instances[iid]["intended_status"] == "stopped"
    iid2 = orch.rent()
    assert iid2 != iid and api.n("create_instance") == 2
    with pytest.raises(V.RentLimitError):
        orch.start(iid)
    assert api.instances[iid]["intended_status"] == "stopped"


def test_custom_query_is_used(api, make):
    api.offers = [raw_offer(1, dph=0.70), raw_offer(2, dph=0.90)]
    iid = make().rent(query=V.OfferQuery(max_dph=0.80))
    assert api.instances[iid]["dph_total"] == pytest.approx(0.70)

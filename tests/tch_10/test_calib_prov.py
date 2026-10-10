"""자체 보정셋 + prov 검사(Soul 방화벽 CT32), 사전 양자화 가중치 금지."""
import pytest
import torch
from tch10_toys import MLP, calib, candidates

from buc.gov import prov as P
from buc.teacher import quant as Q

BAD_TAGS = [{"USER"}, {"SOUL"}, {"SOUL-DERIVED"}, {"SOUL-INSPIRED"}, {"EXTGEN:test"}, {"HOLDOUT"},
            {"PUB-SEALED"}, {"COMP"}, {"ASSET"}, {"EVAL-ONLY"}, {"OPEN-GEN:m:h:bench"}]


def test_selection_deterministic_and_order_free():
    a = calib(16, seed=3)
    b = Q.build_calib_set(list(reversed(candidates())), 16, seed=3)
    assert isinstance(a, Q.CalibSet) and len(a) == 16
    assert a.ids == b.ids and a.digest == b.digest
    assert len(set(a.ids)) == 16 and set(a.ids) <= {f"c{i:02d}" for i in range(24)}
    assert isinstance(a.digest, str) and len(a.digest) == 64 and int(a.digest, 16) >= 0
    c = calib(16, seed=4)
    assert c.ids != a.ids  # 다른 seed → 다른 선택 순서/집합
    full1, full2 = calib(24, seed=1), calib(24, seed=2)
    assert set(full1.ids) == set(full2.ids)
    assert full1.digest == full2.digest  # digest는 선택 집합만 본다(순서 무관)
    assert a.rejected == ()


def test_digest_tracks_content():
    cs = candidates()
    a = Q.build_calib_set(cs, 24)
    cs2 = list(cs)
    d = cs2[5].data.clone()
    d[0, 0] += 1e-3
    cs2[5] = Q.CalibSample(id=cs2[5].id, data=d, prov=cs2[5].prov)
    assert Q.build_calib_set(cs2, 24).digest != a.digest


def test_batches():
    c = calib(10, seed=0)
    bs = c.batches(4)
    assert [b.shape[0] for b in bs] == [4, 4, 2]
    flat = torch.cat(bs)
    for i, s in enumerate(c.samples):
        assert torch.equal(flat[i], s.data)
    with pytest.raises(ValueError):
        c.batches(0)


def test_calib_prov_is_derived_and_train_ok():
    c = calib(8)
    assert isinstance(c.prov, P.ProvRecord) and P.train_ok(c.prov)
    cs = candidates()
    cs[0] = Q.CalibSample(id="c00", data=cs[0].data, prov=P.ProvRecord(taint=frozenset({"CAPTURE"}), roots=("r0",)))
    full = Q.build_calib_set(cs, 24)
    assert "CAPTURE" in full.prov.taint and "r0" in full.prov.roots


@pytest.mark.parametrize("tags", BAD_TAGS, ids=lambda t: sorted(t)[0])
def test_strict_rejects_any_bad_candidate(tags):
    cs = candidates(bad={7: tags})
    with pytest.raises(P.FirewallError):
        Q.build_calib_set(cs, 4, seed=0)  # 선택에 안 들어가도 후보에 있으면 실패(fail closed)


def test_lenient_drops_and_records():
    cs = candidates(bad={1: {"SOUL"}, 9: {"USER"}, 20: {"HOLDOUT"}})
    c = Q.build_calib_set(cs, 21, seed=0, strict=False)
    assert c.rejected == ("c01", "c09", "c20")
    assert not set(c.ids) & {"c01", "c09", "c20"}
    assert P.train_ok(c.prov) and not P.ext_bit(c.prov)
    with pytest.raises(ValueError):
        Q.build_calib_set(cs, 22, strict=False)  # 가용 21


def test_calib_input_validation():
    cs = candidates()
    with pytest.raises(ValueError):
        Q.build_calib_set(cs + [cs[0]], 4)
    with pytest.raises(ValueError):
        Q.build_calib_set(cs, 0)
    with pytest.raises(ValueError):
        Q.build_calib_set(cs, 25)
    odd = cs + [Q.CalibSample(id="zz", data=torch.zeros(3, 3), prov={"ENGINE"})]
    with pytest.raises(ValueError):
        Q.build_calib_set(odd, 4)
    with pytest.raises(TypeError):
        Q.build_calib_set(cs + [Q.CalibSample(id="zz", data=[1.0], prov={"ENGINE"})], 4)


def test_prequantized_state_dict_rejected():
    sd = MLP().state_dict()
    Q.assert_not_prequantized(sd)  # float 원본 통과
    sd_bn = dict(sd, **{"bn.num_batches_tracked": torch.tensor(3)})
    Q.assert_not_prequantized(sd_bn)  # 0-d int64 카운터는 허용
    for k, v in [("up.qweight", torch.zeros(4, 4)), ("up.wscales", torch.ones(4)), ("up.smooth_factor", torch.ones(4)),
                 ("up.proj_down", torch.ones(2, 4)), ("up.weight_scale", torch.ones(1)),
                 ("up.weight", torch.zeros(4, 4, dtype=torch.int8)), ("up.weight", torch.zeros(4, 2, dtype=torch.uint8)),
                 ("x.packed", torch.zeros(4, dtype=torch.int32))]:
        with pytest.raises(P.FirewallError):
            Q.assert_not_prequantized(dict(sd, **{k: v}))


def _sentinel(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


@pytest.mark.parametrize("wprov", [{"USER"}, {"SOUL"}, {"EXTGEN-DERIVED:soul"}, {"HOLDOUT"}])
def test_quantize_model_firewall_is_atomic(wprov):
    m = MLP()
    before = _sentinel(m)
    with pytest.raises(P.FirewallError):
        Q.quantize_model(m, Q.QuantSpec("int8"), calib(8), weight_prov=wprov)
    assert all(isinstance(getattr(m, n), torch.nn.Linear) for n in ("up", "down", "head", "odd", "unused"))
    assert not isinstance(getattr(m, "_buc_quant", None), Q.QuantReport)
    after = m.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())


def test_quantize_model_rejects_prequantized_model():
    m = MLP()
    m.register_buffer("qzeros", torch.zeros(4, dtype=torch.int32))
    with pytest.raises(P.FirewallError):
        Q.quantize_model(m, Q.QuantSpec("int8"), calib(8), weight_prov={"ENGINE"})
    assert isinstance(m.up, torch.nn.Linear)


def test_quantize_model_needs_calibset():
    with pytest.raises(TypeError):
        Q.quantize_model(MLP(), Q.QuantSpec("int8"), [torch.zeros(8, 256)], weight_prov={"ENGINE"})
    with pytest.raises(TypeError):
        Q.quantize_model(MLP(), {"mode": "int8"}, calib(8), weight_prov={"ENGINE"})

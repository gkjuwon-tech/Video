"""정본 해시와 직렬화: 하류 Merkle 캐시·Proof-of-Shot의 바닥."""
import json
import os

import numpy as np
import pytest

from buc.core import contracts as C
from fnd02_fixtures import (make_frames, make_image, make_kpi, make_obs, make_prov, make_read, make_shot,
                            make_world)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _all_objects():
    image = make_image()
    read = make_read(image)
    world = make_world(read)
    shot = make_shot(read.obs)
    return [make_prov("USER", "EXTGEN:soul"), C.Theta(focal_mm=85.0, sun_elev_deg=12.5, extra={"k": [1, 2]}),
            make_obs(mask=np.ones((8, 12), dtype=bool), l_obs_hard=1.0, l_obs_part=2.5),
            image, read, world.layers[0], world.far_field, world, shot, make_frames(shot), make_kpi(),
            C.make_stage_io("S2", "stub", "flat_world", "0.0.1", {"read": read}, {"world": world},
                            params={"cards": 3}, wall_s=1.5)]


def test_constants():
    assert C.CONTRACT_VERSION == "0.1.0"
    assert C.LEVELS == ("L1", "L1.5", "L2a", "L2b")
    assert C.STAGES == ("S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S8.5", "S9-10", "S11")
    assert set(C.STAGE_SIGNATURES) == {"S-1", "S2", "S8", "S8.5", "S9-10", "S11"}
    assert issubclass(C.ContractVersionError, C.ContractError) and issubclass(C.ContractError, ValueError)
    assert {"USER", "EXTGEN", "CAPTURE", "ASSET", "ENGINE", "SI", "HOLDOUT", "COMP"} == set(C.TAINT_BASES)


def test_content_hash_canonical():
    h = C.content_hash
    assert len(h({"a": 1})) == 64 and int(h({"a": 1}), 16) >= 0
    assert h({"a": 1, "b": [1, 2]}) == h({"b": (1, 2), "a": 1})  # 키 순서 무관, list≡tuple
    assert h(1) != h(1.0) and h(1) != h(True) and h(0) != h(False)
    assert h("1") != h(1) and h(None) != h(0) and h([]) != h({})
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert h(a) == h(np.asfortranarray(a)) == h(a.T.copy().T)  # 메모리 배치 무관
    assert h(a) == h(a.astype(">f4"))  # 바이트 순서 정규화
    assert h(a) != h(a.astype(np.float64)) and h(a) != h(a.reshape(4, 3)) and h(a) != h(a.reshape(12))
    b = a.copy()
    b[2, 3] = np.nextafter(b[2, 3], np.float32(100))  # 1 ulp
    assert h(a) != h(b)
    assert h(1.0) != h(np.nextafter(1.0, 2.0))


def test_hash_is_reproducible_across_constructions():
    assert make_read().content_hash() == make_read().content_hash()
    assert make_world().content_hash() == make_world().content_hash()
    assert make_shot().content_hash() == make_shot().content_hash()


@pytest.mark.parametrize("idx", range(12))
def test_roundtrip_bitwise(idx):
    obj = _all_objects()[idx]
    d = obj.to_dict()
    s = json.dumps(d)  # JSON 직렬 가능해야 한다
    d2 = json.loads(s)
    assert d2["__contract__"] == type(obj).__name__ and d2["__v__"] == C.CONTRACT_VERSION
    back = type(obj).from_dict(d2)
    assert type(back) is type(obj)
    assert back.content_hash() == obj.content_hash()
    back2 = C.loads(C.dumps(obj))
    assert type(back2) is type(obj) and back2.content_hash() == obj.content_hash()


def test_roundtrip_preserves_arrays_exactly():
    read = make_read()
    back = C.loads(C.dumps(read))
    assert back.image_lin.dtype == np.float32 and back.depth.dtype == np.float32
    assert np.array_equal(back.image_lin, read.image_lin)
    assert np.isinf(back.depth[0, 0]) and np.array_equal(back.depth, read.depth)
    assert back.obs.K.dtype == np.float64 and np.array_equal(back.obs.K, read.obs.K)
    assert back.theta_in.focal_mm == 50.0 and back.fallbacks == read.fallbacks or list(back.fallbacks) == ["R3"]


def test_ndarray_encoding_shape():
    d = make_obs().to_dict()
    enc = d["K"]
    assert set(enc) >= {"__ndarray__", "dtype", "shape"} and enc["shape"] == [3, 3] and enc["dtype"] == "<f8"


def test_version_and_type_mismatch_rejected():
    d = make_kpi().to_dict()
    d_major = dict(d, __v__="1.0.0")
    with pytest.raises(C.ContractVersionError):
        C.KPI.from_dict(d_major)
    with pytest.raises(C.ContractError):
        C.Theta.from_dict(d)
    ok_minor = dict(d, __v__="0.9.3")  # 같은 주 버전은 읽는다
    assert C.KPI.from_dict(ok_minor).content_hash() == make_kpi().content_hash()


def test_hash_sensitive_to_every_field_change():
    base = make_read()
    variants = [make_read(scale_s=1.26), make_read(level="L2a"), make_read(theta_in=C.Theta(focal_mm=51.0)),
                make_read(prov=make_prov("USER", "EXTGEN:gpt", ns="user_vault", mode="user")),
                make_read(fallbacks=()), make_read(obs=make_obs(k1=0.01))]
    hashes = {v.content_hash() for v in variants}
    assert base.content_hash() not in hashes and len(hashes) == len(variants)


def test_docs_contracts_md_lists_types():
    p = os.path.join(ROOT, "docs", "contracts.md")
    assert os.path.exists(p)
    text = open(p, encoding="utf-8").read()
    for name in ["World", "ReadPacket", "Obs", "Shot", "Theta", "KPI", "StageIO", "Prov", "InputImage", "Frames",
                 "CONTRACT_VERSION", "OpenCV"]:
        assert name in text, name

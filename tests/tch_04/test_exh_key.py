"""TCH-04 키 설계: 골든 재계산, 필드 민감도, (ext,comp,hold,ns) 분리, ext 자격, 입력 검사, teacher_id."""
import dataclasses
import itertools
import random
import struct
from types import SimpleNamespace

import blake3
import pytest
import torch

from buc.core import det
from buc.gov import prov as P
from buc.teacher import exhaust as X
from buc.teacher.sample import DEFAULT_TABLE, SigmaError, SigmaTable
import tch04_fakes as F


def oracle_body(p, rec, proj):
    return det.hash_parts(
        "buc.tch04.key.v1", *P.merkle_parts(rec), P.content_address_kind(rec), proj, p.world, p.view,
        struct.pack("<d", DEFAULT_TABLE.points[3]), p.seed, p.level, p.k, p.teacher_id, p.caption_hash,
        p.table.key, p.height, p.width, p.precision, p.quant)


def test_golden_non_ext_is_plain_hash_parts():
    p = F.parts()
    k = X.compute_key(p, F.ENGINE)
    assert k == oracle_body(p, P.as_record(F.ENGINE), "").hex()
    assert len(k) == 64 and k == k.lower()


def test_golden_ext_is_keyed_blake3_of_body():
    p = F.parts()
    rec = P.as_record(F.USER)
    k = X.compute_key(p, F.USER, project_id=F.PROJ, project_key=F.PK)
    body = oracle_body(p, rec, F.PROJ)
    assert k == blake3.blake3(body, key=F.PK).hexdigest()
    assert k != body.hex()


def test_non_ext_key_ignores_project_args():
    p = F.parts()
    a = X.compute_key(p, F.ENGINE)
    assert a == X.compute_key(p, F.ENGINE, project_id="x", project_key=F.PK)


def test_key_stable_across_calls_and_equal_parts_objects():
    assert X.compute_key(F.parts(), F.ENGINE) == X.compute_key(F.parts(), F.ENGINE)
    assert hash(F.parts()) == hash(F.parts())


FIELD_VARIANTS = {
    "world": bytes(reversed(F.WORLD)),
    "view": "L2/b7/band4",
    "sigma": DEFAULT_TABLE.points[4],
    "seed": 1235,
    "level": 1,
    "k": 1,
    "teacher_id": "zimage@-|torch.bfloat16|aa|bc",
    "caption_hash": "cd" * 32,
    "table": SigmaTable((1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1), name="other"),
    "height": 128,
    "width": 128,
    "precision": "fp16",
    "quant": "int8-sym-ch-v2",
}


@pytest.mark.parametrize("field,val", sorted(FIELD_VARIANTS.items()))
def test_every_field_changes_the_key(field, val):
    over = {field: val}
    if field == "table":
        over["sigma"] = 0.5           # 새 표의 점
    base_over = {"sigma": 0.5} if field == "table" else {}
    assert X.compute_key(F.parts(**over), F.ENGINE) != X.compute_key(F.parts(**base_over), F.ENGINE)


def test_sigma_matched_to_table_point_within_tolerance():
    pt = DEFAULT_TABLE.points[2]
    a = X.compute_key(F.parts(sigma=pt), F.ENGINE)
    assert a == X.compute_key(F.parts(sigma=pt + 1e-12), F.ENGINE)
    with pytest.raises(SigmaError):
        F.parts(sigma=0.8123)             # 표 밖
    with pytest.raises(ValueError):
        F.parts(sigma=0.0)


def test_prov_bits_and_namespace_separate_keys():
    p = F.parts()
    sets = {"engine": F.ENGINE, "comp": F.COMP, "hold": F.HOLD, "user": F.USER, "extgen": F.EXTGEN,
            "optin": {"OPTIN"}}
    keys = {n: X.compute_key(p, s, project_id=F.PROJ, project_key=F.PK) for n, s in sets.items()}
    # engine, optin: bits=0 ns=train_ok -> 같은 키(키는 (bits, ns)만 본다)
    assert keys["engine"] == keys["optin"]
    others = {n: k for n, k in keys.items() if n != "optin"}
    assert len(set(others.values())) == len(others)


def test_same_bits_but_different_ns_differ_and_same_bits_same_ns_agree():
    p = F.parts()
    kw = F.ext_kw()
    # USER(user_vault) vs EXTGEN(ext_vault): 비트 1, ns 다름
    assert X.compute_key(p, F.USER, **kw) != X.compute_key(p, F.EXTGEN, **kw)
    # USER vs USER-DERIVED: 비트 1, ns user_vault
    assert X.compute_key(p, F.USER, **kw) == X.compute_key(p, F.USER_D, **kw)


def test_prov_accepts_record_object_or_tag_iterable():
    p = F.parts()
    rec = P.as_record(F.ENGINE)
    assert X.compute_key(p, rec) == X.compute_key(p, ["ENGINE"]) == X.compute_key(p, F.ENGINE)
    with pytest.raises(TypeError):
        X.compute_key(p, "ENGINE")
    with pytest.raises(TypeError):
        X.compute_key(p, None)


def test_ext_key_depends_on_project_key_and_project_id():
    p = F.parts()
    base = X.compute_key(p, F.USER, project_id=F.PROJ, project_key=F.PK)
    assert base != X.compute_key(p, F.USER, project_id=F.PROJ, project_key=F.PK2)
    assert base != X.compute_key(p, F.USER, project_id=F.PROJ2, project_key=F.PK)
    assert base == X.compute_key(p, F.USER, project_id=F.PROJ, project_key=F.PK)


def test_ext_without_project_credentials_fails_closed():
    p = F.parts()
    for kw in [{}, {"project_id": F.PROJ}, {"project_key": F.PK}, {"project_id": None, "project_key": F.PK}]:
        with pytest.raises(P.FirewallError):
            X.compute_key(p, F.USER, **kw)
    assert X.FirewallError is P.FirewallError


@pytest.mark.parametrize("kw", [
    dict(project_id="", project_key=F.PK), dict(project_id=3, project_key=F.PK),
    dict(project_id=F.PROJ, project_key=b"short"), dict(project_id=F.PROJ, project_key="x" * 32),
])
def test_ext_malformed_credentials_rejected(kw):
    with pytest.raises((X.ExhaustSpecError, P.FirewallError)):
        X.compute_key(F.parts(), F.USER, **kw)


def test_hundred_random_parts_have_distinct_keys_and_no_hmac_plain_collision():
    rng = random.Random(7)
    seen = set()
    for i in range(120):
        p = F.parts(seed=rng.randrange(1 << 64), view=f"v{rng.randrange(10**6)}",
                    sigma=rng.choice(DEFAULT_TABLE.points), level=rng.randrange(4))
        for rec in (F.ENGINE, F.COMP, F.HOLD):
            seen.add(X.compute_key(p, rec))
        seen.add(X.compute_key(p, F.USER, **F.ext_kw()))
    assert len(seen) == 120 * 4


@pytest.mark.parametrize("over", [
    dict(world=b"x" * 31), dict(world="a" * 32), dict(view=""), dict(view=3), dict(seed=-1), dict(seed=1 << 64),
    dict(seed=True), dict(seed=1.5), dict(caption_hash="AB" * 32), dict(caption_hash="ab" * 31),
    dict(caption_hash=b"ab" * 32), dict(teacher_id=""), dict(height=0), dict(width=-8), dict(height=True),
    dict(level=-1), dict(k=-1), dict(precision=""), dict(table="default"),
])
def test_keyparts_validation(over):
    with pytest.raises((X.ExhaustSpecError, SigmaError)):
        F.parts(**over)
    assert issubclass(X.ExhaustSpecError, ValueError)


def test_keyparts_frozen_and_dataclass():
    p = F.parts()
    assert dataclasses.is_dataclass(p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.seed = 3


def test_teacher_id_format_and_requires_fingerprints():
    spec = SimpleNamespace(repo="Tongyi-MAI/Z-Image-Turbo", revision=None)
    fp = {"transformer": bytes([1]) * 32, "vae": bytes([2]) * 32}
    t = SimpleNamespace(spec=spec, dtype=torch.bfloat16, fingerprints=fp)
    assert X.teacher_id(t) == f"Tongyi-MAI/Z-Image-Turbo@-|torch.bfloat16|{'01' * 32}|{'02' * 32}"
    t2 = SimpleNamespace(spec=SimpleNamespace(repo="a/b", revision="r1"), dtype=torch.float16, fingerprints=fp)
    assert X.teacher_id(t2).startswith("a/b@r1|torch.float16|")
    fp2 = dict(fp, vae=bytes([3]) * 32)
    assert X.teacher_id(SimpleNamespace(spec=spec, dtype=torch.bfloat16, fingerprints=fp2)) != X.teacher_id(t)
    with pytest.raises(X.ExhaustSpecError):
        X.teacher_id(SimpleNamespace(spec=spec, dtype=torch.bfloat16, fingerprints=None))


def test_all_prov_kinds_produce_valid_keys():
    p = F.parts()
    for r in itertools.chain([F.ENGINE, F.COMP, F.HOLD, {"CAPTURE"}, {"ASSET"}, set()]):
        assert len(X.compute_key(p, r)) == 64

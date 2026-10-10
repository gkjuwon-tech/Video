"""TCH-04 방화벽: 용도 표, ext 봉인, 구역 분리, 학습 분할 격리, 변조 파일, 프로젝트 열거 금지, TTL, purge."""
import json
import random
import struct

import blake3
import pytest
import torch

from buc.gov import prov as P
from buc.teacher import exhaust as X
import tch04_fakes as F

TTL = 72 * 3600


def mk(tmp_path, **kw):
    clock = kw.pop("clock", None) or F.Clock()
    sealer = kw.pop("sealer", None) or F.FakeSealer()
    return X.ExhaustLedger(tmp_path / "ex", sealer=sealer, clock=clock, **kw), sealer, clock


# ----------------------------------------------------------------------------- 용도 표

@pytest.mark.parametrize("taint,expect", [
    (F.ENGINE, ("resume", "delta", "share", "train")),
    ({"CAPTURE"}, ("resume", "delta", "share", "train")),
    ({"ASSET:CC0"}, ("resume", "delta", "share", "train")),
    (F.USER, ("resume", "delta")), (F.USER_D, ("resume", "delta")), (F.EXTGEN, ("resume", "delta")),
    ({"BIO"}, ("resume", "delta")), (F.COMP, ("resume", "delta")), (F.HOLD, ("resume", "delta")),
    ({"PUB-SEALED"}, ("resume", "delta")),
    ({"ASSET"}, ("resume", "delta")),        # 라이선스 미상 -> 학습·공유 금지
    ({"OPTIN"}, ("resume", "delta")), ({"D3"}, ("resume", "delta")),
])
def test_allowed_purposes_table(taint, expect):
    assert X.allowed_purposes(taint) == expect


def test_allowed_purposes_property_600_cases_matches_prov_oracle():
    pool = ["ENGINE", "CAPTURE", "ASSET", "ASSET:CC0", "ASSET:GPL", "USER", "USER-DERIVED", "BIO", "EXTGEN:a",
            "EXTGEN-DERIVED:a", "COMP:v", "HOLDOUT", "PUB-SEALED", "OPTIN", "D3", "NC", "HUMAN", "TEL-SYS",
            "EVAL-ONLY", "OPEN-GEN:m:h:train", "OPEN-GEN:m:h:bench", "IP-FLAG", "IP-FLAG:masked"]
    rng = random.Random(4)
    for _ in range(600):
        taint = frozenset(rng.sample(pool, rng.randrange(0, 5)))
        rec = P.as_record(taint)
        got = X.allowed_purposes(rec)
        want = ("resume", "delta") + (("share", "train") if rec.bits == 0 and P.train_ok(rec) else ())
        assert got == want, sorted(taint)
        if rec.ext or rec.comp or rec.hold:
            assert "train" not in got and "share" not in got
        if "train" in got:
            assert P.train_ok(rec)


def test_get_denied_purposes_raise_before_any_io(tmp_path):
    led, sealer, _ = mk(tmp_path)
    p = F.parts()
    led.put(p, F.latent(1), F.USER, **F.ext_kw())
    led.put(p, F.latent(2), F.COMP)
    led.put(p, F.latent(3), F.HOLD)
    led.put(p, F.latent(4), {"ASSET"})
    opens = len(sealer.open_calls)
    before = led.stats["denied"]
    cases = [(F.USER, F.ext_kw()), (F.COMP, {}), (F.HOLD, {}), ({"ASSET"}, {})]
    for prov, kw in cases:
        for purpose in ("share", "train"):
            with pytest.raises(P.FirewallError):
                led.get(p, prov, purpose=purpose, **kw)
    assert led.stats["denied"] == before + 8
    assert len(sealer.open_calls) == opens               # 파일을 열기 전에 막혔다
    for prov, kw in cases:
        for purpose in ("resume", "delta"):
            assert led.get(p, prov, purpose=purpose, **kw) is not None


@pytest.mark.parametrize("bad", ["", "SHARE", "bench", None, 3])
def test_unknown_purpose_rejected(tmp_path, bad):
    led, _, _ = mk(tmp_path)
    with pytest.raises(X.ExhaustSpecError):
        led.get(F.parts(), F.ENGINE, purpose=bad)


def test_engine_entry_can_be_shared_and_trained(tmp_path):
    led, _, _ = mk(tmp_path)
    p = F.parts()
    led.put(p, F.latent(5), F.ENGINE)
    for purpose in X.PURPOSES:
        assert led.get(p, F.ENGINE, purpose=purpose) is not None


# ----------------------------------------------------------------------------- ext 봉인

def test_ext_put_requires_sealer_and_writes_nothing(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    with pytest.raises(P.FirewallError):
        led.put(F.parts(), F.latent(6), F.USER, **F.ext_kw())
    assert F.files_under(tmp_path / "ex") == []
    with pytest.raises(P.FirewallError):
        led.get(F.parts(), F.USER, purpose="resume", **F.ext_kw())


def test_ext_put_requires_project_credentials(tmp_path):
    led, _, _ = mk(tmp_path)
    with pytest.raises(P.FirewallError):
        led.put(F.parts(), F.latent(6), F.USER)
    with pytest.raises(P.FirewallError):
        led.get(F.parts(), F.EXTGEN, purpose="resume", project_id=F.PROJ)
    assert F.files_under(tmp_path / "ex") == []


def test_ext_entry_sealed_on_disk_and_roundtrips(tmp_path):
    led, sealer, clock = mk(tmp_path)
    p, x = F.parts(), F.latent(7, scale=2.0)
    r = led.put(p, x, F.USER, stats={"g": 0.5}, **F.ext_kw())
    path = led.entry_path(r.key, F.USER, project_id=F.PROJ)
    rel = path.relative_to(led.root).parts
    assert rel[0] == "user_vault" and len(rel[1]) == 16 and rel[2:] == (r.key[:2], r.key + ".exh")
    assert F.PROJ not in "".join(rel)
    raw = path.read_bytes()
    assert raw[:5] == b"BUCS1" and struct.unpack("<d", raw[5:13]) == (clock.t,)
    q, s = X.quantize_x0(x)
    qb = q.numpy().tobytes()
    assert qb not in raw and qb[:64] not in raw
    assert b"BUCX1" not in raw and b'"shape"' not in raw and b"user_vault" not in raw
    for blob in (F.PROJ.encode(), F.PK):
        assert blob not in raw
    aad = r.key.encode() + struct.pack("<d", clock.t)
    assert sealer.seal_calls == [(aad, F.PROJ)]
    e = led.get(p, F.USER, purpose="resume", **F.ext_kw())
    assert torch.equal(e.q, q) and e.stats == {"g": 0.5} and e.created == clock.t
    assert sealer.open_calls == [(aad, F.PROJ)]
    assert r.nbytes == len(raw)


def test_no_secret_or_project_id_anywhere_on_disk(tmp_path):
    led, _, _ = mk(tmp_path)
    for i, prov in enumerate([F.USER, F.EXTGEN, F.ENGINE, F.COMP, F.HOLD]):
        led.put(F.parts(seed=i), F.latent(i), prov, **F.ext_kw())
    for f in F.files_under(tmp_path / "ex"):
        data = f.read_bytes()
        assert F.PK not in data and F.PROJ.encode() not in data and b"S" * 32 not in data
        assert F.PROJ not in str(f)


def test_ext_wrong_project_key_id_or_sealer_is_a_miss(tmp_path):
    led, _, clock = mk(tmp_path)
    p = F.parts()
    led.put(p, F.latent(8), F.USER, **F.ext_kw())
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw(key=F.PK2)) is None
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw(project=F.PROJ2)) is None
    assert led.get(p, F.ENGINE, purpose="resume") is None            # 비-ext 키로는 못 찾는다
    assert led.get(p, F.COMP, purpose="resume") is None
    other = X.ExhaustLedger(tmp_path / "ex", sealer=F.FakeSealer(secret=b"T" * 32), clock=clock)
    assert other.get(p, F.USER, purpose="resume", **F.ext_kw()) is None
    assert other.stats["corrupt"] == 1


def test_ext_tamper_ciphertext_or_created_is_a_miss(tmp_path):
    led, _, _ = mk(tmp_path)
    p = F.parts()
    r = led.put(p, F.latent(9), F.USER, **F.ext_kw())
    path = led.entry_path(r.key, F.USER, project_id=F.PROJ)
    raw = path.read_bytes()
    path.write_bytes(raw[:60] + bytes([raw[60] ^ 1]) + raw[61:])
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw()) is None
    assert led.stats["corrupt"] == 1
    forged = raw[:5] + struct.pack("<d", struct.unpack("<d", raw[5:13])[0] + 3600.0) + raw[13:]
    path.write_bytes(forged)                           # created를 늘려 TTL을 연장하려는 시도
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw()) is None
    assert led.stats["corrupt"] == 2


# ----------------------------------------------------------------------------- 구역 분리

def test_zones_are_separate_directories(tmp_path):
    led, _, _ = mk(tmp_path)
    for i, prov in enumerate([F.ENGINE, F.USER, F.EXTGEN, F.COMP, F.HOLD]):
        led.put(F.parts(seed=i), F.latent(i), prov, **F.ext_kw())
    tops = {f.relative_to(led.root).parts[0] for f in F.files_under(led.root)}
    assert tops == {"train_ok", "user_vault", "ext_vault", "comp_vault", "pub_bench"}
    assert len(led.iter_keys("train_ok")) == 1 and len(led.iter_keys("comp_vault")) == 1


def test_iter_keys_for_ext_requires_project_and_is_project_scoped(tmp_path):
    led, _, _ = mk(tmp_path)
    ka = [led.put(F.parts(seed=i), F.latent(i), F.USER, **F.ext_kw()).key for i in range(2)]
    kb = [led.put(F.parts(seed=i), F.latent(i), F.USER, **F.ext_kw(F.PROJ2, F.PK2)).key for i in range(3)]
    for ns in ("user_vault", "ext_vault"):
        with pytest.raises(P.FirewallError):
            led.iter_keys(ns)
    assert led.iter_keys("user_vault", project_id=F.PROJ) == sorted(ka)
    assert led.iter_keys("user_vault", project_id=F.PROJ2) == sorted(kb)
    assert led.iter_keys("user_vault", project_id="nobody") == []
    assert led.iter_keys("ext_vault", project_id=F.PROJ) == []
    assert set(ka).isdisjoint(kb)


# ----------------------------------------------------------------------------- 학습 분할

def _forge(raw: bytes, **hdr_over) -> bytes:
    (hl,) = struct.unpack("<I", raw[5:9])
    hdr = json.loads(raw[9:9 + hl])
    hdr.update(hdr_over)
    hb = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    body = raw[:5] + struct.pack("<I", len(hb)) + hb + raw[9 + hl:-32]
    return body + blake3.blake3(body).digest()


def test_iter_train_yields_only_clean_train_ok_entries_in_key_order(tmp_path):
    led, sealer, _ = mk(tmp_path)
    good = []
    for i in range(4):
        good.append(led.put(F.parts(seed=i), F.latent(i), F.ENGINE).key)
    led.put(F.parts(seed=10), F.latent(10), {"CAPTURE"})
    good.append(led.key(F.parts(seed=10), {"CAPTURE"}))
    skipped = {"asset": {"ASSET"}, "optin": {"OPTIN"}, "d3": {"D3"}}   # train_ok 구역이지만 학습 불가
    for i, (n, prov) in enumerate(skipped.items()):
        led.put(F.parts(seed=20 + i), F.latent(20 + i), prov)
    led.put(F.parts(seed=30), F.latent(30), F.USER, **F.ext_kw())
    led.put(F.parts(seed=31), F.latent(31), F.EXTGEN, **F.ext_kw())
    led.put(F.parts(seed=32), F.latent(32), F.COMP)
    led.put(F.parts(seed=33), F.latent(33), F.HOLD)
    # ext/comp/hold 파일을 쓰레기로 바꿔도 (열지 않으므로) 학습 분할은 영향이 없다
    for z in ("user_vault", "ext_vault", "comp_vault", "pub_bench"):
        for f in F.files_under(led.root / z):
            f.write_bytes(b"garbage")
    got = list(led.iter_train())
    assert [e.key for e in got] == sorted(good)
    assert all(isinstance(e, X.ExhaustEntry) and e.prov.bits == 0 and P.train_ok(e.prov) for e in got)
    assert sealer.open_calls == [] and led.stats["corrupt"] == 0
    assert [e.key for e in led.iter_train(branch="main")] == [e.key for e in got]


def test_iter_train_skips_forged_headers_and_sealed_files(tmp_path):
    led, sealer, _ = mk(tmp_path)
    r = led.put(F.parts(seed=1), F.latent(1), F.ENGINE)
    path = led.entry_path(r.key, F.ENGINE)
    raw = path.read_bytes()
    # (a) 유효한 체크섬을 가진 위조: 헤더 prov를 ext로 바꿈
    path.write_bytes(_forge(raw, prov=P.as_record(F.USER).to_dict()))
    assert list(led.iter_train()) == []
    # (b) 위조: 학습 불가 prov (ASSET 라이선스 미상)
    path.write_bytes(_forge(raw, prov=P.as_record({"ASSET"}).to_dict()))
    assert list(led.iter_train()) == []
    # (c) 봉인된 ext 파일을 train_ok 구역에 복사
    re_ = led.put(F.parts(seed=2), F.latent(2), F.USER, **F.ext_kw())
    sealed = led.entry_path(re_.key, F.USER, project_id=F.PROJ).read_bytes()
    dst = led.entry_path(re_.key, F.ENGINE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(sealed)
    assert list(led.iter_train()) == []
    assert sealer.open_calls == []                          # 봉인 파일을 열어 보지도 않는다
    # (d) 원본을 복구하면 다시 나온다
    path.write_bytes(raw)
    assert [e.key for e in led.iter_train()] == [r.key]


def test_iter_train_on_empty_ledger(tmp_path):
    led, _, _ = mk(tmp_path)
    assert list(led.iter_train()) == []


def test_iter_train_unknown_branch_rejected(tmp_path):
    led, _, _ = mk(tmp_path)
    led.put(F.parts(), F.latent(1), F.ENGINE)
    with pytest.raises(ValueError):
        list(led.iter_train(branch="nope"))


def test_iter_train_optin_branch_follows_fnd04_rules(tmp_path):
    led, _, _ = mk(tmp_path)
    k_eng = led.put(F.parts(seed=1), F.latent(1), F.ENGINE).key
    k_opt = led.put(F.parts(seed=2), F.latent(2), {"OPTIN"}).key
    led.put(F.parts(seed=3), F.latent(3), {"D3"})
    led.put(F.parts(seed=4), F.latent(4), F.USER, **F.ext_kw())
    assert [e.key for e in led.iter_train()] == [k_eng]
    assert [e.key for e in led.iter_train(branch="optin-v1")] == sorted([k_eng, k_opt])


# ----------------------------------------------------------------------------- TTL · purge

def test_ext_ttl_boundary_and_deletion(tmp_path):
    led, _, clock = mk(tmp_path)
    p = F.parts()
    r = led.put(p, F.latent(40), F.USER, **F.ext_kw())
    path = led.entry_path(r.key, F.USER, project_id=F.PROJ)
    clock.t += TTL                                             # 정확히 TTL: 아직 유효
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw()) is not None
    clock.t += 1
    assert led.get(p, F.USER, purpose="resume", **F.ext_kw()) is None
    assert led.stats["expired"] == 1 and not path.exists()


def test_non_ext_never_expires(tmp_path):
    led, _, clock = mk(tmp_path)
    p = F.parts()
    led.put(p, F.latent(41), F.ENGINE)
    led.put(p, F.latent(42), F.COMP)
    clock.t += 10 * 365 * 86400
    assert led.get(p, F.ENGINE, purpose="resume") is not None
    assert led.get(p, F.COMP, purpose="resume") is not None
    assert led.expire() == 0


def test_custom_ttl_and_expire_sweep_without_sealer(tmp_path):
    led, sealer, clock = mk(tmp_path, ext_ttl_s=100.0)
    for i in range(3):
        led.put(F.parts(seed=i), F.latent(i), F.USER, **F.ext_kw())
        clock.t += 60.0
    # 시각: 0, 60, 120 에 저장, 지금 180. 나이 180, 120, 60 -> 앞의 둘이 만료
    plain = X.ExhaustLedger(tmp_path / "ex", clock=clock, ext_ttl_s=100.0)   # sealer 없음
    assert plain.expire() == 2
    assert len(led.iter_keys("user_vault", project_id=F.PROJ)) == 1
    assert plain.expire() == 0
    assert sealer.open_calls == []


def test_purge_project_removes_only_that_projects_ext_entries(tmp_path):
    led, _, _ = mk(tmp_path)
    for i in range(2):
        led.put(F.parts(seed=i), F.latent(i), F.USER, **F.ext_kw())
    led.put(F.parts(seed=5), F.latent(5), F.EXTGEN, **F.ext_kw())
    led.put(F.parts(seed=6), F.latent(6), F.USER, **F.ext_kw(F.PROJ2, F.PK2))
    led.put(F.parts(seed=7), F.latent(7), F.ENGINE)
    led.put(F.parts(seed=8), F.latent(8), F.COMP)
    assert led.purge_project(F.PROJ) == 3
    assert led.iter_keys("user_vault", project_id=F.PROJ) == [] and led.iter_keys("ext_vault", project_id=F.PROJ) == []
    assert led.get(F.parts(seed=0), F.USER, purpose="resume", **F.ext_kw()) is None
    assert led.get(F.parts(seed=6), F.USER, purpose="resume", **F.ext_kw(F.PROJ2, F.PK2)) is not None
    assert led.get(F.parts(seed=7), F.ENGINE, purpose="resume") is not None
    assert led.get(F.parts(seed=8), F.COMP, purpose="resume") is not None
    assert led.purge_project(F.PROJ) == 0 and led.purge_project("nobody") == 0
    with pytest.raises(X.ExhaustSpecError):
        led.purge_project("")

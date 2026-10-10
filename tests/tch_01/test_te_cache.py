"""TCH-01 텍스트 임베딩 캐시: 미스만 한 번에 인코딩, 디스크 재사용, 해제 후 동작."""
import os

import pytest
import torch

from buc.core import det
from buc.teacher import zimage as Z

D = 8


def _enc_factory(log, nan=False, bad_len=False, dim=D, grad=False):
    def enc(prompts):
        log.append(list(prompts))
        if bad_len:
            return [torch.zeros(2, dim)]
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(sum(p.encode()) + 1)
            t = torch.randn(len(p) % 5 + 1, dim, generator=g)
            if nan:
                t[0, 0] = float("nan")
            out.append(t.requires_grad_(True) if grad else t)
        return out
    return enc


def _cache(log, **kw):
    enc_kw = {k: kw.pop(k) for k in ("nan", "bad_len", "dim", "grad") if k in kw}
    kw.setdefault("namespace", "ns")
    kw.setdefault("max_len", 16)
    kw.setdefault("text_dim", D)
    return Z.TextEmbedCache(_enc_factory(log, **enc_kw), **kw)


def test_key_is_hash_parts_hex_and_sensitive_to_every_part():
    c = _cache([], namespace="Z@main|bf16", max_len=512)
    assert c.key("a cat") == det.hash_parts("buc.tch01.te.v1", "Z@main|bf16", 512, "a cat").hex()
    assert len(c.key("a cat")) == 64
    ks = {c.key("a cat"), c.key("a cat "), c.key("a dog"),
          _cache([], namespace="other", max_len=512).key("a cat"),
          _cache([], namespace="Z@main|bf16", max_len=256).key("a cat")}
    assert len(ks) == 5


def test_misses_are_deduplicated_and_batched_once_in_first_seen_order():
    log = []
    c = _cache(log)
    out = c.encode(["b", "a", "b", "c"])
    assert log == [["b", "a", "c"]]
    assert len(out) == 4 and torch.equal(out[0], out[2])
    assert out[0] is not out[2]
    assert [o.shape[-1] for o in out] == [D] * 4
    assert c.stats == {"hits": 0, "disk_hits": 0, "misses": 3, "encode_calls": 1}
    out2 = c.encode(["a", "z"])
    assert log[1] == ["z"]                               # a는 히트, z만 인코딩
    assert torch.equal(out2[0], out[1])
    assert c.stats["hits"] == 1 and c.stats["misses"] == 4 and c.stats["encode_calls"] == 2
    c.encode(["a", "b", "c", "z"])
    assert len(log) == 2                                 # 전부 히트면 인코더 호출 0


def test_returns_independent_copies_without_grad():
    c = _cache([], grad=True)
    a = c.encode(["x"])[0]
    assert not a.requires_grad
    ref = a.clone()
    a.zero_()                                            # 호출자가 망쳐도
    b = c.encode(["x"])[0]
    assert torch.equal(b, ref) and not b.requires_grad   # 캐시는 무사


def test_device_and_dtype_conversion_and_empty_and_type_errors():
    c = _cache([])
    assert c.encode([]) == []
    t = c.encode(["x"], device="cpu", dtype=torch.bfloat16)[0]
    assert t.dtype == torch.bfloat16 and t.device.type == "cpu"
    f = c.encode(["x"])[0]
    assert torch.equal(t, f.to(torch.bfloat16))
    for bad in ([1], [None], ["ok", b"bytes"]):
        with pytest.raises(TypeError):
            c.encode(bad)


def test_encoder_output_validation():
    with pytest.raises(Z.TeacherError):
        _cache([], bad_len=True).encode(["a", "b"])                 # 개수 불일치
    with pytest.raises(Z.TeacherSpecError):
        _cache([], dim=D + 1).encode(["a"])                          # 임베딩 차원 불일치
    with pytest.raises(Z.TeacherNumericError):
        _cache([], nan=True).encode(["a"])                           # NaN
    log = []
    c = Z.TextEmbedCache(lambda ps: [torch.zeros(3) for _ in ps], namespace="n", max_len=4)
    with pytest.raises(Z.TeacherError):
        c.encode(["a"])                                              # 2D 아님
    c2 = _cache(log, nan=True)
    with pytest.raises(Z.TeacherNumericError):
        c2.encode(["a"])
    c2.unload()
    # 실패한 결과는 캐시에 남지 않는다: 해제 후 같은 프롬프트는 히트가 아니라 미스
    with pytest.raises(Z.TEUnloadedError):
        c2.encode(["a"])


def test_disk_cache_roundtrip_across_instances_is_bit_identical(tmp_path):
    log1, log2 = [], []
    c1 = _cache(log1, cache_dir=tmp_path / "te")
    a = c1.encode(["hello", "world"], dtype=torch.bfloat16)
    files = sorted(p.name for p in (tmp_path / "te").iterdir())
    assert files == sorted([c1.key("hello") + ".pt", c1.key("world") + ".pt"])   # 임시 파일 잔재 없음

    def boom(prompts):
        raise AssertionError("디스크 히트인데 인코더가 불렸다")
    c2 = Z.TextEmbedCache(boom, namespace="ns", max_len=16, text_dim=D, cache_dir=tmp_path / "te")
    b = c2.encode(["hello", "world"], dtype=torch.bfloat16)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert c2.stats["disk_hits"] == 2 and c2.stats["misses"] == 0 and c2.stats["encode_calls"] == 0
    c2.encode(["hello"])
    assert c2.stats["hits"] == 1                                    # 두 번째부터는 메모리 히트
    # 네임스페이스가 다르면 같은 디렉터리에서도 섞이지 않는다
    c3 = _cache(log2, namespace="other", cache_dir=tmp_path / "te")
    c3.encode(["hello"])
    assert log2 == [["hello"]]


def test_disk_cache_preserves_bf16_encoder_output_exactly(tmp_path):
    t = torch.randn(3, D).to(torch.bfloat16)
    c1 = Z.TextEmbedCache(lambda ps: [t.clone() for _ in ps], namespace="n", max_len=4, cache_dir=tmp_path)
    c1.encode(["p"])
    c2 = Z.TextEmbedCache(lambda ps: pytest.fail("no"), namespace="n", max_len=4, cache_dir=tmp_path)
    out = c2.encode(["p"])[0]
    assert out.dtype == torch.bfloat16 and torch.equal(out, t)


def test_corrupt_disk_entry_is_a_miss_and_gets_repaired(tmp_path):
    log = []
    c = _cache(log, cache_dir=tmp_path)
    (tmp_path / (c.key("x") + ".pt")).write_bytes(b"not a torch file")
    out = c.encode(["x"])[0]
    assert log == [["x"]]
    c2 = Z.TextEmbedCache(lambda ps: pytest.fail("no"), namespace="ns", max_len=16, text_dim=D, cache_dir=tmp_path)
    assert torch.equal(c2.encode(["x"])[0], out)


def test_unload_calls_release_once_and_serves_hits_but_refuses_misses():
    log, rel = [], []
    c = _cache(log, release_fn=lambda: rel.append(1))
    assert not c.unloaded
    a = c.encode(["x"])[0]
    c.unload()
    c.unload()
    assert c.unloaded and rel == [1]
    assert torch.equal(c.encode(["x"])[0], a)                      # 히트는 계속 제공
    n = len(log)
    with pytest.raises(Z.TEUnloadedError):
        c.encode(["x", "new"])                                      # 미스가 하나라도 있으면 거절
    assert len(log) == n                                            # 인코더는 호출되지 않음
    assert isinstance(Z.TEUnloadedError("x"), Z.TeacherError)


def test_unload_without_release_fn_is_fine():
    c = _cache([])
    c.unload()
    assert c.unloaded


def test_unloaded_cache_can_still_serve_from_disk(tmp_path):
    c = _cache([], cache_dir=tmp_path)
    a = c.encode(["x"])[0]
    c2 = _cache([], cache_dir=tmp_path)
    c2.unload()                                                     # 해제 상태로 시작해도 디스크 히트는 된다
    assert torch.equal(c2.encode(["x"])[0], a)
    assert os.listdir(tmp_path) == [c.key("x") + ".pt"]

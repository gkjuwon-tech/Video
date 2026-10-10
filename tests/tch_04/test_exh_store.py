"""TCH-04 저장소: int8 왕복, 파일 배치, 손상 8종, 원자성, write-once, 동시 쓰기, 통계, 임시 파일 복구."""
import json
import struct
import threading

import blake3
import pytest
import torch

from buc.gov import prov as P
from buc.teacher import exhaust as X
import tch04_fakes as F


@pytest.fixture()
def led(tmp_path):
    return X.ExhaustLedger(tmp_path / "ex", clock=F.Clock())


def test_import_surface():
    for n in ["quantize_x0", "dequantize_x0", "KeyParts", "compute_key", "teacher_id", "Sealer", "ExhaustEntry",
              "PutResult", "allowed_purposes", "ExhaustLedger", "ExhaustError", "ExhaustSpecError",
              "ExhaustNumericError", "FirewallError", "KEY_TAG", "QUANT_MODE", "Q_MAX", "EXT_TTL_S", "PURPOSES"]:
        assert hasattr(X, n), n
    assert issubclass(X.ExhaustError, RuntimeError)


def test_roundtrip_is_exactly_the_quantized_value(led):
    p, x = F.parts(), F.latent(11, scale=2.0)
    r = led.put(p, x, F.ENGINE)
    assert r.created and not r.conflict and r.key == led.key(p, F.ENGINE)
    e = led.get(p, F.ENGINE, purpose="resume")
    q, s = X.quantize_x0(x)
    assert torch.equal(e.q, q) and torch.equal(e.scale, s)
    assert torch.equal(e.x0_hat, X.dequantize_x0(q, s))
    assert e.q.dtype == torch.int8 and e.scale.dtype == torch.float32 and e.x0_hat.dtype == torch.float32
    assert not e.x0_hat.requires_grad
    assert e.key == r.key and e.prov == P.as_record(F.ENGINE) and e.created == 1_000_000.0
    assert e.stats == {}


def test_entry_value_is_a_copy(led):
    p = F.parts()
    led.put(p, F.latent(12), F.ENGINE)
    e1 = led.get(p, F.ENGINE, purpose="resume")
    e1.q.zero_()
    e1.x0_hat.zero_()
    e2 = led.get(p, F.ENGINE, purpose="resume")
    assert torch.count_nonzero(e2.q) > 0


def test_stats_roundtrip_and_validation(led):
    p = F.parts()
    led.put(p, F.latent(13), F.ENGINE, stats={"gauge_mean": 0.25, "resid": 3, "z": -1.5})
    e = led.get(p, F.ENGINE, purpose="delta")
    assert e.stats == {"gauge_mean": 0.25, "resid": 3.0, "z": -1.5}
    assert all(isinstance(v, float) for v in e.stats.values())
    for bad in [{"a": float("nan")}, {"a": float("inf")}, {"a": "x"}, {1: 2.0}, [("a", 1.0)]]:
        with pytest.raises(X.ExhaustSpecError):
            led.put(F.parts(seed=99), F.latent(1), F.ENGINE, stats=bad)
    assert led.get(F.parts(seed=99), F.ENGINE, purpose="resume") is None


def test_on_disk_layout_and_size(led, tmp_path):
    p = F.parts(height=256, width=256)                # latent 32x32
    x = F.latent(14, c=16, h=32, w=32)
    r = led.put(p, x, F.ENGINE)
    path = led.entry_path(r.key, F.ENGINE)
    assert path.relative_to(led.root).parts == ("train_ok", r.key[:2], r.key + ".exh")
    raw = path.read_bytes()
    assert r.nbytes == len(raw) == path.stat().st_size
    assert len(raw) < 0.30 * x.numel() * 4            # fp32 대비 ~1/4
    assert len(raw) >= x.numel()                      # int8 1바이트/원소 이상은 든다
    # 레이아웃: 매직 + u32 헤더 길이 + JSON + q + scale + blake3
    assert raw[:5] == b"BUCX1"
    (hl,) = struct.unpack("<I", raw[5:9])
    hdr = json.loads(raw[9:9 + hl].decode())
    assert hdr["key"] == r.key and hdr["shape"] == [16, 32, 32] and hdr["quant"] == X.QUANT_MODE
    assert hdr["prov"] == P.as_record(F.ENGINE).to_dict() and hdr["created"] == 1_000_000.0
    off = 9 + hl
    q, s = X.quantize_x0(x)
    assert raw[off:off + q.numel()] == q.numpy().tobytes()
    off += q.numel()
    assert raw[off:off + 64] == s.numpy().astype("<f4").tobytes()
    assert raw[-32:] == blake3.blake3(raw[:-32]).digest() and len(raw) == off + 64 + 32


@pytest.mark.parametrize("name,prov,ns", [("engine", F.ENGINE, "train_ok"), ("comp", F.COMP, "comp_vault"),
                                          ("hold", F.HOLD, "pub_bench")])
def test_non_ext_zones_by_namespace(led, name, prov, ns):
    r = led.put(F.parts(), F.latent(15), prov)
    assert led.entry_path(r.key, prov).relative_to(led.root).parts[0] == ns
    assert led.entry_path(r.key, prov).is_file()


def test_shape_must_match_height_width_over_8(led):
    with pytest.raises(X.ExhaustSpecError):
        led.put(F.parts(height=128, width=128), F.latent(1), F.ENGINE)       # 16x16 기대, 8x8 제공
    with pytest.raises(X.ExhaustSpecError):
        led.put(F.parts(), torch.zeros(16, 8), F.ENGINE)
    assert F.files_under(led.root) == []


def test_non_finite_latent_rejected_and_nothing_written(led):
    x = F.latent(16)
    x[0, 0, 0] = float("nan")
    with pytest.raises(X.ExhaustNumericError):
        led.put(F.parts(), x, F.ENGINE)
    assert F.files_under(led.root) == []


def test_prov_required_for_put_and_get(led):
    with pytest.raises(TypeError):
        led.put(F.parts(), F.latent(1), None)
    with pytest.raises(TypeError):
        led.get(F.parts(), "ENGINE", purpose="resume")


def test_put_detaches_grad_and_accepts_noncontiguous(led):
    x = F.latent(17).requires_grad_(True)
    led.put(F.parts(), x, F.ENGINE)
    xt = F.latent(18).permute(0, 2, 1).contiguous().permute(0, 2, 1)           # 비연속 뷰
    led.put(F.parts(seed=5), xt, F.ENGINE)
    e = led.get(F.parts(seed=5), F.ENGINE, purpose="resume")
    assert torch.equal(e.q, X.quantize_x0(xt)[0])


def test_write_once_conflict_and_identical_reput(led):
    p = F.parts()
    a, b = F.latent(20), F.latent(21)
    r1 = led.put(p, a, F.ENGINE)
    r2 = led.put(p, a, F.ENGINE)
    assert r1.created and not r2.created and not r2.conflict
    r3 = led.put(p, b, F.ENGINE)
    assert not r3.created and r3.conflict
    e = led.get(p, F.ENGINE, purpose="resume")
    assert torch.equal(e.q, X.quantize_x0(a)[0])           # 첫 값이 이긴다
    st = led.stats
    assert st["writes"] == 1 and st["conflicts"] == 1


def test_missing_returns_none_and_counts(led):
    assert led.get(F.parts(), F.ENGINE, purpose="resume") is None
    st = led.stats
    assert st["misses"] == 1 and st["hits"] == 0
    assert set(st) >= {"hits", "misses", "writes", "conflicts", "corrupt", "expired", "denied"}
    st["hits"] = 99                                         # 복사본
    assert led.stats["hits"] == 0


def _corruptors():
    def truncate_half(b):
        return b[: len(b) // 2]

    def empty(b):
        return b""

    def flip_payload(b):
        i = len(b) - 32 - 70
        return b[:i] + bytes([b[i] ^ 0x5A]) + b[i + 1:]

    def flip_header(b):
        return b[:12] + bytes([b[12] ^ 0x01]) + b[13:]

    def flip_checksum(b):
        return b[:-1] + bytes([b[-1] ^ 0xFF])

    def bad_magic(b):
        return b"XXXXX" + b[5:]

    def append_junk(b):
        return b + b"junk"

    def drop_last_byte(b):
        return b[:-1]
    return [truncate_half, empty, flip_payload, flip_header, flip_checksum, bad_magic, append_junk, drop_last_byte]


@pytest.mark.parametrize("corrupt", _corruptors(), ids=lambda f: f.__name__)
def test_corruption_is_a_miss_never_a_value(led, corrupt):
    p = F.parts()
    r = led.put(p, F.latent(23), F.ENGINE)
    path = led.entry_path(r.key, F.ENGINE)
    path.write_bytes(corrupt(path.read_bytes()))
    assert led.get(p, F.ENGINE, purpose="resume") is None
    assert led.stats["corrupt"] == 1 and led.stats["misses"] == 1
    # 깨진 파일은 없는 것으로 보고 다시 쓸 수 있다
    r2 = led.put(p, F.latent(23), F.ENGINE)
    assert r2.created and led.get(p, F.ENGINE, purpose="resume") is not None


def test_file_moved_to_another_key_is_detected(led):
    pa, pb = F.parts(seed=1), F.parts(seed=2)
    ra = led.put(pa, F.latent(24), F.ENGINE)
    path_b = led.entry_path(led.key(pb, F.ENGINE), F.ENGINE)
    path_b.parent.mkdir(parents=True, exist_ok=True)
    path_b.write_bytes(led.entry_path(ra.key, F.ENGINE).read_bytes())
    assert led.get(pb, F.ENGINE, purpose="resume") is None
    assert led.stats["corrupt"] == 1


def test_header_prov_mismatch_is_detected(led):
    """같은 파일을 다른 prov 구역 경로로 옮겨 놓아도(키가 다르므로) 읽히지 않는다."""
    p = F.parts()
    ra = led.put(p, F.latent(25), F.ENGINE)
    rb_key = led.key(p, F.COMP)
    dst = led.entry_path(rb_key, F.COMP)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(led.entry_path(ra.key, F.ENGINE).read_bytes())
    assert led.get(p, F.COMP, purpose="resume") is None


def test_sealer_failure_leaves_no_partial_files(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex", sealer=F.FakeSealer(fail_seal=True))
    with pytest.raises(RuntimeError):
        led.put(F.parts(), F.latent(26), F.USER, **F.ext_kw())
    assert F.files_under(tmp_path / "ex") == []
    assert led.get(F.parts(), F.USER, purpose="resume", **F.ext_kw()) is None


def test_stray_tmp_files_ignored_and_recovered(led):
    p = F.parts()
    r = led.put(p, F.latent(27), F.ENGINE)
    d = led.entry_path(r.key, F.ENGINE).parent
    stray = d / ("f" * 64 + ".exh.tmp.4242")
    stray.write_bytes(b"half written")
    other = d / "README.keep"
    other.write_bytes(b"x")
    assert led.iter_keys("train_ok") == [r.key]
    assert led.recover() == 1
    assert not stray.exists() and other.exists()
    assert led.recover() == 0
    assert led.get(p, F.ENGINE, purpose="resume") is not None


def test_two_instances_share_the_directory(tmp_path):
    a = X.ExhaustLedger(tmp_path / "ex")
    b = X.ExhaustLedger(tmp_path / "ex")
    p = F.parts()
    a.put(p, F.latent(28), F.ENGINE)
    e = b.get(p, F.ENGINE, purpose="resume")
    assert e is not None and torch.equal(e.x0_hat, a.get(p, F.ENGINE, purpose="resume").x0_hat)
    assert b.stats["hits"] == 1 and a.stats["hits"] == 1


def test_concurrent_identical_puts(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    p, x = F.parts(), F.latent(29)
    errs, results = [], []

    def work():
        try:
            results.append(led.put(p, x, F.ENGINE))
        except Exception as e:           # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert any(r.created for r in results) and not any(r.conflict for r in results)
    e = led.get(p, F.ENGINE, purpose="resume")
    assert torch.equal(e.q, X.quantize_x0(x)[0])
    assert len(F.files_under(led.root)) == 1


def test_iter_keys_sorted_and_zone_scoped(led):
    ks = []
    for i in range(5):
        ks.append(led.put(F.parts(seed=i), F.latent(i), F.ENGINE).key)
    led.put(F.parts(seed=0), F.latent(0), F.COMP)
    assert led.iter_keys("train_ok") == sorted(ks)
    assert len(led.iter_keys("comp_vault")) == 1 and led.iter_keys("pub_bench") == []
    with pytest.raises(X.ExhaustSpecError):
        led.iter_keys("not_a_zone")


@pytest.mark.parametrize("dev", [pytest.param("cuda", marks=pytest.mark.gpu)])
def test_gpu_tensor_equals_cpu(led, dev):
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    x = F.latent(30)
    r1 = led.put(F.parts(seed=1), x, F.ENGINE)
    r2 = led.put(F.parts(seed=2), x.to(dev), F.ENGINE)
    e1 = led.get(F.parts(seed=1), F.ENGINE, purpose="resume")
    e2 = led.get(F.parts(seed=2), F.ENGINE, purpose="resume")
    assert torch.equal(e1.q, e2.q) and torch.equal(e1.scale, e2.scale) and r1.nbytes == r2.nbytes

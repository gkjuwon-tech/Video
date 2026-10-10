"""RUN-08: 장치 판별·분위수·시간 측정 호출 순서·비밀 가림."""
import json
import math

import pytest
from run08_helpers import Clock, FakeTorch, fake_5090, fake_cpu, fake_t4, GIB


# ---------------------------------------------------------------- is_t4_name
@pytest.mark.parametrize("name,expected", [
    ("Tesla T4", True),
    ("NVIDIA T4", True),
    ("NVIDIA T4 16GB", True),
    ("tesla t4", True),
    ("T4", True),
    ("NVIDIA GeForce RTX 5090", False),
    ("NVIDIA GeForce RTX 4090", False),
    ("Quadro T400", False),
    ("Quadro T4000", False),
    ("NVIDIA A100-SXM4-40GB", False),
    ("cpu", False),
    ("", False),
])
def test_is_t4_name(t4, name, expected):
    assert t4.is_t4_name(name) is expected


# -------------------------------------------------------------------- redact
def test_redact_rules(t4):
    secret = "hf_" + "A" * 24
    assert secret not in t4.redact(f"failed with {secret} ok")
    assert t4.redact("sk-abcdef123456") == "***"
    assert t4.redact("ghp_abcdefgh1234") == "***"
    assert t4.redact("api_key = SUPERSECRET") == "api_key=***"
    assert t4.redact("TOKEN: abc") == "TOKEN=***"
    assert t4.redact("x" * 40) == "***"
    assert t4.redact("CUDA out of memory. Tried to allocate 2.00 GiB") == "CUDA out of memory. Tried to allocate 2.00 GiB"


# -------------------------------------------------------------------- is_oom
class OutOfMemoryError(RuntimeError):
    pass


class SubOOM(OutOfMemoryError):
    pass


def test_is_oom(t4):
    assert t4.is_oom(OutOfMemoryError("boom"))
    assert t4.is_oom(SubOOM("boom"))           # MRO 안에 이름이 있으면 된다
    assert t4.is_oom(MemoryError())
    assert t4.is_oom(RuntimeError("CUDA Out Of Memory. Tried to allocate"))
    assert not t4.is_oom(RuntimeError("illegal memory access"))
    assert not t4.is_oom(ValueError("x"))


# --------------------------------------------------------- summarize_samples
def test_summarize_samples_numbers(t4):
    tm = t4.summarize_samples([0.5, 0.1, 0.3, 0.2, 0.4], warmup=2)
    assert tm.n == 5 and tm.warmup == 2
    assert tm.samples_s == (0.5, 0.1, 0.3, 0.2, 0.4)          # 입력 순서 유지
    assert tm.min_s == 0.1 and tm.max_s == 0.5
    assert tm.median_s == pytest.approx(0.3, abs=1e-12)
    assert tm.mean_s == pytest.approx(0.3, abs=1e-12)
    # p95: pos = 0.95*4 = 3.8 -> 0.4 + 0.8*(0.5-0.4)
    assert tm.p95_s == pytest.approx(0.48, abs=1e-12)


def test_summarize_samples_even_and_single(t4):
    tm = t4.summarize_samples([4.0, 1.0, 3.0, 2.0])
    assert tm.median_s == pytest.approx(2.5, abs=1e-12)        # pos 1.5
    assert tm.p95_s == pytest.approx(3.85, abs=1e-12)          # pos 2.85 -> 3 + 0.85*1
    one = t4.summarize_samples([0.7])
    assert one.n == 1
    assert one.median_s == one.p95_s == one.min_s == one.max_s == one.mean_s == 0.7
    assert one.warmup == 0


def test_summarize_samples_is_order_invariant(t4):
    a = t4.summarize_samples([0.9, 0.1, 0.5, 0.7])
    b = t4.summarize_samples([0.1, 0.5, 0.7, 0.9])
    assert (a.median_s, a.p95_s, a.mean_s, a.min_s, a.max_s) == (b.median_s, b.p95_s, b.mean_s, b.min_s, b.max_s)


@pytest.mark.parametrize("bad", [[], [-0.1], [float("nan")], [float("inf")], [0.1, True], [0.1, "x"]])
def test_summarize_samples_rejects_bad(t4, bad):
    with pytest.raises(ValueError):
        t4.summarize_samples(bad)


# ---------------------------------------------------------------- time_calls
def test_time_calls_exact_call_order(t4):
    events = []
    clock = Clock(10.0, events)

    def fn():
        events.append("fn")
        clock.advance(0.5)

    def sync():
        events.append("sync")

    tm = t4.time_calls(fn, repeats=2, warmup=1, sync=sync, clock=clock)
    assert events == (["fn", "sync"]
                      + ["sync", "clock", "fn", "sync", "clock"] * 2)
    assert tm.samples_s == (0.5, 0.5)
    assert tm.n == 2 and tm.warmup == 1
    assert tm.median_s == 0.5


def test_time_calls_samples_follow_clock(t4):
    clock = Clock(100.0)
    durs = iter([0.25, 0.5, 0.125, 1.0])

    def fn():
        clock.advance(next(durs))

    tm = t4.time_calls(fn, repeats=3, warmup=1, clock=clock)   # 워밍업이 첫 값(0.25)을 쓴다
    assert tm.samples_s == (0.5, 0.125, 1.0)
    assert tm.median_s == 0.5
    assert tm.min_s == 0.125 and tm.max_s == 1.0


def test_time_calls_without_sync_and_call_counts(t4):
    calls = []
    clock = Clock(0.0)
    tm = t4.time_calls(lambda: calls.append(1), repeats=4, warmup=3, clock=clock)
    assert len(calls) == 7 and tm.n == 4 and tm.warmup == 3


def test_time_calls_validates_and_propagates(t4):
    with pytest.raises(ValueError):
        t4.time_calls(lambda: None, repeats=0)
    with pytest.raises(ValueError):
        t4.time_calls(lambda: None, repeats=1, warmup=-1)

    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        t4.time_calls(boom, repeats=1, warmup=0)


# ------------------------------------------------------------- detect_device
def test_detect_device_t4(t4):
    d = t4.detect_device(fake_t4())
    assert d.kind == "cuda" and d.is_t4 is True
    assert d.name == "Tesla T4"
    assert d.total_mem_gib == pytest.approx(15.0, abs=1e-9)
    assert d.capability == (7, 5)
    assert d.torch_version == "2.9.1+cu128" and d.cuda_version == "12.8"


def test_detect_device_5090_is_not_t4(t4):
    d = t4.detect_device(fake_5090())
    assert d.kind == "cuda" and d.is_t4 is False
    assert d.total_mem_gib == pytest.approx(32.0, abs=1e-9)
    assert d.capability == (12, 0)


def test_detect_device_cpu(t4):
    d = t4.detect_device(fake_cpu())
    assert d.kind == "cpu" and d.name == "cpu" and d.is_t4 is False
    assert d.total_mem_gib is None and d.capability is None


def test_detect_device_real_torch_on_cpu_container(t4):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA 장치가 있는 호스트")
    d = t4.detect_device()
    assert d.kind == "cpu" and d.is_t4 is False
    assert d.torch_version == torch.__version__


def test_device_info_dict_has_only_whitelisted_keys_and_roundtrips(t4):
    d = t4.detect_device(fake_t4())
    dd = d.to_dict()
    assert set(dd) == {"kind", "name", "total_mem_gib", "capability", "torch_version", "cuda_version", "is_t4"}
    json.dumps(dd)
    assert t4.DeviceInfo.from_dict(json.loads(json.dumps(dd))) == d


# ----------------------------------------------------------------- make_sync
def test_make_sync_cuda_calls_synchronize_once(t4):
    tt = fake_t4()
    sync = t4.make_sync(t4.detect_device(tt), tt)
    sync()
    assert tt.cuda.calls == ["sync"]


def test_make_sync_cpu_is_noop(t4):
    tt = fake_cpu()
    sync = t4.make_sync(t4.detect_device(tt), tt)
    assert sync() is None
    assert tt.cuda.calls == []


def test_constants(t4):
    assert t4.RES_4K == (3840, 2160)
    assert t4.GATE_PRIMS == 3_000_000
    assert t4.GATE_4K_S == 0.6
    assert t4.GIB == GIB
    assert t4.SCHEMA_VERSION == 1
    assert math.isfinite(t4.GATE_4K_S)
    assert isinstance(FakeTorch(), FakeTorch)

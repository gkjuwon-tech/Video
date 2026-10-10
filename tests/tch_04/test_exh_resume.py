"""TCH-04 세션 재개: get_or_compute 의미, 끊긴 8-step 디노이즈의 비트 동일 재개, 호출 수, 델타 재사용."""
import pytest
import torch

from buc.gov import prov as P
from buc.teacher import exhaust as X
from buc.teacher.sample import DEFAULT_TABLE, euler_step
import tch04_fakes as F

SCHED = DEFAULT_TABLE.schedule(0, 8)


class Boom(RuntimeError):
    pass


class Teacher:
    """결정론 가짜 교사: x0 = f(z, sigma). 호출 수를 센다. crash_at번째 호출(1부터)에서 죽는다."""

    def __init__(self, crash_at=None):
        self.calls, self.crash_at = 0, crash_at

    def __call__(self, z, sigma):
        self.calls += 1
        if self.crash_at is not None and self.calls == self.crash_at:
            raise Boom()
        return torch.tanh(z) * (1.0 - sigma) + 0.3 * torch.sin(3.0 * z + sigma)


def denoise(led, teacher, prov, z0, *, world=F.WORLD, view="v0", **kw):
    z, xs = z0, []
    for sig, sig_next in SCHED:
        p = F.parts(world=world, view=view, sigma=sig)
        e = led.get_or_compute(p, prov, lambda z=z, sig=sig: teacher(z, sig), purpose="resume", **kw)
        xs.append(e.x0_hat)
        z = euler_step(z, e.x0_hat, sig, sig_next)
    return z, xs


@pytest.fixture()
def z0():
    return F.latent(100)


def test_get_or_compute_miss_then_hit_returns_the_stored_value(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    calls = []
    x = F.latent(1, scale=2.0)

    def compute():
        calls.append(1)
        return x
    p = F.parts()
    e1 = led.get_or_compute(p, F.ENGINE, compute)
    e2 = led.get_or_compute(p, F.ENGINE, compute)
    assert len(calls) == 1
    assert torch.equal(e1.x0_hat, e2.x0_hat) and torch.equal(e1.q, e2.q)
    assert torch.equal(e1.x0_hat, X.dequantize_x0(*X.quantize_x0(x)))        # 원본 fp32가 아니라 저장된 값
    assert not torch.equal(e1.x0_hat, x)
    st = led.stats
    assert (st["hits"], st["misses"], st["writes"]) == (1, 1, 1)


def test_compute_may_return_tensor_and_stats(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    e = led.get_or_compute(F.parts(), F.ENGINE, lambda: (F.latent(2), {"gauge": 1.5}))
    assert e.stats == {"gauge": 1.5}
    assert led.get(F.parts(), F.ENGINE, purpose="delta").stats == {"gauge": 1.5}


def test_failed_compute_stores_nothing_and_can_be_retried(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    t = Teacher(crash_at=1)
    p = F.parts()
    with pytest.raises(Boom):
        led.get_or_compute(p, F.ENGINE, lambda: t(F.latent(3), 0.5))
    assert F.files_under(tmp_path / "ex") == [] and led.stats["writes"] == 0
    e = led.get_or_compute(p, F.ENGINE, lambda: t(F.latent(3), 0.5))
    assert t.calls == 2 and e is not None


def test_bad_compute_outputs_rejected_and_not_stored(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    nan = F.latent(4)
    nan[1, 1, 1] = float("nan")
    with pytest.raises(X.ExhaustNumericError):
        led.get_or_compute(F.parts(), F.ENGINE, lambda: nan)
    with pytest.raises(X.ExhaustSpecError):
        led.get_or_compute(F.parts(), F.ENGINE, lambda: torch.zeros(16, 4, 4))
    assert F.files_under(tmp_path / "ex") == []


def test_denied_purpose_or_missing_credentials_never_call_compute(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex", sealer=F.FakeSealer())
    calls = []

    def compute():
        calls.append(1)
        return F.latent(5)
    with pytest.raises(P.FirewallError):
        led.get_or_compute(F.parts(), F.USER, compute, purpose="train", **F.ext_kw())
    with pytest.raises(P.FirewallError):
        led.get_or_compute(F.parts(), F.USER, compute)                        # 프로젝트 자격 없음
    with pytest.raises(X.ExhaustSpecError):
        led.get_or_compute(F.parts(), F.ENGINE, compute, purpose="bogus")
    assert calls == [] and F.files_under(tmp_path / "ex") == []


def test_uninterrupted_run_makes_8_calls_and_second_run_zero(tmp_path, z0):
    led = X.ExhaustLedger(tmp_path / "ex")
    t1 = Teacher()
    z_a, xs_a = denoise(led, t1, F.ENGINE, z0)
    assert t1.calls == 8
    t2 = Teacher()
    z_b, xs_b = denoise(X.ExhaustLedger(tmp_path / "ex"), t2, F.ENGINE, z0)
    assert t2.calls == 0
    assert torch.equal(z_a, z_b) and all(torch.equal(a, b) for a, b in zip(xs_a, xs_b))
    assert len(led.iter_keys("train_ok")) == 8


def test_crashed_session_resumes_bit_identical_with_only_remaining_calls(tmp_path, z0):
    ref, _ = denoise(X.ExhaustLedger(tmp_path / "ref"), Teacher(), F.ENGINE, z0)
    root = tmp_path / "live"
    crashed = Teacher(crash_at=5)
    with pytest.raises(Boom):
        denoise(X.ExhaustLedger(root), crashed, F.ENGINE, z0)
    assert crashed.calls == 5 and len(X.ExhaustLedger(root).iter_keys("train_ok")) == 4
    resumed = Teacher()
    z, _ = denoise(X.ExhaustLedger(root), resumed, F.ENGINE, z0)                # 새 인스턴스 = 재시작
    assert resumed.calls == 4
    assert torch.equal(z, ref)                                                  # 끊기지 않은 실행과 비트 동일


def test_ext_session_resume_with_sealer_is_project_scoped(tmp_path, z0):
    sealer = F.FakeSealer()
    root = tmp_path / "ex"
    kw = F.ext_kw()
    ref, _ = denoise(X.ExhaustLedger(tmp_path / "ref", sealer=F.FakeSealer()), Teacher(), F.USER, z0, **kw)
    with pytest.raises(Boom):
        denoise(X.ExhaustLedger(root, sealer=sealer), Teacher(crash_at=4), F.USER, z0, **kw)
    t = Teacher()
    z, _ = denoise(X.ExhaustLedger(root, sealer=sealer), t, F.USER, z0, **kw)
    assert t.calls == 5 and torch.equal(z, ref)
    # 다른 프로젝트·다른 프로젝트 키·비-ext 로는 재사용되지 않는다
    for other in (F.ext_kw(F.PROJ2, F.PK), F.ext_kw(F.PROJ, F.PK2)):
        t2 = Teacher()
        denoise(X.ExhaustLedger(root, sealer=sealer), t2, F.USER, z0, **other)
        assert t2.calls == 8
    t3 = Teacher()
    denoise(X.ExhaustLedger(root, sealer=sealer), t3, F.ENGINE, z0)
    assert t3.calls == 8


def test_world_change_invalidates_only_that_view_delta_reuse(tmp_path, z0):
    led = X.ExhaustLedger(tmp_path / "ex")
    worlds = [bytes([i]) * 32 for i in range(4)]
    t = Teacher()
    for i, w in enumerate(worlds):
        denoise(led, t, F.ENGINE, z0, world=w, view=f"v{i}")
    assert t.calls == 32
    worlds2 = list(worlds)
    worlds2[2] = bytes([99]) * 32                                      # 블록 하나만 편집(Merkle 루트 변경)
    t2 = Teacher()
    for i, w in enumerate(worlds2):
        denoise(led, t2, F.ENGINE, z0, world=w, view=f"v{i}")
    assert t2.calls == 8                                                # v2 만 다시
    assert led.stats["hits"] >= 24


def test_delta_purpose_reads_same_entries(tmp_path):
    led = X.ExhaustLedger(tmp_path / "ex")
    p = F.parts()
    e = led.get_or_compute(p, F.ENGINE, lambda: F.latent(6))
    d = led.get_or_compute(p, F.ENGINE, lambda: pytest.fail("히트여야 한다"), purpose="delta")
    assert torch.equal(e.q, d.q)

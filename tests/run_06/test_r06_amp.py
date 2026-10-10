"""AMP: GradScaler 규칙(65536·2000), inf 스텝 건너뛰기, 클리핑, 정밀도 플래그, fp32 마스터, 상태 왕복."""
import json
import math

import pytest
import torch

from buc.train.amp import Amp, AmpConfig, set_precision_flags


def cpu_amp(**kw):
    base = dict(dtype="fp16", device_type="cpu", init_scale=8.0, growth_interval=3)
    base.update(kw)
    return Amp(AmpConfig(**base))


class Elementwise(torch.nn.Module):
    """원소별 곱만 쓴다 => autocast가 연산을 fp16으로 바꾸지 않아 수치가 정확하다."""

    def __init__(self, n=3):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(n))

    def forward(self, x):
        return (self.w * x).sum()


def one_step(amp, model, opt, x, **kw):
    opt.zero_grad(set_to_none=True)
    with amp.autocast():
        loss = model(x)
    amp.scale(loss).backward()
    return amp.step(opt, **kw)


def test_config_defaults_follow_the_recipe():
    c = AmpConfig()
    assert (c.dtype, c.init_scale, c.growth_factor, c.backoff_factor, c.growth_interval) == ("fp16", 65536.0, 2.0, 0.5, 2000)
    assert c.device_type == "cuda"


@pytest.mark.parametrize("kw", [
    dict(dtype="fp8"), dict(device_type="tpu"), dict(init_scale=0.0), dict(growth_factor=1.0),
    dict(backoff_factor=1.0), dict(backoff_factor=0.0), dict(growth_interval=0),
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        AmpConfig(**kw)


def test_precision_flags():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_precision_flags()
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    Amp(AmpConfig(dtype="fp32", device_type="cpu"))  # 생성이 플래그를 적용한다
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    Amp(AmpConfig(dtype="fp32", device_type="cpu"), apply_backend_flags=False)
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True


def test_scale_grows_after_interval_and_backs_off_on_inf():
    amp, m = cpu_amp(), Elementwise()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    assert amp.scaling and amp.scale_value == 8.0
    x = torch.ones(3)
    for expected in (8.0, 8.0, 16.0):  # 연속 성공 3회 -> x2
        r = one_step(amp, m, opt, x)
        assert r.stepped is True
        assert amp.scale_value == expected and r.scale == expected
    w_before = m.w.detach().clone()
    buf_before = opt.state[m.w]["momentum_buffer"].clone()
    r = one_step(amp, m, opt, torch.tensor([1.0, float("inf"), 1.0]))
    assert r.stepped is False and not math.isfinite(r.grad_norm)
    assert amp.scale_value == 8.0  # x0.5
    assert torch.equal(m.w.detach(), w_before)  # 파라미터 불변
    assert torch.equal(opt.state[m.w]["momentum_buffer"], buf_before)  # 옵티마이저 상태 불변
    assert (amp.skipped_steps, amp.consecutive_skips) == (1, 1)
    # 성장 카운터도 0으로 돌아간다: 성공 2회로는 아직 안 자라고 3회째에 자란다
    for expected in (8.0, 8.0, 16.0):
        assert one_step(amp, m, opt, x).stepped
        assert amp.scale_value == expected
    assert amp.consecutive_skips == 0 and amp.skipped_steps == 1


def test_step_applies_unscaled_gradient_exactly():
    amp, m = cpu_amp(init_scale=1024.0), Elementwise()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    x = torch.tensor([1.0, 2.0, -4.0])
    r = one_step(amp, m, opt, x)
    assert r.stepped
    assert torch.equal(m.w.detach(), torch.ones(3) - 0.5 * x)  # 손실 스케일이 업데이트에 새지 않는다
    assert r.grad_norm == pytest.approx(math.sqrt(21.0), rel=1e-6)  # 보고되는 노름은 언스케일 값


def test_gradient_clipping_reports_preclip_norm():
    amp, m = cpu_amp(init_scale=64.0), Elementwise()
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    x = torch.tensor([3.0, 4.0, 0.0])  # 노름 5
    r = one_step(amp, m, opt, x, max_grad_norm=1.0)
    assert r.grad_norm == pytest.approx(5.0, rel=1e-6)
    assert m.w.detach().tolist() == pytest.approx([1 - 0.6, 1 - 0.8, 1.0], rel=1e-5)
    # 노름이 작으면 건드리지 않는다
    m2 = Elementwise()
    opt2 = torch.optim.SGD(m2.parameters(), lr=1.0)
    one_step(cpu_amp(), m2, opt2, torch.tensor([0.3, 0.4, 0.0]), max_grad_norm=1.0)
    assert m2.w.detach().tolist() == pytest.approx([0.7, 0.6, 1.0], rel=1e-5)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_non_fp16_modes_do_not_scale_but_still_skip_nonfinite(dtype):
    amp, m = cpu_amp(dtype=dtype), Elementwise()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    assert amp.scaling is False and amp.scale_value == 1.0
    loss = torch.tensor(2.0, requires_grad=True)
    assert amp.scale(loss) is loss or torch.equal(amp.scale(loss), loss)
    assert one_step(amp, m, opt, torch.ones(3)).stepped
    step_before = opt.state[m.w]["step"].clone()
    w_before = m.w.detach().clone()
    r = one_step(amp, m, opt, torch.tensor([1.0, float("nan"), 1.0]))
    assert r.stepped is False
    assert torch.equal(m.w.detach(), w_before)
    assert torch.equal(opt.state[m.w]["step"], step_before)  # Adam 스텝 카운트도 안 올라간다
    assert amp.scale_value == 1.0 and amp.skipped_steps == 1


def test_autocast_dtypes_and_fp32_master():
    lin = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    for dtype, want in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
        amp = cpu_amp(dtype=dtype)
        with amp.autocast():
            y = lin(x)
            assert y.dtype == want, dtype
            with amp.fp32_region():
                assert lin(x).dtype == torch.float32
                assert (x @ x.T).dtype == torch.float32
        assert lin(x).dtype == torch.float32  # 바깥은 그대로
    amp = cpu_amp()
    opt = torch.optim.SGD(lin.parameters(), lr=0.01)
    opt.zero_grad()
    with amp.autocast():
        loss = lin(x).float().pow(2).mean()
    amp.scale(loss).backward()
    assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in lin.parameters())
    assert amp.step(opt).stepped


def test_state_dict_roundtrip_is_primitive_and_exact():
    a, m = cpu_amp(), Elementwise()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.ones(3)
    for _ in range(4):  # 8 -> 16, 성장 카운터 1
        one_step(a, m, opt, x)
    one_step(a, m, opt, torch.tensor([float("inf"), 0.0, 0.0]))  # 8, 카운터 0, skipped 1
    one_step(a, m, opt, x)  # 카운터 1
    sd = a.state_dict()
    assert json.loads(json.dumps(sd)) == sd  # 파이썬 기본형만(weights_only 로딩 가능)
    assert sd["dtype"] == "fp16" and sd["scale"] == 8.0 and sd["skipped_steps"] == 1
    b = cpu_amp()
    b.load_state_dict(sd)
    assert b.scale_value == 8.0 and b.skipped_steps == 1
    for amp in (a, b):  # 카운터까지 복원됐는지: 성공 2회 더 -> 정확히 3번째에 성장
        m2 = Elementwise()
        o2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        one_step(amp, m2, o2, x)
        assert amp.scale_value == 8.0
        one_step(amp, m2, o2, x)
        assert amp.scale_value == 16.0
    with pytest.raises(ValueError):
        cpu_amp(dtype="bf16").load_state_dict(sd)
    with pytest.raises(ValueError):
        cpu_amp().load_state_dict({**sd, "v": 2})


def test_step_without_clip_does_not_modify_gradients_beyond_unscale():
    amp, m = cpu_amp(init_scale=16.0), Elementwise()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    x = torch.tensor([100.0, -200.0, 300.0])
    one_step(amp, m, opt, x)
    assert torch.equal(m.w.grad, x)  # 클립 없으면 언스케일된 기울기가 그대로 남는다

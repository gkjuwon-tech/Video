"""TCH-02 회귀: 교사 NaN(`변환기 출력에 NaN/Inf가 있다`) 조사에서 나온 것들.

1. 호출 규약 고정: 진짜 diffusers ZImageTransformer2DModel(작은 무작위 설정) + 진짜 FlowMatchEulerDiscreteScheduler 로 만든
   ZImagePipeline 루프(diffusers 0.41 pipeline_z_image.py 와 같은 줄)와 TeacherSampler.generate 가 일치한다(fp32·bf16).
   -> t=1-sigma, 입력 배치([C,1,h,w] 리스트), 부호(-out), 캡션 형식, 8점 표가 순정 파이프라인과 같음을 CPU 에서 증명한다.
2. 비유한 출력 진단: TeacherNumericError 는 그대로, 메시지에 step·sigma·뷰·dtype·입력 통계·최초 비유한 모듈이 들어간다.
3. 환경 재시도 사다리: 결정론 포이즌, flash/efficient SDPA, 축소정밀 GEMM, 일시 오류가 원인인 비유한은 한 번의 재시도로
   고쳐지고 numeric_events 에 남는다. 안 고쳐지면 같은 TeacherNumericError.
4. 실제 모양의 transformer.layers 에 TapBus(TCH-03)를 걸어도 출력이 안 바뀌고, 이미지 토큰이 오프셋 0·래스터 순서임을 확인."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from buc.teacher import sample as S
from buc.teacher import zimage as Z
from buc.teacher.zimage import Loaders, TeacherNumericError
from tch02_fakes import (H, IMG_H, IMG_W, SPEC, W, FakeDenoiser, _FakeTE, _FakeVae, make_caps, make_hub, rand_latent,
                         target_of)


@pytest.fixture(autouse=True)
def restore_global_numeric_state():
    m = torch.backends.cuda.matmul
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             getattr(torch.utils.deterministic, "fill_uninitialized_memory", None),
             m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        if saved[2] is not None:
            torch.utils.deterministic.fill_uninitialized_memory = saved[2]
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved[3], saved[4]


def _teacher(tmp_path, transformer_factory, dtype=torch.float32):
    loaders = Loaders(
        tokenizer=lambda p: SimpleNamespace(name="fake-tok"),
        text_encoder=lambda p, dt: _FakeTE(SPEC.text_dim).to(dt),
        transformer=lambda p, dt: transformer_factory().to(dt),
        vae=lambda p, dt: _FakeVae(SPEC.latent_channels).to(dt),
        encode_prompts=lambda tok, te, prompts, n: [te.emb(torch.tensor(list(p.encode())[:n] or [0])).detach()
                                                    for p in prompts])
    return Z.load_teacher(SPEC, device="cpu", dtype=dtype, hub_cache=make_hub(tmp_path), loaders=loaders,
                          te_cache_dir=tmp_path / "tec", deterministic=False)


def _sampler(tmp_path, factory, dtype=torch.float32, **kw):
    t = _teacher(tmp_path, factory, dtype)
    return S.TeacherSampler(t, **kw), t


# --------------------------------------------------------------------------- 1. diffusers 와의 호출 규약 일치

def _real_tiny_zimage(seed=0):
    pytest.importorskip("diffusers")
    from diffusers import ZImageTransformer2DModel
    torch.manual_seed(seed)
    m = ZImageTransformer2DModel(in_channels=16, dim=64, n_layers=SPEC.num_layers, n_refiner_layers=1, n_heads=2,
                                 n_kv_heads=2, norm_eps=1e-5, qk_norm=True, cap_feat_dim=SPEC.text_dim,
                                 rope_theta=256.0, t_scale=1000.0, axes_dims=[8, 12, 12], axes_lens=[256, 64, 64])
    with torch.no_grad():
        for p in m.parameters():
            if float(p.abs().sum()) == 0.0:
                p.normal_(0, 0.02)
    return m.eval()


def _pipeline_loop(transformer, caps, z_start, dtype, n_steps):
    """diffusers 0.41 ZImagePipeline.__call__ 의 디노이즈 루프(guidance_scale=0)를 그대로 옮긴 것."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=False, shift=3.0)
    sch.set_timesteps(sigmas=torch.linspace(1.0, 1 / n_steps, n_steps).tolist(), device="cpu")
    sch.set_begin_index(0)
    latents = z_start.clone()
    with torch.no_grad():
        for t in sch.timesteps:
            timestep = (1000 - t.expand(latents.shape[0])) / 1000
            inp = list(latents.to(dtype).unsqueeze(2).unbind(dim=0))
            out = transformer(inp, timestep, caps, return_dict=False)[0]
            noise_pred = -torch.stack([o.float() for o in out], dim=0).squeeze(2)
            latents = sch.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
    return latents, sch


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-3)])
def test_generate_equals_diffusers_zimage_pipeline_loop(tmp_path, dtype, tol):
    smp, t = _sampler(tmp_path, _real_tiny_zimage, dtype)
    cap = make_caps(1, seed=2)[0]
    r = smp.generate([cap], seeds=[5], height=IMG_H, width=IMG_W)
    ref, sch = _pipeline_loop(t.transformer, [cap.to(dtype)], r.z_start, dtype, S.N_STEPS)
    assert torch.isfinite(r.z).all() and torch.isfinite(ref).all()
    assert (r.z - ref).abs().max().item() < tol
    # 우리 표 == 스냅샷 스케줄러(shift 3.0)의 시그마 (GPU 시험에 있던 것을 CPU 에서)
    assert S.DEFAULT_TABLE.points == pytest.approx([float(s) for s in sch.sigmas[:S.N_STEPS]], abs=1e-6)


def test_sdedit_batch_with_unequal_caption_lengths_equals_single_views(tmp_path):
    """캡션 길이가 32 경계를 넘나드는 배치(어텐션 마스크 경로)가 뷰별 단독 호출과 같다."""
    smp, _ = _sampler(tmp_path, _real_tiny_zimage, max_views_per_call=2)
    g = torch.Generator().manual_seed(4)
    caps = [torch.randn(n, SPEC.text_dim, generator=g) * 0.5 for n in (5, 40, 3)]
    x0 = rand_latent(3, seed=1)
    full = smp.sdedit(x0, caps, sigma=0.5, seeds=[7, 8, 9], steps=2)
    assert torch.isfinite(full.z).all()
    for i in range(3):
        one = smp.sdedit(x0[i:i + 1], caps[i:i + 1], sigma=0.5, seeds=[[7, 8, 9][i]], steps=2)
        assert (one.z[0] - full.z[i]).abs().max().item() < 2e-5


# --------------------------------------------------------------------------- 2. 진단

class _NanChild(nn.Module):
    def forward(self, x):
        return x * float("nan")


class NanInChild(FakeDenoiser):
    """출력 직전의 하위 모듈 `bad` 가 NaN 을 낸다 -> first_bad 는 'bad'."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bad = _NanChild()

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        outs = super().forward(x, t, cap_feats, return_dict=False)[0]
        return (list(self.bad(torch.stack(outs)).unbind(0)),)


def test_nonfinite_error_type_is_stable_and_message_is_diagnostic(tmp_path):
    smp, _ = _sampler(tmp_path, lambda: NanInChild(n_layers=SPEC.num_layers))
    with pytest.raises(TeacherNumericError) as ei:
        smp.sdedit(rand_latent(2), make_caps(2), sigma=0.5, seeds=1, steps=2)
    msg = str(ei.value)
    assert msg.startswith("변환기 출력에 NaN/Inf가 있다")
    for key in ("step=0", "views=0:2", "dtype=float32", "bad_rows=[0, 1]", "x_finite=True", "cap_finite=True",
                "first_bad=bad", "tried=probe:nonfinite"):
        assert key in msg, (key, msg)
    assert isinstance(ei.value, FloatingPointError)


def test_retry_can_be_switched_off(tmp_path):
    smp, tr = _sampler(tmp_path, lambda: FakeDenoiser(n_layers=SPEC.num_layers, nan=True))
    smp.nonfinite_retry = False
    with pytest.raises(TeacherNumericError, match="tried=none"):
        smp.predict_x0(rand_latent(1), smp.table.points[2], make_caps(1))
    assert len(smp.teacher.transformer.calls) == 1


# --------------------------------------------------------------------------- 3. 환경 재시도 사다리

class EnvSensitive(FakeDenoiser):
    """`fails()` 가 참인 환경에서만 NaN. 나머지는 해석적 오라클(완전 디노이저)과 같다."""

    def __init__(self, fails, **kw):
        super().__init__(**kw)
        self.fails = fails

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        res = super().forward(x, t, cap_feats, return_dict=return_dict, **kw)
        if not self.fails(len(self.calls)):
            return res
        outs = res[0] if isinstance(res, tuple) else res.sample
        outs = [o * float("nan") for o in outs]
        return (outs,) if isinstance(res, tuple) else SimpleNamespace(sample=outs)


def _oracle_x0(caps):
    return torch.stack([target_of(c) for c in caps])


def _poison_active():
    return torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory


def test_poison_reading_kernel_is_healed_and_logged(tmp_path):
    smp, _ = _sampler(tmp_path, lambda: EnvSensitive(lambda n: _poison_active(), n_layers=SPEC.num_layers))
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    caps = make_caps(2)
    x0 = smp.predict_x0(rand_latent(2), smp.table.points[7], caps)
    assert torch.allclose(x0, _oracle_x0(caps), atol=1e-5)
    ev = smp.numeric_events
    assert len(ev) == 1 and ev[0]["healed_by"] == "no_nan_fill" and ev[0]["tried"] == ["probe:nonfinite"]
    assert torch.utils.deterministic.fill_uninitialized_memory is True          # 재시도가 끝나면 원복
    # 포이즌을 끄는 결정론 모드에서는 재시도 자체가 필요 없다
    smp.numeric_events.clear()
    Z.enable_determinism_mode(seed=0)
    smp.predict_x0(rand_latent(2), smp.table.points[7], caps)
    assert smp.numeric_events == []


def test_flash_sdpa_fault_is_healed_by_math_sdpa(tmp_path):
    smp, _ = _sampler(tmp_path, lambda: EnvSensitive(lambda n: torch.backends.cuda.flash_sdp_enabled(),
                                                      n_layers=SPEC.num_layers))
    caps = make_caps(1)
    x0 = smp.predict_x0(rand_latent(1), smp.table.points[7], caps)
    assert torch.allclose(x0, _oracle_x0(caps), atol=1e-5)
    assert [e["healed_by"] for e in smp.numeric_events] == ["math_sdpa"]
    assert torch.backends.cuda.flash_sdp_enabled()                              # 원복


def test_reduced_precision_gemm_fault_is_healed_by_default_gemm(tmp_path):
    smp, _ = _sampler(tmp_path, lambda: EnvSensitive(
        lambda n: not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, n_layers=SPEC.num_layers))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    caps = make_caps(1)
    x0 = smp.predict_x0(rand_latent(1), smp.table.points[7], caps)
    assert torch.allclose(x0, _oracle_x0(caps), atol=1e-5)
    ev = smp.numeric_events
    assert [e["healed_by"] for e in ev] == ["default_gemm"] and ev[0]["tried"] == ["probe:nonfinite",
                                                                                   "math_sdpa:nonfinite"]
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False


def test_transient_fault_is_healed_by_probe(tmp_path):
    smp, tr = _sampler(tmp_path, lambda: EnvSensitive(lambda n: n == 1, n_layers=SPEC.num_layers))
    caps = make_caps(1)
    x0 = smp.predict_x0(rand_latent(1), smp.table.points[7], caps)
    assert torch.allclose(x0, _oracle_x0(caps), atol=1e-5)
    assert [e["healed_by"] for e in smp.numeric_events] == ["probe"] and len(tr.transformer.calls) == 2


def test_unfixable_nonfinite_still_raises_after_the_ladder(tmp_path):
    smp, _ = _sampler(tmp_path, lambda: FakeDenoiser(n_layers=SPEC.num_layers, nan=True))
    with pytest.raises(TeacherNumericError, match=r"tried=probe:nonfinite,math_sdpa:nonfinite"):
        smp.predict_x0(rand_latent(1), smp.table.points[2], make_caps(1))
    assert smp.numeric_events == []


# --------------------------------------------------------------------------- 4. 실제 transformer.layers 에 TapBus

def test_tapbus_on_real_zimage_layers_is_output_neutral_and_image_tokens_lead(tmp_path):
    from buc.gov import prov as P
    from buc.teacher import taps as T
    smp, t = _sampler(tmp_path, _real_tiny_zimage)
    dim, layers = 64, (0, 2, 3)
    g = torch.Generator().manual_seed(1)
    basis = T.fit_pca_basis({l: torch.randn(64, dim, generator=g) for l in layers}, prov=P.as_record(["ENGINE"]), k=8)
    caps, z = make_caps(2), rand_latent(2, seed=3)
    plain = smp.predict_x0(z, smp.table.points[4], caps)

    full = {}                                    # 오라클: 우리 훅으로 층 전체 출력을 모은다
    hs = [t.layers[l].register_forward_hook(lambda m, a, o, l=l: full.__setitem__(l, o.detach().clone()))
          for l in layers]
    bus = T.TapBus(basis, layers=layers).attach(t.layers)
    grid = (H // 2, W // 2)
    try:
        with bus.capture(P.as_record(["ENGINE"]), grid=grid, token_offset=0) as cap:
            tapped = smp.predict_x0(z, smp.table.points[4], caps)
    finally:
        bus.detach()
        for h in hs:
            h.remove()
    assert torch.equal(plain, tapped)                                       # 훅은 출력을 바꾸지 않는다
    ts = cap.tapset
    for l in layers:
        want = basis.project(l, full[l][:, :grid[0] * grid[1]])             # 이미지 토큰이 앞, 래스터 순서
        want = want.reshape(2, grid[0], grid[1], basis.k).permute(0, 3, 1, 2)
        assert torch.equal(ts.layers[l], want), l
        assert bool(torch.isfinite(ts.features(basis)[l]).all())


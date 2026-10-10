"""실물 Z-Image FP16 rescue (CUDA + 오프라인 가중치 필요; CPU 게이트에서는 제외)."""
import pytest
import torch

from buc.teacher import t4_fp16 as R


def test_zimage_plan_rejects_non_zimage():
    with pytest.raises(R.RescuePlanError):
        R.zimage_plan(torch.nn.Linear(2, 2), 32.0)


@pytest.mark.gpu
def test_zimage_fp16_rescue_real():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher import zimage as Z
    try:
        teacher = Z.load_teacher(device="cuda", dtype=torch.bfloat16, seed=0)
    except (Z.TeacherWeightsMissing, Z.TeacherDependencyError) as e:
        pytest.skip(f"가중치/의존성 없음: {e}")
    tr = teacher.transformer
    cap = teacher.encode_text(["a mossy stone bridge over a creek, overcast"])
    t = torch.tensor([0.5], device="cuda", dtype=torch.float32)

    def run(dtype, seed):
        x = [teacher.make_noise(seed, 1, 512, 512)[0].unsqueeze(1).to(dtype)]
        out = tr(x, t, [cap[0].to(dtype)], return_dict=False)
        out = out[0] if isinstance(out, tuple) else out
        return out[0].float()

    layers = [n for n, _ in tr.named_modules() if n.count(".") == 1 and n.split(".")[0] in
              ("layers", "noise_refiner", "context_refiner")]
    downs = [n for n, _ in tr.named_modules() if n.endswith("feed_forward.w2")]
    assert layers and downs
    with torch.no_grad(), R.MaxAbsProfiler(tr, names=layers + downs) as p:
        refs = [run(torch.bfloat16, s) for s in range(4)]
    s = R.choose_scale({k: p.out_max[k] for k in layers})
    tt = R.choose_scale({k: p.in_max[k] for k in downs})
    plan = R.zimage_plan(tr, s, swiglu_scales=tt)
    plan.validate()
    R.apply_rescue(tr, plan)
    R.cast_fp16(tr)
    with torch.no_grad(), R.fp32_sdpa():
        outs = [run(torch.float16, s) for s in range(4)]
    for o, r in zip(outs, refs):
        assert torch.isfinite(o).all()
        cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
        assert cos >= 0.99, cos

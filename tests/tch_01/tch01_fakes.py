"""TCH-01 공용 가짜 부품: HF 캐시 레이아웃, 가짜 transformer/VAE/TE, 호출을 기록하는 Loaders."""
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from buc.teacher.zimage import ZIMAGE_REPO, Loaders, TeacherSpec

REV = "a" * 40
# 작은 스펙: 층 4개, 텍스트 차원 8. 이미지/VAE 쪽 상수는 기본값(16ch, f8, patch2) 그대로.
SPEC = replace(TeacherSpec(), num_layers=4, text_dim=8, max_text_len=16)


def make_hub(tmp_path, repo=ZIMAGE_REPO, rev=REV, via_ref=True, subdirs=("transformer", "vae", "text_encoder", "tokenizer")):
    """HF 허브 캐시 레이아웃을 만든다. 반환: hub_cache 경로."""
    hub = tmp_path / "hub"
    root = hub / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / rev
    for s in subdirs:
        (snap / s).mkdir(parents=True, exist_ok=True)
        (snap / s / "config.json").write_text("{}")
    if via_ref:
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(rev + "\n")
    return hub


class FakeTransformer(nn.Module):
    def __init__(self, n_layers=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(n_layers))
        for lin in self.layers:
            lin.weight.data = torch.randn(4, 4, generator=g)
            lin.bias.data = torch.randn(4, generator=g)
        self.proj = nn.Linear(4, 2)
        self.proj.weight.data = torch.randn(2, 4, generator=g)
        self.proj.bias.data = torch.randn(2, generator=g)


class FakeTE(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        g = torch.Generator().manual_seed(3)
        self.emb = nn.Embedding(256, hidden)
        self.emb.weight.data = torch.randn(256, hidden, generator=g)


class _Dist:
    def __init__(self, z):
        self._z = z

    def mode(self):
        return self._z

    def sample(self):
        raise AssertionError("VAE encode는 결정론이어야 한다: .sample() 금지, .mode() 사용")


class FakeVae(nn.Module):
    """8x8 평균 풀링 + 3->C 선형. 8x8 블록 상수 영상은 encode/decode 왕복이 정확하다."""

    def __init__(self, scaling=0.5, shift=0.25, channels=16, nan_decode=False):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=channels, scaling_factor=scaling, shift_factor=shift)
        g = torch.Generator().manual_seed(7)
        self.mix = nn.Parameter(torch.randn(channels, 3, generator=g))  # requires_grad=True로 시작
        self.nan_decode = nan_decode

    def encode(self, x):
        p = F.avg_pool2d(x, 8)
        return SimpleNamespace(latent_dist=_Dist(torch.einsum("cr,brhw->bchw", self.mix, p)))

    def decode(self, z):
        pinv = torch.linalg.pinv(self.mix.float()).to(z.dtype)
        out = F.interpolate(torch.einsum("rc,bchw->brhw", pinv, z), scale_factor=8, mode="nearest")
        if self.nan_decode:
            out = out * float("nan")
        return SimpleNamespace(sample=out)


def raw_latent(vae, x):
    """오라클: 스케일/시프트 전의 원시 잠재."""
    return torch.einsum("cr,brhw->bchw", vae.mix.detach().float(), F.avg_pool2d(x.float(), 8))


def block_image(b=2, hb=4, wb=5, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    base = (torch.rand(b, 3, hb, wb, generator=g) * 2 - 1) * scale
    return F.interpolate(base, scale_factor=8, mode="nearest")


class Spy:
    """호출 기록 + 모듈 약한 참조."""

    def __init__(self):
        self.calls = []        # (name, path.name or None, dtype or None)
        self.encode_calls = []  # 각 호출의 prompts
        self.te_weak = None
        self.te_device_seen = None


def make_loaders(spy=None, spec=SPEC, te_hidden=None, vae_kwargs=None, tr_layers=None, tr_dtype_override=None,
                 tr_seed=0):
    spy = spy or Spy()
    te_hidden = spec.text_dim if te_hidden is None else te_hidden
    n_layers = spec.num_layers if tr_layers is None else tr_layers

    def tokenizer(path):
        spy.calls.append(("tokenizer", path.name, None))
        return SimpleNamespace(name="fake-tok")

    def text_encoder(path, dtype):
        spy.calls.append(("text_encoder", path.name, dtype))
        m = FakeTE(te_hidden).to(dtype)
        spy.te_weak = weakref.ref(m)
        return m

    def transformer(path, dtype):
        spy.calls.append(("transformer", path.name, dtype))
        m = FakeTransformer(n_layers, seed=tr_seed)
        return m.to(tr_dtype_override or dtype)

    def vae(path, dtype):
        spy.calls.append(("vae", path.name, dtype))
        return FakeVae(**(vae_kwargs or {})).to(dtype)

    def encode_prompts(tok, te, prompts, max_len):
        spy.encode_calls.append(list(prompts))
        spy.te_device_seen = next(te.parameters()).device.type
        out = []
        for p in prompts:
            ids = torch.tensor(list(p.encode("utf-8"))[:max_len] or [0], dtype=torch.long)
            out.append(te.emb(ids).detach())
        return out

    loaders = Loaders(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae,
                      encode_prompts=encode_prompts)
    return loaders, spy


def collect():
    gc.collect()
    gc.collect()

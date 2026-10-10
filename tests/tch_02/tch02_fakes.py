"""TCH-02 공용 가짜 부품: 해석적 오라클을 가진 가짜 변환기와, 진짜 load_teacher로 만든 교사."""
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from buc.teacher.zimage import ZIMAGE_REPO, Loaders, TeacherSpec, load_teacher

REV = "b" * 40
SPEC = replace(TeacherSpec(), num_layers=4, text_dim=8, max_text_len=16)
C = 16
H, W = 4, 6                      # 잠재 크기(둘 다 짝수: patch 2)
IMG_H, IMG_W = 8 * H, 8 * W      # 이미지 32x48 (16의 배수)


def make_hub(tmp_path, repo=ZIMAGE_REPO, rev=REV):
    hub = tmp_path / "hub"
    root = hub / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / rev
    for s in ("transformer", "vae", "text_encoder", "tokenizer"):
        (snap / s).mkdir(parents=True, exist_ok=True)
        (snap / s / "config.json").write_text("{}")
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(rev + "\n")
    return hub


def target_of(cap, c=C, h=H, w=W):
    """조건 cap -> 가짜 '데이터' x* [c,h,w]. cap 평균 + 채널 기울기 + 열 기울기."""
    m = cap.float().mean()
    ramp = torch.arange(c, dtype=torch.float32).view(c, 1, 1) * 0.1
    col = torch.arange(w, dtype=torch.float32).view(1, 1, w) * 0.05
    return m + ramp + col + torch.zeros(1, h, 1)


class FakeDenoiser(nn.Module):
    """diffusers ZImage 변환기 규약을 흉내 낸다: forward(x_list, t, cap_list, return_dict) -> -v.

    t = 1 - sigma. mode='perfect': x0_hat = x*(cap).  mode='affine': x0_hat = a*z + x*(cap).
    속도 v = (z - x0_hat)/sigma 이므로 x0 = z - sigma*v 가 위 값과 정확히 일치한다."""

    def __init__(self, mode="perfect", a=0.5, n_layers=4, nan=False, drop_last=False, bad_shape=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))
        self.mode, self.a, self.nan, self.drop_last, self.bad_shape = mode, a, nan, drop_last, bad_shape
        self.calls = []

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls.append(SimpleNamespace(
            n=len(x), t=t.detach().clone(), t_dtype=t.dtype, x_dtype=x[0].dtype,
            x_shapes=[tuple(a.shape) for a in x], cap_lens=[int(c.shape[0]) for c in cap_feats],
            return_dict=return_dict, grad_enabled=torch.is_grad_enabled()))
        assert len(cap_feats) == len(x) == int(t.shape[0]), "x, t, cap 길이가 같아야 한다"
        z = torch.stack([a.float() for a in x])                       # [n,C,1,h,w]
        sig = (1.0 - t.float()).view(-1, 1, 1, 1, 1)
        h, w = z.shape[-2:]
        tgt = torch.stack([target_of(c, z.shape[1], h, w) for c in cap_feats]).unsqueeze(2)
        x0 = tgt if self.mode == "perfect" else self.a * z + tgt
        out = -(z - x0) / sig
        if self.nan:
            out = out * float("nan")
        outs = list(out.to(x[0].dtype).unbind(0))
        if self.drop_last:
            outs = outs[:-1]
        if self.bad_shape:
            outs = [o[..., :-1] for o in outs]
        return SimpleNamespace(sample=outs) if return_dict else (outs,)


class _FakeTE(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        g = torch.Generator().manual_seed(3)
        self.emb = nn.Embedding(256, hidden)
        self.emb.weight.data = torch.randn(256, hidden, generator=g)


class _FakeVae(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=channels, scaling_factor=0.5, shift_factor=0.25)
        self.mix = nn.Parameter(torch.ones(channels, 3))


def make_teacher(tmp_path, dtype=torch.float32, **fake_kw):
    def tokenizer(path):
        return SimpleNamespace(name="fake-tok")

    def text_encoder(path, dt):
        return _FakeTE(SPEC.text_dim).to(dt)

    def transformer(path, dt):
        return FakeDenoiser(n_layers=SPEC.num_layers, **fake_kw).to(dt)

    def vae(path, dt):
        return _FakeVae(SPEC.latent_channels).to(dt)

    def encode_prompts(tok, te, prompts, max_len):
        out = []
        for p in prompts:
            ids = torch.tensor(list(p.encode("utf-8"))[:max_len] or [0], dtype=torch.long)
            out.append(te.emb(ids).detach())
        return out

    loaders = Loaders(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae,
                      encode_prompts=encode_prompts)
    return load_teacher(SPEC, device="cpu", dtype=dtype, hub_cache=make_hub(tmp_path), loaders=loaders,
                        te_cache_dir=tmp_path / "tec", deterministic=False)


def rand_latent(b, seed=0, h=H, w=W):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, C, h, w, generator=g)


def make_caps(n, seed=0):
    """길이가 뷰마다 다른 조건 임베딩 n개(평균이 서로 다르다)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3 + i, SPEC.text_dim, generator=g) * 0.5 + 0.3 * i for i in range(n)]


def targets(caps):
    return torch.stack([target_of(c) for c in caps])

"""TCH-08 공용 가짜 부품: 가짜 Z 교사(Ring 시험용, 진짜 load_teacher), 가짜 klein 로더·스냅샷."""
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from buc.teacher.zimage import ZIMAGE_REPO, Loaders, TeacherSpec, load_teacher

REV = "c" * 40
ZSPEC = replace(TeacherSpec(), num_layers=4, text_dim=8, max_text_len=16)
C = 16                  # Z 잠재 채널
STRIP = 2               # 띠 잠재 행(= 16 px)
HB, W = 4, 6            # 본문 잠재 크기
H = STRIP + HB          # 캔버스 잠재 높이 6 -> 이미지 48x48
APACHE = "                                 Apache License\n                           Version 2.0, January 2004\n"


def make_snapshot(tmp_path, repo, rev=REV, license_text=APACHE, readme=None, subdirs=None):
    hub = tmp_path / "hub"
    root = hub / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / rev
    for s in (subdirs if subdirs is not None else ("transformer", "vae", "text_encoder", "tokenizer")):
        (snap / s).mkdir(parents=True, exist_ok=True)
        (snap / s / "config.json").write_text("{}")
    snap.mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(rev + "\n")
    if license_text is not None:
        (snap / "LICENSE").write_text(license_text)
    if readme is not None:
        (snap / "README.md").write_text(readme)
    return hub, snap


# ----------------------------------------------------------------------------- 가짜 Z 교사(Ring 시험용)

def target_of(cap, c=C, h=H, w=W):
    """조건 cap -> 가짜 데이터 x* [c,h,w]."""
    m = cap.float().mean()
    ramp = torch.arange(c, dtype=torch.float32).view(c, 1, 1) * 0.1
    col = torch.arange(w, dtype=torch.float32).view(1, 1, w) * 0.05
    row = torch.arange(h, dtype=torch.float32).view(1, h, 1) * -0.03
    return m + ramp + col + row


class FakeDenoiser(nn.Module):
    """ZImage 변환기 규약: forward(x_list, t, cap_list, return_dict) -> -v.  x0_hat = a*z + x*(cap) (a=0 이면 완전)."""

    def __init__(self, a=0.5, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))
        self.a = a
        self.calls = []

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls.append(SimpleNamespace(n=len(x), sigma=(1.0 - t.double()).tolist()))
        z = torch.stack([v.float() for v in x])                  # [n,C,1,h,w]
        sig = (1.0 - t.float()).view(-1, 1, 1, 1, 1)
        h, w = z.shape[-2:]
        tgt = torch.stack([target_of(c, z.shape[1], h, w) for c in cap_feats]).unsqueeze(2)
        x0 = self.a * z + tgt
        out = -(z - x0) / sig
        outs = list(out.to(x[0].dtype).unbind(0))
        return SimpleNamespace(sample=outs) if return_dict else (outs,)


class FakeTE(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        g = torch.Generator().manual_seed(3)
        self.emb = nn.Embedding(256, hidden)
        self.emb.weight.data = torch.randn(256, hidden, generator=g)


class FakeVae(nn.Module):
    """encode: 8x 평균 풀 + 3->C 혼합. decode: 최근접 업샘플 + C->3 평균."""

    def __init__(self, channels=16):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=channels, scaling_factor=1.0, shift_factor=0.0)
        g = torch.Generator().manual_seed(5)
        self.mix = nn.Parameter(torch.randn(channels, 3, generator=g))
        self.encode_calls = 0

    def encode(self, x):
        self.encode_calls += 1
        p = F.avg_pool2d(x, 8)
        lat = torch.einsum("ck,bkhw->bchw", self.mix.to(x.dtype), p)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: lat))

    def decode(self, z):
        y = F.interpolate(z, scale_factor=8, mode="nearest")[:, :3]
        return SimpleNamespace(sample=y)


def encode_prompts(tok, te, prompts, max_len):
    out = []
    for p in prompts:
        ids = torch.tensor(list(p.encode("utf-8"))[:max_len] or [0], dtype=torch.long)
        out.append(te.emb(ids).detach())
    return out


def make_z_teacher(tmp_path, a=0.5):
    hub, _ = make_snapshot(tmp_path / "z", ZIMAGE_REPO, license_text=None)
    loaders = Loaders(tokenizer=lambda p: SimpleNamespace(name="tok"),
                      text_encoder=lambda p, dt: FakeTE(ZSPEC.text_dim).to(dt),
                      transformer=lambda p, dt: FakeDenoiser(a=a, n_layers=ZSPEC.num_layers).to(dt),
                      vae=lambda p, dt: FakeVae(ZSPEC.latent_channels).to(dt),
                      encode_prompts=encode_prompts)
    return load_teacher(ZSPEC, device="cpu", dtype=torch.float32, hub_cache=hub, loaders=loaders,
                        te_cache_dir=tmp_path / "tec", deterministic=False)


def make_caps(n, seed=0, d=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3 + i, d, generator=g) + 0.3 * i for i in range(n)]


def rand_latent(b, seed=0, c=C, h=H, w=W):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, generator=g)


# ----------------------------------------------------------------------------- 가짜 klein

KC = 8                  # klein 잠재 채널(테스트용 축소)


class FakeKleinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class KleinRecorder:
    """가짜 klein 부품 묶음과 호출 기록."""

    def __init__(self, transformer_dtype=None, vae_channels=KC):
        self.log = []
        self.forward_refs = []
        self.transformer_dtype = transformer_dtype
        self.vae_channels = vae_channels
        self.vae_obj = None

    def tokenizer(self, path):
        self.log.append("tokenizer")
        return SimpleNamespace(name="tok")

    def text_encoder(self, path, dt):
        self.log.append("text_encoder")
        return FakeTE(8).to(dt)

    def transformer(self, path, dt):
        self.log.append("transformer")
        return FakeKleinTransformer().to(self.transformer_dtype or dt)

    def vae(self, path, dt):
        self.log.append("vae")
        self.vae_obj = FakeVae(self.vae_channels).to(dt)
        return self.vae_obj

    def encode_prompts(self, tok, te, prompts, max_len):
        self.log.append("encode_prompts")
        return encode_prompts(tok, te, prompts, max_len)

    def forward(self, raw, x_list, t, cap_list, refs):
        """완전 디노이저: x0 = x*(cap) + sum(ref.mean()).  -v 를 돌려준다."""
        assert isinstance(raw, FakeKleinTransformer)
        self.forward_refs.append([r.detach().clone() for r in refs])
        shift = sum(float(r.float().mean()) for r in refs) if refs else 0.0
        outs = []
        for xi, ti, ci in zip(x_list, t, cap_list):
            z = xi.float()
            sig = 1.0 - float(ti)
            x0 = target_of(ci, z.shape[0], z.shape[-2], z.shape[-1]).unsqueeze(1) + shift
            outs.append((-(z - x0) / sig).to(xi.dtype))
        return outs

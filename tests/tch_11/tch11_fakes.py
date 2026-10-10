"""TCH-11 공용 가짜 부품: 진짜 load_teacher + 아핀 가짜 변환기 + 디코드 가능한 가짜 VAE.

가짜 변환기는 TCH-02 규약(t=1-sigma, 출력 -v)을 따르고 x0_hat = a*z + x*(cap) 이다. a != 0 이라 최종 상태가
시드에 의존한다(완전 디노이저라면 시드를 무시하므로 시드 검증이 공허해진다)."""
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from buc.teacher.sample import TeacherSampler
from buc.teacher.zimage import ZIMAGE_REPO, Loaders, TeacherSpec, load_teacher

REV = "c" * 40
SPEC = replace(TeacherSpec(), num_layers=4, text_dim=8, max_text_len=512)
C = 16
IMG_H, IMG_W = 32, 48            # 16의 배수. 잠재는 4x6


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


def target_of(cap, c, h, w):
    m = cap.float().mean()
    ramp = torch.arange(c, dtype=torch.float32).view(c, 1, 1) * 0.1
    col = torch.arange(w, dtype=torch.float32).view(1, 1, w) * 0.05
    return m * 4.0 + ramp + col + torch.zeros(1, h, 1)


class AffineDenoiser(nn.Module):
    def __init__(self, a=0.5, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))
        self.a = a
        self.calls = []

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls.append(SimpleNamespace(n=len(x), t=t.detach().clone(), cap_lens=[int(c.shape[0]) for c in cap_feats]))
        z = torch.stack([a.float() for a in x])                       # [n,C,1,h,w]
        sig = (1.0 - t.float()).view(-1, 1, 1, 1, 1)
        h, w = z.shape[-2:]
        tgt = torch.stack([target_of(c, z.shape[1], h, w) for c in cap_feats]).unsqueeze(2)
        x0 = self.a * z + tgt
        out = -(z - x0) / sig
        outs = list(out.to(x[0].dtype).unbind(0))
        return SimpleNamespace(sample=outs) if return_dict else (outs,)


class _FakeTE(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        g = torch.Generator().manual_seed(5)
        self.emb = nn.Embedding(256, hidden)
        self.emb.weight.data = torch.randn(256, hidden, generator=g) * 0.3


class _FakeVae(nn.Module):
    """decode: 채널 5/5/6 묶음 평균 -> tanh -> 최근접 x8. 입력에 대해 엄밀히 결정적이고 값이 (-1,1) 안에 있다."""

    def __init__(self, channels=16):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=channels, scaling_factor=0.5, shift_factor=0.25)
        self.mix = nn.Parameter(torch.ones(channels, 3))

    def decode(self, z):
        rgb = torch.stack([z[:, :5].mean(1), z[:, 5:10].mean(1), z[:, 10:].mean(1)], 1)
        img = torch.tanh(rgb * 0.5)
        return SimpleNamespace(sample=torch.repeat_interleave(torch.repeat_interleave(img, 8, 2), 8, 3))


def make_teacher(tmp_path, dtype=torch.float32, a=0.5):
    def tokenizer(path):
        return SimpleNamespace(name="fake-tok")

    def text_encoder(path, dt):
        return _FakeTE(SPEC.text_dim).to(dt)

    def transformer(path, dt):
        return AffineDenoiser(a=a, n_layers=SPEC.num_layers).to(dt)

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


class Spy:
    """teacher.encode_text 를 감싸 실제로 인코딩에 들어간 문장을 기록한다(구현이 sampler.generate 를 어떻게 부르든 보인다)."""

    def __init__(self, teacher):
        self.prompts = []
        orig = teacher.encode_text

        def spy(prompts):
            self.prompts.extend(list(prompts))
            return orig(prompts)

        teacher.encode_text = spy


def make_sampler(tmp_path, *, table=None, a=0.5, **kw):
    t = make_teacher(tmp_path, a=a)
    spy = Spy(t)
    return TeacherSampler(t, table=table, **kw), t, spy

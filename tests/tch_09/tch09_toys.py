"""TCH-09 장난감 DiT: pre-norm(선형 gate) · sandwich-norm(tanh gate) · 직접 readout.

eps를 일부러 크게(1e-2) 둬서 stream_norms의 eps 보상(1/s²)을 빠뜨리면 동치 테스트가 깨지게 한다.
"""
import torch
import torch.nn.functional as F
from torch import nn

from buc.teacher import t4_fp16 as R


class RMS(nn.Module):
    def __init__(self, d, eps=1e-2, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w_gate = nn.Linear(d, h)
        self.w_up = nn.Linear(d, h)
        self.w_down = nn.Linear(h, d)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Attn(nn.Module):
    def __init__(self, d, heads=2):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        b, t, d = x.shape

        def sp(z):
            return z.view(b, t, self.h, d // self.h).transpose(1, 2)

        y = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(x)), sp(self.v(x)))
        return self.o(y.transpose(1, 2).reshape(b, t, d))


class PreBlock(nn.Module):
    def __init__(self, d, dc, hid):
        super().__init__()
        self.norm1 = RMS(d)
        self.norm2 = RMS(d)
        self.ada = nn.Linear(dc, 2 * d)  # [scale | gate], gate 선형
        self.attn = Attn(d)
        self.mlp = SwiGLU(d, hid)

    def forward(self, x, c):
        scale, gate = self.ada(c).chunk(2, -1)
        x = x + gate[:, None] * self.attn(self.norm1(x) * (1 + scale[:, None]))
        return x + self.mlp(self.norm2(x))


class SandBlock(nn.Module):
    def __init__(self, d, dc, hid):
        super().__init__()
        self.pre1, self.post1, self.pre2, self.post2 = RMS(d), RMS(d), RMS(d), RMS(d)
        self.ada = nn.Linear(dc, 4 * d)
        self.attn = Attn(d)
        self.mlp = SwiGLU(d, hid)

    def forward(self, x, c):
        sm, gm, sf, gf = self.ada(c).chunk(4, -1)
        x = x + torch.tanh(gm)[:, None] * self.post1(self.attn(self.pre1(x) * (1 + sm[:, None])))
        return x + torch.tanh(gf)[:, None] * self.post2(self.mlp(self.pre2(x) * (1 + sf[:, None])))


class ToyDiT(nn.Module):
    """kind='pre' | 'sand'. 입력 x[B,T,din], c[B,dc] -> [B,T+1,dout]."""

    def __init__(self, kind="pre", din=16, d=32, dc=8, hid=64, depth=2, dout=8):
        super().__init__()
        self.kind = kind
        self.d = d
        self.embed = nn.Linear(din, d)
        self.pad_token = nn.Parameter(torch.randn(1, 1, d))
        blk = PreBlock if kind == "pre" else SandBlock
        self.blocks = nn.ModuleList([blk(d, dc, hid) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, eps=1e-2, elementwise_affine=False)
        self.head = nn.Linear(d, dout)

    def forward(self, x, c):
        h = self.embed(x)
        h = torch.cat([self.pad_token.expand(h.shape[0], 1, -1).to(h.dtype), h], 1)
        for b in self.blocks:
            h = b(h, c)
        return self.head(self.final_norm(h))


def plan_for(m: ToyDiT, s, t=1.0, islands=False):
    n = len(m.blocks)
    swiglu = tuple(R.SwiGLUSite(up=f"blocks.{i}.mlp.w_up", gate=f"blocks.{i}.mlp.w_gate",
                                down=f"blocks.{i}.mlp.w_down", scale=t) for i in range(n))
    if m.kind == "pre":
        norms = tuple(f"blocks.{i}.norm{j}" for i in range(n) for j in (1, 2)) + ("final_norm",)
        return R.RescuePlan(
            s=s, embed=("embed",), stream_params=("pad_token",),
            gate_rows=tuple((f"blocks.{i}.ada", m.d, 2 * m.d) for i in range(n)),
            residual_out=tuple(f"blocks.{i}.mlp.w_down" for i in range(n)),
            stream_norms=norms, swiglu=swiglu,
            fp32_islands=norms if islands else ())
    stream = tuple(f"blocks.{i}.pre{j}" for i in range(n) for j in (1, 2)) + ("final_norm",)
    post = tuple(f"blocks.{i}.post{j}" for i in range(n) for j in (1, 2))
    return R.RescuePlan(
        s=s, embed=("embed",), stream_params=("pad_token",),
        residual_out=post, stream_norms=stream, swiglu=swiglu,
        fp32_islands=(stream + post) if islands else ())


def make(kind="pre", seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    m = ToyDiT(kind)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.to(dtype).eval()


def inputs(seed=1, dtype=torch.float64, b=2, t=7, din=16, dc=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, din, generator=g).to(dtype), torch.randn(b, dc, generator=g).to(dtype)

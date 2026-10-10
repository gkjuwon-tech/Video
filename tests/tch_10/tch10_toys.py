"""TCH-10 장난감: 이상치 채널 활성, 저랭크 구조 가중치, 작은 MLP, 보정 후보."""
import torch
from torch import nn

from buc.teacher import quant as Q

K, O = 256, 128
OUTLIERS = (3, 17, 40, 101)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def acts(n, seed, *, k=K, outliers=OUTLIERS, mag=60.0):
    x = torch.randn(n, k, generator=gen(seed))
    if outliers:
        x[:, list(outliers)] *= mag
    return x


def linear(seed=1, *, lowrank=0):
    lin = nn.Linear(K, O)
    g = gen(seed)
    with torch.no_grad():
        w = torch.randn(O, K, generator=g) / 16
        if lowrank:
            w += torch.randn(O, lowrank, generator=g) @ torch.randn(lowrank, K, generator=g)
        lin.weight.copy_(w)
        lin.bias.copy_(torch.randn(O, generator=g) / 4)
    return lin


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


class MLP(nn.Module):
    """토큰 [B,T,K] → [B,T,40]. odd(48→40)는 w4a4(g=64)에서 K%g≠0, unused는 보정 중 안 불린다."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.up = nn.Linear(K, 512)
        self.down = nn.Linear(512, K)
        self.head = nn.Linear(K, 48)
        self.odd = nn.Linear(48, 40)  # w4a4(g=64)에서 K%g≠0 → float 유지
        self.unused = nn.Linear(K, K)  # 보정 중 안 불림 → float 유지

    def forward(self, x):
        h = x + self.down(torch.nn.functional.gelu(self.up(x)))
        return self.odd(self.head(h))


def mlp_inputs(n, seed, t=8):
    return acts(n * t, seed).reshape(n, t, K)


def candidates(n=24, *, seed=10, bad=()):
    """id c00..; bad = {index: taint set}."""
    xs = mlp_inputs(n, seed)
    out = []
    for i in range(n):
        tags = bad.get(i, {"ENGINE"}) if isinstance(bad, dict) else {"ENGINE"}
        out.append(Q.CalibSample(id=f"c{i:02d}", data=xs[i], prov=frozenset(tags)))
    return out


def calib(n=16, seed=0, **kw):
    return Q.build_calib_set(candidates(**kw), n, seed=seed)

"""TCH-03 테스트 공용 가짜 교사와 데이터 생성기. 결정론(지역 Generator만 쓴다)."""
from __future__ import annotations

import torch
from torch import nn

from buc.gov import prov as P
from buc.teacher import taps as T

D = 32          # 가짜 은닉 차원 (실제 3840)
K = 8           # 가짜 PCA 차원 (실제 128)
NUM_LAYERS = 24  # 층 23까지 필요
GRID = (3, 4)   # 이미지 토큰 격자 (gh, gw), 실제는 H/16 x W/16
OFF = 2         # 이미지 토큰 앞의 비이미지 토큰 수
TAIL = 3        # 뒤의 비이미지 토큰 수
SEQ = OFF + GRID[0] * GRID[1] + TAIL
ENGINE = P.as_record(["ENGINE"])


def gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class FakeBlock(nn.Module):
    """토큰별 독립 블록(어텐션 없음): 다른 토큰 값이 출력에 영향을 주지 않는다."""

    def __init__(self, seed: int, tuple_out: bool = False, dim: int = D):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        g = gen(seed)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(dim, dim, generator=g) / dim ** 0.5)
            self.lin.bias.copy_(torch.randn(dim, generator=g) * 0.1)
        self.tuple_out = tuple_out

    def forward(self, x):
        y = x + 0.5 * torch.tanh(self.lin(x))
        return (y, 0) if self.tuple_out else y


class FakeDiT(nn.Module):
    def __init__(self, n: int = NUM_LAYERS, dim: int = D, tuple_layers=(), seed: int = 100):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(seed + i, i in tuple_layers, dim) for i in range(n)])

    def forward(self, x, upto=None, twice=()):
        h = x
        for i, blk in enumerate(self.layers):
            if upto is not None and i >= upto:
                break
            for _ in range(2 if i in twice else 1):
                h = blk(h)
                if isinstance(h, tuple):
                    h = h[0]
        return h


def make_input(seed: int = 1, batch: int = 2, dim: int = D, seq: int = SEQ) -> torch.Tensor:
    return torch.randn(batch, seq, dim, generator=gen(seed))


def collect_acts(model: FakeDiT, x: torch.Tensor, layers=T.TAP_LAYERS) -> dict:
    """오라클용: 우리가 직접 건 훅으로 각 층 출력(텐서)을 모은다."""
    got, hs = {}, []
    for l in layers:
        def mk(l):
            def h(m, a, out):
                got[l] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return h
        hs.append(model.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hs:
            h.remove()
    return got


def fit_samples(seed: int = 7, n: int = 96, dim: int = D, layers=T.TAP_LAYERS) -> dict:
    """층마다 다른 평균·스케일을 가진 가짜 활성에서 PCA 표본을 만든다."""
    out = {}
    for j, l in enumerate(layers):
        g = gen(seed + j)
        sc = torch.linspace(3.0, 0.3, dim)
        out[l] = torch.randn(n, dim, generator=g) * sc + torch.randn(dim, generator=g)
    return out


def make_basis(seed: int = 7, k: int = K, dim: int = D) -> "T.PCABasis":
    return T.fit_pca_basis(fit_samples(seed, dim=dim), prov=ENGINE, k=k)


def oracle_quant(basis: "T.PCABasis", layer: int, x: torch.Tensor) -> torch.Tensor:
    """float64 오라클: round(((x-mean)@V)/scale) 클램프 ±127. 반환 [..., k] int64."""
    i = basis.layers.index(layer)
    mean = basis.mean[i].double()
    comps = basis.comps[i].double()
    scale = basis.scale[i].double()
    z = (x.double() - mean) @ comps
    return torch.round(z / scale).clamp(-127, 127).to(torch.int64)


def assert_quant_close(q: torch.Tensor, q_or: torch.Tensor, max_frac: float = 0.01):
    """int8 결과가 오라클과 같다. float32 대 float64 반올림 경계 때문에 일부는 ±1 허용."""
    d = (q.to(torch.int64) - q_or).abs()
    assert int(d.max()) <= 1, f"최대 차이 {int(d.max())} > 1 LSB"
    assert float((d > 0).double().mean()) <= max_frac, "경계 불일치 비율이 너무 크다"

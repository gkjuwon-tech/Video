"""TCH-05 공용 테스트 부품: 자연 영상 합성기, 코덱, 기록용 가짜 샘플러, 오라클 샘플러, 합성 결정 곡선."""
from types import SimpleNamespace

import numpy as np
import torch

from buc.core.pyramid import gaussian_blur
from buc.teacher import decision_curve as dc
from buc.teacher.sample import SigmaTable

TABLE = SigmaTable.from_flow_shift()


def natural(h, w, seed, channels=3):
    """1/f 진폭 스펙트럼 영상(자연 영상 흉내), 표준편차 0.4, float64 (h,w,C)."""
    g = np.random.default_rng(seed)
    f = np.fft.fft2(g.standard_normal((h, w, channels)), axes=(0, 1))
    fy = np.fft.fftfreq(h)[:, None, None]
    fx = np.fft.fftfreq(w)[None, :, None]
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    img = np.real(np.fft.ifft2(f / r, axes=(0, 1)))
    return img / img.std() * 0.4


def batch(n, h=64, w=64, seed0=0):
    """[N,3,h,w] float32 텐서."""
    a = np.stack([natural(h, w, seed0 + i) for i in range(n)])
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().contiguous()


class IdCodec:
    """잠재 = 픽셀(항등). 호출을 센다."""

    def __init__(self):
        self.enc = 0
        self.dec = 0

    def encode(self, x):
        self.enc += 1
        return x.clone()

    def decode(self, z):
        self.dec += 1
        return z.clone()


class PoolCodec:
    """손실 코덱: 2x2 평균 풀링 -> 최근접 2배 확대. 왕복이 항등이 아니다."""

    def encode(self, x):
        return torch.nn.functional.avg_pool2d(x, 2)

    def decode(self, z):
        return z.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


class RecSampler:
    """TeacherSampler 가짜. sdedit 인자를 기록하고 x0_hat = x0 (항등 교사)."""

    def __init__(self, table=TABLE, fn=None):
        self.table = table
        self.calls = []
        self.fn = fn

    def sdedit(self, x0, cond, *, sigma, seeds, steps=2, level=0, k=0, **kw):
        self.calls.append(SimpleNamespace(sigma=sigma, seeds=seeds, steps=steps, level=level, k=k,
                                          cond=cond, extra=kw, shape=tuple(x0.shape)))
        out = x0.clone() if self.fn is None else self.fn(x0, sigma, seeds)
        return SimpleNamespace(x0_hat=out)


class OracleSampler(RecSampler):
    """알려진 결정 구조를 가진 교사: 저주파는 입력 보존, 고주파는 시드 노이즈로 교체.
    y = G_c(x) + (n - G_c(n)), c = c0 * 2^(-j/2) (j = 표 인덱스, σ가 클수록 c가 커서 더 굵은 옥타브까지 교체)."""

    def __init__(self, table=TABLE, c0=6.0):
        super().__init__(table)
        self.c0 = c0

    def sdedit(self, x0, cond, *, sigma, seeds, steps=2, level=0, k=0, **kw):
        self.calls.append(SimpleNamespace(sigma=sigma, seeds=seeds, steps=steps, level=level, k=k,
                                          cond=cond, extra=kw, shape=tuple(x0.shape)))
        j = self.table.index_of(sigma)
        c = self.c0 * 2.0 ** (-j / 2.0)
        g = torch.Generator().manual_seed(int(seeds))
        n = torch.randn(x0.shape, generator=g, dtype=torch.float64) * float(x0.std())
        outs = []
        for b in range(x0.shape[0]):
            x = x0[b].permute(1, 2, 0).double().numpy()
            nb = n[b].permute(1, 2, 0).numpy()
            y = gaussian_blur(x, c) + (nb - gaussian_blur(nb, c))
            outs.append(torch.from_numpy(y).permute(2, 0, 1).float())
        return SimpleNamespace(x0_hat=torch.stack(outs))


class ExplodingSampler(RecSampler):
    def sdedit(self, *a, **kw):
        raise AssertionError("검증 실패는 sdedit 호출 전에 나야 한다")


def logistic_rows(edges, k, levels):
    """D(o) = 1/(1+exp(k (o - e))): 옥타브 o 가 클수록(굵을수록) D가 작다."""
    o = np.arange(levels, dtype=np.float64)
    return np.array([1.0 / (1.0 + np.exp(k * (o - e))) for e in edges])


def make_curve(edges, k, levels, *, steps=2, indices=None, n_seeds=2, spread=None):
    n = len(edges)
    idx = tuple(range(n)) if indices is None else tuple(indices)
    d = logistic_rows(edges, k, levels)
    sp = np.zeros_like(d) if spread is None else spread
    return dc.DecisionCurve(idx, tuple(float(TABLE.points[i]) for i in idx), d, sp, np.ones(levels), levels,
                            steps, 1, n_seeds, TABLE.key)


def ref_edge(d, tau, step=1e-4):
    """독립 오라클: 조각 선형 보간한 D를 촘촘한 격자에서 굵은 쪽부터 훑어 처음 tau를 넘는 위치."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if d[-1] > tau:
        return np.inf
    pos = np.arange(n - 1, -step, -step)
    vals = np.interp(pos, np.arange(n), d)
    hit = np.nonzero(vals > tau)[0]
    return -np.inf if hit.size == 0 else float(pos[hit[0]])

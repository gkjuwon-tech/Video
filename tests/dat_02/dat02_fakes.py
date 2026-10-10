"""DAT-02 테스트 공용 가짜 교사(덕 타이핑). 결정론: 지역 Generator만 쓴다.

가짜 변환기는 diffusers ZImage 규약(forward(x_list, t, cap_list, return_dict) -> -v, t=1-σ)을 따르고,
층 24개(토큰별 독립 블록)를 실제로 통과시켜 TCH-03 탭 훅이 [B,S,D] 은닉을 보게 한다.
토큰 배치: [레지스터 OFF개 | 이미지 토큰 gh*gw (행 우선, 패치 2x2) | 캡션 1개].
예측: x̂0 = z/(1-σ) + δ·checker(잠재 h,w) + b·mean(cap) (b=0.01, 무노이즈 모드에서는 0). 잠재 체커는 영상에서 주기 16px 정사각파 -> 옥타브 ≤3.
가짜 VAE: encode = 8x8 평균 풀링 후 16x3 행렬 M, decode = pinv(M) 후 최근접 8배 확대 (블록 상수 영상은 정확 왕복).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from buc.core import det
from buc.gov import prov as P
from buc.teacher import taps as T
from buc.teacher.sample import TeacherSampler
from buc.teacher.zimage import TeacherSpec
from buc.teacher.zimage import make_noise as z_make_noise

D = 32           # 가짜 은닉 차원
K = 8            # 가짜 PCA 차원
NUM_LAYERS = 24
OFF = 2          # 이미지 토큰 앞 레지스터 토큰 수
C = 16
TEXT_DIM = 8
IMG = 64         # 기본 영상 크기 64x64 -> 잠재 8x8 -> 탭 격자 4x4
GRID = (IMG // 16, IMG // 16)
SPEC = replace(TeacherSpec(), num_layers=NUM_LAYERS, hidden_dim=D, text_dim=TEXT_DIM, max_text_len=16)
ENGINE = P.as_record(["ENGINE"])


def gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class FakeBlock(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.lin = nn.Linear(D, D)
        g = gen(seed)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(D, D, generator=g) / D ** 0.5)
            self.lin.bias.copy_(torch.randn(D, generator=g) * 0.1)

    def forward(self, x):
        return x + 0.5 * torch.tanh(self.lin(x))


def latent_checker(h: int, w: int) -> torch.Tensor:
    i = torch.arange(h).view(h, 1)
    j = torch.arange(w).view(1, w)
    return (1.0 - 2.0 * ((i + j) % 2)).float().view(1, 1, h, w)


class FakeDiT(nn.Module):
    def __init__(self, delta: float = 0.05, seed: int = 100, cap_bias: float = 0.01):
        super().__init__()
        self.cap_bias = float(cap_bias)
        self.layers = nn.ModuleList([FakeBlock(seed + i) for i in range(NUM_LAYERS)])
        g = gen(seed - 1)
        self.register_buffer("embed", torch.randn(C * 4, D, generator=g) / (C * 4) ** 0.5)
        self.register_buffer("reg", torch.randn(OFF, D, generator=g))
        self.register_buffer("cap_proj", torch.randn(TEXT_DIM, D, generator=g))
        self.delta = float(delta)
        self.calls = 0
        self.batch_sizes = []

    def tokens(self, z: torch.Tensor, t: torch.Tensor, cap_feats) -> torch.Tensor:
        n, c, h, w = z.shape
        gh, gw = h // 2, w // 2
        pt = z.reshape(n, c, gh, 2, gw, 2).permute(0, 2, 4, 1, 3, 5).reshape(n, gh * gw, c * 4)
        tok = pt @ self.embed
        reg = self.reg.unsqueeze(0).expand(n, -1, -1) + 3.0 * t.float().view(n, 1, 1)
        cap = torch.stack([cf.float().mean(0) @ self.cap_proj for cf in cap_feats]).unsqueeze(1)
        return torch.cat([reg, tok, cap], dim=1)

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls += 1
        self.batch_sizes.append(len(x))
        assert len(cap_feats) == len(x) == int(t.shape[0])
        z = torch.stack([a.float() for a in x]).squeeze(2)            # [n,C,h,w]
        n, c, h, w = z.shape
        hs = self.tokens(z, t, cap_feats).to(x[0].dtype)
        for blk in self.layers:
            hs = blk(hs)
        sig = (1.0 - t.float()).view(n, 1, 1, 1)
        capm = torch.stack([cf.float().mean() for cf in cap_feats]).view(n, 1, 1, 1)
        x0 = z / (1.0 - sig) + self.delta * latent_checker(h, w) + self.cap_bias * capm
        out = -(z - x0) / sig
        outs = list(out.unsqueeze(2).to(x[0].dtype).unbind(0))
        return {"sample": outs} if return_dict else (outs,)


class FakeVae:
    def __init__(self, seed: int = 5):
        g = gen(seed)
        m = torch.randn(C, 3, generator=g, dtype=torch.float64)
        m[:, 0] += 1.0                       # pinv(M)·1 이 0이 되지 않게 비대칭
        self.M = m.float()
        self.Minv = torch.linalg.pinv(m).float()
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        assert x.ndim == 4 and x.shape[1] == 3 and x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0
        assert bool(torch.isfinite(x).all())
        p = Fn.avg_pool2d(x.float(), 8)
        return torch.einsum("ck,bkhw->bchw", self.M, p).contiguous()

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        self.decode_calls += 1
        p = torch.einsum("kc,bchw->bkhw", self.Minv, z.float())
        y = p.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        return y.clamp(-1.0, 1.0) if clamp else y


class FakeTeacher:
    """ZImageTeacher 덕 타입: spec, device, dtype, transformer, layers, encode_text, encode_image,
    decode_latent, make_noise."""

    def __init__(self, delta: float = 0.05, zero_noise: bool = False):
        self.spec = SPEC
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.transformer = FakeDiT(delta, cap_bias=0.0 if zero_noise else 0.01).eval()
        self.layers = self.transformer.layers
        self.vae = FakeVae()
        self.zero_noise = zero_noise
        g = gen(3)
        self._emb = torch.randn(256, TEXT_DIM, generator=g)
        self.text_calls = 0

    def encode_text(self, prompts) -> list:
        self.text_calls += 1
        out = []
        for p in prompts:
            ids = torch.tensor(list(p.encode("utf-8"))[:16] or [0], dtype=torch.long)
            out.append(self._emb[ids].clone())
        return out

    def encode_image(self, x):
        return self.vae.encode(x)

    def decode_latent(self, z, clamp: bool = True):
        return self.vae.decode(z, clamp=clamp)

    def make_noise(self, seed, batch, height, width, *, level=0, k=0):
        shape = self.spec.latent_shape(height, width, batch)
        if self.zero_noise:
            return torch.zeros(shape, dtype=torch.float32)
        return z_make_noise(seed, shape, level=level, k=k, dtype=torch.float32, device="cpu")

    @property
    def calls(self) -> int:
        return self.transformer.calls

    def total_calls(self) -> int:
        return self.transformer.calls + self.vae.encode_calls + self.vae.decode_calls


_BASIS = None


def basis() -> T.PCABasis:
    """가짜 DiT의 이미지 토큰 활성으로 1회 적합한 PCA 기저(k=8)."""
    global _BASIS
    if _BASIS is None:
        dit = FakeDiT()
        samples = {l: [] for l in T.TAP_LAYERS}
        hooks = []
        for l in T.TAP_LAYERS:
            def mk(l):
                def h(m, a, out):
                    samples[l].append(out[:, OFF:OFF + GRID[0] * GRID[1]].reshape(-1, D).detach().clone())
                return h
            hooks.append(dit.layers[l].register_forward_hook(mk(l)))
        g = gen(11)
        caps = [torch.randn(4, TEXT_DIM, generator=g) for _ in range(6)]
        with torch.no_grad():
            for s in (0.3, 0.5, 9 / 14):
                z = torch.randn(6, C, IMG // 8, IMG // 8, generator=g) * 0.6
                dit(list(z.unsqueeze(2).unbind(0)), torch.full((6,), 1.0 - s), caps, return_dict=False)
        for h in hooks:
            h.remove()
        _BASIS = T.fit_pca_basis({l: torch.cat(v) for l, v in samples.items()}, prov=ENGINE, k=K)
    return _BASIS


def make_rig(delta: float = 0.05, zero_noise: bool = False, max_views_per_call=None):
    teacher = FakeTeacher(delta=delta, zero_noise=zero_noise)
    sampler = TeacherSampler(teacher, max_views_per_call=max_views_per_call)
    bus = T.TapBus(basis()).attach(teacher.layers)
    return teacher, sampler, bus


def block_image(seed: int, h: int = IMG, w: int = IMG, lo: float = 0.3, hi: float = 0.7) -> np.ndarray:
    """8x8 블록 상수 영상 (H,W,3) f32: 가짜 VAE 왕복이 정확하다."""
    rng = np.random.default_rng(seed)
    b = rng.uniform(lo, hi, size=(h // 8, w // 8, 3)).astype(np.float32)
    return np.ascontiguousarray(np.repeat(np.repeat(b, 8, axis=0), 8, axis=1))


def smooth_image(seed: int, h: int = IMG, w: int = IMG) -> np.ndarray:
    """일반 영상(블록 상수 아님) (H,W,3) f32 [0.1,0.9]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base = 0.5 + 0.25 * np.sin(2 * np.pi * (xx * rng.uniform(0.5, 2) + yy * rng.uniform(0.5, 2)))
    img = np.stack([base, base[::-1], base[:, ::-1]], -1) + rng.uniform(-0.1, 0.1, size=(h, w, 3))
    return np.clip(img, 0.1, 0.9).astype(np.float32)


def oracle_octave_map(h: int, w: int, n_oct: int) -> np.ndarray:
    """독립 오라클: 경계 비교로 옥타브를 매긴다(log2 를 쓰지 않는다)."""
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    out = np.zeros((h, w), dtype=np.int64)
    for i in range(h):
        for j in range(w):
            r = max(abs(fy[i]), abs(fx[j]))
            o = n_oct - 1
            for cand in range(n_oct - 1):
                if 0.5 * 2.0 ** -(cand + 1) < r <= 0.5 * 2.0 ** -cand:
                    o = cand
                    break
            out[i, j] = o
    return out


def oracle_band_project(x: np.ndarray, valid) -> np.ndarray:
    """x[...,H,W] float64 를 유효 옥타브로 사영."""
    h, w = x.shape[-2:]
    om = oracle_octave_map(h, w, len(valid))
    mask = np.isin(om, [o for o, v in enumerate(valid) if v])
    return np.real(np.fft.ifft2(np.fft.fft2(x.astype(np.float64)) * mask))


def oracle_lowband_err(img_out: np.ndarray, img_in: np.ndarray, valid) -> float:
    d = (img_out.astype(np.float64) - img_in.astype(np.float64)).transpose(2, 0, 1)
    p = oracle_band_project(d, valid)
    return float(np.sqrt(np.mean(p ** 2)))


def item_seeds(seed: int, keys) -> list:
    return [det.derive_seed(seed, k) for k in keys]


def oracle_sdedit(teacher, sampler, images, captions, keys, *, sigma, steps, seed):
    """계약 2단계의 오라클: encode -> sampler.sdedit -> decode(마지막 x̂0)."""
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1) for im in images]) * 2 - 1
    x0 = teacher.encode_image(x.float())
    res = sampler.sdedit(x0, list(captions), sigma=sigma, seeds=item_seeds(seed, keys), steps=steps)
    img = ((teacher.decode_latent(res.x0_hat) + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return res, x0, img


def oracle_taps(teacher, sampler, bus, images, captions, keys, *, sigma, steps, seed, tap_step, token_offset=OFF):
    """탭 step 의 상태 z_j 에서 직접 캡처한 탭 {l: int8[B,k,gh,gw]}."""
    j = tap_step % steps
    res, x0, _ = oracle_sdedit(teacher, sampler, images, captions, keys, sigma=sigma, steps=steps, seed=seed)
    if j == 0:
        zj = res.z_start
    else:
        zj = sampler.sdedit(x0, list(captions), sigma=sigma, seeds=item_seeds(seed, keys), steps=j).z
    snap = sampler.table.snap(sigma)
    sj = sampler.table.points[snap.index + j]
    h, w = images[0].shape[:2]
    with bus.capture(ENGINE, grid=(h // 16, w // 16), token_offset=token_offset) as cap:
        sampler.predict_x0(zj, sj, list(captions))
    return cap.tapset.layers, sj


def assert_tap_close(q: torch.Tensor, q_or: torch.Tensor, max_frac: float = 0.02):
    assert q.dtype == torch.int8
    assert tuple(q.shape) == tuple(q_or.shape), (tuple(q.shape), tuple(q_or.shape))
    d = (q.to(torch.int64) - q_or.to(torch.int64)).abs()
    assert int(d.max()) <= 1, f"탭 최대 차이 {int(d.max())} > 1 LSB"
    assert float((d > 0).double().mean()) <= max_frac

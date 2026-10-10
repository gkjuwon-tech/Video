"""RD-07 테스트 공용 도구. 결정론: 지역 Generator/default_rng만 쓴다(전역 RNG 불변).

- numpy 오라클: 타일 가중치(계약서 수식), 가중 affine(a>=0), 상대 잔차.
- StubReader: 영상의 국소 함수로 판독을 내는 가짜 읽기기(게이지·교란을 주입할 수 있다).
- 가짜 DiT(토큰별 블록 24개, 행 우선 패치) + 가짜 VAE + 진짜 load_teacher/TeacherSampler/TapBus + 소형 MRH.
RD-07 구현(buc.read.mrh_call)은 StubReader.__call__ 안에서만 지연 import 한다.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn

from buc.gov import prov as P
from buc.teacher import taps as T
from buc.teacher.sample import TeacherSampler
from buc.teacher.zimage import Loaders, TeacherSpec, ZIMAGE_REPO, load_teacher

LAYERS = (8, 15, 23)
D = 16            # 가짜 은닉 차원
K = 8             # 가짜 PCA 차원
OFF = 2           # 이미지 토큰 앞의 비이미지 토큰 수
LAT_C = 16
REV = "c" * 40
SPEC = replace(TeacherSpec(), num_layers=24, text_dim=8, max_text_len=16, hidden_dim=D)

ENGINE = P.as_record(["ENGINE"])
USER = P.as_record(["USER"])


# --------------------------------------------------------------------------- 영상·오라클

def make_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """선형 장면값 (h,w,3) float32 ∈ [0.05, 0.95]. 저주파 무작위 장."""
    rng = np.random.default_rng(seed)
    x = gaussian_filter(rng.normal(size=(h, w, 3)), sigma=(5, 5, 0))
    for c in range(3):
        a = x[..., c]
        x[..., c] = 0.05 + 0.9 * (a - a.min()) / (a.max() - a.min())
    return x.astype(np.float32)


def canvas_display(img_lin: np.ndarray, canvas_hw=None) -> torch.Tensor:
    """계약서 2단계의 표시값: 오른쪽·아래 edge 패딩 후 2*linear_to_display-1. [1,3,Hc,Wc] f32."""
    from buc.core import color
    h, w = img_lin.shape[:2]
    hc = 16 * -(-h // 16) if canvas_hw is None else canvas_hw[0]
    wc = 16 * -(-w // 16) if canvas_hw is None else canvas_hw[1]
    pad = np.pad(img_lin, ((0, hc - h), (0, wc - w), (0, 0)), mode="edge")
    x = 2.0 * color.linear_to_display(pad).astype(np.float32) - 1.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float()


def oracle_tile_weights(boxes, canvas_hw, feather_px) -> np.ndarray:
    h4, w4 = canvas_hw[0] // 4, canvas_hw[1] // 4
    fc = feather_px / 4.0
    out = np.zeros((len(boxes), h4, w4))

    def axis(c0, c1, n):
        i = np.arange(c0, c1, dtype=np.float64)
        d = np.full(i.shape, np.inf)
        if c0 > 0:
            d = np.minimum(d, i - c0 + 0.5)
        if c1 < n:
            d = np.minimum(d, c1 - i - 0.5)
        if fc == 0:
            return np.ones_like(d)
        u = np.clip(d / fc, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    for t, b in enumerate(boxes):
        y0, y1, x0, x1 = b.y0 // 4, b.y1 // 4, b.x0 // 4, b.x1 // 4
        out[t, y0:y1, x0:x1] = axis(y0, y1, h4)[:, None] * axis(x0, x1, w4)[None, :]
    return out


def oracle_affine(src, ref, w=None):
    """가중 LS a*src+b≈ref, a>=0. 퇴화/음수 기울기면 (0, ref 가중평균). float64."""
    s = np.asarray(src, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    ww = np.ones_like(s) if w is None else np.asarray(w, np.float64).ravel()
    tot = ww.sum()
    ms, mr = (ww * s).sum() / tot, (ww * r).sum() / tot
    var = (ww * (s - ms) ** 2).sum() / tot
    cov = (ww * (s - ms) * (r - mr)).sum() / tot
    if var <= 1e-12:
        return 0.0, mr
    a = cov / var
    if a < 0:
        return 0.0, mr
    return a, mr - a * ms


def oracle_rel(src, ref, w=None, floor=0.05) -> float:
    s = np.asarray(src, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    ww = np.ones_like(s) if w is None else np.asarray(w, np.float64).ravel()
    a, b = oracle_affine(s, r, ww)
    tot = ww.sum()
    rms = np.sqrt((ww * (a * s + b - r) ** 2).sum() / tot)
    mr = (ww * r).sum() / tot
    std = np.sqrt((ww * (r - mr) ** 2).sum() / tot)
    return float(rms / max(std, floor))


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


# --------------------------------------------------------------------------- StubReader

def stub_outputs(image: torch.Tensor, gauge=(1.0, 0.0), depth_fn=None) -> dict:
    """image [1,3,h,w] -> 8개 헤드(H/4). 4x4 블록의 평균(g4)의 국소 함수라 크롭·전역이 같은 값을 낸다."""
    g4 = F.avg_pool2d(image.mean(dim=1, keepdim=True), 4)
    d = 2.0 * g4 + 0.3 * g4 ** 2 if depth_fn is None else depth_fn(g4)
    nrm = unit(torch.cat([g4, 0.5 * g4 ** 2, torch.ones_like(g4)], 1))
    sem = unit(g4 * torch.linspace(-1, 1, 8).view(1, 8, 1, 1) + torch.linspace(0.5, 1.5, 8).view(1, 8, 1, 1))
    return {
        "log_depth": gauge[0] * d + gauge[1],
        "normal": nrm,
        "albedo": torch.sigmoid(g4 * torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)),
        "rough_metal": torch.sigmoid(g4 * torch.tensor([1.0, -1.0]).view(1, 2, 1, 1)),
        "shading": F.softplus(g4),
        "defocus": F.softplus(-g4),
        "sem": sem,
        "sigma_hat": F.softplus(g4 * torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)) + 1e-3,
    }


class StubReader:
    """읽기기 프로토콜: snap(sigma) -> float, __call__(image, sigma, *, prov, key) -> ViewRead."""

    def __init__(self, points=(1.0, 0.75, 0.5, 0.35, 0.25, 0.2), gauge=None, depth_fn=None, post=None,
                 ephemeral=None):
        self.points = tuple(points)
        self.gauge = dict(gauge or {})        # key -> (A, B)
        self.depth_fn = dict(depth_fn or {})  # key -> callable(g4) -> depth
        self.post = dict(post or {})          # key -> callable(outputs) -> outputs
        self.ephemeral = ephemeral            # None -> ext 비트를 따른다
        self.calls = []

    def snap(self, sigma) -> float:
        return float(min(self.points, key=lambda p: (abs(p - float(sigma)), -p)))

    def __call__(self, image, sigma, *, prov, key):
        from buc.read.mrh_call import ViewRead
        self.calls.append(SimpleNamespace(image=image.clone(), sigma=sigma, key=key, prov=prov))
        out = stub_outputs(image, self.gauge.get(key, (1.0, 0.0)), self.depth_fn.get(key))
        if key in self.post:
            out = self.post[key](out)
        eph = bool(P.ext_bit(prov)) if self.ephemeral is None else bool(self.ephemeral)
        return ViewRead(outputs=out, sigma=self.snap(sigma), digest=bytes(32), ephemeral=eph)


# --------------------------------------------------------------------------- 가짜 DiT·VAE·교사

def _patchify(z: torch.Tensor) -> torch.Tensor:
    c, h, w = z.shape
    return z.reshape(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4).reshape((h // 2) * (w // 2), c * 4)


def _unpatchify(tok: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    return tok.reshape(h // 2, w // 2, c, 2, 2).permute(2, 0, 3, 1, 4).reshape(c, h, w)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x):
        return x + 0.5 * torch.tanh(self.lin(x))


class FakeDiT(nn.Module):
    """ZImage 변환기 규약을 흉내 낸다. 은닉 = [비이미지 OFF개 | 행 우선 패치 토큰], 블록은 토큰별 독립."""

    def __init__(self, seed: int = 0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.ModuleList(_Block() for _ in range(24))
            self.inp = nn.Linear(LAT_C * 4, D)
            self.cap = nn.Linear(SPEC.text_dim, D)
            self.off_bias = nn.Parameter(torch.randn(OFF, D))
            self.out = nn.Linear(D, LAT_C * 4)
        self.calls = []

    def trunk(self, x_list, cap_list, upto=None) -> torch.Tensor:
        rows = []
        for a, c in zip(x_list, cap_list):
            img = self.inp(_patchify(a[:, 0].float()))
            pre = self.cap(c.float().mean(0))[None] + self.off_bias
            rows.append(torch.cat([pre, img], 0))
        h = torch.stack(rows)
        for i, blk in enumerate(self.layers):
            if upto is not None and i >= upto:
                break
            h = blk(h)
        return h

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls.append(SimpleNamespace(
            n=len(x), t=t.detach().clone(), t_dtype=t.dtype, x_shapes=[tuple(a.shape) for a in x],
            return_dict=return_dict, grad_enabled=torch.is_grad_enabled()))
        h = self.trunk(x, cap_feats)
        outs = []
        for i, a in enumerate(x):
            tok = 0.1 * self.out(h[i, OFF:])
            outs.append(_unpatchify(tok, a.shape[0], a.shape[2], a.shape[3]).unsqueeze(1).to(a.dtype))
        return (outs,) if not return_dict else SimpleNamespace(sample=outs)


class _FakeTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=SPEC.text_dim)
        g = torch.Generator().manual_seed(3)
        self.emb = nn.Embedding(256, SPEC.text_dim)
        self.emb.weight.data = torch.randn(256, SPEC.text_dim, generator=g)


class _FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=LAT_C, scaling_factor=0.5, shift_factor=0.25)
        g = torch.Generator().manual_seed(5)
        self.mix = nn.Parameter(torch.randn(LAT_C, 3, generator=g))

    def encode(self, x):
        z = torch.einsum("oc,bchw->bohw", self.mix, F.avg_pool2d(x, 8))
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


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


def make_teacher(tmp_path, dit: FakeDiT):
    def encode_prompts(tok, te, prompts, max_len):
        return [te.emb(torch.tensor(list(p.encode("utf-8"))[:max_len] or [0], dtype=torch.long)).detach()
                for p in prompts]

    loaders = Loaders(tokenizer=lambda path: SimpleNamespace(name="fake-tok"),
                      text_encoder=lambda path, dt: _FakeTE().to(dt),
                      transformer=lambda path, dt: dit.to(dt),
                      vae=lambda path, dt: _FakeVae().to(dt),
                      encode_prompts=encode_prompts)
    return load_teacher(SPEC, device="cpu", dtype=torch.float32, hub_cache=make_hub(tmp_path), loaders=loaders,
                        te_cache_dir=tmp_path / "tec", deterministic=False)


def make_basis(seed: int = 0) -> T.PCABasis:
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(len(LAYERS), D, generator=g) * 0.1
    comps = torch.randn(len(LAYERS), D, K, generator=g) / D ** 0.5
    scale = 0.02 + 0.02 * torch.rand(len(LAYERS), K, generator=g)
    return T.PCABasis(LAYERS, mean, comps, scale)


def tiny_mrh_cfg(**kw):
    from buc.models.mrh.neck import MRHConfig
    base = dict(tap_layers=LAYERS, tap_dim=K, neck_ch=16, rcu_per_fusion=1, head_ch=8, sem_dim=8,
                n_sigma_bands=4, t_embed_dim=16)
    base.update(kw)
    return MRHConfig(**base)


class SpyMRH(nn.Module):
    """MRH를 감싸 호출 인수(t, 탭, grad 모드)를 기록한다."""

    def __init__(self, mrh):
        super().__init__()
        self.mrh = mrh
        self.cfg = mrh.cfg
        self.calls = []

    def forward(self, taps, t):
        self.calls.append(SimpleNamespace(t=t, taps={l: v.detach().clone() for l, v in taps.items()},
                                          grad=torch.is_grad_enabled()))
        return self.mrh(taps, t)


def make_stack(tmp_path, *, ring=None, seed=0, token_offset=OFF, caption=None, table=None):
    """진짜 TapBus·TeacherSampler·load_teacher(가짜 부품) + 소형 MRH + MrhReader."""
    from buc.models.mrh.heads import MRH
    from buc.read.mrh_call import SURVEY_CAPTION, MrhReader
    dit = FakeDiT()
    teacher = make_teacher(tmp_path, dit)
    basis = make_basis()
    bus = T.TapBus(basis).attach(teacher.layers)
    sampler = TeacherSampler(teacher, table=table)
    mrh = MRH(tiny_mrh_cfg(), seed=1).eval()
    spy = SpyMRH(mrh)
    caption = SURVEY_CAPTION if caption is None else caption
    reader = MrhReader(sampler, bus, spy, token_offset=token_offset, seed=seed, ring=ring, caption=caption)
    return SimpleNamespace(caption=caption, dit=dit, teacher=teacher, basis=basis, bus=bus, sampler=sampler, mrh=mrh, spy=spy,
                           reader=reader)


def weights_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def oracle_taps(stack, image: torch.Tensor, sigma: float, key: str, seed: int = 0, token_offset: int = OFF):
    """계약서 수식으로 독립 재계산한 int8 탭 {층: [1,K,gh,gw]}와 z."""
    from buc.core import det
    from buc.teacher.sample import add_noise
    t = stack.teacher
    h, w = image.shape[-2:]
    x0 = t.encode_image(image)
    eps = t.make_noise(det.derive_seed(seed, key), 1, h, w)
    z = add_noise(x0, sigma, eps)
    cap = t.encode_text([stack.caption])
    gh, gw = h // 16, w // 16
    out = {}
    with torch.no_grad():
        for l in LAYERS:
            hid = stack.dit.trunk([z[0].unsqueeze(1)], cap, upto=l + 1)
            tok = hid[:, token_offset:token_offset + gh * gw]
            q = stack.basis.project(l, tok)                       # [1, n, K] int8
            out[l] = q.reshape(1, gh, gw, K).permute(0, 3, 1, 2).contiguous()
    return out, z

"""GOV-03 refl_bake_lint: 반사 영역을 입력 픽셀로 덮거나 굽지 않고, 잔차 흡수는 M_refl 을 뺀다(R-NR11)."""
import textwrap
from pathlib import Path

import pytest

from buc.gov import lints as G

LINT = "refl_bake_lint"
REPO = Path(__file__).resolve().parents[2]


def scan(src, path="buc/world/m.py"):
    vs = G.scan_refl_source(textwrap.dedent(src), path=path, module="buc.world.m")
    for v in vs:
        assert isinstance(v, G.Violation) and v.lint == LINT and v.path == path and v.line >= 1
    return vs


def got(src):
    return sorted({(v.rule, v.detail) for v in scan(src)})


def test_constants():
    assert G.REFL_DIRS == G.CANVAS_DIRS
    for s in ("M_refl", "m_refl", "mrefl", "m_refl_soft", "refl_mask", "reflection_mask", "mask_refl",
              "ReflMask", "mask_reflection"):
        assert G.REFL_RE.match(s), s
    for s in ("refl", "reflectance", "reflect_pad", "mask", "m_obs", "is_reflected", "prefl_mask"):
        assert not G.REFL_RE.match(s), s


BAD = [
    ("import numpy as np\ndef f(F0, I_w, M_refl):\n    return np.where(M_refl[..., None], I_w, F0)\n",
     [("REFL_INPUT", "where")]),
    ("import torch\ndef f(F0, read, w):\n    return torch.where(w.refl_mask, F0, read.image_lin)\n",
     [("REFL_INPUT", "where")]),
    ("def f(frame, read, m_refl):\n    frame[m_refl] = read.image_lin[m_refl]\n    return frame\n",
     [("REFL_INPUT", "assign")]),
    ("def f(tex, I_w, mask_refl):\n    tex[..., mask_refl] += 0.5 * I_w[..., mask_refl]\n",
     [("REFL_INPUT", "assign")]),
    ("def f(F, I_w, m_refl):\n    return F * (1 - m_refl) + I_w * m_refl\n", [("REFL_INPUT", "mult")]),
    ("def f(F, img, m_refl):\n    return F + m_refl[..., None] * img.pixels\n", [("REFL_INPUT", "mult")]),
    ("def f(F, I_w, m_refl):\n    return lerp(F, I_w, m_refl)\n", [("REFL_INPUT", "blend")]),
    ("def f(F, I_w, m):\n    return F.masked_scatter(m.refl_mask, I_w)\n", [("REFL_INPUT", "masked")]),
    ("import numpy as np\ndef f(F, I_w, m_refl):\n    np.copyto(F, I_w, where=m_refl)\n",
     [("REFL_INPUT", "masked")]),
    # 잔차 흡수: M_refl 파라미터가 없거나 쓰이지 않거나, 호출이 그것을 넘기지 않거나 None 으로 끈다
    ("def absorb_low_band_residual(world, r, eps):\n    return world\n",
     [("ABSORB_NO_REFL", "absorb_low_band_residual")]),
    ("def absorb_residual(world, r, m_refl=None):\n    world.L_d += r\n    return world\n",
     [("ABSORB_NO_REFL", "absorb_residual")]),
    ("def f(w, r):\n    return absorb_low_band_residual(w, r, eps=0.02)\n",
     [("ABSORB_NO_REFL", "absorb_low_band_residual")]),
    ("def f(w, r):\n    return w.absorb_residual(r, m_refl=None)\n", [("ABSORB_NO_REFL", "absorb_residual")]),
    ("def f(w, r):\n    return AbsorbLowBandResidual(w, r, refl_mask=False)\n",
     [("ABSORB_NO_REFL", "AbsorbLowBandResidual")]),
    ("def f(:\n", [("SYNTAX", None)]),
]

GOOD = [
    # 잔차에서 반사 영역을 빼는 것은 의무 규칙 그 자체다(1-m 은 '직접' 마스크가 아니다)
    "def f(I_w, F0, m_refl):\n    r = (I_w - F0) * (1 - m_refl)\n    return r\n",
    "def f(r, m_refl):\n    r[m_refl] = 0.0\n    return r\n",
    "def absorb_low_band_residual(world, r, *, m_refl, eps=0.02):\n    r = r * (1 - m_refl)\n    return world\n",
    "def AbsorbLowBandResidual(world, r, refl_mask):\n    keep = ~refl_mask\n    return world, keep\n",
    "def f(w, r, M_refl):\n    return absorb_low_band_residual(w, r, m_refl=M_refl)\n",
    "def f(w, r, M_refl):\n    return absorb_low_band_residual(w, r, M_refl)\n",
    "def f(w, r, s):\n    return w.absorb_residual(r, mask=s.refl_mask)\n",
    # 반사 관측(세계)로 반사 영역을 채우는 것은 정상
    "import numpy as np\ndef f(F0, refl_render, m_refl):\n    return np.where(m_refl, refl_render, F0)\n",
    "def f(frame, probe, m_refl):\n    frame[m_refl] = probe[m_refl]\n",
    # 물리 흡수(볼륨)는 잔차 흡수가 아니다
    "def absorb_coeff(sigma_a, d):\n    return sigma_a * d\n",
    "def f(x):\n    return absorb(x)\n",
    # 반사 마스크를 만드는 판독 코드는 입력을 읽지만 덮어쓰지 않는다
    "def f(I_w):\n    m_refl = detect(I_w) > 0.5\n    return m_refl\n",
]


@pytest.mark.parametrize("src,want", BAD)
def test_bad(src, want):
    g = got(src)
    if want[0][1] is None:
        assert [r for r, _ in g] == [want[0][0]]
    else:
        assert g == want, src


@pytest.mark.parametrize("src", GOOD)
def test_good(src):
    assert scan(src) == [], src


def test_lines():
    src = "def f(F, I_w, m_refl):\n    a = 1\n    return F * (1 - m_refl) + I_w * m_refl\n"
    assert [(v.line, v.rule) for v in scan(src)] == [(3, "REFL_INPUT")]
    src = "x = 0\n\n\ndef absorb_residual(w, r):\n    return w\n"
    assert [(v.line, v.rule) for v in scan(src)] == [(4, "ABSORB_NO_REFL")]


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def test_scan_paths_scope(tmp_path):
    _write(tmp_path, "buc/world/absorb.py", "def absorb_low_band_residual(world, r):\n    return world\n")
    _write(tmp_path, "buc/render/comp.py", "def f(F, I_w, m_refl):\n    return F + I_w * m_refl\n")
    _write(tmp_path, "buc/read/refl.py", "def f(F, I_w, m_refl):\n    return F + I_w * m_refl\n")
    vs = G.scan_paths(LINT, tmp_path)
    assert [(v.path, v.line, v.rule) for v in vs] == [("buc/render/comp.py", 2, "REFL_INPUT"),
                                                     ("buc/world/absorb.py", 1, "ABSORB_NO_REFL")]


def test_repo_refl_clean():
    vs = G.scan_paths(LINT, REPO)
    assert vs == [], "\n".join(f"{v.path}:{v.line} {v.rule} {v.detail}" for v in vs)

"""GOV-03 canvas_extension_lint: 입력 크롭·가장자리로 교사를 초기화하거나 캔버스를 넓히는 경로가 없다(R-NR1)."""
import textwrap
from pathlib import Path

import pytest

from buc.gov import lints as G

LINT = "canvas_extension_lint"
REPO = Path(__file__).resolve().parents[2]


def scan(src, path="buc/teacher/m.py"):
    vs = G.scan_canvas_source(textwrap.dedent(src), path=path, module="buc.teacher.m")
    for v in vs:
        assert isinstance(v, G.Violation) and v.lint == LINT and v.path == path and v.line >= 1
    return vs


def rules(src):
    return sorted({v.rule for v in scan(src)})


def test_constants():
    assert G.CANVAS_DIRS == ("buc/teacher", "buc/world", "buc/invent", "buc/render", "buc/lens", "buc/camera",
                             "buc/models")
    assert {"init", "init_image", "init_latent", "x0", "image", "latents"} <= G.INIT_KWARGS
    assert {"pad", "copyMakeBorder"} <= G.PAD_FUNCS
    for s in ("outpaint", "Outpaint_Border", "out_paint", "uncrop", "extend_canvas", "expand_canvas",
              "canvas_extension", "border_extend"):
        assert G.OUTPAINT_RE.search(s), s
    for s in ("paint", "extend", "canvas", "crop", "inpaint_mask_dilate", "expand_dims"):
        assert not G.OUTPAINT_RE.search(s), s


BAD = [
    # 아웃페인팅 API: 정의·호출·속성·import
    ("def outpaint_border(world, px):\n    return world\n", ["OUTPAINT_API"]),
    ("class CanvasExtension:\n    pass\n", ["OUTPAINT_API"]),
    ("y = teacher.outpaint(x, margin=64)\n", ["OUTPAINT_API"]),
    ("from diffusers import StableDiffusionOutpaintPipeline\n", ["OUTPAINT_API"]),
    ("z = uncrop(x)\n", ["OUTPAINT_API"]),
    ("def f(t, w):\n    return t.extend_canvas(w, 0.1)\n", ["OUTPAINT_API"]),
    # 교사 초기화에 입력 픽셀
    ("def f(teacher, read):\n    return teacher.sample(prompt, init=read.image_lin[:, -64:])\n", ["INPUT_INIT"]),
    ("def f(t, img):\n    return t.img2img(image=img.pixels, strength=0.3)\n", ["INPUT_INIT"]),
    ("def f(t, vae, image_lin):\n    return t.sample(x0=vae.encode(image_lin[0:32]))\n", ["INPUT_INIT"]),
    ("def f(s, I_w):\n    return s.run(init_latent=enc(crop(I_w, 0, 0, 64, 64)))\n", ["INPUT_INIT"]),
    ("def f(s, rp):\n    return s(latents=rp.input_rgb)\n", ["INPUT_INIT"]),
    # 입력 픽셀 패딩 = 캔버스 확장
    ("import numpy as np\ndef f(read):\n    return np.pad(read.image_lin, ((0, 0), (32, 32), (0, 0)))\n",
     ["PAD_INPUT"]),
    ("import torch.nn.functional as F\ndef f(x):\n    return F.pad(x.input_pixels, (8, 8, 8, 8), mode='reflect')\n",
     ["PAD_INPUT"]),
    ("import cv2\ndef f(input_image):\n    return cv2.copyMakeBorder(input_image, 4, 4, 4, 4, 0)\n", ["PAD_INPUT"]),
    ("from PIL import ImageOps\ndef f(img):\n    return ImageOps.expand(img.pixels, border=16)\n", ["PAD_INPUT"]),
    ("import numpy as np\ndef f(rp):\n    return np.pad(array=rp.I_w, pad_width=4)\n", ["PAD_INPUT"]),
    # 둘 다
    ("def outpaint(t, read):\n    return t.sample(init=read.image_lin)\n", ["INPUT_INIT", "OUTPAINT_API"]),
    ("def f(:\n", ["SYNTAX"]),
]

GOOD = [
    # Enc(R_v(W)) 초기화는 정상 경로다
    "def f(t, vae, world, view):\n    return t.sample(init=vae.encode(render(world, view)), sigma=0.6)\n",
    "def f(t, z_world):\n    return t.img2img(image=z_world, strength=0.3)\n",
    "import numpy as np\ndef f(depth):\n    return np.pad(depth, 1, mode='edge')\n",
    "import torch.nn.functional as F\ndef f(emb, d):\n    return F.pad(emb, (0, d - emb.shape[-1]))\n",
    # 관측항(ObsTerm)으로 입력을 쓰는 건 초기화가 아니다
    "def f(t, world, I_w):\n    return t.polish(world, obs=I_w, views=8)\n",
    "def inpaint_hole(world, mask):\n    return world\n",
    "x = arr.expand_dims(0)\ny = t.expand(3, 4)\n",
    "s = 'outpaint the border'\n# def outpaint(): pass\n",
]


@pytest.mark.parametrize("src,want", BAD)
def test_bad(src, want):
    assert rules(src) == want, src


@pytest.mark.parametrize("src", GOOD)
def test_good(src):
    assert scan(src) == [], src


def test_line_and_detail():
    src = "a = 1\n\ndef f(t, read):\n    b = 2\n    return t.sample(\n        init=read.image_lin)\n"
    vs = scan(src)
    assert len(vs) == 1 and vs[0].rule == "INPUT_INIT" and vs[0].detail == "init"
    assert vs[0].line == 5  # 호출이 시작하는 줄
    vs = scan("def go():\n    return outpaint_left(1)\n")
    assert [(v.line, v.detail) for v in vs] == [(2, "outpaint_left")]


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def test_scan_paths_scope(tmp_path):
    _write(tmp_path, "buc/teacher/a.py", "def outpaint(x):\n    return x\n")
    _write(tmp_path, "buc/world/b.py", "import numpy as np\ndef f(r):\n    return np.pad(r.image_lin, 2)\n")
    _write(tmp_path, "buc/models/c.py", "def f(t, r):\n    return t(image=r.pixels)\n")
    # 판독 단계(buc/read)는 필터링용 패딩을 할 수 있다: 기본 범위 밖
    _write(tmp_path, "buc/read/d.py", "import numpy as np\ndef f(image_lin):\n    return np.pad(image_lin, 2)\n")
    vs = G.scan_paths(LINT, tmp_path)
    assert [(v.path, v.rule) for v in vs] == [("buc/models/c.py", "INPUT_INIT"),
                                             ("buc/teacher/a.py", "OUTPAINT_API"),
                                             ("buc/world/b.py", "PAD_INPUT")]
    assert vs == sorted(vs)
    assert [v.path for v in G.scan_paths(LINT, tmp_path, dirs=("buc/read",))] == ["buc/read/d.py"]
    with pytest.raises(ValueError):
        G.scan_paths("no_such_lint", tmp_path)


def test_repo_canvas_clean():
    vs = G.scan_paths(LINT, REPO)
    assert vs == [], "\n".join(f"{v.path}:{v.line} {v.rule} {v.detail}" for v in vs)

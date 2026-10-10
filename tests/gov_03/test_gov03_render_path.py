"""GOV-03 no_input_pixels_in_render_path: 렌더 경로(Render/Lens/Camera)에 입력 픽셀이 닿는 코드가 없다(I6)."""
import textwrap
from pathlib import Path

import pytest

from buc.gov import lints as G

LINT = "no_input_pixels_in_render_path"
REPO = Path(__file__).resolve().parents[2]


def scan(src, path="buc/render/m.py", module="buc.render.m"):
    vs = G.scan_render_source(textwrap.dedent(src), path=path, module=module)
    for v in vs:
        assert isinstance(v, G.Violation)
        assert v.lint == LINT and v.path == path and v.line >= 1
    return vs


def rules(src, **kw):
    return sorted({v.rule for v in scan(src, **kw)})


def test_constants():
    assert G.LINTS == ("no_input_pixels_in_render_path", "canvas_extension_lint", "refl_bake_lint",
                       "preset_lineage_check")
    assert G.RENDER_PATH_DIRS == ("buc/render", "buc/lens", "buc/camera")
    assert G.INPUT_MODULES == ("buc.read",)
    assert G.INPUT_IDENTS == frozenset({"image_lin", "input_image", "input_pixels", "input_rgb", "input_buffer",
                                        "I_w"})
    assert G.INPUT_ATTRS == G.INPUT_IDENTS | {"pixels"}
    assert G.INPUT_TYPES == frozenset({"InputImage", "ReadPacket"})
    assert Path(G.REPO_ROOT).resolve() == REPO


BAD = [
    # 판독 모듈 import(절대·from·별칭·패키지 이름 가져오기)
    ("import buc.read.ingest\n", ["INPUT_IMPORT"]),
    ("from buc.read import ingest as ing\n", ["INPUT_IMPORT"]),
    ("from buc.read.ingest import ingest_bytes\n", ["INPUT_IMPORT"]),
    ("from buc import read\n", ["INPUT_IMPORT"]),
    ("import buc.read as R\n", ["INPUT_IMPORT"]),
    # 상대 import 는 module 기준으로 해석한다(buc.render.m → .. = buc)
    ("from ..read import ingest\n", ["INPUT_IMPORT"]),
    ("from .. import read\n", ["INPUT_IMPORT"]),
    ("from ..read.edge_unmix import edge_unmix\n", ["INPUT_IMPORT"]),
    # 동적 import: 리터럴이면 해석, 아니면 검증 불가(닫힌 쪽)
    ("import importlib\nm = importlib.import_module('buc.read.ingest')\n", ["INPUT_IMPORT"]),
    ("m = __import__('buc.read')\n", ["INPUT_IMPORT"]),
    ("import importlib\nname = 'x'\nm = importlib.import_module(name)\n", ["DYNAMIC_IMPORT"]),
    # 입력 버퍼 식별자: 이름·인자·키워드·속성
    ("def render(world, image_lin):\n    return world\n", ["INPUT_NAME"]),
    ("def f(read):\n    return read.image_lin * 0.5\n", ["INPUT_NAME"]),
    ("def f(img):\n    return img.pixels\n", ["INPUT_NAME"]),
    ("def f(w):\n    return g(w, input_rgb=w)\n", ["INPUT_NAME"]),
    ("def f(*, I_w=None):\n    return 0\n", ["INPUT_NAME"]),
    ("frame0 = input_image\n", ["INPUT_NAME"]),
    ("def f(ctx):\n    return ctx.input_buffer\n", ["INPUT_NAME"]),
    # 입력 계약 타입
    ("from buc.core.contracts import ReadPacket\n", ["INPUT_TYPE"]),
    ("from buc.core import contracts as C\ndef f(x):\n    return isinstance(x, C.InputImage)\n", ["INPUT_TYPE"]),
    ("def f(rp: 'x') -> None:\n    y = ReadPacket\n", ["INPUT_TYPE"]),
    # 이미지 파일 읽기
    ("from PIL import Image\nim = Image.open('in.png')\n", ["IMAGE_READ"]),
    ("import PIL.Image\nim = PIL.Image.open('in.png')\n", ["IMAGE_READ"]),
    ("import cv2\nim = cv2.imread('in.png')\n", ["IMAGE_READ"]),
    ("import imageio.v3 as iio\nim = iio.imread('a.png')\n", ["IMAGE_READ"]),
    ("from torchvision.io import read_image\nt = read_image('a.png')\n", ["IMAGE_READ"]),
    # 파싱 실패(닫힌 쪽)
    ("def f(:\n  pass\n", ["SYNTAX"]),
]

GOOD = [
    "from buc.core import color, geom\nfrom buc.core import contracts as C\n",
    "from buc.render.quads import render_ldi\n",
    "from . import quads\nfrom .quads import RenderOpts\n",
    "import importlib\nm = importlib.import_module('gsplat')\n",
    "def render(world, *, c2w, K, width, height):\n    return world.layers[0].rgba\n",
    # θ_in 은 렌더 함수의 정당한 인자다(F_0 = Render(W, C_0, θ_in))
    "def shoot(world, shot):\n    th = world.theta_in\n    return th\n",
    # 문자열·주석은 코드가 아니다
    "s = 'image_lin InputImage buc.read'\n# from buc.read import ingest\n",
    # 비슷하지만 다른 이름
    "rgb_lin = 1\nimage = 2\npixel = 3\nn_pixels = 4\nimage_linear = 5\n",
    "from PIL import Image\nImage.fromarray(x).save('out.png')\n",
    "import numpy as np\na = np.pad(x, 1, mode='edge')\n",
    "from buc.reader import x\nimport buc.readout\n",
]


@pytest.mark.parametrize("src,want", BAD)
def test_bad(src, want):
    got = rules(src)
    assert got == want, (src, got)


@pytest.mark.parametrize("src", GOOD)
def test_good(src):
    assert scan(src) == [], src


def test_line_numbers_and_detail_and_order():
    src = "x = 1\nfrom buc.read import ingest\n\n\ndef f(a):\n    return a.image_lin\n"
    vs = scan(src)
    assert [(v.line, v.rule) for v in vs] == [(2, "INPUT_IMPORT"), (6, "INPUT_NAME")]
    assert vs[0].detail.startswith("buc.read")
    assert vs[1].detail == "image_lin"
    assert vs == sorted(vs)
    assert scan(src) == vs  # 결정론


def test_relative_import_needs_module():
    # 같은 소스라도 module 이 다르면 해석 결과가 다르다: buc.render.sub.m 에서 '..read' = buc.render.read
    assert rules("from ..read import x\n", path="buc/render/sub/m.py", module="buc.render.sub.m") == []
    assert rules("from ...read import x\n", path="buc/render/sub/m.py", module="buc.render.sub.m") == \
        ["INPUT_IMPORT"]
    # __init__.py 는 그 자신이 패키지다: buc.render 패키지에서 '..read' = buc.read
    assert rules("from ..read import x\n", path="buc/render/__init__.py", module="buc.render") == \
        ["INPUT_IMPORT"]


def test_does_not_execute(tmp_path):
    marker = tmp_path / "ran"
    src = f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(3)\n"
    assert scan(src) == []
    assert not marker.exists()


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_transitive_import_chain(tmp_path):
    _write(tmp_path, "buc/__init__.py", "")
    _write(tmp_path, "buc/read/__init__.py", "")
    _write(tmp_path, "buc/read/ingest.py", "def ingest(b):\n    return b\n")
    _write(tmp_path, "buc/core/helper.py", "import os\n\n\ndef h():\n    from buc.read import ingest\n    return ingest\n")
    _write(tmp_path, "buc/core/clean.py", "import numpy\n")
    _write(tmp_path, "buc/render/a.py", "import math\nfrom buc.core import clean\nfrom buc.core.helper import h\n")
    _write(tmp_path, "buc/render/b.py", "from buc.core import clean\n")
    _write(tmp_path, "buc/lens/c.py", "from ..render import a\n")
    vs = G.check_render_path(tmp_path)
    assert all(v.lint == LINT for v in vs)
    got = [(v.path, v.line, v.rule, v.detail) for v in vs]
    assert ("buc/render/a.py", 3, "TRANSITIVE_IMPORT", "buc.render.a -> buc.core.helper -> buc.read.ingest") in got
    c = [g for g in got if g[0] == "buc/lens/c.py"]
    assert len(c) == 1 and c[0][1] == 1 and c[0][2] == "TRANSITIVE_IMPORT"
    assert c[0][3] == "buc.lens.c -> buc.render.a -> buc.core.helper -> buc.read.ingest"
    assert not any(g[0] == "buc/render/b.py" for g in got)
    # 사슬 없는 직접 import 는 TRANSITIVE 가 아니라 INPUT_IMPORT 로만
    _write(tmp_path, "buc/camera/d.py", "from buc.read import ingest\n")
    vs2 = G.check_render_path(tmp_path)
    d = [(v.rule, v.line) for v in vs2 if v.path == "buc/camera/d.py"]
    assert d == [("INPUT_IMPORT", 1)]
    assert vs2 == sorted(vs2) and G.check_render_path(tmp_path) == vs2


def test_render_dirs_only(tmp_path):
    # 판독·월드 단계는 입력을 만질 수 있다(솔브). 렌더 경로 밖은 검사하지 않는다.
    _write(tmp_path, "buc/read/x.py", "def f(image_lin):\n    return image_lin\n")
    _write(tmp_path, "buc/world/y.py", "from buc.read import x\n")
    _write(tmp_path, "buc/render/ok.py", "from buc.world import y as wy\n")
    vs = G.check_render_path(tmp_path)
    assert [(v.path, v.rule) for v in vs] == [("buc/render/ok.py", "TRANSITIVE_IMPORT")]
    assert vs[0].detail == "buc.render.ok -> buc.world.y -> buc.read.x"
    assert vs[0].line == 1
    # 렌더 경로 디렉터리를 바꿔 줄 수 있다
    assert G.check_render_path(tmp_path, dirs=("buc/read",))[0].rule == "INPUT_NAME"


def test_syntax_error_file_is_violation(tmp_path):
    _write(tmp_path, "buc/render/broken.py", "def f(:\n")
    vs = G.check_render_path(tmp_path)
    assert [(v.path, v.rule) for v in vs] == [("buc/render/broken.py", "SYNTAX")]


def test_repo_render_path_clean():
    vs = G.check_render_path(REPO)
    assert vs == [], "\n".join(f"{v.path}:{v.line} {v.rule} {v.detail}" for v in vs)


def test_repo_canary_injection(tmp_path):
    """실제 렌더 파일 사본에 바꿔치기 한 줄을 넣으면 잡혀야 한다(카나리아)."""
    src = (REPO / "buc/render/quads.py").read_text(encoding="utf-8")
    assert G.scan_render_source(src, path="buc/render/quads.py", module="buc.render.quads") == []
    bad = src + "\n\ndef _frame0(world, read):\n    return read.image_lin  # crossfade\n"
    vs = G.scan_render_source(bad, path="buc/render/quads.py", module="buc.render.quads")
    assert [v.rule for v in vs] == ["INPUT_NAME"]
    assert vs[0].line == bad.splitlines().index("    return read.image_lin  # crossfade") + 1

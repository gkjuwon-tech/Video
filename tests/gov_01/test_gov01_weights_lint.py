"""GOV-01 weights=None 린트(AST): 사전학습 가중치 기본값 함정 진리표."""
import textwrap

import pytest

from buc.gov import lineage as L


def rules(src):
    vs = L.scan_source(textwrap.dedent(src), path="m.py")
    for v in vs:
        assert isinstance(v, L.Violation) and v.path == "m.py" and v.line >= 1
    return [v.rule for v in vs]


BAD = [
    ("import torchvision\nm = torchvision.models.resnet50(weights='DEFAULT')", ["WEIGHTS"]),
    ("from torchvision.models import resnet50, ResNet50_Weights\n"
     "m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)", ["WEIGHTS"]),
    ("from torchvision.models.optical_flow import raft_large\nm = raft_large(pretrained=True)", ["PRETRAINED"]),
    ("import torchvision.models as tvm\nw = None\nm = tvm.hrnet(weights=w)", ["WEIGHTS"]),
    ("from torchvision import models\nm = models.segmentation.deeplabv3_resnet50(weights_backbone='X')",
     ["WEIGHTS"]),
    ("from torchvision import models as M\nm = M.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=1)",
     ["PRETRAINED"]),
    ("import timm\nm = timm.create_model('vit_base', pretrained=True)", ["PRETRAINED"]),
    ("import timm\nflag = False\nm = timm.create_model('vit_base', pretrained=flag)", ["PRETRAINED"]),
    ("import timm\nkw = {}\nm = timm.create_model('vit', **kw)", ["KWARGS"]),
    ("from transformers import AutoModel\nm = AutoModel.from_pretrained('bert')", ["FROM_PRETRAINED"]),
    ("class X:\n    pass\nm = X.from_pretrained('anything')", ["FROM_PRETRAINED"]),
    ("from mylib import from_pretrained\nm = from_pretrained('x')", ["FROM_PRETRAINED"]),
    ("import torch\nm = torch.hub.load('pytorch/vision', 'resnet50')", ["HUB"]),
    ("from torch.hub import load_state_dict_from_url\nsd = load_state_dict_from_url('http://x')", ["HUB"]),
    ("import torch\nsd = torch.utils.model_zoo.load_url('http://x')", ["HUB"]),
    ("from torch import hub as H\nm = H.load('a', 'b')", ["HUB"]),
    ("def f(:\n  pass", ["SYNTAX"]),
]

GOOD = [
    "import torchvision\nm = torchvision.models.resnet50(weights=None)",
    "from torchvision.models.optical_flow import raft_large\nm = raft_large(weights=None, progress=False)",
    "from torchvision.models import resnet50\nm = resnet50()",
    "import timm\nm = timm.create_model('vit_base', pretrained=False)",
    "import timm\nm = timm.create_model('vit_base', pretrained=None, num_classes=3)",
    # 감시 모듈 밖의 weights= 키워드는 우리 코드의 정상 인자다(FND-03 segment_reduce 등)
    "from buc.core.det import segment_reduce\nr = segment_reduce(v, s, 3, weights=w)",
    "def f(weights=None, pretrained=True):\n    return g(weights=weights, pretrained=pretrained)",
    "import torch\nsd = torch.load('x.pt', weights_only=True)",
    "x = {'pretrained': True}\n# torchvision.models.resnet50(weights='DEFAULT')\n",
]


@pytest.mark.parametrize("src,exp", BAD)
def test_bad_sources_flagged(src, exp):
    assert rules(src) == exp


@pytest.mark.parametrize("src", GOOD)
def test_good_sources_clean(src):
    assert rules(src) == []


def test_line_numbers_and_sorting():
    src = (
        "import timm\n"  # 1
        "import torchvision as tv\n"  # 2
        "a = tv.models.resnet18(weights=None)\n"  # 3
        "b = tv.models.resnet18(weights='DEFAULT', pretrained=True)\n"  # 4
        "c = timm.create_model('x', pretrained=True)\n"  # 5
        "d = Foo.from_pretrained('y')\n"  # 6
    )
    vs = L.scan_source(src, path="z.py")
    assert [(v.line, v.rule) for v in vs] == [(4, "PRETRAINED"), (4, "WEIGHTS"), (5, "PRETRAINED"),
                                              (6, "FROM_PRETRAINED")]


def test_scan_paths_recursive(tmp_path):
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "sub" / "b.py").write_text("import timm\nm = timm.create_model('v', pretrained=True)\n")
    (tmp_path / "pkg" / "sub" / "c.txt").write_text("timm.create_model('v', pretrained=True)\n")
    (tmp_path / "solo.py").write_text("m = M.from_pretrained('q')\n")
    vs = L.scan_paths([str(tmp_path / "pkg"), str(tmp_path / "solo.py")])
    got = [(v.path.replace("\\", "/").split("/")[-1], v.line, v.rule) for v in vs]
    assert got == [("b.py", 2, "PRETRAINED"), ("solo.py", 1, "FROM_PRETRAINED")]
    paths = [v.path for v in vs]
    assert paths == sorted(paths)

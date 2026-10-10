"""PRD-03: 모듈 API, 등록, 메타데이터, import 청결."""
import os
import subprocess
import sys

import pytest

from buc.core import contracts as C
from buc.pipeline import spine as S
from prd03_helpers import NAME_FOR, STUB_STAGES, impls


def test_modules_expose_name_version_fn():
    from buc.pipeline.stubs import move_fixed, read_classic, world_flat
    for mod, name, fn in ((read_classic, "classic_read", "classic_read"), (world_flat, "flat_world", "flat_world"),
                          (move_fixed, "fixed_move", "fixed_move")):
        assert mod.NAME == name and mod.VERSION == "0.1.0"
        assert callable(getattr(mod, fn))


def test_stub_impls_shape_and_order():
    d = impls()
    assert tuple(d) == STUB_STAGES
    for st, im in d.items():
        assert isinstance(im, S.Impl) and im.stage == st and im.kind == "stub"
        assert im.name == NAME_FOR[st] and im.version == "0.1.0"
        assert im.ref == f"stub:{NAME_FOR[st]}@0.1.0"
        assert st in C.STAGE_SIGNATURES


def test_register_stubs_and_resolve():
    from buc.pipeline.stubs import pack
    reg = S.Registry()
    assert pack.register_stubs(reg) is reg
    for st in STUB_STAGES:
        assert reg.get(st, "stub").name == NAME_FOR[st]
        assert reg.get(st, "real") is None
        assert reg.resolve(st, "real").kind == "stub"          # 실물이 없으면 스텁으로 해석
    with pytest.raises(S.SpineError):
        reg.resolve("S9-10")                                   # 렌더 스텁은 이 팩에 없다
    with pytest.raises(ValueError):
        pack.register_stubs(reg)                               # 중복
    assert pack.register_stubs(reg, replace=True) is reg


def test_identity_polish_returns_same_world():
    from buc.pipeline.stubs import pack
    from prd03_helpers import call, make_image
    read = call("S-1", {"image": make_image()})["read"]
    world = call("S2", {"read": read})["world"]
    out = call("S8.5", {"world": world, "read": read})
    assert set(out) == {"world"} and out["world"] is world
    assert pack.identity_polish is not None


def test_stub_sources_do_not_import_torch_or_scipy():
    import ast
    import glob
    import os

    import buc
    root = os.path.dirname(os.path.abspath(buc.__file__))
    files = glob.glob(os.path.join(root, "pipeline", "stubs", "*.py"))
    assert len(files) >= 4
    for f in files:
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert n.split(".")[0] not in ("torch", "scipy", "PIL"), f"{f}: {n}"


def test_import_has_no_global_rng_side_effects():
    code = ("import numpy as np, random\n"
            "s = np.random.get_state()[1].tobytes()\n"
            "r = random.getstate()\n"
            "import buc.pipeline.stubs.pack as p\n"
            "import buc.pipeline.stubs.read_classic, buc.pipeline.stubs.world_flat, buc.pipeline.stubs.move_fixed\n"
            "assert np.random.get_state()[1].tobytes() == s\n"
            "assert random.getstate() == r\n"
            "print('ok')\n")
    import os
    import buc
    root = os.path.dirname(os.path.dirname(os.path.abspath(buc.__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50, cwd=root)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]

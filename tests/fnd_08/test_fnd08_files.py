"""FND-08 파일·CLI: 커밋된 픽스처 = 재생성, manifest, 비트 무손실 왕복, --check 변조 탐지, torch 미로드."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import fnd08_support as S

NAMES = S.NAMES
ROOT = str(S.ROOT)


def _run(args, **kw):
    # PYTHONPATH 없이도 돌아야 한다: 도구가 저장소 루트를 스스로 sys.path에 넣는다
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(S.TOOL), *args], cwd=str(S.TOOL.parent), capture_output=True, text=True, timeout=50, env=env, **kw)


def _same_scene(a, b):
    assert a.name == b.name and len(a.views) == len(b.views)
    for va, vb in zip(a.views, b.views):
        for f in ("image_lin", "depth", "normals", "instance"):
            x, y = getattr(va, f), getattr(vb, f)
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f
        assert va.obs.content_hash() == vb.obs.content_hash()
    assert set(a.gt) == set(b.gt)
    for k in a.gt:
        x, y = a.gt[k], b.gt[k]
        assert type(x) is type(y), k
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), k
        else:
            assert x == y, k
    assert a.content_hash() == b.content_hash()


def test_committed_fixtures_exist_and_match_manifest_and_regeneration():
    t = S.tool()
    d = S.FIXDIR
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["version"] == "1" and set(man["scenes"]) == set(NAMES)
    total = 0
    for n in NAMES:
        ent = man["scenes"][n]
        f = d / ent["file"]
        assert f.name == f"{n}.npz" and f.is_file()
        total += f.stat().st_size
        assert isinstance(ent["hash"], str) and len(ent["hash"]) == 64
        loaded = t.load_scene(n)
        assert loaded.content_hash() == ent["hash"] == S.built()[n].content_hash()
        _same_scene(loaded, S.built()[n])
    assert 0 < total <= 6 * 1024 * 1024
    assert t.check_fixtures(d) is True


def test_write_fixtures_roundtrip_bit_exact(tmp_path):
    t = S.tool()
    man = t.write_fixtures(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == man
    assert man == json.loads((S.FIXDIR / "manifest.json").read_text(encoding="utf-8"))
    for n in NAMES:
        assert (tmp_path / f"{n}.npz").is_file()
        _same_scene(t.load_scene(n, tmp_path), S.built()[n])
    assert t.check_fixtures(tmp_path) is True


def test_check_detects_tampering(tmp_path):
    t = S.tool()
    d = tmp_path / "fx"
    shutil.copytree(S.FIXDIR, d)
    assert t.check_fixtures(d) is True
    # manifest 해시 변조
    mp = d / "manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    h = man["scenes"]["corridor"]["hash"]
    man["scenes"]["corridor"]["hash"] = ("0" if h[0] != "0" else "1") + h[1:]
    mp.write_text(json.dumps(man), encoding="utf-8")
    assert t.check_fixtures(d) is False
    # 파일 삭제
    d2 = tmp_path / "fx2"
    shutil.copytree(S.FIXDIR, d2)
    (d2 / "sphere_lit.npz").unlink()
    assert t.check_fixtures(d2) is False
    # 파일 내용 변조: 다른 장면의 파일로 바꿔치기
    d3 = tmp_path / "fx3"
    shutil.copytree(S.FIXDIR, d3)
    shutil.copyfile(d3 / "plane_tilted.npz", d3 / "plane_frontal.npz")
    assert t.check_fixtures(d3) is False
    # manifest 없음
    d4 = tmp_path / "fx4"
    d4.mkdir()
    assert t.check_fixtures(d4) is False


def test_cli_check_and_out(tmp_path):
    p = _run(["--check"])
    assert p.returncode == 0, p.stderr[-800:]
    d = tmp_path / "fx"
    shutil.copytree(S.FIXDIR, d)
    assert _run(["--check", "--out", str(d)]).returncode == 0
    (d / "box_pair.npz").unlink()
    q = _run(["--check", "--out", str(d)])
    assert q.returncode == 1 and "Traceback" not in q.stderr
    assert not (d / "box_pair.npz").exists()          # --check는 쓰지 않는다
    out = tmp_path / "gen"
    r = _run(["--out", str(out)])
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads((out / "manifest.json").read_text()) == json.loads((S.FIXDIR / "manifest.json").read_text())
    assert _run(["--check", "--out", str(out)]).returncode == 0


def test_tool_is_torch_free_and_side_effect_free():
    code = ("import sys, importlib.util, numpy as np; s0 = np.random.get_state()[1].copy();"
            f"spec = importlib.util.spec_from_file_location('mf', {str(S.TOOL)!r}); m = importlib.util.module_from_spec(spec);"
            "sys.modules['mf'] = m; spec.loader.exec_module(m);"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert (np.random.get_state()[1] == s0).all();"
            "m.build_scene('plane_frontal'); m.build_scene('distortion_dots');"
            "assert 'torch' not in sys.modules; assert (np.random.get_state()[1] == s0).all()")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stderr[-800:]


@pytest.mark.parametrize("name", ["plane_frontal", "color_chart"])
def test_load_scene_default_root_is_repo_fixtures(name):
    t = S.tool()
    a = t.load_scene(name)
    b = t.load_scene(name, S.FIXDIR)
    _same_scene(a, b)

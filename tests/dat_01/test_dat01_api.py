"""DAT-01 공개 상수·자료형·검증·import 위생·결정론 (계약서 DAT-01.md)."""
import ast
import dataclasses
import math
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from buc.data import synth_render as S
from buc.invent.grammar import core as G

import dat01_helpers as H

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------- 상수
def test_gbuffer_channel_layout_is_12():
    assert S.N_GBUF == 12
    assert len(S.GBUF_CHANNELS) == 12 and len(set(S.GBUF_CHANNELS)) == 12
    assert S.GBUF_CHANNELS == ("log_depth", "normal_x", "normal_y", "normal_z", "albedo_r", "albedo_g", "albedo_b",
                               "sem_structure", "sem_opening", "sem_furniture", "sem_accessory", "level_mask")
    assert S.SEM_GROUP_NAMES == ("structure", "opening", "furniture", "accessory")


def test_albedo_table_covers_materials_and_is_physical():
    for m in list(G.MATERIALS) + ["door", "window"]:
        a = np.asarray(S.ALBEDO[m], float)
        assert a.shape == (3,) and np.isfinite(a).all(), m
        assert (a >= 0.02).all() and (a <= 0.95).all(), (m, a)  # 비물리적 알베도(0 또는 >0.95) 금지


def test_default_material_by_class_is_total_and_valid():
    assert set(S.DEFAULT_MATERIAL_BY_CLASS) >= set(G.SLOT_CLASSES)
    for c in G.SLOT_CLASSES:
        assert S.DEFAULT_MATERIAL_BY_CLASS[c] in G.MATERIALS, c


def test_sem_group_mapping():
    assert S.sem_group(G.SEM_VOID) == -1
    for s in (G.SEM_FLOOR, G.SEM_WALL, G.SEM_CEILING):
        assert S.sem_group(s) == 0
    for s in (G.SEM_DOOR, G.SEM_WINDOW):
        assert S.sem_group(s) == 1
    for c in G.SLOT_CLASSES:
        want = 2 if c in G.FURNITURE_CLASSES else 3
        assert S.sem_group(G.sem_id(c)) == want, c
    for bad in (6, 7, G.SEM_SLOT_BASE + len(G.SLOT_CLASSES), -3):
        with pytest.raises(ValueError):
            S.sem_group(bad)


# --------------------------------------------------------------------------- 자료형 검증
def test_camera_validation():
    cam = H.cam_center()
    ok = dict(c2w=cam.c2w, K=cam.K, width=64, height=48)
    S.SynthCamera(**ok)
    bad_pose = cam.c2w.copy()
    bad_pose[:3, :3] *= 1.1
    bad_pose_nan = cam.c2w.copy()
    bad_pose_nan[0, 3] = np.nan
    badK = cam.K.copy()
    badK[0, 0] = -1.0
    for kw in (dict(c2w=bad_pose), dict(c2w=bad_pose_nan), dict(K=badK), dict(width=0), dict(height=-2),
               dict(width=1.5), dict(k1=float("nan")), dict(c2w=np.eye(3)), dict(K=np.eye(4))):
        with pytest.raises(ValueError):
            S.SynthCamera(**{**ok, **kw})


def test_opts_validation_and_defaults():
    o = S.SynthOpts()
    assert o.spp == 1 and o.ambient == 0.05 and o.exposure == 0.05 and tuple(o.bg) == (0.0, 0.0, 0.0)
    for spp in (1, 4, 9, 16):
        S.SynthOpts(spp=spp)
    for kw in (dict(spp=2), dict(spp=0), dict(spp=3), dict(spp=25), dict(ambient=-0.1), dict(exposure=float("nan")),
               dict(ambient=float("inf")), dict(bg=(0.0, 0.0)), dict(bg=(0.0, -1.0, 0.0)), dict(bg=(0, 0, np.nan))):
        with pytest.raises(ValueError):
            S.SynthOpts(**kw)
    assert dataclasses.is_dataclass(o)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.spp = 4


def test_render_gbuffer_argument_types():
    prog = H.box_program()
    cam = H.cam_center(16, 12)
    with pytest.raises(TypeError):
        S.render_gbuffer(prog, cam, None)  # 위치 인자 금지(키워드 전용)
    with pytest.raises(TypeError):
        S.render_gbuffer("room", cam)
    with pytest.raises(TypeError):
        S.render_gbuffer(prog, "cam")
    with pytest.raises(TypeError):
        S.render_gbuffer(prog, cam, opts={"spp": 1})
    with pytest.raises(TypeError):
        S.render_gbuffer(prog, cam, flow_to="cam")
    with pytest.raises(ValueError):  # 광류 목표 카메라의 영상 크기가 달라서
        S.render_gbuffer(prog, cam, flow_to=H.cam_center(20, 12))


# --------------------------------------------------------------------------- 결정론·부작용
def test_deterministic_bytes_and_no_global_rng_change():
    prog = H.furnished_program()
    cam = H.cam_center(32, 24)
    np.random.seed(1234)
    random.seed(99)
    st_np, st_py = np.random.get_state(), random.getstate()
    a = S.render_gbuffer(prog, cam, flow_to=H.shifted(cam, (0.05, 0, 0)), opts=S.SynthOpts(spp=4))
    b = S.render_gbuffer(prog, cam, flow_to=H.shifted(cam, (0.05, 0, 0)), opts=S.SynthOpts(spp=4))
    st_np2, st_py2 = np.random.get_state(), random.getstate()
    assert st_py == st_py2
    assert st_np[0] == st_np2[0] and np.array_equal(st_np[1], st_np2[1]) and st_np[2:] == st_np2[2:]
    for k, v in a.to_arrays().items():
        assert v.tobytes() == b.to_arrays()[k].tobytes(), k
    assert a.content_hash() == b.content_hash()
    assert a.meta == b.meta


def test_program_is_not_mutated():
    prog = H.furnished_program()
    h0 = prog.content_hash()
    S.render_gbuffer(prog, H.cam_center(24, 16))
    assert prog.content_hash() == h0


# --------------------------------------------------------------------------- import 위생 (디노이저·교사 금지)
FORBIDDEN_PREFIXES = ("oidn", "optix", "openimagedenoise", "diffusers", "transformers", "cv2", "skimage",
                      "scipy.ndimage", "scipy.signal", "buc.teacher", "buc.models", "buc.train")


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.append(n.module)
    return names


def test_source_imports_no_denoiser_or_teacher():
    path = os.path.join(ROOT, "buc", "data", "synth_render.py")
    for name in _imports(path):
        low = name.lower()
        assert not any(low == p or low.startswith(p + ".") for p in FORBIDDEN_PREFIXES), name


def test_import_does_not_load_heavy_or_gpu_stacks():
    code = ("import sys, buc.data.synth_render\n"
            "bad = [m for m in ('diffusers', 'transformers', 'cv2', 'oidn', 'skimage') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "import torch\n"
            "assert not torch.cuda.is_initialized()\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stderr[-800:]


def test_metadata_declares_denoiser_off():
    s = S.render_gbuffer(H.box_program(lights=H.two_lights()), H.cam_center(16, 12))
    assert s.meta["denoiser"] == "off"
    assert s.meta["renderer"] == "buc.data.synth_render.v1"
    assert s.meta["spp"] == 1
    assert math.isfinite(float(s.rgb.sum()))

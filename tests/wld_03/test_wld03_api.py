"""WLD-03 API·검증·지연 import·결정론·S2 이음매·meta."""
import dataclasses
import json
import os
import subprocess
import sys

import buc.world.depth_world as dw
import numpy as np
import pytest
from wld03_helpers import H, Matte, W, box_scene, make_read

from buc.core import contracts as C


def _read():
    img, d = box_scene()
    return make_read(img, d)


def test_constants_and_names():
    assert dw.NAME == "depth_world"
    assert dw.VERSION == "0.1.0"
    assert dw.N_LAYERS == 2
    assert issubclass(dw.DepthWorldError, C.ContractError)


def test_config_defaults():
    c = dw.DepthWorldConfig()
    assert c.rel_jump == pytest.approx(0.10)
    assert c.ext_px == pytest.approx(16.0)
    assert c.near_m == pytest.approx(1e-3)
    assert c.far_m == pytest.approx(1e4)
    assert c.tsq_stride == 2
    assert c.alpha_eps == pytest.approx(1e-3)
    assert c.opacity == pytest.approx(0.99)
    assert c.scale_mult == pytest.approx(0.5)
    assert c.edge_n == pytest.approx(2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.rel_jump = 0.2  # frozen


@pytest.mark.parametrize("kw", [
    {"rel_jump": 0.0}, {"rel_jump": float("nan")}, {"ext_px": -1.0}, {"ext_px": float("inf")},
    {"near_m": 0.0}, {"far_m": 1e-4}, {"far_m": float("inf")}, {"tsq_stride": 0}, {"tsq_stride": 1.5},
    {"tsq_stride": True}, {"alpha_eps": -0.1}, {"alpha_eps": 0.5}, {"opacity": 0.0}, {"opacity": 1.1},
    {"scale_mult": 0.0}, {"edge_n": 1.0}, {"edge_n": 65.0},
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        dw.DepthWorldConfig(**kw)


def test_input_type_errors():
    r = _read()
    with pytest.raises(TypeError):
        dw.build_depth_world({"depth": 1})
    with pytest.raises(TypeError):
        dw.build_depth_world(r, cfg={"rel_jump": 0.1})
    with pytest.raises(TypeError):
        dw.build_depth_world(r, far_field=np.ones((4, 4, 3)))


def test_matte_validation():
    r = _read()
    ok = Matte(np.zeros((H, W)), np.zeros((H, W, 3)), np.zeros((H, W, 3)), np.zeros((H, W), bool))

    class NoBand:
        alpha, fg, bg = ok.alpha, ok.fg, ok.bg

    with pytest.raises(TypeError):
        dw.build_depth_world(r, matte=NoBand())
    with pytest.raises(ValueError):
        dw.build_depth_world(r, matte=Matte(np.zeros((H, W - 1)), ok.fg, ok.bg, ok.band))
    with pytest.raises(ValueError):
        dw.build_depth_world(r, matte=Matte(np.full((H, W), 1.5), ok.fg, ok.bg, ok.band))
    with pytest.raises(ValueError):
        dw.build_depth_world(r, matte=Matte(ok.alpha, -np.ones((H, W, 3)), ok.bg, ok.band))
    bad = ok.bg.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        dw.build_depth_world(r, matte=Matte(ok.alpha, ok.fg, bad, ok.band))
    # 유효한 빈 매트는 통과
    dw.build_depth_world(r, matte=ok)


def test_import_is_lazy():
    code = ("import sys; import buc.world.depth_world as m; "
            "print('buc.render.quads' in sys.modules, 'buc.render.splat2d' in sys.modules, "
            "'buc.render.fit' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=root, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False False"


def test_result_shapes_and_world_identity():
    r = _read()
    res = dw.build_depth_world(r, cfg=dw.DepthWorldConfig(tsq_stride=4))
    assert isinstance(res.world, C.World)
    assert len(res.world.layers) == 2
    assert res.z.shape == (H, W) and res.z.dtype == np.float64
    for name in ("sky", "near", "far", "hidden", "matted"):
        a = getattr(res, name)
        assert a.shape == (H, W) and a.dtype == bool, name
    assert res.tear_x.shape == (H, W - 1) and res.tear_y.shape == (H - 1, W)
    assert res.tsq_layer.shape == (res.n_tsq,) and res.tsq_layer.dtype == np.int8
    assert res.strata.tsq.n == res.n_tsq
    assert res.world.content_hash() == res.strata.to_world().content_hash()
    assert res.world.meta["strata_root"] == res.strata.merkle_root().hex()
    res.strata.validate()
    assert res.strata.blocks == {} and res.strata.charts == {}


def test_s2_seam():
    r = _read()
    res = dw.build_depth_world(r)
    C.check_stage_io("S2", {"read": r}, {"world": res.world})
    w = res.world
    assert w.read_hash == r.content_hash()
    assert w.obs.content_hash() == r.obs.content_hash()
    assert w.theta_in.content_hash() == r.theta_in.content_hash()
    assert w.level == r.level
    assert w.prov.taint >= r.prov.taint


def test_far_field_passthrough():
    r = _read()
    ff = C.FarField(rgb=np.full((8, 8, 3), 0.25, np.float32))
    res = dw.build_depth_world(r, far_field=ff)
    assert res.world.far_field is not None
    assert np.array_equal(res.world.far_field.rgb, ff.rgb)
    assert res.meta()["far_field"] is True
    assert dw.build_depth_world(r).meta()["far_field"] is False


def test_determinism_and_input_immutability():
    r = _read()
    h0 = r.content_hash()
    img0, d0 = r.image_lin.copy(), r.depth.copy()
    a = dw.build_depth_world(r, cfg=dw.DepthWorldConfig(tsq_stride=1))
    b = dw.build_depth_world(r, cfg=dw.DepthWorldConfig(tsq_stride=1))
    assert a.strata.merkle_root() == b.strata.merkle_root()
    assert a.strata.to_bytes() == b.strata.to_bytes()
    assert r.content_hash() == h0
    assert np.array_equal(r.image_lin, img0) and np.array_equal(r.depth, d0)


def test_meta_json():
    r = _read()
    res = dw.build_depth_world(r)
    m = res.meta()
    s = json.dumps(m, allow_nan=False)
    assert isinstance(s, str)
    for k in ("name", "version", "cfg", "n_layers", "n_tsq", "n_tear_pairs", "hidden_frac", "matted_frac",
              "sky_frac", "matte", "far_field"):
        assert k in m, k
    assert m["name"] == "depth_world" and m["n_layers"] == 2 and m["n_tsq"] == res.n_tsq
    assert m["n_tear_pairs"] == int(res.tear_x.sum() + res.tear_y.sum())
    assert m["hidden_frac"] == pytest.approx(res.hidden.mean())
    assert m["matte"] is False
    assert isinstance(m["cfg"], dict) and m["cfg"]["rel_jump"] == pytest.approx(0.1)
    assert res.strata.meta["depth_world"]["n_tsq"] == res.n_tsq


def test_stage_fn_and_impl():
    from buc.pipeline import spine as S
    r = _read()
    impl = dw.make_world_impl()
    assert isinstance(impl, S.Impl)
    assert (impl.stage, impl.kind, impl.name, impl.version) == ("S2", "real", "depth_world", "0.1.0")
    ctx = S.StageContext(stage="S2", seed=1, params={"_seed": 5, "tsq_stride": 3, "ext_px": 8.0}, impl=impl)
    out = impl.fn({"read": r}, ctx)
    assert set(out) == {"world"}
    C.check_stage_io("S2", {"read": r}, out)
    ref = dw.build_depth_world(r, cfg=dw.DepthWorldConfig(tsq_stride=3, ext_px=8.0)).world
    assert out["world"].content_hash() == ref.content_hash()
    # _seed 는 결과에 영향 없음
    ctx2 = S.StageContext(stage="S2", seed=9, params={"_seed": 77, "tsq_stride": 3, "ext_px": 8.0}, impl=impl)
    assert impl.fn({"read": r}, ctx2)["world"].content_hash() == ref.content_hash()
    bad = S.StageContext(stage="S2", seed=1, params={"_seed": 5, "bogus": 1}, impl=impl)
    with pytest.raises(ValueError):
        impl.fn({"read": r}, bad)


def test_no_runtime_warnings():
    import warnings
    img, d = box_scene()
    d = d.copy()
    d[:5, :] = np.inf  # 하늘
    r = make_read(img, d)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = dw.build_depth_world(r, cfg=dw.DepthWorldConfig(tsq_stride=1))
    for ly in res.world.layers:
        assert np.isfinite(ly.depth).all() and np.isfinite(ly.rgba).all()
    for k, col in res.strata.tsq.columns.items():
        if col.dtype.kind == "f":
            assert np.isfinite(col).all(), k

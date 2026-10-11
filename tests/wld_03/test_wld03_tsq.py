"""WLD-03 TSQ 행(2DGS 초기화) 열 오라클 · 재투영 · RDR-01 surfel 이음매."""
import buc.world.depth_world as dw
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wld03_helpers import box_scene, halo_scene, make_read, obs_of

from buc.core import geom
from buc.world import strata as ST


def _rows_oracle(res, read, cfg):
    """계약서 6절을 그대로 다시 쓴 기대 행(층 0 → 층 1, 격자 행 우선)."""
    s = cfg.tsq_stride
    obs = read.obs
    K = obs.K
    fx, fy = K[0, 0], K[1, 1]
    H, W = read.depth.shape
    vv, uu = np.mgrid[0:H:s, 0:W:s]
    out = []
    for li, ly in enumerate(res.world.layers):
        a = ly.rgba[..., 3][vv, uu].astype(np.float64)
        ok = a > cfg.alpha_eps
        u = uu[ok].astype(np.float64)
        v = vv[ok].astype(np.float64)
        z = ly.depth[vv, uu][ok].astype(np.float64)
        out.append({"layer": np.full(u.size, li), "u": u, "v": v, "z": z, "a": a[ok], "fx": fx, "fy": fy})
    return out


def test_tsq_columns_oracle_with_distortion_and_pose():
    img, d = box_scene()
    R = Rotation.from_euler("xyz", [8.0, -12.0, 5.0], degrees=True).as_matrix()
    t = np.array([0.3, -0.2, 1.1])
    obs = obs_of(R=R, t=t, k1=0.04, k2=-0.01)
    conf = (np.random.default_rng(1).integers(0, 256, size=d.shape) / 255.0).astype(np.float32)  # rint 동률 회피
    read = make_read(img, d, obs=obs, depth_conf=conf)
    cfg = dw.DepthWorldConfig(tsq_stride=3, ext_px=6.0, scale_mult=0.7, opacity=0.9, edge_n=3.0)
    res = dw.build_depth_world(read, cfg=cfg)
    cols = res.strata.tsq.columns
    exp = _rows_oracle(res, read, cfg)
    n0, n1 = exp[0]["u"].size, exp[1]["u"].size
    assert n1 > 0
    assert res.n_tsq == n0 + n1 == res.strata.tsq.n
    assert np.array_equal(res.tsq_layer, np.r_[np.zeros(n0), np.ones(n1)].astype(np.int8))
    u = np.r_[exp[0]["u"], exp[1]["u"]]
    v = np.r_[exp[0]["v"], exp[1]["v"]]
    z = np.r_[exp[0]["z"], exp[1]["z"]]
    a = np.r_[exp[0]["a"], exp[1]["a"]]
    mu_exp = geom.unproject_world(obs.K, R, t, np.stack([u, v], 1), z, obs.k1, obs.k2)
    assert np.allclose(cols["mu"], mu_exp, rtol=1e-5, atol=1e-5)
    # 재투영: 관측 카메라로 찍으면 원래 화소·깊이로 돌아온다
    uv, zz = obs.project(cols["mu"].astype(np.float64))
    assert np.abs(uv - np.stack([u, v], 1)).max() < 1e-3
    assert np.allclose(zz, z, rtol=1e-5)
    n_exp = -R[2, :]
    assert np.allclose(cols["normal"], np.tile(n_exp, (res.n_tsq, 1)), atol=1e-6)
    fx, fy = obs.K[0, 0], obs.K[1, 1]
    s_exp = np.stack([0.7 * 3 * z / fx, 0.7 * 3 * z / fy], 1)
    assert np.allclose(cols["s_uv"], s_exp, rtol=1e-5)
    assert np.allclose(cols["opacity"], 0.9 * a, atol=1e-6)
    assert np.all(cols["edge_n"] == np.float32(3.0))
    lv = np.clip(np.rint(np.log2(8 * fx / (3 * z))), -8, 24).astype(np.int8)
    assert np.array_equal(cols["level"], lv)
    assert np.all(cols["sem"] == 0)
    own = cols["own"]
    assert np.all(own[:n0] == ST.OWNERS["INPUT"])
    assert np.all(own[n0:] == ST.OWNERS["ENGINE"])
    unc = cols["unc"]
    conf_s = conf.astype(np.float64)[v[:n0].astype(int), u[:n0].astype(int)]
    assert np.array_equal(unc[:n0], np.rint(255 * (1 - conf_s)).astype(np.uint8))
    assert np.all(unc[n0:] == 255)
    # 색: 층 0은 입력 그대로, 층 1은 숨은 층 색
    c0 = img.astype(np.float64)[v[:n0].astype(int), u[:n0].astype(int)]
    assert np.allclose(cols["color"][:n0], c0, atol=1e-6)
    l1 = res.world.layers[1].rgba[..., :3]
    c1 = l1[v[n0:].astype(int), u[n0:].astype(int)]
    assert np.allclose(cols["color"][n0:], c1, atol=1e-6)


def test_unc_default_without_depth_conf():
    img, d = box_scene()
    res = dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(tsq_stride=4))
    assert np.all(res.strata.tsq.columns["unc"] == 255)


def test_matted_rows_own_input_part():
    img, d, m = halo_scene()
    cfg = dw.DepthWorldConfig(tsq_stride=1, ext_px=16.0)
    res = dw.build_depth_world(make_read(img, d), cfg=cfg, matte=m)
    cols = res.strata.tsq.columns
    own = cols["own"]
    lay = res.tsq_layer
    # 행 → 화소 복원(재투영)
    uv, _ = res.world.obs.project(cols["mu"].astype(np.float64))
    pu = np.rint(uv[:, 0]).astype(int)
    pv = np.rint(uv[:, 1]).astype(int)
    matted = res.matted[pv, pu]
    hidden = res.hidden[pv, pu]
    assert np.all(own[matted] == ST.OWNERS["INPUT_PART"])
    assert np.all(own[(lay == 0) & ~matted] == ST.OWNERS["INPUT"])
    assert np.all(own[(lay == 1) & hidden] == ST.OWNERS["ENGINE"])
    # α ≤ eps 인 매팅 층 0 화소는 행이 없다
    a0 = res.world.layers[0].rgba[..., 3]
    n0_exp = int((a0 > cfg.alpha_eps).sum())
    assert int((lay == 0).sum()) == n0_exp
    assert int((lay == 1).sum()) == int((res.world.layers[1].rgba[..., 3] > cfg.alpha_eps).sum())


def test_stride_reduces_rows():
    img, d = box_scene()
    n1 = dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(tsq_stride=1, ext_px=0.0)).n_tsq
    n2 = dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(tsq_stride=2, ext_px=0.0)).n_tsq
    assert n1 == d.size
    assert n2 == ((d.shape[0] + 1) // 2) * ((d.shape[1] + 1) // 2)


def test_tsq_to_surfels_matches_init_from_rgbd():
    torch = pytest.importorskip("torch")
    from buc.render import fit, splat2d
    h, w = 16, 24
    rng = np.random.default_rng(0)
    img = rng.random((h, w, 3)).astype(np.float32)
    d = (2.0 + 0.1 * rng.random((h, w))).astype(np.float32)  # 인접 차 < 0.1 < 0.1·2 -> 찢김 없음
    obs = obs_of(w, h, f=30.0)
    read = make_read(img, d, obs=obs)
    res = dw.build_depth_world(read, cfg=dw.DepthWorldConfig(tsq_stride=1))
    assert not res.hidden.any() and res.n_tsq == h * w
    cols = res.strata.tsq.columns
    s = dw.tsq_to_surfels(cols, obs)
    assert isinstance(s, splat2d.Surfels) and s.dtype == torch.float32 and s.n == h * w
    c2w = obs.c2w()
    ref = fit.init_from_rgbd(img, d, K=obs.K, c2w=c2w, stride=1, opacity=0.99, scale_mult=0.5)
    assert torch.allclose(s.means, ref.means, atol=1e-5)
    assert torch.allclose(s.scales(), ref.scales(), rtol=1e-5)
    assert torch.allclose(s.opacities(), ref.opacities(), atol=1e-5)
    assert torch.allclose(s.colors, ref.colors, atol=1e-6)
    assert torch.allclose(s.normals(), torch.as_tensor(cols["normal"]), atol=1e-5)
    rot = s.rotations()
    assert torch.allclose(torch.linalg.det(rot), torch.ones(s.n), atol=1e-5)
    opts = splat2d.SplatOpts(backend="torch")
    a = splat2d.render(s, c2w=c2w, K=obs.K, width=w, height=h, opts=opts).numpy()
    b = splat2d.render(ref, c2w=c2w, K=obs.K, width=w, height=h, opts=opts).numpy()
    for k in ("rgb", "alpha", "depth"):
        assert np.allclose(a[k], b[k], atol=1e-4), k
    s64 = dw.tsq_to_surfels(cols, obs, dtype=torch.float64)
    assert s64.dtype == torch.float64

"""RD-14 이음매(RD-01 r_eff, RD-08 σ_s·미터 깊이, RD-04 MTF)와 import 위생, 공개 이름."""
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import rd14_helpers as H

om = H.om
ROOT = str(H.ROOT)


def _env():
    return dict(os.environ, PYTHONPATH=ROOT)


def test_public_names_present():
    names = [
        "ObsMapError", "NoiseModel", "ObsOpts", "ObsMap", "ObsBlock", "ObsSet",
        "texel_pitch_m", "cell_size_m", "block_extent_m", "footprint_m", "jpeg_band_factor", "effective_footprint",
        "levels_from_footprint", "owner_of_level", "defocus_mtf", "motion_mtf", "w_of_nu", "nu_obs_cutoff",
        "mrh_nu_cap", "build_obs_map", "build_obs_set",
        "LEVEL_RANGE", "H0_M", "BLOCK_CELLS", "OCT_HI", "OCT_CENTER", "NU_GRID_N", "W_TRUST", "NO_LEVEL",
        "OWN_INPUT", "OWN_INPUT_PART", "OWN_INPUT_REFL",
    ]
    missing = [n for n in names if not hasattr(om, n)]
    assert not missing, missing
    assert set(names) <= set(om.__all__)
    assert issubclass(om.ObsMapError, ValueError)


def test_import_has_no_heavy_deps_and_no_side_effects():
    code = (
        "import sys, numpy as np\n"
        "np.random.seed(7); a = np.random.get_state()[1].copy()\n"
        "import buc.read.obs_map\n"
        "b = np.random.get_state()[1]\n"
        "bad = [m for m in ('torch', 'cv2', 'skimage', 'yaml', 'diffusers') if m in sys.modules]\n"
        "print(bad, (a == b).all()); sys.exit(0 if (not bad and (a == b).all()) else 1)"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr


def test_build_runs_without_torch():
    code = (
        "import sys, numpy as np\n"
        "from buc.read import obs_map as om\n"
        "from buc.core import geom\n"
        "K = geom.intrinsics_from_fov(60.0, 64, 48)\n"
        "m = om.build_obs_map(np.full((48, 64), 3.0), K, coc_px=2.0)\n"
        "s = om.build_obs_set(m, np.full((48, 64), 3.0), K, np.eye(3), np.zeros(3))\n"
        "print(m.obs_levels(), s.n_cells(), 'torch' in sys.modules); sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr


# ------------------------------------------------------------------ RD-01: r_eff.ratio 가 곧 r_ratio
def _field(h, w, seed):
    rng = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    nu = np.hypot(fx, fy)
    nu[0, 0] = 1.0
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) / nu
    spec[0, 0] = 0.0
    x = np.fft.ifft2(spec).real
    x = (x - x.mean()) / x.std() * 0.15 + 0.5
    return x


def test_rd01_reff_ratio_feeds_r_ratio_and_cancels_pure_upsampling():
    ing = pytest.importorskip("buc.read.ingest")
    base = (np.clip(_field(256, 256, 1), 0, 1) * 255).astype(np.uint8)
    base = np.stack([base] * 3, axis=-1)
    up = np.asarray(Image.fromarray(base).resize((512, 512), Image.BILINEAR))
    r0, r1 = ing.ingest(base), ing.ingest(up)
    q0, q1 = r0.r_eff / r0.r_in, r1.r_eff / r1.r_in              # RD-01: r_eff = ratio · r_in (짧은 변 기준)
    assert q0 >= 0.85 and 0.3 <= q1 <= 0.7 and 1 / 16 <= q1 <= 1
    z = 3.0
    maps = {}
    for tag, ratio, side in (("orig", q0, 256), ("up", q1, 512)):
        K = H.K_from_fov(side, side, 60.0)
        maps[tag] = om.build_obs_map(np.full((side, side), z), K, r_ratio=min(1.0, ratio))
    m0, m1 = maps["orig"], maps["up"]
    assert m1.r_ratio == pytest.approx(q1) and m0.r_ratio == pytest.approx(min(1.0, q0))
    # 2배 업샘플은 정보가 같다: φ_eff 가 거의 같고(소유 대역이 한 옥타브 이상 늘지 않는다)
    ratio = float(m1.phi_eff[0, 0] / m0.phi_eff[0, 0])
    assert 0.7 <= ratio <= 1.5, ratio
    assert abs(int(m1.l_hard[0, 0]) - int(m0.l_hard[0, 0])) <= 1
    # r_ratio 를 무시했다면 업샘플 입력이 한 단 더 높은 레벨을 주장했을 것이다
    naive = om.build_obs_map(np.full((512, 512), z), H.K_from_fov(512, 512, 60.0))
    assert int(naive.l_hard[0, 0]) >= int(m1.l_hard[0, 0]) and naive.phi_eff[0, 0] < m1.phi_eff[0, 0]


# ------------------------------------------------------------------ RD-08: 미터 깊이와 σ_s
def test_rd08_metric_scale_feeds_depth_and_sigma():
    ms = pytest.importorskip("buc.read.metric_scale")

    class NoTable:
        def resolve(self, text):
            raise KeyError(text)

    anchor = ms.anchor_subject_distance(3.0, 1.0, sigma_log=0.2)
    scale = ms.estimate_metric_scale([], table=NoTable(), anchors=[anchor])
    assert scale.s == pytest.approx(3.0) and scale.sigma_s == pytest.approx(0.2)
    depth_rel = np.full((48, 64), 1.0)
    K = H.K_from_fov(64, 48, 60.0)
    m = om.build_obs_map(scale.to_meters(depth_rel), K, sigma_log_s=scale.sigma_s,
                         opts=om.ObsOpts(scale_margin_k=2.0))
    ref = om.build_obs_map(np.full((48, 64), 3.0), K)
    assert m.sigma_log_s == pytest.approx(0.2)
    # 여유 e^{2·0.2} = 1.49 배 -> 레벨은 ref 와 같거나 (log2 1.49 = 0.58 옥타브) 1 단 낮다
    d = ref.l_hard.astype(int) - m.l_hard.astype(int)
    assert ((d == 0) | (d == 1)).all() and np.allclose(m.phi_in, ref.phi_in * np.exp(0.4), rtol=1e-5)
    assert m.to_meta()["l_sigma_octaves"] == pytest.approx(0.2 / np.log(2.0))


# ------------------------------------------------------------------ RD-04: 같은 MTF, 같은 옥타브
def test_rd04_mtf_and_octaves_agree():
    ea = pytest.importorskip("buc.read.explain_away")
    nu = np.linspace(0.0, 0.5, 201)
    for c in (0.0, 1.5, 4.0):
        assert np.allclose(om.defocus_mtf(nu, c), ea.defocus_mtf(nu, c), atol=1e-12)
    for L in (0.0, 2.0, 5.0):
        assert np.allclose(om.motion_mtf(nu, L), ea.motion_mtf(nu, L), atol=1e-12)
    assert tuple(om.OCT_HI) == tuple(ea.OCTAVE_HI)

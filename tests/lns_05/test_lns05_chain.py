"""LNS-05 이음매: OpticsSpec, θ -> 스펙 사상, apply_optics 체인 순서, Frames 어댑터, 격리·성능."""
import dataclasses
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.lens import optics
from buc.lens import theta as T
from lns05_helpers import blob, hdr_scene

REPO = Path(__file__).resolve().parents[2]
G = optics.GhostSpec


def full_spec(**kw):
    base = dict(k1=0.05, k2=0.0, ca_red_px=2.0, ca_blue_px=-1.5, catseye=0.4, blades=6, blade_rot_deg=10.0,
                glare=0.05, ghosts=(G(-0.5, 0.06, 0.05, (1.0, 0.8, 0.6)), G(1.5, 0.08, 0.03)), ghost_threshold=1.0,
                star_amount=0.1, star_size_frac=0.5, star_threshold=1.0)
    base.update(kw)
    return optics.OpticsSpec(**base)


def manual(x, s):
    """계약서 체인: (글레어 + 고스트 + 별; 셋 다 원본 x 에서) -> 횡색수차 -> 왜곡."""
    out = optics.veiling_glare(x, s.glare, r0_frac=s.glare_r0_frac)
    out = out + optics.ghost_layer(x, s.ghosts, threshold=s.ghost_threshold, blades=s.blades,
                                   blade_rot_deg=s.blade_rot_deg)
    out = out + s.star_amount * optics.star_layer(x, blades=s.blades, rotation_deg=s.blade_rot_deg,
                                                  threshold=s.star_threshold, size_frac=s.star_size_frac)
    out = optics.apply_lateral_ca(out, s.ca_red_px, s.ca_blue_px)
    return optics.apply_distortion(out, s.k1, s.k2)


# ----------------------------------------------------------------------------------------------- OpticsSpec
def test_spec_fields_defaults_and_validation():
    s = optics.OpticsSpec()
    assert [f.name for f in dataclasses.fields(s)] == [
        "k1", "k2", "ca_red_px", "ca_blue_px", "catseye", "blades", "blade_rot_deg", "glare", "glare_r0_frac",
        "ghosts", "ghost_threshold", "star_amount", "star_size_frac", "star_threshold"]
    assert (s.k1, s.k2, s.ca_red_px, s.ca_blue_px, s.catseye, s.blades, s.blade_rot_deg, s.glare) == \
        (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    assert s.glare_r0_frac == optics.GLARE_R0_FRAC and s.ghosts == () and s.ghost_threshold == 0.8
    assert s.star_amount == 0.0 and s.star_size_frac == 0.25 and s.star_threshold == 0.8
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.k1 = 0.1
    with pytest.raises(TypeError):
        optics.OpticsSpec(0.1)  # 키워드 전용
    assert isinstance(optics.OpticsSpec(ghosts=[G(0.5, 0.05, 0.1)]).ghosts, tuple)  # 시퀀스는 tuple 로 정규화
    for bad in (dict(k1=float("nan")), dict(k1=-0.4), dict(k1=True), dict(k2=3.0), dict(ca_red_px=float("inf")),
                dict(catseye=1.1), dict(catseye=-0.1), dict(blades=2), dict(blades=19), dict(blades=6.0),
                dict(blade_rot_deg=float("nan")), dict(glare=1.1), dict(glare=-0.01), dict(glare_r0_frac=0.0),
                dict(glare_r0_frac=1.5), dict(ghosts=(1, 2)), dict(ghost_threshold=-1.0),
                dict(star_amount=1.1), dict(star_size_frac=0.0), dict(star_size_frac=1.2), dict(star_threshold=-0.5)):
        with pytest.raises(ValueError):
            optics.OpticsSpec(**bad)
    assert optics.OpticsSpec(glare=1.0, catseye=1.0, star_amount=1.0, blades=18, k1=-0.3) is not None


# ----------------------------------------------------------------------------------------------- apply_optics
def test_default_spec_is_copy_and_only_catseye_is_not_consumed():
    for dt in (np.float32, np.float64):
        x = hdr_scene().astype(dt)
        keep = x.copy()
        y = optics.apply_optics(x, optics.OpticsSpec())
        assert y is not x and y.dtype == dt and np.array_equal(y, x) and np.array_equal(x, keep)
    x = hdr_scene()
    # catseye 는 apply_catseye 호출자(LNS-06/08)가 쓰는 값이다. apply_optics 는 소비하지 않는다.
    assert np.array_equal(optics.apply_optics(x, optics.OpticsSpec(catseye=1.0, blades=6)), x)


def test_apply_optics_single_stages_equal_public_pieces():
    x = hdr_scene()
    s = optics.OpticsSpec(k1=0.06, k2=-0.01)
    assert np.array_equal(optics.apply_optics(x, s), optics.apply_distortion(x, 0.06, -0.01))
    s = optics.OpticsSpec(ca_red_px=2.0, ca_blue_px=-3.0)
    assert np.array_equal(optics.apply_optics(x, s), optics.apply_lateral_ca(x, 2.0, -3.0))
    s = optics.OpticsSpec(glare=0.1, glare_r0_frac=0.05)
    assert np.allclose(optics.apply_optics(x, s), optics.veiling_glare(x, 0.1, r0_frac=0.05), atol=1e-6)
    gs = (G(-0.5, 0.06, 0.05), G(1.4, 0.05, 0.02, (0.5, 1.0, 1.0)))
    s = optics.OpticsSpec(ghosts=gs, ghost_threshold=1.0, blades=7, blade_rot_deg=5.0)
    ref = x + optics.ghost_layer(x, gs, threshold=1.0, blades=7, blade_rot_deg=5.0)
    assert np.allclose(optics.apply_optics(x, s), ref, atol=2e-5 * float(x.max()))
    for blades in (0, 6):
        s = optics.OpticsSpec(star_amount=0.2, blades=blades, star_size_frac=0.4, star_threshold=1.5)
        ref = x + 0.2 * optics.star_layer(x, blades=blades, threshold=1.5, size_frac=0.4)
        assert np.allclose(optics.apply_optics(x, s), ref, atol=2e-5 * float(x.max()))


def test_apply_optics_chain_order_matches_contract():
    x = hdr_scene()
    s = full_spec()
    keep = x.copy()
    out = optics.apply_optics(x, s)
    ref = manual(x, s)
    assert np.array_equal(x, keep) and out.dtype == np.float32 and out.shape == x.shape
    assert np.abs(out - ref).max() < 2e-5 * float(x.max())
    assert np.all(np.isfinite(out)) and out.min() >= -1e-5
    # 순서에 민감한 시험인지 확인: 왜곡을 먼저 하면 결과가 눈에 띄게 달라진다
    wrong = optics.apply_distortion(x, s.k1, s.k2)
    wrong = manual(wrong, dataclasses.replace(s, k1=0.0, k2=0.0, ca_red_px=0.0, ca_blue_px=0.0))
    assert np.abs(out - wrong).max() > 5e-3 * float(x.max())
    # 광속: 광원에서 떨어진 고스트도 왜곡/색수차 뒤의 최종 영상에 남아 있다
    assert float(out.sum()) > float(x.sum()) * 1.001


def test_apply_optics_deterministic_and_validation():
    x = hdr_scene(48, 64)
    s = full_spec()
    a, b = optics.apply_optics(x, s), optics.apply_optics(x, s)
    assert np.array_equal(a, b)
    with pytest.raises(TypeError):
        optics.apply_optics(x, {"k1": 0.1})
    with pytest.raises(TypeError):
        optics.apply_optics((x * 10).astype(np.uint8), s)
    xn = x.copy()
    xn[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        optics.apply_optics(xn, s)
    with pytest.raises(ValueError):
        optics.apply_optics(x[..., :2], s)
    with pytest.raises(ValueError):
        optics.apply_optics(x[0], s)


def test_ghost_lands_at_distorted_position_end_to_end():
    """광원 하나 -> 고스트가 (이상 위치) 의 왜곡 상(像)에 나타난다. 기하 사슬 전체의 위치 검증."""
    h, w = 121, 161
    cx, cy = (w - 1) / 2, (h - 1) / 2
    x = np.zeros((h, w, 3), np.float32)
    x[int(cy - 10), int(cx + 30), :] = 20.0  # 광원
    s = optics.OpticsSpec(k1=0.1, ghosts=(G(-1.0, 0.08, 0.5),), ghost_threshold=1.0)
    out = optics.apply_optics(x, s)
    ideal_ghost = np.array([cx - 30.0, cy + 10.0])  # c + (-1) (p - c)
    want = optics.ideal_to_distorted(ideal_ghost, 0.1, 0.0, (h, w))
    sub = out[..., 1].astype(np.float64).copy()
    sub[: int(cy)] = 0  # 광원이 있는 위쪽 절반은 제외
    ys, xs = np.mgrid[0:h, 0:w]
    gx, gy = float((sub * xs).sum() / sub.sum()), float((sub * ys).sum() / sub.sum())
    assert np.hypot(gx - want[0], gy - want[1]) < 1.0, ((gx, gy), want)


# ----------------------------------------------------------------------------------------------- θ -> 스펙
def mk(lens_rx="spherical_clean", **kw):
    extra = dict(T.EXTRA_DEFAULTS)
    extra.update(kw.pop("extra", {}))
    return C.Theta(lens_rx=lens_rx, extra=extra, **kw)


def gain(spec):
    return sum(g.gain * max(g.tint) for g in spec.ghosts)


def test_spec_from_theta_lens_ordering_and_validity():
    specs = {n: optics.spec_from_theta(mk(n), 640) for n in T.LENS_RX_NAMES}
    for n, s in specs.items():
        assert isinstance(s, optics.OpticsSpec) and len(s.ghosts) >= 3, n
        assert optics.spec_from_theta(mk(n), 640) == s  # 결정론
        y = optics.apply_optics(hdr_scene(40, 56), s)  # 모든 처방이 실제로 적용 가능
        assert np.all(np.isfinite(y))
    clean = specs["spherical_clean"]
    assert clean.k1 == 0.0 and clean.k2 == 0.0 and clean.ca_red_px == 0.0 and clean.ca_blue_px == 0.0
    assert clean.catseye <= 0.1 and clean.glare <= 0.03
    pz = specs["vintage_petzval"]
    assert pz.catseye >= 0.5 and abs(pz.ca_red_px) > 0 and pz.ca_red_px * pz.ca_blue_px <= 0
    assert (pz.k1, pz.k2) != (0.0, 0.0)
    sc = specs["soft_single_coat"]
    assert sc.glare >= 3 * clean.glare and sc.glare >= 0.05
    assert gain(sc) >= 2 * gain(clean)
    for n in T.LENS_RX_NAMES:
        assert gain(specs[n]) >= gain(clean) - 1e-12 and specs[n].catseye >= clean.catseye


def test_spec_from_theta_scaling_blades_and_fstop_star():
    t = mk("vintage_double_gauss")
    a, b = optics.spec_from_theta(t, 320), optics.spec_from_theta(t, 640)
    assert b.ca_red_px == pytest.approx(2 * a.ca_red_px, rel=1e-9) and b.ca_blue_px == pytest.approx(2 * a.ca_blue_px, rel=1e-9)
    assert (a.k1, a.k2, a.glare, a.catseye) == (b.k1, b.k2, b.glare, b.catseye)  # 해상도 불변량
    assert optics.spec_from_theta(mk(extra={"blades": 7}), 320).blades == 7
    assert optics.spec_from_theta(C.Theta(), 320).blades == 0  # extra 에 없으면 LNS-02 와 같이 0
    stars = [optics.spec_from_theta(mk(f_stop=f, extra={"blades": 6}), 320).star_amount
             for f in (1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0)]
    assert stars[0] == 0.0 and stars[2] == 0.0 and stars[-1] > 0.0
    assert all(b >= a for a, b in zip(stars, stars[1:])) and max(stars) <= 1.0
    for bad in (0, -3, 1.5, True, "320"):
        with pytest.raises(ValueError):
            optics.spec_from_theta(mk(), bad)
    with pytest.raises(ValueError):
        optics.spec_from_theta(C.Theta(lens_rx="no_such_lens"), 320)
    with pytest.raises(TypeError):
        optics.spec_from_theta({"lens_rx": "spherical_clean"}, 320)


# ----------------------------------------------------------------------------------------------- Frames
def make_frames(n=2, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return C.Frames(rgb=rng.random((n, h, w, 3)).astype(np.float32), fps=24.0,
                    aov={"depth": (1.0 + 5 * rng.random((n, h, w))).astype(np.float32),
                         "coc_px": rng.random((n, h, w)).astype(np.float32),
                         "meta": rng.random((n, 5)).astype(np.float32)},
                    shot_hash="shot-abc", world_hash="world-xyz")


def test_apply_optics_frames_matches_per_frame_chain():
    fr = make_frames()
    keep = fr.rgb.copy()
    th = mk("vintage_petzval", f_stop=11.0, extra={"blades": 6})
    out = optics.apply_optics_frames(fr, th)
    assert isinstance(out, C.Frames) and out.rgb.shape == fr.rgb.shape and out.rgb.dtype == np.float32
    assert out.fps == fr.fps and out.shot_hash == fr.shot_hash and out.world_hash == fr.world_hash
    assert np.array_equal(fr.rgb, keep)
    assert out.rgb.min() >= 0 and out.rgb.max() <= 1
    spec = optics.spec_from_theta(th, fr.rgb.shape[2])
    assert spec.k1 != 0.0
    for i in range(2):
        ref = color.linear_to_display(optics.apply_optics(color.srgb_to_linear(fr.rgb[i]), spec))
        assert np.abs(out.rgb[i] - ref).max() < 1e-5
        for k in ("depth", "coc_px"):  # (N,H,W) 맵은 같은 왜곡으로 최근접 사상
            want = optics.apply_distortion(fr.aov[k][i], spec.k1, spec.k2, nearest=True)
            assert np.array_equal(out.aov[k][i], want)
    assert set(out.aov) == set(fr.aov) and np.array_equal(out.aov["meta"], fr.aov["meta"])
    # 사용자 지정 스펙: 항등이면 표시값 왕복
    ident = optics.apply_optics_frames(fr, th, spec=optics.OpticsSpec())
    assert np.abs(ident.rgb - fr.rgb).max() < 2e-5
    assert all(np.array_equal(ident.aov[k], fr.aov[k]) for k in fr.aov)
    with pytest.raises(TypeError):
        optics.apply_optics_frames(fr.rgb, th)
    with pytest.raises(TypeError):
        optics.apply_optics_frames(fr, th, optics.OpticsSpec())  # spec 은 키워드 전용


# ----------------------------------------------------------------------------------------------- 격리·성능
def test_import_isolation_global_rng_and_warnings():
    code = ("import sys, numpy as np\nnp.random.seed(5)\ns0 = np.random.get_state()[1].copy()\n"
            "import buc.lens.optics as op\nfrom buc.core.contracts import Theta\n"
            "assert 'torch' not in sys.modules, 'torch 금지'\n"
            "assert (np.random.get_state()[1] == s0).all(), 'import 가 전역 RNG 를 건드렸다'\n"
            "x = np.zeros((48, 64, 3), np.float32); x[20, 30] = 30.0; x += 0.05\n"
            "s = op.spec_from_theta(Theta(lens_rx='vintage_petzval', f_stop=11.0, extra={'blades': 6}), 64)\n"
            "y = op.apply_optics(x, s)\nassert np.isfinite(y).all()\n"
            "op.apply_catseye(x, np.full((48, 64), 8.0), strength=0.8, blades=6)\n"
            "assert (np.random.get_state()[1] == s0).all(), '전역 RNG 사용 금지'\n")
    r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


def test_no_runtime_warnings_in_process():
    x = hdr_scene(40, 56)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        optics.apply_optics(x, full_spec())
        optics.apply_catseye(x, np.full((40, 56), 6.0), strength=1.0, blades=5)
        optics.undistort_image(x, 0.1, -0.02)
        optics.star_psf(7, 3.0)


def test_full_stack_speed_cpu():
    h, w = 180, 320
    x = 0.04 + 0.05 * np.random.default_rng(0).random((h, w, 3)).astype(np.float32)
    for (bx, by, a) in ((60, 50, 30.0), (250, 40, 20.0), (170, 140, 12.0)):
        x += (blob(h, w, bx, by, 2.0, a)[..., None] * np.array([1.0, 0.95, 0.85])).astype(np.float32)
    th = mk("soft_single_coat", f_stop=16.0, extra={"blades": 8})
    s = optics.spec_from_theta(th, w)
    optics.apply_optics(x[:48, :64].copy(), optics.spec_from_theta(th, 64))  # 캐시 예열 허용
    t0 = time.perf_counter()
    y = optics.apply_optics(x, s)
    dt = time.perf_counter() - t0
    assert dt < 6.0, f"{dt:.1f}s"
    assert y.shape == x.shape and np.all(np.isfinite(y))
    # 큰 CoC cat-eye 한 프레임 (96x160, CoC 20 px) 도 6 초 안
    xs = x[::2, ::2].copy()
    t0 = time.perf_counter()
    optics.apply_catseye(xs, np.full(xs.shape[:2], 20.0), strength=0.8)
    assert time.perf_counter() - t0 < 6.0

"""LNS-06 이음매: RX_TABLE · prescription_from_theta · apply_prescription 체인 · Frames 어댑터 · 격리·성능."""
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
from buc.lens import prescriptions as P
from buc.lens import theta as T
from lns06_helpers import hdr_scene, hf_energy, impulse, noise_img, smooth_img

REPO = Path(__file__).resolve().parents[2]


def th(rx, **kw):
    extra = dict(kw.pop("extra", {}))
    if rx == "anamorphic_2x":
        extra.setdefault("squeeze", 2.0)
        kw.setdefault("aspect", 2.39)
    return C.Theta(lens_rx=rx, extra=extra, **kw)


def rxs(width=320, **kw):
    return {name: P.prescription_from_theta(th(name, **kw), width) for name in T.LENS_RX_NAMES}


# ----------------------------------------------------------------------------------------------- 표 · θ 사상
def test_rx_table_covers_every_registered_lens():
    assert set(P.RX_TABLE) == set(T.LENS_RX_NAMES)
    for name, row in P.RX_TABLE.items():
        assert {"swirl", "edge_blur", "streak"} <= set(row), name
        P.Prescription(optics=optics.OpticsSpec(), **row)          # 행이 곧 Prescription 키워드


def test_prescription_from_theta_wraps_lns05_spec():
    for name, rx in rxs().items():
        assert isinstance(rx, P.Prescription) and rx.name == name
        want = optics.spec_from_theta(th(name), 320)
        assert rx.optics == want                                    # 왜곡·색수차·글레어·고스트·별은 LNS-05 그대로
        row = P.RX_TABLE[name]
        for k in ("swirl", "edge_blur", "streak"):
            assert getattr(rx, k) == row[k]
        assert rx.squeeze == (2.0 if name == "anamorphic_2x" else 1.0)
    # extra.blades, f_stop 이 optics 로 흘러간다
    rx = P.prescription_from_theta(th("vintage_petzval", f_stop=11.0, extra={"blades": 6}), 320)
    assert rx.blades == 6 and rx.optics.star_amount > 0.0
    # 해상도 불변: 비율은 같고 (swirl, edge_blur, streak, len_frac, squeeze), 수차 px 는 폭에 선형
    a, b = rxs(160)["vintage_petzval"], rxs(640)["vintage_petzval"]
    for k in ("swirl", "edge_blur", "streak", "streak_len_frac", "streak_tint", "squeeze"):
        assert getattr(a, k) == getattr(b, k)
    assert b.optics.ca_red_px == pytest.approx(4.0 * a.optics.ca_red_px, rel=1e-9)


def test_prescription_ordering_constraints():
    r = rxs()
    clean, pet, dg = r["spherical_clean"], r["vintage_petzval"], r["vintage_double_gauss"]
    ana, soft = r["anamorphic_2x"], r["soft_single_coat"]
    assert clean.swirl == 0.0 and clean.edge_blur <= 0.1 and clean.streak == 0.0 and clean.squeeze == 1.0
    assert pet.swirl >= 0.5 and pet.edge_blur >= 0.5 and pet.optics.catseye >= 0.5
    assert 0.1 <= dg.swirl <= 0.4 and pet.swirl >= 2.0 * dg.swirl
    assert clean.edge_blur < dg.edge_blur < pet.edge_blur
    assert ana.squeeze == 2.0 and ana.streak >= 0.3 and ana.swirl <= 0.1 and ana.streak_tint[2] > ana.streak_tint[0]
    assert soft.edge_blur >= 0.2 and soft.optics.glare >= 0.05 and soft.streak == 0.0
    for name, rx in r.items():
        if name != "anamorphic_2x":
            assert rx.squeeze == 1.0 and rx.streak == 0.0, name
        assert 0.0 <= rx.swirl <= 1.0 and 0.0 <= rx.edge_blur <= 1.0 and 0.0 <= rx.streak <= 1.0, name
        assert rx.swirl >= clean.swirl and rx.edge_blur >= clean.edge_blur, name


def test_prescription_from_theta_errors():
    with pytest.raises(TypeError):
        P.prescription_from_theta({"lens_rx": "spherical_clean"}, 320)
    for w in (0, -3, 1.5, True, "320"):
        with pytest.raises(ValueError):
            P.prescription_from_theta(th("spherical_clean"), w)
    with pytest.raises(ValueError):
        P.prescription_from_theta(C.Theta(lens_rx="not_a_lens"), 320)
    # squeeze 와 lens_rx 의 불일치는 거부 (LNS-01 규칙), anamorphic 의 squeeze 누락은 2.0 으로 본다
    with pytest.raises(ValueError):
        P.prescription_from_theta(C.Theta(lens_rx="spherical_clean", extra={"squeeze": 2.0}), 320)
    with pytest.raises(ValueError):
        P.prescription_from_theta(C.Theta(lens_rx="anamorphic_2x", aspect=2.39, extra={"squeeze": 1.0}), 320)
    assert P.prescription_from_theta(C.Theta(lens_rx="anamorphic_2x", aspect=2.39), 320).squeeze == 2.0


# ----------------------------------------------------------------------------------------------- 체인
@pytest.mark.parametrize("name", list(T.LENS_RX_NAMES))
def test_apply_prescription_equals_documented_composition(name):
    x = hdr_scene(72, 96)
    keep = x.copy()
    rx = P.prescription_from_theta(th(name, f_stop=11.0, extra={"blades": 6}), 96)
    out = P.apply_prescription(x, rx)
    want = optics.apply_optics(P.edge_softness(x, rx.edge_blur), rx.optics)
    if rx.streak > 0.0:
        want = want + P.streak_layer(x, gain=rx.streak, length_frac=rx.streak_len_frac, tint=rx.streak_tint,
                                     threshold=rx.streak_threshold)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert np.array_equal(x, keep)
    assert np.abs(out - want).max() <= 1e-5 * max(1.0, float(np.abs(want).max()))
    assert np.isfinite(out).all() and out.min() >= -1e-4


def test_apply_prescription_neutral_and_single_stage():
    x = hdr_scene(48, 64)
    neutral = P.apply_prescription(x, P.Prescription())
    assert np.array_equal(neutral, x) and neutral is not x
    streak_only = P.apply_prescription(x, P.Prescription(streak=0.6, streak_len_frac=0.2))
    want = x + P.streak_layer(x, gain=0.6, length_frac=0.2, tint=P.DEFAULT_STREAK_TINT, threshold=0.8)
    assert np.abs(streak_only - want).max() < 1e-5 * float(want.max())
    edge_only = P.apply_prescription(noise_img(48, 64), P.Prescription(edge_blur=0.8))
    assert np.array_equal(edge_only, P.edge_softness(noise_img(48, 64), 0.8))
    xd = x.astype(np.float64)
    assert P.apply_prescription(xd, P.Prescription(streak=0.5)).dtype == np.float64


def test_vintage_edges_are_softer_than_clean():
    x = noise_img(96, 160, seed=9)                                   # 임계 아래 -> 고스트·스트릭 없음
    clean = P.apply_prescription(x, rxs(160)["spherical_clean"])
    pet = P.apply_prescription(x, rxs(160)["vintage_petzval"])
    ratio = lambda y: hf_energy(y, 0.85, 1.01) / hf_energy(y, 0.0, 0.3)
    assert ratio(pet) < 0.6 * ratio(clean)
    assert ratio(clean) > 0.5


def test_anamorphic_light_gets_horizontal_streak():
    h, w = 64, 97
    x = impulse(h, w, 48, 32, 20.0, base=0.02)
    out = P.apply_prescription(x, rxs(w)["anamorphic_2x"]).astype(np.float64)
    horiz = out[32, 48 + 30, :] - 0.02          # 광원에서 가로 30 px
    vert = out[32 - 30, 48, :] - 0.02           # 세로 30 px
    assert (horiz > 3.0 * vert).all() and (horiz > 0).all()
    assert horiz[2] / horiz[0] > 1.2             # 푸른 기운 (tint B > R)


# ----------------------------------------------------------------------------------------------- Frames
def make_frames(n=2, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return C.Frames(rgb=rng.random((n, h, w, 3)).astype(np.float32), fps=24.0,
                    aov={"depth": (1.0 + 5 * rng.random((n, h, w))).astype(np.float32),
                         "meta": rng.random((n, 5)).astype(np.float32)},
                    shot_hash="shot-abc", world_hash="world-xyz")


def test_apply_prescription_frames_matches_per_frame_chain():
    fr = make_frames()
    keep = fr.rgb.copy()
    t = th("anamorphic_2x", f_stop=11.0, extra={"blades": 6})
    out = P.apply_prescription_frames(fr, t)
    rx = P.prescription_from_theta(t, fr.rgb.shape[2])
    assert isinstance(out, C.Frames) and out.rgb.shape == fr.rgb.shape and out.rgb.dtype == np.float32
    assert out.fps == fr.fps and out.shot_hash == fr.shot_hash and out.world_hash == fr.world_hash
    assert np.array_equal(fr.rgb, keep) and out.rgb.min() >= 0 and out.rgb.max() <= 1
    assert rx.optics.k1 != 0.0
    for i in range(2):
        ref = color.linear_to_display(P.apply_prescription(color.srgb_to_linear(fr.rgb[i]), rx))
        assert np.abs(out.rgb[i] - ref).max() < 1e-5
        want = optics.apply_distortion(fr.aov["depth"][i], rx.optics.k1, rx.optics.k2, nearest=True)
        assert np.array_equal(out.aov["depth"][i], want)             # (N,H,W) AOV 는 같은 왜곡의 최근접 사상
    assert np.array_equal(out.aov["meta"], fr.aov["meta"])
    # 사용자 지정 rx: 항등이면 표시값 왕복
    ident = P.apply_prescription_frames(fr, t, rx=P.Prescription())
    assert np.abs(ident.rgb - fr.rgb).max() < 2e-5
    assert np.array_equal(ident.aov["depth"], fr.aov["depth"])
    with pytest.raises(TypeError):
        P.apply_prescription_frames(fr.rgb, t)
    with pytest.raises(TypeError):
        P.apply_prescription_frames(fr, t, rx=optics.OpticsSpec())
    with pytest.raises(TypeError):
        P.apply_prescription_frames(fr, t, P.Prescription())        # rx 는 키워드 전용


def test_apply_prescription_type_errors():
    x = hdr_scene(24, 32)
    with pytest.raises(TypeError):
        P.apply_prescription(x, optics.OpticsSpec())
    with pytest.raises(TypeError):
        P.apply_prescription((x * 255).astype(np.uint8), P.Prescription())
    with pytest.raises(ValueError):
        P.apply_prescription(x[..., 0], P.Prescription())


# ----------------------------------------------------------------------------------------------- 격리 · 성능
def test_import_isolation_global_rng_and_warnings():
    code = ("import sys, numpy as np\nnp.random.seed(5)\ns0 = np.random.get_state()[1].copy()\n"
            "import buc.lens.prescriptions as pr\nfrom buc.core.contracts import Theta\n"
            "assert 'torch' not in sys.modules, 'torch 금지'\n"
            "assert (np.random.get_state()[1] == s0).all(), 'import 가 전역 RNG 를 건드렸다'\n"
            "x = np.zeros((48, 64, 3), np.float32); x[20, 30] = 30.0; x += 0.05\n"
            "rx = pr.prescription_from_theta(Theta(lens_rx='anamorphic_2x', aspect=2.39, f_stop=11.0, "
            "extra={'blades': 6, 'squeeze': 2.0}), 64)\n"
            "y = pr.apply_prescription(x, rx)\nassert np.isfinite(y).all()\n"
            "pr.apply_prescription_bokeh(x, np.full((48, 64), 6.0), rx)\n"
            "assert 'torch' not in sys.modules\n"
            "assert (np.random.get_state()[1] == s0).all(), '전역 RNG 사용 금지'\n")
    r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


def test_no_runtime_warnings_in_process():
    x = hdr_scene(40, 56)
    rx = P.Prescription(optics=optics.OpticsSpec(catseye=1.0, blades=5), squeeze=3.0, swirl=1.0, edge_blur=1.0,
                        streak=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        P.apply_prescription(x, rx)
        P.apply_prescription_bokeh(x, np.full((40, 56), 6.0), rx)
        P.exit_pupil_kernel(9.0, 1.0, 45.0, rx)
        P.streak_layer(x, gain=1.0)
        P.edge_softness(x, 1.0)


def test_stack_speed_cpu():
    x = hdr_scene(180, 320)
    rx = rxs(320, f_stop=8.0, extra={"blades": 8})["anamorphic_2x"]
    t0 = time.perf_counter()
    P.apply_prescription(x, rx)
    t_chain = time.perf_counter() - t0
    ramp = np.tile(np.linspace(0.0, 10.0, 96), (72, 1)).astype(np.float32)
    t0 = time.perf_counter()
    P.apply_prescription_bokeh(hdr_scene(72, 96), ramp, rx)
    t_bokeh = time.perf_counter() - t0
    assert t_chain < 12.0, t_chain
    assert t_bokeh < 12.0, t_bokeh

"""LNS-02 Frames 이음매: RDR-02 render_shot 출력(표시값 rgb + depth AOV) 뒤에 붙는 apply_dof_frames."""
import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.lens import dof
from lns02_helpers import coc_ref, tex

Hh, Ww = 48, 64


def _shot(focus, theta=None):
    n = len(focus)
    th = theta or C.Theta(focal_mm=85.0, f_stop=1.4, sensor_width_mm=36.0, focus_m=float(focus[0]))
    K = np.array([[Ww * 85.0 / 36.0, 0, (Ww - 1) / 2], [0, Ww * 85.0 / 36.0, (Hh - 1) / 2], [0, 0, 1]])
    return C.Shot(c2w=np.repeat(np.eye(4)[None], n, 0), t=np.arange(n) / 24.0, K=np.repeat(K[None], n, 0),
                  focus_m=np.asarray(focus, np.float64), width=Ww, height=Hh, fps=24.0, theta=th)


def _frames(n):
    lin = tex(Ww, Hh)
    disp = color.linear_to_display(lin)
    depth = np.full((Hh, Ww), 6.0, np.float32)
    depth[:, :Ww // 2] = 1.0
    alpha = np.ones((Hh, Ww), np.float32)
    return C.Frames(rgb=np.repeat(disp[None], n, 0), fps=24.0,
                    aov={"depth": np.repeat(depth[None], n, 0), "alpha": np.repeat(alpha[None], n, 0),
                         "layer_id": np.zeros((n, Hh, Ww), np.int16)},
                    shot_hash="s" * 8, world_hash="w" * 8), lin, depth


def test_frames_seam():
    fr, lin, depth = _frames(3)
    shot = _shot([1.0, 2.5, 6.0])
    out = dof.apply_dof_frames(fr, shot)
    assert isinstance(out, C.Frames)
    out.validate()
    assert out.rgb.shape == fr.rgb.shape and out.rgb.dtype == np.float32
    assert out.fps == fr.fps and out.shot_hash == fr.shot_hash and out.world_hash == fr.world_hash
    assert set(out.aov) >= {"depth", "alpha", "layer_id", "coc_px"}
    np.testing.assert_array_equal(out.aov["alpha"], fr.aov["alpha"])
    np.testing.assert_array_equal(out.aov["layer_id"], fr.aov["layer_id"])
    assert out.aov["coc_px"].shape == (3, Hh, Ww) and out.aov["coc_px"].dtype == np.float32
    for i, f in enumerate([1.0, 2.5, 6.0]):
        np.testing.assert_allclose(out.aov["coc_px"][i], np.minimum(coc_ref(depth, f, width_px=Ww), 64.0),
                                   rtol=1e-4, atol=1e-4)
        ref = dof.apply_dof(color.srgb_to_linear(fr.rgb[i]), depth, focal_mm=85.0, f_stop=1.4,
                            sensor_width_mm=36.0, focus_m=f, opts=dof.DofOpts(blades=0))
        np.testing.assert_allclose(out.rgb[i], color.linear_to_display(ref.rgb), atol=1e-6)
    # 프레임 0: 왼쪽(1 m) 초점 -> 왼쪽 내부는 원본(표시값 왕복 오차 이내), 프레임 2: 오른쪽 초점
    np.testing.assert_allclose(out.rgb[0][:, :Ww // 2 - 2], fr.rgb[0][:, :Ww // 2 - 2], atol=2 / 255)
    mg = Ww // 2 + int(np.ceil(float(coc_ref(1.0, 6.0, width_px=Ww)) / 2)) + 2  # 녹은 전경 번짐 폭 밖
    np.testing.assert_allclose(out.rgb[2][:, mg:], fr.rgb[2][:, mg:], atol=2 / 255)
    assert np.abs(out.rgb[0][:, Ww // 2 + 12:] - fr.rgb[0][:, Ww // 2 + 12:]).mean() > 0.02
    # 입력 Frames 불변
    fr2, _, _ = _frames(3)
    np.testing.assert_array_equal(fr.rgb, fr2.rgb)


def test_frames_blades_from_theta_extra():
    fr, _, depth = _frames(1)
    th = C.Theta(focal_mm=85.0, f_stop=1.4, sensor_width_mm=36.0, focus_m=1.0, extra={"blades": 6})
    out = dof.apply_dof_frames(fr, _shot([1.0], th))
    ref = dof.apply_dof(color.srgb_to_linear(fr.rgb[0]), depth, focal_mm=85.0, f_stop=1.4,
                        sensor_width_mm=36.0, focus_m=1.0, opts=dof.DofOpts(blades=6))
    np.testing.assert_allclose(out.rgb[0], color.linear_to_display(ref.rgb), atol=1e-6)


def test_frames_breathing_reference_is_first_frame():
    fr, _, _ = _frames(2)
    shot = _shot([6.0, 1.0])
    o = dof.DofOpts(breathing_beta=0.5)
    out = dof.apply_dof_frames(fr, shot, o)
    plain = dof.apply_dof_frames(fr, shot, dof.DofOpts())
    np.testing.assert_allclose(out.rgb[0], plain.rgb[0], atol=1e-6)  # 프레임 0 은 기준 배율 1
    np.testing.assert_array_equal(out.aov["depth"][0], fr.aov["depth"][0])
    s = dof.breathing_scale(85.0, 1.0, 0.5, ref_focus_m=6.0)
    assert s > 1.01
    # 프레임 1 의 깊이 AOV 는 주점 기준 s 배 확대(최근접): 깊이 경계 x 위치가 바깥으로 이동
    cx = (Ww - 1) / 2
    edge0 = Ww // 2 - 0.5
    d1 = out.aov["depth"][1]
    row = d1[Hh // 2]
    edge1 = np.where(row > 3.0)[0].min() - 0.5
    assert edge1 == pytest.approx(cx + s * (edge0 - cx), abs=1.0)
    assert not np.allclose(out.rgb[1], plain.rgb[1], atol=1e-3)


def test_frames_validation():
    fr, _, _ = _frames(2)
    with pytest.raises(ValueError):
        dof.apply_dof_frames(fr, _shot([1.0, 2.0, 3.0]))  # 프레임 수 불일치
    nodepth = C.Frames(rgb=fr.rgb, fps=24.0, aov={"alpha": fr.aov["alpha"]})
    with pytest.raises(ValueError):
        dof.apply_dof_frames(nodepth, _shot([1.0, 2.0]))
    small = C.Frames(rgb=fr.rgb[:, :-1], fps=24.0, aov={"depth": fr.aov["depth"][:, :-1]})
    with pytest.raises(ValueError):
        dof.apply_dof_frames(small, _shot([1.0, 2.0]))

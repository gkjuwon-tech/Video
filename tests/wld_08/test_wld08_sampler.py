"""WLD-08 ↔ 샘플러 규약: lockstep 1회 호출, σ·steps·cond·seeds, 입력 영상 = 렌더, 16 비배수 크기, 오류."""
import numpy as np
import pytest

from buc.core import color
from buc.render.quads import render_ldi
from buc.teacher.sample import SigmaTable

import wld08_helpers as h

S = h.mod()


def _fill(sam, dw=None, **kw):
    dw = dw if dw is not None else h.build()[1]
    return dw, S.sdedit_avg_fill(dw, sam, **kw)


def test_single_lockstep_call_convention():
    sam = h.FakeSampler(h.const_transform([0.5] * 4))
    dw, res = _fill(sam, seed=7)
    assert len(sam.calls) == 1, "SDEdit 는 뷰 전체를 묶어 정확히 1회"
    c = sam.calls[0]
    assert c.x0_shape == (4, 16, h.H // 8, h.W // 8)
    assert c.sigma == 0.5 and c.seeds == 7 and c.steps == 2   # 기본 표: σ=0.5 는 6번 점, 끝까지 2 step
    assert list(c.cond) == [S.SDEditAvgConfig().prompt]
    assert res.steps == 2 and res.sigma_snapped == pytest.approx(0.5)
    assert len(sam.teacher.encoded) == 2, "렌더 인코딩 1회 + 교사 출력 인코딩(가짜) 1회"
    assert sam.teacher.encoded[0].shape == (4, 3, h.H, h.W)


def test_prompt_and_sigma_are_passed_through_and_table_drives_steps():
    table = SigmaTable((1.0, 0.7, 0.45, 0.3, 0.1))
    sam = h.FakeSampler(h.const_transform([0.5] * 2), table=table)
    cfg = S.SDEditAvgConfig(sigma=0.5, n_per_side=1, prompt="hello")
    _, res = _fill(sam, cfg=cfg, seed=0)
    c = sam.calls[0]
    assert list(c.cond) == ["hello"] and c.sigma == 0.5
    assert table.snap(0.5).index == 2 and c.steps == 3     # 표 끝까지 = len(table) - index
    assert res.steps == 3 and res.sigma_snapped == pytest.approx(0.45)
    assert res.meta()["sigma_snapped"] == pytest.approx(0.45) and res.meta()["steps"] == 3


def test_teacher_input_is_the_srgb_render_of_each_view_without_black_holes():
    sam = h.FakeSampler(h.const_transform([0.5] * 4))
    dw, res = _fill(sam)
    x = sam.teacher.encoded[0].numpy().astype(np.float64)
    assert np.isfinite(x).all() and x.min() >= -1.0 and x.max() <= 1.0
    for i, v in enumerate(res.views):
        r = render_ldi(dw.world, c2w=v.c2w, K=v.K, width=v.width, height=v.height, k1=v.k1, k2=v.k2)
        exp = 2.0 * color.linear_to_srgb(np.clip(r.rgb.astype(np.float64), 0, 1)) - 1.0
        got = np.moveaxis(x[i], 0, -1)
        solid = r.alpha >= 0.99
        assert solid.sum() > 0.5 * solid.size
        assert np.abs(got[solid] - exp[solid]).max() <= 2e-5
        # 구멍(화면 밖 등)도 순수 검정(-1)이 아니라 가장 가까운 alpha>=0.99 화소의 색으로 채워 준다
        assert not np.all(got < -0.99, axis=-1).any()
        hole = ~solid
        if hole.any():
            _assert_nearest_solid_color(got, solid, hole)


def _assert_nearest_solid_color(got, solid, hole):
    from scipy.spatial import cKDTree
    sc = np.argwhere(solid)
    tree = cKDTree(sc)
    hc = np.argwhere(hole)
    dist, _ = tree.query(hc)
    for (y, x), d in zip(hc, dist):          # 동률 거리의 후보 중 하나여야 한다
        cand = tree.query_ball_point([y, x], d + 1e-9)
        cols = got[sc[cand][:, 0], sc[cand][:, 1]]
        assert np.abs(cols - got[y, x]).max(axis=1).min() <= 2e-5, (y, x)


def test_sizes_not_multiple_of_16_are_padded_for_the_teacher_and_cropped_back():
    w, hh = 100, 70
    _, dw = h.build(w, hh)
    sam = h.FakeSampler(h.const_transform([0.5] * 4))
    res = S.sdedit_avg_fill(dw, sam, seed=0)
    c = sam.calls[0]
    assert c.x0_shape == (4, 16, 80 // 8, 112 // 8)
    assert sam.teacher.encoded[0].shape == (4, 3, 80, 112)
    # 올림 영역은 오른쪽·아래 edge 복제
    x = sam.teacher.encoded[0].numpy()
    assert np.array_equal(x[:, :, 70:, :100], np.repeat(x[:, :, 69:70, :100], 10, axis=2))
    assert np.array_equal(x[:, :, :, 100:], np.repeat(x[:, :, :, 99:100], 12, axis=3))
    assert res.filled.shape == (hh, w) and res.filled.any()
    gray = float(color.srgb_to_linear(0.5))
    assert np.abs(res.world.layers[1].rgba[res.filled][:, :3] - gray).max() <= 1e-5
    assert res.world.layers[0].rgba.shape == (hh, w, 4)


def test_steps_must_reach_table_end_and_bad_teacher_outputs_are_errors():
    _, dw = h.build()

    class EarlyStop(h.FakeSampler):
        def sdedit(self, *a, **k):
            r = super().sdedit(*a, **k)
            r.sigma_end = 0.3
            return r

    with pytest.raises(S.SDEditAvgError):
        S.sdedit_avg_fill(dw, EarlyStop(h.const_transform([0.5] * 4)))

    class BadShape(h.FakeSampler):
        def sdedit(self, x0, *a, **k):
            r = super().sdedit(x0, *a, **k)
            r.z = r.z[:, :, :-1, :]
            return r

    with pytest.raises(S.SDEditAvgError):
        S.sdedit_avg_fill(dw, BadShape(h.const_transform([0.5] * 4)))

    class FewViews(h.FakeSampler):
        def sdedit(self, x0, *a, **k):
            r = super().sdedit(x0, *a, **k)
            r.z = r.z[:3]
            return r

    with pytest.raises(S.SDEditAvgError):
        S.sdedit_avg_fill(dw, FewViews(h.const_transform([0.5] * 4)))

    class NanTeacher(h.HandleTeacher):
        """샘플러 안의 첫 decode(입력 복원)는 정상, 모듈의 decode(교사 출력)는 NaN 을 돌려준다."""

        def __init__(self):
            super().__init__()
            self.n_decode = 0

        def decode_latent(self, z, clamp=True):
            y = super().decode_latent(z, clamp)
            self.n_decode += 1
            if self.n_decode >= 2:
                y = y.clone()
                y[0, 0, 0, 0] = float("nan")
            return y

    sam = h.FakeSampler(h.const_transform([0.5] * 4), teacher=NanTeacher())
    with pytest.raises(S.SDEditAvgError):
        S.sdedit_avg_fill(dw, sam)


def test_sampler_protocol_attributes_are_checked_before_any_work():
    _, dw = h.build()

    class NoTable:
        teacher = h.HandleTeacher()

        def sdedit(self, *a, **k):
            raise AssertionError("호출되면 안 된다")

    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw, NoTable())

    class NoTeacher:
        table = h.DEFAULT_TABLE

        def sdedit(self, *a, **k):
            raise AssertionError("호출되면 안 된다")

    with pytest.raises(TypeError):
        S.sdedit_avg_fill(dw, NoTeacher())


def test_views_override_is_used_in_order():
    _, dw = h.build()
    views = S.plan_views(dw, S.SDEditAvgConfig(max_angle_deg=12.0, n_per_side=1))[::-1]   # 순서를 뒤집어 준다
    sam = h.FakeSampler(h.const_transform([0.2, 0.8]))
    res = S.sdedit_avg_fill(dw, sam, views=views, seed=0)
    assert [v.angle_deg for v in res.views] == [v.angle_deg for v in views]
    assert sam.calls[0].x0_shape[0] == 2
    assert res.meta()["angles_deg"] == pytest.approx([12.0, -12.0])
    # 입력 영상 i 는 views[i] 의 렌더다
    x = sam.teacher.encoded[0].numpy().astype(np.float64)
    for i, v in enumerate(views):
        r = render_ldi(dw.world, c2w=v.c2w, K=v.K, width=v.width, height=v.height)
        solid = r.alpha >= 0.99
        exp = 2.0 * color.linear_to_srgb(np.clip(r.rgb.astype(np.float64), 0, 1)) - 1.0
        assert np.abs(np.moveaxis(x[i], 0, -1)[solid] - exp[solid]).max() <= 2e-5


def test_view_stats_report_disocclusion_and_holes():
    sam = h.FakeSampler(h.const_transform([0.5] * 4))
    dw, res = _fill(sam)
    for i, v in enumerate(res.views):
        r = render_ldi(dw.world, c2w=v.c2w, K=v.K, width=v.width, height=v.height)
        assert res.view_disocc_frac[i] == pytest.approx(float((r.layer_id == 1).mean()), abs=1e-12)
        assert res.view_hole_frac[i] == pytest.approx(float(((r.alpha < 0.5) & (r.layer_id != 1)).mean()), abs=1e-12)
    # ±20° 뷰가 ±10° 뷰보다 더 많이 열린다(좌우 각각)
    d = res.view_disocc_frac
    assert d[0] > d[1] and d[3] > d[2]

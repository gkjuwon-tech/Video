"""DAT-01 셰이딩 공식(람베르트·점광원·그림자 없음), 슈퍼샘플링 의미, 디노이저 OFF(화소 독립) 검증."""
import dataclasses

import numpy as np
import pytest

from buc.core import geom
from buc.data import synth_render as S
from buc.invent.grammar import core as G

import dat01_helpers as H

W, HT = 64, 48


def _oracle_rgb(prog, cam, ambient, exposure):
    depth, ax, side, P = H.box_oracle(cam)
    sem, nw = H.face_arrays(ax, side)
    mats = dict(prog.matter)
    tab = {G.SEM_FLOOR: mats["floor"], G.SEM_WALL: mats["wall"], G.SEM_CEILING: mats["ceiling"]}
    alb = np.zeros(depth.shape + (3,))
    for sv, name in tab.items():
        alb[sem == sv] = S.ALBEDO[name]
    return H.shade_oracle(P, nw, alb, prog.lights, ambient, exposure), sem


@pytest.fixture(scope="module")
def box():
    prog = H.box_program(lights=H.two_lights(), matter={"floor": "wood", "wall": "plaster", "ceiling": "paint_white"})
    cam = H.cam_center(W, HT, fov=90.0, eye=(2.0, 1.0, 1.5))
    return prog, cam


@pytest.mark.parametrize("ambient,exposure", [(0.05, 0.05), (0.2, 0.01), (0.0, 0.1), (0.0, 0.0)])
def test_lambert_point_light_formula(box, ambient, exposure):
    prog, cam = box
    s = S.render_gbuffer(prog, cam, opts=S.SynthOpts(ambient=ambient, exposure=exposure))
    want, sem = _oracle_rgb(prog, cam, ambient, exposure)
    ok = np.ones(sem.shape, bool)
    # 두 구현이 동점으로 갈라지는 모서리 화소만 제외
    ok &= (s.sem == sem)
    assert ok.mean() > 0.995
    np.testing.assert_allclose(s.rgb[ok], want[ok], rtol=1e-3, atol=2e-6)


def test_light_distance_is_floored_at_half_meter():
    """광원이 면에 0.15 m 로 붙어도 d_eff = max(d, 0.5) 라서 값이 폭주하지 않는다."""
    lights = (H.lamp(0, (2.0, 3.85, 1.5), (1.0, 1.0, 1.0), 300.0),)
    prog = H.box_program(lights=lights, matter={"floor": "wood", "wall": "plaster", "ceiling": "paint_white"})
    cam = H.cam_center(W, HT, fov=90.0, eye=(2.0, 1.0, 1.5))
    s = S.render_gbuffer(prog, cam, opts=S.SynthOpts(ambient=0.0, exposure=0.05))
    want, sem = _oracle_rgb(prog, cam, 0.0, 0.05)
    ok = s.sem == sem
    np.testing.assert_allclose(s.rgb[ok], want[ok], rtol=1e-3, atol=2e-6)
    # 폭주 방지: 뒷벽 중앙 화소 = albedo * exposure * P cos / (4 pi^2 * 0.5^2)
    c = s.rgb[HT // 2, W // 2].astype(np.float64)
    a = np.asarray(S.ALBEDO["plaster"])
    assert (c <= a * 0.05 * 300.0 / (4 * np.pi ** 2 * 0.25) * 1.001).all()


def test_back_facing_surfaces_get_ambient_only_and_formula_holds_with_furniture():
    """광원 반대쪽을 향한 면은 max(0, n.w)=0 이라 환경광만 받는다. 전 화소를 G-buffer 로 재계산해 대조."""
    lights = (H.lamp(0, (2.0, 3.8, 1.0), (1.0, 1.0, 1.0), 400.0), H.lamp(1, (0.5, 1.0, 2.5), (0.7, 0.8, 1.0), 300.0))
    prog = H.furnished_program(lights=lights)
    cam = H.cam_center(W, HT, fov=110.0)
    s = S.render_gbuffer(prog, cam, opts=S.SynthOpts(ambient=0.07, exposure=0.04))
    Xc = geom.unproject(cam.K, geom.pixel_grid(HT, W), s.depth.astype(np.float64), cam.k1, cam.k2)
    P = Xc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
    nw = np.moveaxis(s.normal, 0, -1).astype(np.float64) @ cam.c2w[:3, :3].T
    alb = np.moveaxis(s.albedo, 0, -1).astype(np.float64)
    want = H.shade_oracle(P, nw, alb, lights, 0.07, 0.04)
    np.testing.assert_allclose(s.rgb, want, rtol=2e-3, atol=2e-6)
    # 램프 0 은 가구 앞면(법선 -y) 뒤쪽 y=3.8 에 있다 -> 그 면에서 램프 0 의 기여는 0 이어야 한다
    front = (s.inst == 3) & (nw[..., 1] < -0.99)
    assert front.sum() >= 5
    only0 = H.shade_oracle(P, nw, alb, lights[:1], 0.0, 0.04)
    assert np.abs(only0[front]).max() == 0.0


def test_shading_is_affine_in_exposure_and_ambient(box):
    prog, cam = box
    r = lambda a, e: S.render_gbuffer(prog, cam, opts=S.SynthOpts(ambient=a, exposure=e)).rgb.astype(np.float64)
    r00, r10, r01, r02 = r(0.0, 0.0), r(0.1, 0.0), r(0.0, 0.05), r(0.0, 0.10)
    assert np.abs(r00).max() == 0.0  # 광원 기여 0, 환경 0 -> 검정
    np.testing.assert_allclose(r02, 2.0 * r01, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(r(0.1, 0.05), r10 + r01, rtol=1e-5, atol=1e-7)
    assert r10.min() > 0.0  # ambient 만으로도 알베도 비율의 밝기


def _segment_hits_aabb(P, L, lo, hi):
    """선분 P->L (P: (...,3)) 이 축 정렬 상자 [lo,hi] 와 만나는가(슬랩 법). 끝점 근처 1e-3 은 제외."""
    d = np.asarray(L, float) - P
    t0 = np.full(P.shape[:-1], 1e-3)
    t1 = np.full(P.shape[:-1], 1.0 - 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in range(3):
            ta = (lo[a] - P[..., a]) / d[..., a]
            tb = (hi[a] - P[..., a]) / d[..., a]
            lo_t, hi_t = np.minimum(ta, tb), np.maximum(ta, tb)
            par = d[..., a] == 0
            inside = (P[..., a] >= lo[a]) & (P[..., a] <= hi[a])
            lo_t = np.where(par, np.where(inside, -np.inf, np.inf), lo_t)
            hi_t = np.where(par, np.where(inside, np.inf, -np.inf), hi_t)
            t0, t1 = np.maximum(t0, lo_t), np.minimum(t1, hi_t)
    return t0 <= t1


def test_no_shadows_in_v0():
    """그림자 없음(v0): 기하학적으로 광원이 가려진 바닥 화소의 밝기도 람베르트 공식 그대로다."""
    lights = (H.lamp(0, (2.0, 3.5, 2.5)),)
    cam = H.cam_center(W, HT, fov=120.0, eye=(2.0, 1.0, 1.5), target=(2.0, 2.0, 0.6))
    pillar = G.Slot(id=1, cls="cabinet", center=(2.0, 2.2, 1.45), size=(0.5, 0.4, 2.9), material="plaster")
    matter = {"floor": "wood", "wall": "plaster", "ceiling": "paint_white"}
    prog = H.box_program(lights=lights, slots=(pillar,), matter=matter)
    s = S.render_gbuffer(prog, cam)
    from buc.core import geom
    Xc = geom.unproject(cam.K, geom.pixel_grid(HT, W), s.depth.astype(np.float64), cam.k1, cam.k2)
    P = Xc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
    lo, hi = np.array([1.75, 2.0, 0.0]), np.array([2.25, 2.4, 2.9])
    shadowed = _segment_hits_aabb(P, lights[0].pos, lo, hi) & (s.sem == G.SEM_FLOOR)
    assert shadowed.sum() >= 20, "테스트 설계 오류: 그림자 영역에 화소가 거의 없다"
    nw = (np.moveaxis(s.normal, 0, -1)) @ cam.c2w[:3, :3].T
    alb = np.moveaxis(s.albedo, 0, -1).astype(np.float64)
    want = H.shade_oracle(P, nw, alb, lights, 0.05, 0.05)
    np.testing.assert_allclose(s.rgb[shadowed], want[shadowed], rtol=2e-3, atol=2e-6)


def test_light_color_and_power_scale_linearly(box):
    prog, cam = box
    base = S.render_gbuffer(prog, cam, opts=S.SynthOpts(ambient=0.0, exposure=0.05)).rgb.astype(np.float64)
    prog2 = dataclasses.replace(prog, lights=tuple(
        G.Light(id=l.id, kind=l.kind, pos=l.pos, rgb=l.rgb, power=l.power * 3.0, source=l.source)
        for l in prog.lights))
    triple = S.render_gbuffer(prog2, cam, opts=S.SynthOpts(ambient=0.0, exposure=0.05)).rgb.astype(np.float64)
    np.testing.assert_allclose(triple, 3.0 * base, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- 슈퍼샘플링
def _shifted_K_cam(cam, dx, dy):
    K = cam.K.copy()
    K[0, 2] -= dx
    K[1, 2] -= dy
    return S.SynthCamera(c2w=cam.c2w, K=K, width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)


@pytest.mark.parametrize("spp", [4, 9, 16])
def test_spp_is_mean_of_regular_subpixel_samples(spp):
    prog = H.furnished_program()
    cam = H.cam_center(32, 24, fov=110.0)
    n = int(round(spp ** 0.5))
    offs = [((i + 0.5) / n - 0.5) for i in range(n)]
    acc = 0.0
    for dy in offs:
        for dx in offs:
            acc = acc + S.render_gbuffer(prog, _shifted_K_cam(cam, dx, dy)).rgb.astype(np.float64)
    want = acc / spp
    got = S.render_gbuffer(prog, cam, opts=S.SynthOpts(spp=spp))
    np.testing.assert_allclose(got.rgb, want, rtol=0, atol=2e-5)
    assert got.meta["spp"] == spp


def test_aov_channels_do_not_depend_on_spp():
    prog = H.furnished_program()
    cam = H.cam_center(32, 24, fov=110.0)
    fcam = H.shifted(cam, (0.05, 0, 0))
    a = S.render_gbuffer(prog, cam, flow_to=fcam, opts=S.SynthOpts(spp=1))
    b = S.render_gbuffer(prog, cam, flow_to=fcam, opts=S.SynthOpts(spp=4))
    for k in ("gbuf", "depth", "sem", "inst", "flow", "flow_valid"):
        assert a.to_arrays()[k].tobytes() == b.to_arrays()[k].tobytes(), k
    assert not np.array_equal(a.rgb, b.rgb)  # 안티앨리어싱이 rgb 에는 작용한다


def test_supersampling_reduces_edge_aliasing():
    """가구 에지 화소: spp=16 의 rgb 는 spp=1 의 두 이웃 색 사이의 중간값을 가진다(계단 감소)."""
    prog = H.furnished_program()
    cam = H.cam_center(48, 36, fov=110.0)
    a = S.render_gbuffer(prog, cam).rgb.astype(np.float64)
    b = S.render_gbuffer(prog, cam, opts=S.SynthOpts(spp=16)).rgb.astype(np.float64)
    s1 = S.render_gbuffer(prog, cam)
    edge = np.zeros(s1.inst.shape, bool)
    edge[:, 1:-1] = (s1.inst[:, 1:-1] != s1.inst[:, :-2]) | (s1.inst[:, 1:-1] != s1.inst[:, 2:])
    assert edge.sum() > 20
    # 에지 화소의 변화량이 비에지의 변화량보다 훨씬 크다
    d = np.abs(a - b).sum(axis=-1)
    assert d[edge].mean() > 3.0 * d[~edge].mean()


# --------------------------------------------------------------------------- 디노이저 OFF = 화소 독립
@pytest.mark.parametrize("spp", [1, 4])
def test_crop_equivariance_no_cross_pixel_filtering(spp):
    prog = H.furnished_program()
    cam = H.cam_center(64, 48, fov=110.0)
    x0, y0, w, h = 10, 7, 30, 20
    crop = S.SynthCamera(c2w=cam.c2w, K=geom.crop_intrinsics(cam.K, x0, y0), width=w, height=h)
    full = S.render_gbuffer(prog, cam, opts=S.SynthOpts(spp=spp))
    part = S.render_gbuffer(prog, crop, opts=S.SynthOpts(spp=spp))
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    np.testing.assert_allclose(part.rgb, full.rgb[sl], rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(part.gbuf, full.gbuf[(slice(None),) + sl], rtol=1e-5, atol=2e-5)
    assert (part.sem == full.sem[sl]).mean() > 0.999
    assert (part.inst == full.inst[sl]).mean() > 0.999

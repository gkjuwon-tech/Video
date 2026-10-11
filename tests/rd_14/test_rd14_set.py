"""RD-14 관측 집합 O: 레벨별 (블록 키, 16³ 셀 비트셋) 경/부분 소유. 전수 오라클 + 손계산 평면 + STRATA 이음매."""
import json
import math
import time

import numpy as np
import pytest

import rd14_helpers as H

om = H.om


def pose():
    R = H.rot("y", 7.3) @ H.rot("x", -3.1)
    t = np.array([0.0371, -0.113, 0.2057])
    return R, t


def scene(Hh=40, Ww=56, coc_right=5.0):
    K = H.K_from_fov(Ww, Hh, 60.0)
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    z = 3.137 + 0.011 * (xx - 28) + 0.017 * (yy - 20) + 0.05 * np.sin(0.7 * xx) * np.cos(0.5 * yy)
    coc = np.zeros((Hh, Ww))
    coc[:, Ww // 2:] = coc_right
    return K, z, coc


def make(Hh=40, Ww=56, coc_right=5.0, valid=None, **kw):
    K, z, coc = scene(Hh, Ww, coc_right)
    R, t = pose()
    m = om.build_obs_map(z, K, coc_px=coc, valid=valid)
    s = om.build_obs_set(m, z, K, R, t, **kw)
    return K, R, t, z, m, s


# ------------------------------------------------------------------ 손계산: 정면 균일 평면, 단위 자세
def test_hand_computed_fronto_plane_counts_and_bits():
    Hh, Ww = 40, 56
    K = H.K_from_fov(Ww, Hh, 60.0)
    z = np.full((Hh, Ww), 3.137)
    m = om.build_obs_map(z, K)
    assert (m.l_hard == 5).all() and (m.l_part == 6).all()          # φ = 3.137/48.497 = 0.0647 -> a = 5.95
    s = om.build_obs_set(m, z, K, np.eye(3), np.zeros(3))
    assert s.levels == tuple(range(0, 7)) and s.kind == "direct" and s.stride == 1
    # x ∈ ±1.779, y ∈ ±1.261, z = 3.137. 셀 변 = 64·2^-l
    exp = {6: (0, 16), 5: (4, 0), 4: (4, 0), 3: (4, 0), 2: (4, 0), 1: (4, 0), 0: (4, 0)}      # level: (hard, part)
    for l, (nh, npart) in exp.items():
        assert s.n_cells(l, "hard") == nh and s.n_cells(l, "part") == npart, l
        assert s.n_cells(l) == nh + npart
    assert s.n_cells() == 16 + 6 * 4 and s.n_cells(None, "hard") == 24 and s.n_cells(None, "part") == 16
    assert s.n_cells(7) == 0 and s.keys(7) == []
    # 레벨 6 블록: i,j ∈ {-1, 0}, k = 0. 셀 인덱스 x ∈ {-2,-1,0,1}, y 같음, z = 3 -> 비트 = x%16 + 16·(y%16) + 256·3
    assert s.keys(6) == [(6, -1, -1, 0), (6, -1, 0, 0), (6, 0, -1, 0), (6, 0, 0, 0)]
    d = H.set_as_dict(s)
    xs = {-1: (14, 15), 0: (0, 1)}
    for (l, i, j, k), (hard, part) in d.items():
        if l != 6:
            continue
        assert hard == set() and k == 0
        assert part == {x + 16 * y + 256 * 3 for x in xs[i] for y in xs[j]}
    # 레벨 5 (셀 2 m): 4 셀 전부 경소유, 블록 (5, -1|0, -1|0, 0), 셀 (x,y) ∈ {-1,0}², z = 1
    d5 = {k: v for k, v in d.items() if k[0] == 5}
    assert sorted(d5) == [(5, -1, -1, 0), (5, -1, 0, 0), (5, 0, -1, 0), (5, 0, 0, 0)]
    for (l, i, j, k), (hard, part) in d5.items():
        assert part == set() and hard == {(15 if i < 0 else 0) + 16 * (15 if j < 0 else 0) + 256 * 1}
    # 블록 단위 소유: 레벨 5 는 INPUT, 레벨 6 은 INPUT_PART
    assert all(s.block_own(k) == om.OWN_INPUT for k in s.keys(5))
    assert all(s.block_own(k) == om.OWN_INPUT_PART for k in s.keys(6))


# ------------------------------------------------------------------ 전수 오라클
def test_matches_bruteforce_oracle_and_semantics():
    K, R, t, z, m, s = make()
    levels = tuple(range(0, int(m.l_part.max()) + 1))
    assert s.levels == levels
    X, ii, jj = H.world_points(K, R, t, z, 1)
    exp = H.oracle_obs_set(X, ii, jj, m.l_hard, m.l_part, m.valid, levels)
    got = H.set_as_dict(s)
    assert set(got) == set(exp)
    for key in exp:
        assert got[key] == exp[key], key
    # 의미 확인: 최소 경소유 레벨 아래에는 부분 셀이 없고, 최대 부분 레벨 위에는 아무것도 없다
    lo = int(m.l_hard[m.valid].min())
    hi = int(m.l_part[m.valid].max())
    assert any(s.n_cells(l, "part") > 0 for l in range(lo + 1, hi + 1)) and s.n_cells(hi) > 0
    assert all(s.n_cells(l, "part") == 0 for l in range(0, lo + 1))
    assert all(s.n_cells(l, "hard") == 0 for l in range(int(m.l_hard.max()) + 1, hi + 1))
    # 한 셀은 경·부분 둘 다일 수 없다
    for blk in s.blocks.values():
        assert not (blk.hard & blk.part).any() and blk.hard.dtype == np.uint64 and blk.hard.shape == (64,)


def test_invalid_pixels_do_not_observe():
    valid = np.ones((40, 56), bool)
    valid[:, 28:] = False
    K, R, t, z, m, s = make(valid=valid)
    X, ii, jj = H.world_points(K, R, t, z, 1)
    exp = H.oracle_obs_set(X, ii, jj, m.l_hard, m.l_part, m.valid, s.levels)
    assert H.set_as_dict(s) == exp
    _, _, _, _, _, full = make()
    assert s.n_cells() < full.n_cells()
    assert om.build_obs_set(om.build_obs_map(np.full((40, 56), np.nan), K), z, K, R, t).n_cells() == 0


def test_stride_subsamples_and_matches_oracle():
    K, R, t, z, m, s1 = make()
    s3 = om.build_obs_set(m, z, K, R, t, stride=3)
    X, ii, jj = H.world_points(K, R, t, z, 3)
    assert ii.shape == (14, 19) and s3.stride == 3
    exp = H.oracle_obs_set(X, ii, jj, m.l_hard, m.l_part, m.valid, s3.levels)
    assert H.set_as_dict(s3) == exp
    # 표본이 적으면 관측 셀이 늘 수는 없다
    assert s3.n_cells() <= s1.n_cells()
    d1, d3 = H.set_as_dict(s1), H.set_as_dict(s3)
    for key, (h3, p3) in d3.items():
        h1, p1 = d1[key]
        assert (h3 | p3) <= (h1 | p1)


def test_distortion_follows_geom_unproject():
    """가까운 장면(작은 셀)에서 k1, k2 가 가장자리 셀 집합을 실제로 바꾸는지까지 확인한다(자체 민감도 점검)."""
    from buc.core import geom
    Hh, Ww = 40, 56
    K = H.K_from_fov(Ww, Hh, 60.0)
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    z = 0.6 + 0.0013 * (xx - 28) + 0.002 * (yy - 20) + 0.004 * np.sin(0.7 * xx) * np.cos(0.5 * yy)
    R, t = pose()
    k1, k2 = -0.3, 0.05
    m = om.build_obs_map(z, K)
    s = om.build_obs_set(m, z, K, R, t, k1=k1, k2=k2)
    plain = om.build_obs_set(m, z, K, R, t)
    ii, jj = np.mgrid[0:Hh, 0:Ww]
    uv = np.stack([jj, ii], axis=-1).astype(np.float64)
    X = geom.unproject_world(K, R, t, uv, z, k1, k2)
    X0, _, _ = H.world_points(K, R, t, z, 1)
    exp = H.oracle_obs_set(X, ii, jj, m.l_hard, m.l_part, m.valid, s.levels)
    exp0 = H.oracle_obs_set(X0, ii, jj, m.l_hard, m.l_part, m.valid, s.levels)
    assert exp != exp0                                           # 이 장면에서 왜곡이 집합을 바꾼다
    assert H.set_as_dict(s) == exp and H.set_as_dict(plain) == exp0
    assert plain.digest() != s.digest()


# ------------------------------------------------------------------ levels 인자
def test_levels_argument():
    K, R, t, z, m, s = make()
    sub = om.build_obs_set(m, z, K, R, t, levels=(5, 3))
    assert sub.levels == (3, 5)
    d_all, d_sub = H.set_as_dict(s), H.set_as_dict(sub)
    assert d_sub == {k: v for k, v in d_all.items() if k[0] in (3, 5)}
    assert om.build_obs_set(m, z, K, R, t, levels=()).n_cells() == 0
    for bad in ((2, 2), (-9,), (25,), (1.5,), ("3",)):
        with pytest.raises((ValueError, TypeError)):
            om.build_obs_set(m, z, K, R, t, levels=bad)
    # 음수 레벨도 허용(LEVEL_RANGE 안): 한 셀로 뭉친다
    neg = om.build_obs_set(m, z, K, R, t, levels=(-3, 0))
    assert neg.n_cells(-3) >= 1 and neg.n_cells(-3) <= neg.n_cells(0)


# ------------------------------------------------------------------ 점 질의와 STRATA 이음매
def test_cell_state_queries_and_strata_seam():
    strata = pytest.importorskip("buc.world.strata")
    assert (strata.H0_M, strata.BLOCK_CELLS, tuple(strata.LEVEL_RANGE)) == (om.H0_M, om.BLOCK_CELLS, om.LEVEL_RANGE)
    assert strata.OWNERS["INPUT"] == om.OWN_INPUT and strata.OWNERS["INPUT_PART"] == om.OWN_INPUT_PART
    assert strata.OWNERS["INPUT_REFL"] == om.OWN_INPUT_REFL
    for l in (-8, 0, 7, 24):
        assert om.cell_size_m(l) == strata.cell_size(l) and om.block_extent_m(l) == strata.block_extent(l)
        assert om.texel_pitch_m(l) == strata.texel_pitch(l)
    K, R, t, z, m, s = make()
    X, ii, jj = H.world_points(K, R, t, z, 1)
    P = X.reshape(-1, 3)
    exp = H.oracle_obs_set(X, ii, jj, m.l_hard, m.l_part, m.valid, s.levels)
    for l in s.levels:
        cs = H.H0_M * 2.0 ** (-l)
        st = s.cell_state(P, l)
        assert st.shape == (P.shape[0],) and st.dtype == np.uint8
        # 키 규약: block_of 가 만든 키가 곧 블록 키다
        keys = strata.block_of(P, l)
        for row in range(0, P.shape[0], 37):
            key = tuple(int(v) for v in keys[row])
            if st[row]:
                assert key in s.blocks
            c = tuple(int(math.floor(v / cs)) for v in P[row])
            loc = tuple(v % 16 for v in c)
            idx = loc[0] + 16 * loc[1] + 256 * loc[2]
            hard, part = exp.get(key, (set(), set()))
            want = 1 if idx in hard else 2 if idx in part else 0
            assert st[row] == want, (l, row)
    # 관측 안 된 점은 0
    far = np.array([[1e3, 1e3, 1e3], [-500.0, 20.0, 9.0]])
    assert (s.cell_state(far, 4) == 0).all()
    assert s.cell_state(np.zeros((0, 3)), 4).shape == (0,)
    assert 12 not in s.levels and (s.cell_state(P, 12) == 0).all()        # 만들지 않은 레벨은 관측 없음
    for bad_l in (-9, 25):
        with pytest.raises(ValueError):
            s.cell_state(P[:2], bad_l)
    with pytest.raises(ValueError):
        s.cell_state(np.zeros((3, 2)), 4)
    with pytest.raises(ValueError):
        s.cell_state(np.array([[np.nan, 0, 0]]), 4)
    # 키는 정렬되고 (l,i,j,k) int 4-튜플
    keys = s.keys()
    assert keys == sorted(keys) and all(len(k) == 4 and all(isinstance(v, int) for v in k) for k in keys)
    assert s.keys(5) == [k for k in keys if k[0] == 5]


# ------------------------------------------------------------------ 합집합, 다이제스트, kind
def test_union_digest_and_kind():
    K, R, t, z, m, s = make()
    sharp = om.build_obs_map(z, K)                                  # 전부 선명: 경소유가 한 단 높다
    blurry = om.build_obs_map(z, K, coc_px=5.0)
    a = om.build_obs_set(sharp, z, K, R, t)
    b = om.build_obs_set(blurry, z, K, R, t, levels=a.levels)
    u = a.union(b)
    assert u.digest() == b.union(a).digest() and a.union(a).digest() == a.digest()
    assert u.levels == a.levels and u.kind == "direct"
    da, db, du = H.set_as_dict(a), H.set_as_dict(b), H.set_as_dict(u)
    assert set(du) == set(da) | set(db)
    conflicts = 0
    for key in du:
        ha, pa = da.get(key, (set(), set()))
        hb, pb = db.get(key, (set(), set()))
        hard = ha | hb
        assert du[key] == (hard, (pa | pb) - hard), key
        conflicts += len(ha & pb) + len(hb & pa)
    assert conflicts > 0                                            # 경소유가 부분을 이기는 셀이 실제로 있었다
    with pytest.raises(ValueError):
        a.union(om.build_obs_set(sharp, z, K, R, t, levels=(1, 2)))     # 레벨 목록이 다르면 합칠 수 없다
    # 결정론, 다이제스트
    s2 = om.build_obs_set(m, z, K, R, t)
    assert s.digest() == s2.digest() and len(s.digest()) == 64
    z2 = z.copy()
    z2[5, 5] += 3.0
    m2 = om.build_obs_map(z2, K, coc_px=scene()[2])
    assert om.build_obs_set(m2, z2, K, R, t).digest() != s.digest()
    # kind
    refl = om.build_obs_set(m, z, K, R, t, kind="refl")
    assert refl.kind == "refl" and H.set_as_dict(refl) == H.set_as_dict(s) and refl.digest() != s.digest()
    assert all(refl.block_own(k) == om.OWN_INPUT_REFL for k in refl.keys())
    with pytest.raises(ValueError):
        s.union(refl)
    with pytest.raises(ValueError):
        om.build_obs_set(m, z, K, R, t, kind="mirror")
    with pytest.raises(TypeError):
        s.union(42)
    # block_own: 부분 셀이 하나라도 있으면 INPUT_PART
    for k, blk in s.blocks.items():
        want = om.OWN_INPUT_PART if blk.part.any() else om.OWN_INPUT
        assert s.block_own(k) == want
    with pytest.raises(KeyError):
        s.block_own((0, 10 ** 6, 0, 0))


def test_meta_json_and_counts():
    K, R, t, z, m, s = make()
    meta = s.to_meta()
    json.dumps(meta, allow_nan=False)
    assert meta["kind"] == "direct" and meta["n_blocks"] == len(s.blocks) and meta["levels"] == list(s.levels)
    assert meta["n_cells_hard"] == s.n_cells(None, "hard") and meta["n_cells_part"] == s.n_cells(None, "part")
    assert meta["digest"] == s.digest()
    total = sum(bin(int(w)).count("1") for b in s.blocks.values() for w in np.concatenate([b.hard, b.part]))
    assert total == s.n_cells()


# ------------------------------------------------------------------ 검증
def test_build_obs_set_validation():
    K, R, t, z, m, s = make()
    f = om.build_obs_set
    with pytest.raises(TypeError):
        f("map", z, K, R, t)
    with pytest.raises(ValueError):
        f(m, z[:-1], K, R, t)                              # 깊이 크기 != 지도 크기
    with pytest.raises(ValueError):
        f(m, z, np.eye(2), R, t)
    with pytest.raises(ValueError):
        f(m, z, K, R * 2.0, t)                             # 회전 아님
    with pytest.raises(ValueError):
        f(m, z, K, R, t[:2])
    with pytest.raises(ValueError):
        f(m, z, K, R, np.array([0.0, np.nan, 0.0]))
    for st in (0, -1):
        with pytest.raises(ValueError):
            f(m, z, K, R, t, stride=st)
    for st in (1.5, True, "2"):
        with pytest.raises((TypeError, ValueError)):
            f(m, z, K, R, t, stride=st)
    with pytest.raises(ValueError):
        f(m, z, K, R, t, k1=float("nan"))
    # 입력 불변
    z0, R0, t0 = z.copy(), R.copy(), t.copy()
    f(m, z, K, R, t)
    assert np.array_equal(z, z0) and np.array_equal(R, R0) and np.array_equal(t, t0)


def test_time_budget_larger_scene():
    Hh, Ww = 540, 960
    K = H.K_from_fov(Ww, Hh, 60.0)
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    z = 2.0 + 0.004 * xx + 0.002 * yy
    coc = 0.01 * xx
    R, t = pose()
    t0 = time.perf_counter()
    m = om.build_obs_map(z, K, coc_px=coc)
    s = om.build_obs_set(m, z, K, R, t, stride=4)
    dt = time.perf_counter() - t0
    assert dt < 15.0, dt
    assert len(s.blocks) > 0
    assert s.n_cells() > 100

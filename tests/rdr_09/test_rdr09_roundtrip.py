"""RDR-09 왕복 비트 동일: 정확히 대칭인 경로는 돌아오는 프레임이 나가는 프레임과 바이트까지 같다."""
import copy

import numpy as np
import pytest
from rdr09_helpers import forward_shot, shot_from_arrays, synth_frames, two_layer_world

from buc.core import contracts as C
from buc.render import aov as A
from buc.render import pos

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


@pytest.fixture(scope="module")
def world():
    return two_layer_world()


@pytest.fixture(scope="module")
def fwd():
    return forward_shot(3)


@pytest.fixture(scope="module")
def rt_shot(fwd):
    return pos.out_and_back(fwd)


@pytest.fixture(scope="module")
def rt_frames(world, rt_shot):
    return A.render_shot_aov(world, rt_shot)


# ---------------------------------------------------------------- 경로 구성
def test_mirror_index():
    assert pos.mirror_index(5).tolist() == [4, 3, 2, 1, 0] and pos.mirror_index(5).dtype == np.int64
    assert pos.mirror_index(1).tolist() == [0] and pos.mirror_index(0).tolist() == []
    with pytest.raises(ValueError):
        pos.mirror_index(-1)


def test_out_and_back_structure(fwd, rt_shot):
    n = fwd.n_frames
    assert isinstance(rt_shot, C.Shot) and rt_shot.n_frames == 2 * n - 1 == 5
    j = list(range(n)) + list(range(n - 2, -1, -1))
    for name in ("c2w", "K", "focus_m"):
        a, b = getattr(fwd, name), getattr(rt_shot, name)
        for k, src in enumerate(j):
            assert a[src].tobytes() == b[k].tobytes(), (name, k)                 # 비트 단위로 복사
    assert np.allclose(rt_shot.t, np.arange(2 * n - 1) / fwd.fps, atol=1e-12)    # 시간축은 다시 매긴다
    assert (rt_shot.width, rt_shot.height, rt_shot.fps) == (fwd.width, fwd.height, fwd.fps)
    assert rt_shot.theta == fwd.theta and rt_shot.move == fwd.move and rt_shot.rig == fwd.rig and rt_shot.seed == fwd.seed
    assert rt_shot.content_hash() != fwd.content_hash()
    assert pos.is_palindromic_shot(rt_shot) and not pos.is_palindromic_shot(fwd)
    assert pos.out_and_back(forward_shot(2)).n_frames == 3
    with pytest.raises(ValueError):
        pos.out_and_back(forward_shot(1))


def test_is_palindromic_shot_exact_and_tolerance(rt_shot):
    assert pos.is_palindromic_shot(rt_shot, atol=0.0)
    for name in ("c2w", "K", "focus_m"):
        arr = {k: copy.deepcopy(getattr(rt_shot, k)) for k in ("c2w", "K", "focus_m")}
        arr[name][4] = arr[name][4] + (1e-12 if name != "c2w" else 0.0)
        if name == "c2w":
            arr[name][4][0, 3] += 1e-12
        s = shot_from_arrays(arr["c2w"], arr["K"], arr["focus_m"], rt_shot)
        assert not pos.is_palindromic_shot(s), name                              # 기본은 비트 단위 정확 일치
        assert pos.is_palindromic_shot(s, atol=1e-9), name                        # 허용오차를 주면 통과
        assert not pos.is_palindromic_shot(s, atol=1e-15), name
    one = pos.out_and_back(forward_shot(2))
    assert pos.is_palindromic_shot(one)


# ---------------------------------------------------------------- 렌더 왕복
def test_roundtrip_is_bit_identical_on_real_render(rt_frames):
    fr = rt_frames
    n = fr.rgb.shape[0]
    assert n == 5
    r = pos.roundtrip_check(fr)
    assert isinstance(r, pos.RoundTrip)
    assert r.identical is True and r.n_frames == 5 and r.pairs == 2
    assert r.pair_identical == (True, True) and r.first_mismatch is None
    assert r.keys == pos.ROUNDTRIP_KEYS
    assert r.max_abs_diff == 0.0 and r.max_lsb_diff == 0
    assert r.diff_map.shape == (2, fr.rgb.shape[1], fr.rgb.shape[2]) and r.diff_map.dtype == np.float32
    assert not r.diff_map.any()                                                  # 차이 맵이 새까맣다
    for i in range(n):
        j = n - 1 - i
        assert fr.rgb[i].tobytes() == fr.rgb[j].tobytes(), i
        for k in pos.ROUNDTRIP_KEYS:
            assert fr.aov[k][i].tobytes() == fr.aov[k][j].tobytes(), (i, k)
    d = pos.frame_digests(fr, keys=pos.ROUNDTRIP_KEYS)
    assert d == d[::-1]
    assert fr.rgb[0].std() > 0.02 and len(set(pos.frame_digests(fr, keys=pos.ROUNDTRIP_KEYS))) == 3   # 자명한 상수 영상이 아니다


def test_flow_is_why_roundtrip_keys_exclude_flow(rt_frames):
    """flow 는 i->i+1 이라 왕복에서 방향이 뒤집힌다. 전체 aov 지문은 짝이 달라야 하고 ROUNDTRIP_KEYS 지문은 같아야 한다."""
    fr = rt_frames
    n = fr.rgb.shape[0]
    assert "flow" not in pos.ROUNDTRIP_KEYS and "flow_valid" not in pos.ROUNDTRIP_KEYS
    assert pos.frame_leaf(fr, 0) != pos.frame_leaf(fr, n - 1)
    assert pos.frame_leaf(fr, 0, keys=pos.ROUNDTRIP_KEYS) == pos.frame_leaf(fr, n - 1, keys=pos.ROUNDTRIP_KEYS)
    assert fr.aov["flow_valid"][0].any() and not fr.aov["flow_valid"][n - 1].any()


def test_verify_roundtrip_renders_and_checks(world, rt_shot, fwd):
    r = pos.verify_roundtrip(world, rt_shot)
    assert r.identical and r.pairs == 2 and r.max_abs_diff == 0.0
    with pytest.raises(ValueError):
        pos.verify_roundtrip(world, fwd)                                         # 대칭이 아닌 경로는 거부


def test_roundtrip_is_independent_of_frame_order(world, rt_shot, rt_frames):
    """순서 불변: 경로를 거꾸로 렌더해도 같은 포즈의 프레임은 같은 바이트다."""
    n = rt_shot.n_frames
    rev = shot_from_arrays(rt_shot.c2w[::-1].copy(), rt_shot.K[::-1].copy(), rt_shot.focus_m[::-1].copy(), rt_shot)
    fr = A.render_shot_aov(world, rev)
    a = pos.frame_digests(rt_frames, keys=pos.ROUNDTRIP_KEYS)
    b = pos.frame_digests(fr, keys=pos.ROUNDTRIP_KEYS)
    assert b == a[::-1]
    # 그리고 서로 다른 포즈의 프레임은 서로 다르다(지문이 상수가 아니다)
    assert len({a[0], a[1], a[2]}) == 3
    assert n == 5


# ---------------------------------------------------------------- 음성 대조군
def test_one_ulp_in_the_return_leg_is_detected(rt_frames):
    fr = copy.deepcopy(rt_frames)
    n = fr.rgb.shape[0]
    fr.rgb[n - 2, 5, 7, 1] = np.nextafter(fr.rgb[n - 2, 5, 7, 1], np.float32(2.0))     # 프레임 3 (짝: 1)
    r = pos.roundtrip_check(fr)
    assert not r.identical and r.pair_identical == (True, False) and r.first_mismatch == 1
    assert 0.0 < r.max_abs_diff < 1e-6 and r.max_lsb_diff == 0                   # 1 ulp 는 8-bit 로는 안 보이지만 비트로는 잡힌다
    assert r.diff_map[1].max() == r.max_abs_diff and r.diff_map[0].max() == 0.0
    assert np.argwhere(r.diff_map[1] > 0).tolist() == [[5, 7]]


def test_aov_difference_alone_breaks_identity(rt_frames):
    fr = copy.deepcopy(rt_frames)
    fr.aov["atom_id"][4, 0, 0] += 1                                              # 프레임 4 의 지배 원자만 다르다
    r = pos.roundtrip_check(fr)
    assert not r.identical and r.pair_identical == (False, True) and r.first_mismatch == 0
    assert r.max_abs_diff == 0.0                                                 # rgb 는 같다
    ok = pos.roundtrip_check(fr, keys=["depth", "alpha"])                        # keys 로 비교 대상을 좁힐 수 있다
    assert ok.identical and ok.keys == ("alpha", "depth")


def test_generative_style_noise_is_visible_as_lsb_difference():
    """'돌아오는 길에 세계를 다시 상상하는' 모델 모사: 같은 포즈의 두 프레임이 약간씩 다르다."""
    fr = synth_frames(n=5, with_aov=False)
    fr.rgb[4] = fr.rgb[0]
    fr.rgb[3] = fr.rgb[1]
    assert pos.roundtrip_check(fr).identical
    rng = np.random.default_rng(1)
    noisy = copy.deepcopy(fr)
    noisy.rgb[4] = np.clip(noisy.rgb[4] + rng.normal(0, 0.02, noisy.rgb[4].shape).astype(np.float32), 0, 1)
    r = pos.roundtrip_check(noisy)
    assert not r.identical and r.first_mismatch == 0 and r.pair_identical == (False, True)
    assert r.max_lsb_diff >= 3 and r.max_abs_diff > 0.02
    assert r.keys == ()                                                          # aov 가 없으면 rgb 만 비교한다
    assert r.diff_map.shape == (2, 6, 5)


def test_roundtrip_check_small_cases():
    one = synth_frames(n=1)
    r = pos.roundtrip_check(one)
    assert r.identical and r.pairs == 0 and r.pair_identical == () and r.first_mismatch is None
    assert r.diff_map.shape == (0, 6, 5) and r.max_abs_diff == 0.0 and r.max_lsb_diff == 0
    odd = synth_frames(n=3)                                                      # 가운데 프레임은 자기 자신과만 짝
    odd.rgb[2] = odd.rgb[0]
    for k in odd.aov:
        odd.aov[k][2] = odd.aov[k][0]
    assert pos.roundtrip_check(odd).identical
    even = synth_frames(n=2)
    even.rgb[1] = even.rgb[0]
    for k in even.aov:
        even.aov[k][1] = even.aov[k][0]
    assert pos.roundtrip_check(even).pairs == 1 and pos.roundtrip_check(even).identical
    with pytest.raises(ValueError):
        pos.roundtrip_check(odd, keys=["nope"])


def test_signed_zero_is_not_identical():
    fr = synth_frames(n=2, with_aov=False)
    fr.rgb[1] = fr.rgb[0]
    fr.rgb[0, 0, 0, 0] = 0.0
    fr.rgb[1, 0, 0, 0] = -0.0
    r = pos.roundtrip_check(fr)
    assert not r.identical and r.max_abs_diff == 0.0                             # 값은 같지만 바이트가 다르다


def test_near_palindromic_path_is_close_but_not_bit_identical(world, fwd):
    """근사 대칭(포즈가 1e-3 m 어긋남)은 비트 동일이 아니지만 8-bit 로도 거의 같다 -- '정확히 대칭' 조건이 왜 필요한지."""
    rt = pos.out_and_back(fwd)
    c2w = rt.c2w.copy()
    c2w[4, 0, 3] += 1e-3
    s = shot_from_arrays(c2w, rt.K.copy(), rt.focus_m.copy(), rt)
    assert not pos.is_palindromic_shot(s) and pos.is_palindromic_shot(s, atol=2e-3)
    with pytest.raises(ValueError):
        pos.verify_roundtrip(world, s)
    fr = A.render_shot_aov(world, s)
    r = pos.roundtrip_check(fr)
    assert not r.identical and r.pair_identical[0] is False and r.pair_identical[1] is True
    assert 0.0 < r.max_abs_diff < 0.1


def test_lsb_diff():
    a = np.zeros((2, 2, 3), np.float32)
    b = a.copy()
    assert pos.lsb_diff(a, b) == 0
    b[0, 0, 0] = 1.0 / 255.0
    assert pos.lsb_diff(a, b) == 1
    b[1, 1, 2] = 0.5
    assert pos.lsb_diff(a, b) == 128                                             # rint(0.5*255)=128 (짝수 반올림)
    b[1, 1, 2] = 0.0004                                                          # 0.1 LSB 는 0 으로 양자화
    b[0, 0, 0] = 0.0
    assert pos.lsb_diff(a, b) == 0
    assert pos.lsb_diff(np.ones(3, np.float32), np.zeros(3, np.float32)) == 255
    with pytest.raises(ValueError):
        pos.lsb_diff(a, b[:1])

"""RDR-09 프레임 지문(leaf)·Merkle 루트·툴체인 pin. 바이트 배치는 계약이므로 FND-03 det 로 독립 오라클을 만든다."""
import copy

import numpy as np
import pytest
from rdr09_helpers import synth_frames

from buc.core import contracts as C
from buc.core import det
from buc.render import pos


def oracle_leaf(frames, i, keys):
    parts = [b"buc.pos.frame.v1", det.hash_array(frames.rgb[i])]
    for k in sorted(keys):
        parts += [k, det.hash_array(frames.aov[k][i])]
    return det.hash_parts(*parts).hex()


def test_module_constants():
    assert pos.SCHEMA == "buc.pos/v1"
    assert pos.ROUNDTRIP_KEYS == ("alpha", "atom_id", "depth", "layer_id", "normal")
    assert issubclass(pos.PosError, ValueError)
    assert isinstance(pos.RENDERER_PIN, str) and pos.RENDERER_PIN


def test_toolchain_pin_shape_and_stability():
    a, b = pos.toolchain_pin(), pos.toolchain_pin()
    assert a == b and a is not b
    assert set(a) == {"renderer", "python", "numpy", "scipy", "torch", "blake3", "machine", "system"}
    assert all(isinstance(v, str) and v for v in a.values())
    assert a["renderer"] == pos.RENDERER_PIN and a["numpy"] == np.__version__
    a["numpy"] = "tampered"                                  # 반환 dict 를 고쳐도 다음 호출에는 영향이 없다
    assert pos.toolchain_pin()["numpy"] == np.__version__


def test_frame_leaf_matches_independent_oracle():
    fr = synth_frames(n=4)
    allk = sorted(fr.aov)
    for i in range(4):
        got = pos.frame_leaf(fr, i)
        assert got == oracle_leaf(fr, i, allk), i
        assert isinstance(got, str) and len(got) == 64 and got == got.lower()
    sub = ("depth", "atom_id")
    assert pos.frame_leaf(fr, 2, keys=sub) == oracle_leaf(fr, 2, sub)
    assert pos.frame_leaf(fr, 2, keys=["atom_id", "depth", "depth"]) == oracle_leaf(fr, 2, sub)     # 순서·중복 무관
    assert pos.frame_leaf(fr, 2, keys=()) == oracle_leaf(fr, 2, ())                                # rgb 만


def test_frame_leaf_ignores_dict_insertion_order_and_frame_index():
    fr = synth_frames(n=4)
    rev = C.Frames(rgb=fr.rgb, fps=fr.fps, aov={k: fr.aov[k] for k in reversed(list(fr.aov))})
    assert pos.frame_digests(rev) == pos.frame_digests(fr)
    same = synth_frames(n=4)
    same.rgb[3] = same.rgb[0]
    for k in same.aov:
        same.aov[k][3] = same.aov[k][0]
    d = pos.frame_digests(same)
    assert d[0] == d[3]                                       # 지문은 인덱스를 담지 않는다(왕복 비교의 전제)
    assert len(set(d)) == 3


def test_frame_leaf_is_bit_sensitive_in_rgb_and_every_aov():
    base = synth_frames(n=2)
    ref = pos.frame_leaf(base, 1)
    fr = copy.deepcopy(base)
    fr.rgb[1, 2, 3, 0] = np.nextafter(fr.rgb[1, 2, 3, 0], np.float32(2.0))     # 1 ulp
    assert pos.frame_leaf(fr, 1) != ref
    assert pos.frame_leaf(fr, 0) == pos.frame_leaf(base, 0)                     # 다른 프레임은 그대로
    for k, a in base.aov.items():
        fr = copy.deepcopy(base)
        v = fr.aov[k]
        if v.dtype == bool:
            v[1, 0, 0] = ~v[1, 0, 0]
        elif v.dtype.kind == "i":
            v[1, 0, 0] += 1
        else:
            v[1, 0, 0] = np.nextafter(v[1, 0, 0], np.float32(100.0))
        assert pos.frame_leaf(fr, 1) != ref, k
        assert pos.frame_leaf(fr, 1, keys=[x for x in base.aov if x != k]) == pos.frame_leaf(base, 1, keys=[x for x in base.aov if x != k]), k


def test_frame_leaf_distinguishes_signed_zero_and_dtype():
    fr = synth_frames(n=1)
    fr.rgb[0, 0, 0, 0] = 0.0
    a = pos.frame_leaf(fr, 0)
    fr.rgb[0, 0, 0, 0] = -0.0
    assert pos.frame_leaf(fr, 0) != a                          # 비트 동일은 값 동일(−0.0==+0.0)보다 엄격하다
    fr2 = synth_frames(n=1)
    fr3 = C.Frames(rgb=fr2.rgb, fps=24.0, aov={**fr2.aov, "atom_id": fr2.aov["atom_id"].astype(np.int32)})
    assert pos.frame_leaf(fr3, 0) != pos.frame_leaf(fr2, 0)


def test_frame_digests_and_errors():
    fr = synth_frames(n=3)
    d = pos.frame_digests(fr)
    assert isinstance(d, tuple) and len(d) == 3
    assert d == tuple(pos.frame_leaf(fr, i) for i in range(3))
    assert pos.frame_digests(fr, keys=["depth"]) == tuple(pos.frame_leaf(fr, i, keys=["depth"]) for i in range(3))
    with pytest.raises(ValueError):
        pos.frame_leaf(fr, 0, keys=["nope"])
    with pytest.raises(ValueError):
        pos.frame_digests(fr, keys=["depth", "nope"])
    with pytest.raises(IndexError):
        pos.frame_leaf(fr, 3)
    with pytest.raises(IndexError):
        pos.frame_leaf(fr, -1)
    bare = synth_frames(n=2, with_aov=False)
    assert pos.frame_leaf(bare, 0) == oracle_leaf(bare, 0, ())


def test_leaf_root_is_merkle_root_of_leaf_bytes():
    fr = synth_frames(n=5)
    d = pos.frame_digests(fr)
    assert pos.leaf_root(d) == det.merkle_root([bytes.fromhex(x) for x in d]).hex()
    assert pos.leaf_root(()) == det.EMPTY_ROOT.hex()
    assert pos.leaf_root(d[::-1]) != pos.leaf_root(d)           # 순서가 루트에 들어간다
    assert pos.leaf_root(d[:4]) != pos.leaf_root(d)
    one = pos.frame_digests(synth_frames(n=1))
    assert pos.leaf_root(one) == det.merkle_leaf(bytes.fromhex(one[0])).hex()
    with pytest.raises(ValueError):
        pos.leaf_root(["zz"])

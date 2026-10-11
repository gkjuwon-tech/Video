"""RDR-09 Proof-of-Shot: 만들기·직렬화·검증·변조 탐지·Merkle 포함 증명."""
import copy
import dataclasses
import json

import numpy as np
import pytest
from rdr09_helpers import forward_shot, shot_from_arrays, two_layer_world

from buc.core import contracts as C
from buc.core import det
from buc.render import aov as A
from buc.render import pos, quads
from buc.render.quads import RenderOpts

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

AOV_KEYS = ("alpha", "atom_id", "depth", "flow", "flow_valid", "layer_id", "normal")
DEFAULT_OPTS = {"bg": [0.0, 0.0, 0.0], "edge_px": 1.0, "z_tol_rel": 0.1, "alpha_eps": 0.001, "near_m": 0.001,
                "max_footprint_px": 8.0}


@pytest.fixture(scope="module")
def world():
    return two_layer_world()


@pytest.fixture(scope="module")
def shot():
    return pos.out_and_back(forward_shot(3))             # 5 프레임, 정확히 대칭


@pytest.fixture(scope="module")
def frames(world, shot):
    return A.render_shot_aov(world, shot)


@pytest.fixture(scope="module")
def proof(world, shot):
    return pos.make_proof(world, shot)


def _bad(v, *names):
    return all(not getattr(v, n) for n in names) and not v.ok


# ---------------------------------------------------------------- 만들기
def test_opts_dict():
    assert pos.opts_dict() == DEFAULT_OPTS == pos.opts_dict(None) == pos.opts_dict(RenderOpts())
    d = pos.opts_dict(RenderOpts(bg=(0.2, 0.1, 0.0), edge_px=2.0))
    assert d["bg"] == [0.2, 0.1, 0.0] and d["edge_px"] == 2.0 and set(d) == set(DEFAULT_OPTS)
    json.dumps(d, allow_nan=False)


def test_make_proof_fields(proof, world, shot, frames):
    p = proof
    assert isinstance(p, pos.ProofOfShot) and p.schema == pos.SCHEMA == "buc.pos/v1"
    assert p.shot_hash == shot.content_hash() and p.world_hash == world.content_hash()
    assert p.theta_hash == shot.theta.content_hash()
    assert p.toolchain == pos.toolchain_pin()
    assert p.opts == DEFAULT_OPTS
    assert (p.n_frames, p.width, p.height, p.fps) == (5, shot.width, shot.height, 24.0)
    assert p.aov_keys == AOV_KEYS and isinstance(p.aov_keys, tuple)
    assert isinstance(p.leaves, tuple) and p.leaves == pos.frame_digests(frames)
    assert len(set(p.leaves)) == 5                                        # 전체 aov(flow 포함) 지문은 프레임마다 다르다
    assert p.root == pos.leaf_root(p.leaves) == det.merkle_root([bytes.fromhex(x) for x in p.leaves]).hex()
    assert p.extra == {}
    # IR
    r = pos.ir_geo(frames.aov["atom_id"], A.observed_atoms(world))
    assert set(p.ir) == {"obs_hash", "ir_geo_mean", "ir_geo_max", "per_frame"}
    assert p.ir["obs_hash"] == A.observed_atoms(world).content_hash()
    assert p.ir["ir_geo_mean"] == r.mean and p.ir["ir_geo_max"] == r.max and p.ir["per_frame"] == list(r.per_frame)
    # 왕복 주장
    assert p.roundtrip == {"keys": list(pos.ROUNDTRIP_KEYS), "pairs": 2, "identical": True, "max_abs_diff": 0.0}
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.root = "0" * 64


def test_make_proof_is_deterministic_across_independent_objects(proof, shot):
    world2 = two_layer_world()                                            # 완전히 새로 만든 같은 월드
    shot2 = pos.out_and_back(forward_shot(3))
    p2 = pos.make_proof(world2, shot2)
    assert p2 == proof and p2.to_json() == proof.to_json() and p2.content_hash() == proof.content_hash()


def test_make_proof_non_palindromic_and_single_frame_have_no_roundtrip_claim(world):
    fwd = forward_shot(3)
    p = pos.make_proof(world, fwd)
    assert p.roundtrip is None and p.n_frames == 3
    assert pos.verify_proof(p, world, fwd).ok
    one = forward_shot(1)
    p1 = pos.make_proof(world, one)
    assert p1.roundtrip is None and p1.n_frames == 1 and len(p1.leaves) == 1
    assert p1.root == det.merkle_leaf(bytes.fromhex(p1.leaves[0])).hex()
    assert pos.verify_proof(p1, world, one).ok


def test_make_proof_with_given_frames_and_validation(world, shot, frames, proof):
    assert pos.make_proof(world, shot, frames=frames) == proof
    other = forward_shot(3)
    with pytest.raises(ValueError):
        pos.make_proof(world, other, frames=frames)                        # shot_hash 가 다른 프레임은 거부
    with pytest.raises(ValueError):
        pos.make_proof(two_layer_world(seed=41), shot, frames=frames)      # world_hash 가 다른 프레임은 거부
    plain = quads.render_shot(world, shot)                                  # atom_id 가 없는 Frames (RDR-02)
    with pytest.raises(pos.PosError):
        pos.make_proof(world, shot, frames=plain)
    with pytest.raises(TypeError):
        pos.make_proof(None, shot)
    with pytest.raises(TypeError):
        pos.make_proof(world, None)
    with pytest.raises(TypeError):
        pos.make_proof(world, shot, None)                                   # world, shot 외는 키워드 전용


def test_make_proof_extra(world, shot, proof):
    ex = {"soul_asset_hashes": ["ab" * 32], "plan": ["S2", "S8"], "still_polish": False, "n": 3, "nest": {"k": [1, 2.5]}}
    snapshot = copy.deepcopy(ex)
    p = pos.make_proof(world, shot, extra=ex)
    ex["plan"].append("S11")                                                # 호출 뒤 원본을 고쳐도 증명은 그대로
    assert p.extra == snapshot
    assert p.root == proof.root and p.leaves == proof.leaves and p.ir == proof.ir
    assert p.content_hash() != proof.content_hash()
    assert pos.verify_proof(p, world, shot).ok                              # extra 는 검증 대상이 아니다
    for bad in ({"x": object()}, {"x": float("nan")}, {"x": float("inf")}, {1: "non-str key"}, {"x": {"y": {3}}}):
        with pytest.raises(pos.PosError):
            pos.make_proof(world, shot, extra=bad)


# ---------------------------------------------------------------- 직렬화
def test_serialization_canonical_and_lossless(proof):
    d = proof.to_dict()
    assert set(d) == {f.name for f in dataclasses.fields(pos.ProofOfShot)}
    assert set(d) == {"schema", "shot_hash", "world_hash", "theta_hash", "toolchain", "opts", "n_frames", "width", "height",
                      "fps", "aov_keys", "leaves", "root", "ir", "roundtrip", "extra"}
    assert d["aov_keys"] == list(AOV_KEYS) and d["leaves"] == list(proof.leaves)
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    s = proof.to_json()
    assert s == json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    back = pos.ProofOfShot.from_json(s)
    assert back == proof and back.to_json() == s and isinstance(back.leaves, tuple) and isinstance(back.aov_keys, tuple)
    assert pos.ProofOfShot.from_dict(d) == proof
    assert d == proof.to_dict()                                             # to_dict 가 내부 상태를 새어 보내지 않는다
    d["leaves"].append("x")
    d["ir"]["per_frame"].append(9.0)
    assert proof.to_dict()["leaves"] == list(proof.leaves) and len(proof.ir["per_frame"]) == 5
    assert proof.content_hash() == det.hash_parts(b"buc.pos.v1", s.encode("utf-8")).hex()
    assert len(proof.content_hash()) == 64


def test_content_hash_changes_with_every_field(proof):
    h = proof.content_hash()
    flip = "0" * 64 if proof.root[0] != "0" else "1" * 64
    for name, val in (("shot_hash", flip), ("world_hash", flip), ("theta_hash", flip), ("root", flip),
                      ("toolchain", {**proof.toolchain, "numpy": "0"}), ("opts", {**proof.opts, "edge_px": 2.0}),
                      ("fps", 25.0), ("width", proof.width + 1), ("height", proof.height + 1),
                      ("ir", {**proof.ir, "ir_geo_max": 0.123}), ("roundtrip", None), ("extra", {"a": 1})):
        assert dataclasses.replace(proof, **{name: val}).content_hash() != h, name
    assert dataclasses.replace(proof, aov_keys=proof.aov_keys[:-1]).content_hash() != h
    assert dataclasses.replace(proof, leaves=proof.leaves[::-1]).content_hash() != h


def test_from_dict_is_strict(proof):
    d = proof.to_dict()
    pos.ProofOfShot.from_dict(d)
    bad = []
    bad.append({**d, "schema": "buc.pos/v0"})
    bad.append({**d, "surprise": 1})
    bad.append({k: v for k, v in d.items() if k != "root"})
    bad.append({**d, "root": d["root"].upper() if d["root"].upper() != d["root"] else "Z" * 64})
    bad.append({**d, "root": d["root"][:-1]})
    bad.append({**d, "leaves": d["leaves"][:-1]})                              # n_frames 와 길이 불일치
    bad.append({**d, "n_frames": 4})
    bad.append({**d, "leaves": d["leaves"][:-1] + ["xyz"]})
    for b in bad:
        with pytest.raises(pos.PosError):
            pos.ProofOfShot.from_dict(b)
    for notdict in (None, [], "x", 3):
        with pytest.raises(pos.PosError):
            pos.ProofOfShot.from_dict(notdict)
    with pytest.raises(pos.PosError):
        pos.ProofOfShot.from_json("{not json")
    with pytest.raises(pos.PosError):
        pos.ProofOfShot.from_json("[]")


def test_write_and_read_proof(tmp_path, proof):
    path = tmp_path / "a" / "b" / "shot.pos.json"
    out = pos.write_proof(proof, path)
    assert out == path and path.read_bytes() == (proof.to_json() + "\n").encode("utf-8")
    assert [p.name for p in path.parent.iterdir()] == ["shot.pos.json"]       # 임시 파일이 남지 않는다
    assert pos.read_proof(path) == proof
    pos.write_proof(dataclasses.replace(proof, extra={"v": 2}), str(path))     # str 경로, 덮어쓰기
    assert pos.read_proof(path).extra == {"v": 2}
    with pytest.raises(FileNotFoundError):
        pos.read_proof(tmp_path / "nope.json")
    junk = tmp_path / "junk.json"
    junk.write_text("not a proof", encoding="utf-8")
    with pytest.raises(pos.PosError):
        pos.read_proof(junk)


# ---------------------------------------------------------------- 검증
def test_verify_ok_by_rerendering(proof, world, shot):
    v = pos.verify_proof(proof, world, shot)
    assert isinstance(v, pos.Verdict) and v.ok is True and v.reason == "" and v.bad_frames == ()
    for name in ("world_ok", "shot_ok", "theta_ok", "toolchain_ok", "shape_ok", "root_ok", "frames_ok", "ir_ok", "roundtrip_ok"):
        assert getattr(v, name) is True, name


def test_verify_with_supplied_frames_and_independent_reconstruction(proof, world, shot, frames):
    assert pos.verify_proof(proof, world, shot, frames=frames).ok
    # 시청자 시나리오: 월드·샷을 직렬화해 다른 객체로 복원하고, 디스크의 증명 파일로 검증한다
    w2 = C.loads(C.dumps(world))
    s2 = C.loads(C.dumps(shot))
    p2 = pos.ProofOfShot.from_json(proof.to_json())
    v = pos.verify_proof(p2, w2, s2)
    assert v.ok and v.world_ok and v.shot_ok
    with pytest.raises(TypeError):
        pos.verify_proof(proof, world, shot, frames)                           # frames 는 키워드 전용


def test_verify_detects_changed_world(proof, shot):
    other = two_layer_world(seed=41)                                           # 앞 카드 텍스처만 다르다
    v = pos.verify_proof(proof, other, shot)
    assert _bad(v, "world_ok", "frames_ok") and v.shot_ok and v.shape_ok and v.root_ok
    assert len(v.bad_frames) >= 1 and list(v.bad_frames) == sorted(v.bad_frames) and "world_ok" in v.reason
    assert all(0 <= i < 5 for i in v.bad_frames)


def test_verify_detects_world_change_invisible_to_pixels_via_hash_and_ir(proof, shot):
    mask = np.ones((32, 48), bool)
    mask[:, :10] = False                                                       # 렌더에는 영향이 없지만 입력이 본 영역이 다르다
    other = two_layer_world(mask=mask)
    v = pos.verify_proof(proof, other, shot)
    assert v.frames_ok and v.bad_frames == ()                                  # 픽셀·aov 는 그대로
    assert not v.world_ok and not v.ir_ok and not v.ok                         # 하지만 world 해시와 IR 이 잡는다
    assert "ir_ok" in v.reason and "world_ok" in v.reason


def test_verify_detects_changed_shot(proof, world, shot):
    c2w = shot.c2w.copy()
    c2w[1, 0, 3] += 0.01
    c2w[3, 0, 3] += 0.01                                                       # 대칭도 유지(1<->3)
    moved = shot_from_arrays(c2w, shot.K.copy(), shot.focus_m.copy(), shot)
    v = pos.verify_proof(proof, world, moved)
    assert not v.shot_ok and not v.frames_ok and not v.ok and v.theta_ok and v.world_ok
    assert 1 in v.bad_frames and 3 in v.bad_frames and 4 not in v.bad_frames      # 0, 2 는 flow(i->i+1)가 달라져 같이 걸릴 수 있다
    # 프레임 수가 다른 샷
    v2 = pos.verify_proof(proof, world, forward_shot(3))
    assert not v2.shape_ok and not v2.shot_ok and not v2.ok and v2.bad_frames == () and not v2.frames_ok
    # theta 만 다른 샷
    th = C.Theta(f_stop=2.0)
    s3 = C.Shot(c2w=shot.c2w, t=shot.t, K=shot.K, focus_m=shot.focus_m, width=shot.width, height=shot.height, fps=shot.fps,
                theta=th, move=shot.move)
    v3 = pos.verify_proof(proof, world, s3)
    assert not v3.theta_ok and not v3.shot_ok and not v3.ok


@pytest.mark.parametrize("field,check", [("shot_hash", "shot_ok"), ("world_hash", "world_ok"), ("theta_hash", "theta_ok")])
def test_verify_detects_forged_hash_fields(proof, world, shot, field, check):
    forged = dataclasses.replace(proof, **{field: "f" * 64 if getattr(proof, field)[0] != "f" else "e" * 64})
    v = pos.verify_proof(forged, world, shot)
    assert getattr(v, check) is False and v.ok is False and check in v.reason
    assert v.frames_ok and v.root_ok and v.ir_ok and v.roundtrip_ok            # 나머지는 영향이 없다


def test_verify_detects_tampered_leaf_and_root(proof, world, shot):
    leaves = list(proof.leaves)
    leaves[2] = "0" * 64 if leaves[2][0] != "0" else "1" * 64
    forged = dataclasses.replace(proof, leaves=tuple(leaves))
    v = pos.verify_proof(forged, world, shot)
    assert not v.root_ok and not v.frames_ok and v.bad_frames == (2,) and not v.ok      # 루트와 잎이 모순, 렌더와도 모순
    # 잎과 루트를 함께 위조해 서로 일관되게 만들어도 재렌더가 잡는다
    consistent = dataclasses.replace(forged, root=pos.leaf_root(forged.leaves))
    v = pos.verify_proof(consistent, world, shot)
    assert v.root_ok and not v.frames_ok and v.bad_frames == (2,) and not v.ok
    # 루트만 바꾸면 잎은 맞지만 루트 모순
    v = pos.verify_proof(dataclasses.replace(proof, root="a" * 64 if proof.root[0] != "a" else "b" * 64), world, shot)
    assert not v.root_ok and v.frames_ok and v.bad_frames == () and not v.ok


def test_verify_detects_forged_ir_and_roundtrip_claims(proof, world, shot):
    ir = dict(proof.ir)
    ir["ir_geo_mean"] = 0.0 if ir["ir_geo_mean"] != 0.0 else 0.5
    v = pos.verify_proof(dataclasses.replace(proof, ir=ir), world, shot)
    assert not v.ir_ok and not v.ok and v.frames_ok and v.root_ok
    ir2 = dict(proof.ir)
    ir2["per_frame"] = [9.0] * 5
    assert not pos.verify_proof(dataclasses.replace(proof, ir=ir2), world, shot).ir_ok
    rt = dict(proof.roundtrip, identical=False)
    v = pos.verify_proof(dataclasses.replace(proof, roundtrip=rt), world, shot)
    assert not v.roundtrip_ok and not v.ok and v.frames_ok
    assert not pos.verify_proof(dataclasses.replace(proof, roundtrip=None), world, shot).roundtrip_ok   # 주장 삭제도 불일치
    fwd = forward_shot(3)
    pf = pos.make_proof(world, fwd)
    assert not pos.verify_proof(dataclasses.replace(pf, roundtrip=dict(proof.roundtrip)), world, fwd).roundtrip_ok  # 대칭 아닌데 주장


def test_toolchain_mismatch_is_informational(proof, world, shot):
    other = dataclasses.replace(proof, toolchain={**proof.toolchain, "machine": "other-arch", "numpy": "0.0.0"})
    v = pos.verify_proof(other, world, shot)
    assert v.toolchain_ok is False
    assert v.ok is True and v.reason == ""                                     # 바이트가 같으면 증명은 성립한다(툴체인이 달라도)


def test_verify_with_supplied_frames_detects_tampering_and_shape(proof, world, shot, frames):
    fr = copy.deepcopy(frames)
    fr.rgb[3, 4, 5, 2] = np.nextafter(fr.rgb[3, 4, 5, 2], np.float32(2.0))
    v = pos.verify_proof(proof, world, shot, frames=fr)
    assert not v.frames_ok and v.bad_frames == (3,) and not v.ok and v.shape_ok
    fr = copy.deepcopy(frames)
    fr.aov["depth"][0, 0, 0] += np.float32(1e-3)
    assert pos.verify_proof(proof, world, shot, frames=fr).bad_frames == (0,)
    short = C.Frames(rgb=frames.rgb[:4], fps=frames.fps, aov={k: v[:4] for k, v in frames.aov.items()},
                     shot_hash=frames.shot_hash, world_hash=frames.world_hash)
    v = pos.verify_proof(proof, world, shot, frames=short)
    assert not v.shape_ok and not v.ok and v.bad_frames == () and not v.frames_ok
    plain = quads.render_shot(world, shot)                                      # aov 키가 다른 프레임
    v = pos.verify_proof(proof, world, shot, frames=plain)
    assert not v.shape_ok and not v.ok


def test_opts_are_part_of_the_proof(world, shot, proof):
    opts = RenderOpts(bg=(0.2, 0.1, 0.0), edge_px=2.0, z_tol_rel=0.05)
    p = pos.make_proof(world, shot, opts=opts)
    assert p.opts == pos.opts_dict(opts) and p.opts["bg"] == [0.2, 0.1, 0.0]
    assert p.leaves != proof.leaves and p.root != proof.root                    # 옵션이 실제로 결과를 바꾼다
    assert pos.verify_proof(p, world, shot).ok                                  # 검증은 증명에 적힌 옵션으로 렌더한다
    assert pos.verify_proof(pos.ProofOfShot.from_json(p.to_json()), world, shot).ok
    default_frames = A.render_shot_aov(world, shot)
    v = pos.verify_proof(p, world, shot, frames=default_frames)
    assert not v.frames_ok and not v.ok
    forged = dataclasses.replace(p, opts=dict(DEFAULT_OPTS))                    # 옵션을 숨긴 위조
    v = pos.verify_proof(forged, world, shot)
    assert not v.frames_ok and not v.ok


# ---------------------------------------------------------------- Merkle 포함 증명
def test_inclusion_proofs(proof):
    n = proof.n_frames
    for i in range(n):
        path = pos.inclusion_proof(proof, i)
        assert isinstance(path, list) and all(isinstance(x, str) and len(x) == 64 for x in path)
        assert path == [x.hex() for x in det.merkle_proof([bytes.fromhex(x) for x in proof.leaves], i)]
        assert pos.verify_inclusion(proof, i, proof.leaves[i], path) is True
        assert pos.verify_inclusion(proof, i, proof.leaves[(i + 1) % n], path) is False        # 잎이 다르다
        assert pos.verify_inclusion(proof, (i + 1) % n, proof.leaves[i], path) is False        # 인덱스가 다르다
        if path:
            bad = [("0" * 64 if path[0][0] != "0" else "1" * 64)] + path[1:]
            assert pos.verify_inclusion(proof, i, proof.leaves[i], bad) is False
    assert pos.verify_inclusion(proof, 0, "zz", []) is False                   # 예외 없이 False
    assert pos.verify_inclusion(proof, 99, proof.leaves[0], []) is False
    assert pos.verify_inclusion(proof, 0, proof.leaves[0], ["nothex"]) is False
    with pytest.raises(IndexError):
        pos.inclusion_proof(proof, n)
    with pytest.raises(IndexError):
        pos.inclusion_proof(proof, -1)


def test_single_frame_can_be_checked_without_the_rest(proof, frames):
    """시청자가 프레임 하나만 받아 증명에 대해 확인한다: 프레임 -> 잎 -> 포함 증명 -> 공개된 루트."""
    i = 3
    one = C.Frames(rgb=frames.rgb[i:i + 1], fps=frames.fps, aov={k: v[i:i + 1] for k, v in frames.aov.items()})
    leaf = pos.frame_leaf(one, 0)
    assert leaf == proof.leaves[i]
    assert pos.verify_inclusion(proof, i, leaf, pos.inclusion_proof(proof, i))

"""WLD-01: 직렬화 왕복·결정론·변조 거부, World 이음매."""
import numpy as np
import pytest

from buc.core import contracts as C
from wld01_fixtures import H, W, make_obs, make_prov, make_strata, populate

from buc.world import strata as S


def _assert_same(a, b):
    assert a.merkle_root() == b.merkle_root()
    assert a.seed == b.seed and a.meta == b.meta and a.level == b.level and a.read_hash == b.read_hash
    assert a.obs == b.obs and a.theta_in == b.theta_in and a.prov == b.prov
    assert len(a.layers) == len(b.layers)
    for x, y in zip(a.layers, b.layers):
        assert x.rgba.tobytes() == y.rgba.tobytes() and x.depth.tobytes() == y.depth.tobytes()
    assert (a.far_field is None) == (b.far_field is None)
    if a.far_field is not None:
        assert a.far_field.rgb.tobytes() == b.far_field.rgb.tobytes()
    assert a.blocks == b.blocks and a.charts == b.charts
    for k in S.TSQ_FIELDS:
        assert a.tsq.columns[k].dtype == b.tsq.columns[k].dtype
        assert a.tsq.columns[k].tobytes() == b.tsq.columns[k].tobytes()


def test_roundtrip_bit_exact():
    st = populate(S, make_strata(S, seed=2 ** 127 + 99, meta={"note": "한옥", "x": [1, 2.5, None, True]}))
    # 특수 비트 패턴: -0.0, 비정규 수, inf 깊이(하늘)는 계약상 금지이므로 유한 값만
    st.tsq.columns["mu"][0] = np.array([-0.0, 1e-45, 3.4e38], dtype=np.float32)
    b = st.to_bytes()
    assert isinstance(b, bytes) and b.startswith(S.MAGIC)
    st2 = S.Strata.from_bytes(b)
    _assert_same(st, st2)
    assert st2.to_bytes() == b
    assert np.signbit(st2.tsq.columns["mu"][0, 0])


def test_roundtrip_minimal():
    st = S.Strata(obs=make_obs(), theta_in=C.Theta(), level="L1", read_hash="h", prov=C.Prov())
    st2 = S.Strata.from_bytes(st.to_bytes())
    _assert_same(st, st2)


def test_deterministic_bytes():
    a = populate(S, make_strata(S)).to_bytes()
    b = populate(S, make_strata(S)).to_bytes()
    assert a == b
    c = populate(S, make_strata(S, seed=1)).to_bytes()
    assert c != a


def test_no_pickle():
    b = populate(S, make_strata(S)).to_bytes()
    assert b"numpy.core.multiarray" not in b and b"\x80\x04\x95" not in b[:64]


def test_tamper_rejected():
    b = populate(S, make_strata(S)).to_bytes()
    n = len(b)
    positions = sorted(set([0, len(S.MAGIC), n - 1, n - 2, n // 2] + list(np.linspace(0, n - 1, 64).astype(int))))
    for p in positions:
        bad = bytearray(b)
        bad[p] ^= 0x01
        with pytest.raises(S.StrataError):
            S.Strata.from_bytes(bytes(bad))
    for cut in (0, 5, len(S.MAGIC), n // 3, n - 1):
        with pytest.raises(S.StrataError):
            S.Strata.from_bytes(b[:cut])
    with pytest.raises(S.StrataError):
        S.Strata.from_bytes(b + b"\x00")


def test_save_load(tmp_path):
    st = populate(S, make_strata(S))
    p = tmp_path / "w.strata"
    st.save(p)
    assert p.read_bytes() == st.to_bytes()
    _assert_same(st, S.Strata.load(p))


def test_to_world_contract():
    st = populate(S, make_strata(S, seed=2 ** 100 + 7))
    w = st.to_world()
    assert isinstance(w, C.World)
    w.validate()
    assert w.schema == C.WORLD_SCHEMA and w.level == "L1"
    assert w.meta["strata_root"] == st.merkle_root().hex()
    assert w.meta["strata_seed"] == f"{2 ** 100 + 7:032x}"
    assert {k: v for k, v in w.meta.items() if not k.startswith("strata_")} == st.meta
    assert len(w.layers) == 2
    for x, y in zip(w.layers, st.layers):
        assert x.rgba.tobytes() == y.rgba.tobytes() and x.depth.tobytes() == y.depth.tobytes()
    assert w.far_field.rgb.tobytes() == st.far_field.rgb.tobytes()
    # FND-02 JSON 왕복 후에도 같은 World
    w2 = C.loads(C.dumps(w))
    assert w2.content_hash() == w.content_hash()


def test_to_world_needs_layer():
    st = make_strata(S, layers=False)
    with pytest.raises(S.StrataError):
        st.to_world()


def test_from_world_roundtrip():
    st = make_strata(S, seed=2 ** 100 + 7)  # 블록·차트·TSQ 없음
    w = st.to_world()
    back = S.Strata.from_world(C.loads(C.dumps(w)))
    assert back.seed == st.seed and back.meta == st.meta
    assert back.merkle_root() == st.merkle_root()
    assert back.blocks == {} and back.charts == {} and back.tsq.n == 0
    assert back.to_world().content_hash() == w.content_hash()


def test_from_world_without_strata_keys():
    st = make_strata(S)
    w = st.to_world()
    plain = C.World(obs=w.obs, layers=w.layers, theta_in=w.theta_in, level=w.level, read_hash=w.read_hash,
                    prov=w.prov, far_field=None, meta={"lot": "near"})
    back = S.Strata.from_world(plain)
    assert back.seed == 0 and back.meta == {"lot": "near"} and back.far_field is None


def test_stage_s2_seam():
    """S2 이음매: ReadPacket -> to_world() 가 FND-02 check_stage_io 를 통과한다."""
    obs = make_obs()
    rng = np.random.default_rng(1)
    read = C.ReadPacket(image_lin=rng.random((H, W, 3)).astype(np.float32),
                        depth=np.linspace(1.0, 5.0, H * W, dtype=np.float32).reshape(H, W), obs=obs,
                        theta_in=C.Theta(focal_mm=50.0), input_hash="cd" * 32,
                        prov=C.Prov(taint=frozenset({"USER"}), ns="user_vault", mode="user"))
    st = make_strata(S)
    st.read_hash = read.content_hash()
    st.prov = C.Prov.merge(read.prov, make_prov())
    C.check_stage_io("S2", {"read": read}, {"world": st.to_world()})

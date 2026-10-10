"""WLD-01 테스트 공용 도우미: 작은 결정론 장면(6x8)과 Merkle 섹션 오라클."""
import json
import math

import numpy as np

from buc.core import contracts as C
from buc.core import det

H, W = 6, 8


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_obs():
    K = C.Theta().intrinsics(W, H)
    return C.Obs(K=K, R=rot_y(7.0), t=np.array([0.2, -0.1, 0.4]), width=W, height=H)


def make_prov():
    return C.Prov(taint=frozenset({"USER", "ENGINE"}), mode="mixed", roots=("r1",), ns="user_vault")


def make_layer(depth, alpha, seed, h=H, w=W):
    rng = np.random.default_rng(seed)
    rgba = rng.random((h, w, 4)).astype(np.float32)
    rgba[..., 3] = alpha
    return C.LDILayer(rgba=rgba, depth=np.full((h, w), depth, dtype=np.float32))


def make_far_field(seed=5, s=4):
    rng = np.random.default_rng(seed)
    return C.FarField(rgb=(rng.random((s, s, 3)) * 3.0).astype(np.float32))


def make_tsq_cols(n, seed=0):
    rng = np.random.default_rng(seed)
    nrm = rng.normal(size=(n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return dict(
        mu=rng.normal(size=(n, 3)).astype(np.float32),
        normal=nrm.astype(np.float32),
        s_uv=(rng.random((n, 2)) * 0.1 + 0.01).astype(np.float32),
        opacity=rng.random(n).astype(np.float32),
        edge_n=np.full(n, 2.0, dtype=np.float32),
        color=rng.random((n, 3)).astype(np.float32),
        level=np.full(n, 10, dtype=np.int8),
        sem=np.arange(n, dtype=np.uint16),
        own=np.full(n, 1, dtype=np.uint16),
        unc=np.full(n, 7, dtype=np.uint8),
    )


def make_strata(S, *, seed=12345, meta=None, layers=True, far=True):
    st = S.Strata(obs=make_obs(), theta_in=C.Theta(focal_mm=50.0), level="L1", read_hash="ab" * 32,
                  prov=make_prov(), seed=seed, meta=meta if meta is not None else {"lot": "near", "n": 3})
    if layers:
        st.add_layer(make_layer(1.0, 0.6, 1))
        st.add_layer(make_layer(2.5, 1.0, 2))
    if far:
        st.set_far_field(make_far_field())
    return st


def populate(S, st, order=0):
    """blocks·charts·tsq를 채운다. order에 따라 삽입 순서만 바꾼다(내용은 같다)."""
    st.add_tsq(**make_tsq_cols(5, seed=3))
    st.add_tsq(**make_tsq_cols(3, seed=4))
    blocks = [((10, 0, 0, 2), S.Block(state="committed", own=1, unc=0, tsq_start=0, tsq_stop=5)),
              ((10, -1, 0, 2), S.Block(state="proposed", own=4, unc=128, tsq_start=5, tsq_stop=8)),
              ((6, 3, -2, 0), S.Block(state="bedrock", own=6))]
    charts = [("c0", {"axis": "+z", "level": 10, "uv": [0, 0, 64, 64]}),
              ("wall.1", {"axis": "-x", "level": 11, "uv": [64, 0, 128, 32]})]
    if order:
        blocks = blocks[::-1]
        charts = charts[::-1]
    for k, b in blocks:
        st.set_block(k, b)
    for cid, rec in charts:
        st.set_chart(cid, rec)
    return st


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def oracle_entries(st):
    """계약서의 섹션 바이트 배치를 독립적으로 재현한다."""
    e = {
        "schema": C.WORLD_SCHEMA.encode("utf-8"),
        "level": st.level.encode("utf-8"),
        "read_hash": st.read_hash.encode("utf-8"),
        "obs": bytes.fromhex(st.obs.content_hash()),
        "theta_in": bytes.fromhex(st.theta_in.content_hash()),
        "prov": bytes.fromhex(st.prov.content_hash()),
        "seed": int(st.seed).to_bytes(16, "little"),
        "meta": det.hash_parts(b"buc.strata.meta.v0", canon(st.meta)),
    }
    for i, ly in enumerate(st.layers):
        e[f"layer/{i:04d}"] = det.hash_parts(b"buc.strata.layer.v0", det.hash_array(ly.rgba), det.hash_array(ly.depth))
    if st.far_field is not None:
        e["far_field"] = det.hash_array(st.far_field.rgb)
    for (l, i, j, k), b in st.blocks.items():
        e[f"block/{l}/{i}/{j}/{k}"] = det.hash_parts(b"buc.strata.block.v0", b.state, b.own, b.unc,
                                                      b.tsq_start, b.tsq_stop)
    for cid, rec in st.charts.items():
        e[f"chart/{cid}"] = det.hash_parts(b"buc.strata.chart.v0", canon(rec))
    for name in ("mu", "normal", "s_uv", "opacity", "edge_n", "color", "level", "sem", "own", "unc"):
        e[f"tsq/{name}"] = det.hash_array(st.tsq.columns[name])
    return e

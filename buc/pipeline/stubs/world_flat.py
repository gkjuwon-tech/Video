"""PRD-03 평면 세계 스텁(S2 ``{read}→{world}``): 깊이를 기하 경계로 n층에 나눠 상수 깊이 카드로 만든다.

계약서: ops/harness/contracts/PRD-03.md. 층은 앞→뒤(깊이 오름차순), 알파는 직선 0/1 로 화소마다 한 층만 1.
"""
import numpy as np

from buc.core import contracts as C
from buc.pipeline.stubs._common import check_keys, int_in

NAME = "flat_world"
VERSION = "0.1.0"
_PARAMS = ("n_layers",)


def flat_world(inputs: dict, ctx) -> dict:
    read = inputs["read"]
    p = check_keys(ctx.params, _PARAMS)
    n = int_in("n_layers", p.get("n_layers", 3), 1, 16)

    d = np.asarray(read.depth, dtype=np.float64)
    fin = d[np.isfinite(d)]
    if fin.size:
        dmin, dmax = float(fin.min()), float(fin.max())
    else:
        dmin = dmax = 1.0
    e = dmin * (dmax / dmin) ** (np.arange(n + 1, dtype=np.float64) / n)

    # 화소 구간: 내부 경계 e[1:-1] 에 대한 searchsorted. inf 와 최대 깊이는 마지막 구간.
    b = np.searchsorted(e[1:-1], d, side="right")
    b = np.clip(b, 0, n - 1)

    H, W = d.shape
    rgb = np.clip(np.asarray(read.image_lin, dtype=np.float32), 0.0, 1.0).astype(np.float32)
    layers = []
    for j in range(n):
        rgba = np.empty((H, W, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = (b == j).astype(np.float32)
        z = float(np.sqrt(e[j] * e[j + 1]))
        layers.append(C.LDILayer(rgba=rgba, depth=np.full((H, W), z, dtype=np.float32)))

    world = C.World(
        obs=read.obs,
        layers=tuple(layers),
        theta_in=read.theta_in,
        level=read.level,
        read_hash=read.content_hash(),
        prov=read.prov,
        far_field=None,
        meta={"stub": NAME, "n_layers": n, "bin_edges": [float(x) for x in e]},
    )
    return {"world": world}

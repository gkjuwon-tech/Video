"""PRD-03 고정 무브 스텁(S8 ``{world, read}→{shot}``): 관측 자세를 고정하고 광축을 따라 선형으로 민다.

계약서: ops/harness/contracts/PRD-03.md. world 입력은 무시한다. 첫 프레임은 관측 포즈와 정확히 같다.
"""
import numpy as np

from buc.core import contracts as C
from buc.pipeline.stubs._common import check_keys, finite_float, int_in

NAME = "fixed_move"
VERSION = "0.1.0"
_PARAMS = ("n_frames", "fps", "push_frac", "z_s")


def fixed_move(inputs: dict, ctx) -> dict:
    read = inputs["read"]
    p = check_keys(ctx.params, _PARAMS)
    n = int_in("n_frames", p.get("n_frames", 25), 1)
    theta = read.theta_in
    fps_raw = p.get("fps")
    if fps_raw is None:
        fps = float(theta.fps)
    else:
        fps = finite_float("fps", fps_raw, positive=True, hi=240.0)
    push = finite_float("push_frac", p.get("push_frac", 0.10), lo=-0.5, hi=0.5)
    zs_raw = p.get("z_s")
    if zs_raw is None:
        fin = np.asarray(read.depth, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        zs = float(np.median(fin)) if fin.size else 1.0   # 유한 깊이가 없으면 1.0 (가정)
    else:
        zs = finite_float("z_s", zs_raw, positive=True)

    obs = read.obs
    C0 = obs.c2w()
    tau = np.arange(n, dtype=np.float64) / (n - 1) if n > 1 else np.zeros(1)
    d = push * zs * tau                                   # 광축 방향 이동량 (N,)

    # c2w_k = C0 · T(0,0,d_k): 회전은 고정, 이동 = C0 의 위치 + d_k · (C0 의 3번째 열)
    c2w = np.repeat(C0[None], n, axis=0)
    c2w[:, :3, 3] = C0[:3, 3][None, :] + d[:, None] * C0[:3, 2][None, :]

    t = np.arange(n, dtype=np.float64) / fps
    K = np.repeat(np.asarray(obs.K, dtype=np.float64)[None], n, axis=0)
    focus = zs - d
    if float(theta.fps) == fps:
        th = theta
    else:
        th = C.Theta.from_dict({**theta.to_dict(), "fps": fps})

    shot = C.Shot(
        c2w=c2w,
        t=t,
        K=K,
        focus_m=focus,
        width=obs.width,
        height=obs.height,
        fps=fps,
        theta=th,
        move="stub_push_in",
        rig="locked",
        seed=int(ctx.seed),
    )
    return {"shot": shot}

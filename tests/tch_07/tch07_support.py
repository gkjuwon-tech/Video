"""TCH-07 공용 도우미: 후보 증거 생성기. 모든 K 지표를 같은 비율 d만큼 올리면 ES_L1 = ES_L2b = d 이다(가중치 합 1)."""
from buc.teacher import tct as T

Z = T.INCUMBENT
KLEIN = T.CHALLENGER
ZT4H = 1.345            # 0.5*1.0 + 0.35*1.6 + 0.15*1.9


def gates(**kw):
    base = dict(license_ok=True, fp16_nan=0, fp16_trials=10_000, fp16_ref_nan=0, fp16_ref_trials=2_000,
                mem_peak_gib=12.0, host_completed=True, nvf_r=0.40)
    base.update(kw)
    return T.GateInputs(**base)


def metrics(d=0.0):
    return {k: 1.0 + d for k in T.CRITERIA}


def cand(cid, d=0.0, t4h=ZT4H, *, sources=("R-real", "R-syn", "O-photo-dev"), ref=False, swap=False, nvf_s=None,
         ref_cost=None, **gate_kw):
    return T.Candidate(id=cid, metrics=metrics(d), t4h_world=t4h, gates=gates(**gate_kw), sources=tuple(sources),
                       ref_contract=ref, swap_ok=swap, nvf_s=nvf_s, ref_t4h_world=ref_cost)


def l2b(nvf, z_style=1.0, dphoto=0.1, tau=0.2):
    return T.L2bEvidence(nvf=nvf, z_style=z_style, delta_photo=dphoto, tau_photo=tau)


def ring(ids, nvf=0.40):
    return {i: l2b(nvf) for i in ids}

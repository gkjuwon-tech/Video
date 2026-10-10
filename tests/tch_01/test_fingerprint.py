"""TCH-01 모듈 지문과 바이트 계산."""
import torch
from torch import nn

from buc.core import det
from buc.teacher import zimage as Z
from tch01_fakes import FakeTransformer


def test_module_bytes_counts_parameters_only():
    m = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))                 # 12+3+6+2 = 23
    assert Z.module_bytes(m) == 23 * 4
    assert Z.module_bytes(m.to(torch.bfloat16)) == 23 * 2
    m.register_buffer("b", torch.zeros(1000))
    assert Z.module_bytes(m) == 23 * 2                                   # 버퍼는 세지 않는다
    assert Z.module_bytes(nn.Linear(2, 2).to("meta")) == 6 * 4


def test_fingerprint_matches_merkle_oracle_fp32():
    m = FakeTransformer(2, seed=1)
    exp = det.merkle_root_keyed({f"{k}:{v.dtype}": det.hash_array(v.detach().cpu().numpy())
                                 for k, v in m.state_dict().items()})
    fp = Z.module_fingerprint(m)
    assert isinstance(fp, bytes) and len(fp) == 32 and fp == exp


def test_fingerprint_bf16_uses_int16_bit_view():
    m = FakeTransformer(2, seed=1).to(torch.bfloat16)
    ent = {}
    for k, v in m.state_dict().items():
        arr = v.detach().cpu().contiguous().view(torch.int16).numpy()
        ent[f"{k}:{v.dtype}"] = det.hash_array(arr)
    assert Z.module_fingerprint(m) == det.merkle_root_keyed(ent)


def test_fingerprint_sensitivity_and_instance_independence():
    a, b = FakeTransformer(2, seed=1), FakeTransformer(2, seed=1)
    assert Z.module_fingerprint(a) == Z.module_fingerprint(b)             # 다른 인스턴스, 같은 가중치
    assert Z.module_fingerprint(a) != Z.module_fingerprint(FakeTransformer(2, seed=2))
    assert Z.module_fingerprint(a) != Z.module_fingerprint(FakeTransformer(3, seed=1))
    before = Z.module_fingerprint(a)
    with torch.no_grad():
        w = a.layers[0].weight
        w.view(-1)[0] = torch.nextafter(w.view(-1)[0], torch.tensor(float("inf")))   # 1 ulp
    assert Z.module_fingerprint(a) != before
    assert Z.module_fingerprint(b) != Z.module_fingerprint(b.to(torch.bfloat16))      # dtype도 지문의 일부


def test_fingerprint_bf16_single_bit_flip_detected():
    m = FakeTransformer(2, seed=1).to(torch.bfloat16)
    before = Z.module_fingerprint(m)
    with torch.no_grad():
        bits = m.layers[1].bias.view(torch.int16)
        bits[0] ^= 1
    assert Z.module_fingerprint(m) != before


def test_fingerprint_includes_buffers_and_ignores_requires_grad():
    m = nn.Linear(2, 2)
    base = Z.module_fingerprint(m)
    m.register_buffer("rope", torch.arange(4.0))
    withbuf = Z.module_fingerprint(m)
    assert withbuf != base
    m.requires_grad_(False)
    assert Z.module_fingerprint(m) == withbuf

"""TCH-03 고정 PCA-128 int8 기저: 적합 오라클, 양자화 오차 한계, 지문, 저장·적재, prov 검사."""

import numpy as np
import pytest
import torch

import tch03_fakes as F
from buc.gov import prov as P
from buc.teacher import taps as T


def lowrank(n=120, dim=24, k=6, seed=3, noise=0.0, layers=T.TAP_LAYERS):
    """고유값 간격이 큰 데이터: 주성분 부분공간이 유일하다."""
    out = {}
    for j, l in enumerate(layers):
        g = F.gen(seed + j)
        A, _ = torch.linalg.qr(torch.randn(dim, k, generator=g, dtype=torch.float64))
        s = torch.tensor([10.0, 8.0, 6.0, 4.0, 3.0, 2.5][:k], dtype=torch.float64)
        z = torch.randn(n, k, generator=g, dtype=torch.float64) * s
        x = z @ A.T + torch.randn(dim, generator=g, dtype=torch.float64) * 2.0
        if noise:
            x = x + noise * torch.randn(n, dim, generator=g, dtype=torch.float64)
        out[l] = x.float()
    return out


def test_constants():
    assert T.TAP_LAYERS == (8, 15, 23)
    assert T.PCA_DIM == 128
    assert T.INT8_MAX == 127


def test_fit_matches_float64_oracle():
    k = 6
    samples = lowrank(noise=0.05, k=k)
    b = T.fit_pca_basis(samples, prov=F.ENGINE, k=k)
    assert b.layers == T.TAP_LAYERS and b.k == k and b.dim == 24
    assert b.mean.shape == (3, 24) and b.comps.shape == (3, 24, k) and b.scale.shape == (3, k)
    for i, l in enumerate(b.layers):
        x = samples[l].double().numpy()
        mu = x.mean(0)
        assert np.allclose(b.mean[i].double().numpy(), mu, atol=1e-5)
        xc = x - mu
        w, V = np.linalg.eigh(xc.T @ xc / (len(x) - 1))
        V = V[:, np.argsort(w)[::-1][:k]]
        Vb = b.comps[i].double().numpy()
        # 부분공간 투영자 일치 (부호·회전 무관)
        assert np.allclose(Vb @ Vb.T, V @ V.T, atol=2e-4)
        # 직교정규
        assert np.allclose(Vb.T @ Vb, np.eye(k), atol=1e-5)
        # 부호 규약: 각 성분에서 절댓값이 최대인 원소는 양수
        for j in range(k):
            assert Vb[np.argmax(np.abs(Vb[:, j])), j] > 0
        # 성분은 분산 내림차순
        var = ((xc @ Vb) ** 2).mean(0)
        assert np.all(np.diff(var) <= 1e-6 * var[0])
        # scale = 적합 표본 투영의 채널별 최대 절댓값 / 127
        zmax = np.abs(xc @ Vb).max(0)
        assert np.allclose(b.scale[i].double().numpy(), zmax / 127.0, rtol=1e-3)


def test_fit_is_deterministic_and_fixed_tensors():
    s = lowrank()
    b1 = T.fit_pca_basis(s, prov=F.ENGINE, k=6)
    b2 = T.fit_pca_basis(s, prov=F.ENGINE, k=6)
    assert b1.fingerprint == b2.fingerprint and isinstance(b1.fingerprint, bytes) and len(b1.fingerprint) == 32
    for t in (b1.mean, b1.comps, b1.scale):
        assert t.dtype == torch.float32 and not t.requires_grad


def test_fit_default_k_is_128_and_size_ratio():
    g = F.gen(5)
    x = torch.randn(300, 256, generator=g) * torch.linspace(3, 0.1, 256)
    b = T.fit_pca_basis({8: x}, prov=F.ENGINE)
    assert b.k == 128 and b.layers == (8,)
    q = b.project(8, x[:10])
    assert q.shape == (10, 128) and q.dtype == torch.int8
    # int8 128채널은 bf16 은닉 3840차원 대비 60배 작다 (3840*2 / 128)
    assert (3840 * 2) / q.shape[-1] == 60


@pytest.mark.parametrize("bad", ["few_samples", "k_gt_dim", "ndim", "empty", "nonfinite", "dim_mismatch", "k_zero"])
def test_fit_input_validation(bad):
    ok = F.fit_samples()
    if bad == "few_samples":
        s = {8: ok[8][:F.K]}                      # n=k: 공분산 랭크가 k 미만일 수 있다 (n >= k+1 필요)
        exc, kw = T.TapSpecError, {}
    elif bad == "k_gt_dim":
        s, exc, kw = {8: ok[8]}, T.TapSpecError, {"k": F.D + 1}
    elif bad == "ndim":
        s, exc, kw = {8: ok[8][0]}, T.TapSpecError, {}
    elif bad == "empty":
        s, exc, kw = {}, T.TapSpecError, {}
    elif bad == "nonfinite":
        x = ok[8].clone()
        x[3, 4] = float("nan")
        s, exc, kw = {8: x}, T.TapNumericError, {}
    elif bad == "dim_mismatch":
        s, exc, kw = {8: ok[8], 15: ok[15][:, :16]}, T.TapSpecError, {}
    else:
        s, exc, kw = {8: ok[8]}, T.TapSpecError, {"k": 0}
    kw.setdefault("k", F.K)
    with pytest.raises(exc):
        T.fit_pca_basis(s, prov=F.ENGINE, **kw)


def test_fit_requires_prov_and_blocks_firewall_tags():
    s = F.fit_samples()
    with pytest.raises(TypeError):
        T.fit_pca_basis(s, k=F.K)                      # prov 키워드 필수
    with pytest.raises(TypeError):
        T.fit_pca_basis(s, prov=None, k=F.K)
    for tags in (["USER"], ["EXTGEN:test"], ["USER-DERIVED"], ["HOLDOUT"], ["COMP"], ["ENGINE", "USER"],
                 ["PUB-SEALED"], ["EVAL-ONLY"], ["NC"]):
        with pytest.raises(P.FirewallError):
            T.fit_pca_basis(s, prov=tags, k=F.K)
    for tags in (["ENGINE"], ["CAPTURE"], ["ASSET:CC0"], ["HUMAN"]):
        assert T.fit_pca_basis(s, prov=tags, k=F.K).k == F.K


def test_project_matches_oracle_and_error_bound():
    b = F.make_basis()
    x = torch.randn(5, 7, F.D, generator=F.gen(11)) * 1.5
    for l in b.layers:
        q = b.project(l, x)
        assert q.dtype == torch.int8 and q.shape == (5, 7, F.K)
        F.assert_quant_close(q, F.oracle_quant(b, l, x))
        # 범위 안(클램프 안 된) 성분의 역양자화 오차 <= scale/2
        i = b.layers.index(l)
        z = (x.double() - b.mean[i].double()) @ b.comps[i].double()
        deq = b.dequantize(l, q).double()
        inside = q.abs() < 127
        err = (deq - z).abs()
        bound = 0.5 * b.scale[i].double() * (1 + 1e-3) + 1e-5
        assert bool((err <= bound.expand_as(err))[inside].all())


def test_project_never_emits_minus_128_and_saturates():
    b = F.make_basis()
    x = torch.randn(64, F.D, generator=F.gen(2)) * 1e4
    q = b.project(8, x)
    assert int(q.min()) >= -127 and int(q.max()) <= 127
    assert int(q.max()) == 127 and int(q.min()) == -127


def test_project_rejects_bad_inputs():
    b = F.make_basis()
    with pytest.raises(T.TapSpecError):
        b.project(9, torch.zeros(2, F.D))            # 기저에 없는 층
    with pytest.raises(T.TapSpecError):
        b.project(8, torch.zeros(2, F.D + 1))        # 차원 불일치
    x = torch.zeros(2, F.D)
    x[0, 0] = float("inf")
    with pytest.raises(T.TapNumericError):
        b.project(8, x)
    with pytest.raises(T.TapSpecError):
        b.dequantize(8, torch.zeros(2, F.K, dtype=torch.float32))   # int8만


def test_project_accepts_bf16_and_equals_float32_path():
    b = F.make_basis()
    x = (torch.randn(4, F.D, generator=F.gen(3))).to(torch.bfloat16)
    assert torch.equal(b.project(8, x), b.project(8, x.float()))


def test_reconstruct_rank_k_data_within_quantization_bound():
    k = 6
    s = lowrank(k=k, noise=0.0)
    b = T.fit_pca_basis(s, prov=F.ENGINE, k=k)
    for i, l in enumerate(b.layers):
        x = s[l]
        q = b.project(l, x)
        rec = b.reconstruct(l, q)
        err = (rec.double() - x.double()).norm(dim=-1)
        bound = 0.5 * b.scale[i].double().norm() * (1 + 1e-3) + 2e-3
        assert float(err.max()) <= float(bound)
        assert float((x.double() - rec.double()).abs().max()) < 0.2 * float(x.double().abs().max())


def test_fingerprint_sensitivity():
    b = F.make_basis()
    base = b.fingerprint
    for name in ("mean", "comps", "scale"):
        kw = dict(layers=b.layers, mean=b.mean.clone(), comps=b.comps.clone(), scale=b.scale.clone())
        t = kw[name]
        t.view(-1)[0] = t.view(-1)[0] * 1.001 + 1e-3
        assert T.PCABasis(**kw).fingerprint != base, name
    same = T.PCABasis(b.layers, b.mean.clone(), b.comps.clone(), b.scale.clone(), meta={"anything": 1})
    assert same.fingerprint == base                      # meta는 지문에 들어가지 않는다
    relabeled = T.PCABasis((8, 15, 22), b.mean.clone(), b.comps.clone(), b.scale.clone())
    assert relabeled.fingerprint != base                 # 층 번호도 지문에 들어간다


def test_verify_detects_in_place_tampering():
    b = F.make_basis()
    b.verify()
    b.comps[0, 0, 0] += 1e-3
    with pytest.raises(T.TapBasisError):
        b.verify()


def test_constructor_validation():
    b = F.make_basis()
    ok = dict(layers=b.layers, mean=b.mean, comps=b.comps, scale=b.scale)
    with pytest.raises(T.TapSpecError):
        T.PCABasis(**{**ok, "layers": (8, 8, 15)})
    with pytest.raises(T.TapSpecError):
        T.PCABasis(**{**ok, "layers": (8, 15)})                       # 개수 불일치
    with pytest.raises(T.TapSpecError):
        T.PCABasis(**{**ok, "scale": b.scale[:, :-1]})
    with pytest.raises(T.TapSpecError):
        T.PCABasis(**{**ok, "scale": torch.zeros_like(b.scale)})      # scale > 0
    bad = b.comps.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(T.TapNumericError):
        T.PCABasis(**{**ok, "comps": bad})
    # 생성자는 입력을 복사한다: 원본을 바꿔도 지문은 그대로
    m = b.mean.clone()
    nb = T.PCABasis(b.layers, m, b.comps, b.scale)
    fp = nb.fingerprint
    m += 1.0
    nb.verify()
    assert nb.fingerprint == fp


def test_save_load_roundtrip(tmp_path):
    b = T.fit_pca_basis(F.fit_samples(), prov=F.ENGINE, k=F.K, meta={"note": "x", "n": 3})
    p = tmp_path / "basis.pt"
    b.save(p)
    assert not list(tmp_path.glob("*.tmp"))                # 원자적 저장: 임시 파일이 남지 않는다
    b2 = T.PCABasis.load(p)
    assert b2.fingerprint == b.fingerprint and b2.layers == b.layers and b2.meta == b.meta
    for a, c in ((b.mean, b2.mean), (b.comps, b2.comps), (b.scale, b2.scale)):
        assert torch.equal(a, c)
    x = torch.randn(3, F.D, generator=F.gen(1))
    assert torch.equal(b.project(8, x), b2.project(8, x))


def test_load_rejects_missing_corrupt_and_tampered(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.PCABasis.load(tmp_path / "nope.pt")
    b = F.make_basis()
    p = tmp_path / "b.pt"
    b.save(p)
    raw = p.read_bytes()
    trunc = tmp_path / "trunc.pt"
    trunc.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(T.TapBasisError):
        T.PCABasis.load(trunc)
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(T.TapBasisError):
        T.PCABasis.load(junk)
    # 파일은 유효하지만 텐서만 바꿔치기: 저장된 지문과 불일치
    d = torch.load(p, weights_only=True)
    d["comps"] = d["comps"].clone()
    d["comps"][0, 0, 0] += 0.01
    tam = tmp_path / "tam.pt"
    torch.save(d, tam)
    with pytest.raises(T.TapBasisError):
        T.PCABasis.load(tam)

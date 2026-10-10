"""TCH-01 VAE 코덱: 스케일/시프트 산식, 결정론 encode, 왕복, NaN 거부."""
import pytest
import torch

from buc.teacher import zimage as Z
from tch01_fakes import FakeVae, block_image, raw_latent


def _codec(**kw):
    vae = FakeVae(**kw)
    return vae, Z.VaeCodec(vae, scaling_factor=vae.config.scaling_factor, shift_factor=vae.config.shift_factor)


def test_encode_applies_shift_then_scale_exactly_and_uses_mode_not_sample():
    vae, codec = _codec(scaling=0.5, shift=0.25)
    x = block_image()
    z = codec.encode(x)                                # FakeVae의 .sample()은 AssertionError를 낸다
    assert z.dtype == torch.float32 and tuple(z.shape) == (2, 16, 4, 5)
    exp = (raw_latent(vae, x) - 0.25) * 0.5
    assert torch.allclose(z, exp, atol=1e-6, rtol=1e-6)
    assert torch.equal(z, codec.encode(x))             # 반복 호출 비트 동일


def test_decode_inverts_scale_and_shift_and_roundtrips_block_images():
    vae, codec = _codec(scaling=0.3611, shift=0.1159)
    x = block_image(scale=0.9)
    y = codec.decode(codec.encode(x))
    assert y.dtype == torch.float32 and y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)
    z = codec.encode(x)
    y2 = codec.decode(z, clamp=False)
    assert torch.allclose(y2, x, atol=1e-4)


def test_decode_clamp_and_out_dtype():
    _, codec = _codec()
    x = block_image(scale=1.0) * 2.0                   # 범위를 벗어난 입력 (|x| 최대 약 2)
    z = codec.encode(x)
    hi = codec.decode(z, clamp=False)
    lo = codec.decode(z, clamp=True)
    assert hi.abs().max() > 1.5 and lo.abs().max() <= 1.0
    assert torch.equal(lo, hi.clamp(-1, 1))
    assert codec.decode(z, out_dtype=torch.bfloat16).dtype == torch.bfloat16
    assert codec.encode(x, out_dtype=torch.bfloat16).dtype == torch.bfloat16


def test_shift_none_equivalent_to_zero_and_no_grad_in_outputs():
    vae = FakeVae(shift=0.0)
    codec = Z.VaeCodec(vae, scaling_factor=2.0)        # shift_factor 기본 0.0
    x = block_image()
    z = codec.encode(x)
    assert torch.allclose(z, raw_latent(vae, x) * 2.0, atol=1e-6)
    assert vae.mix.requires_grad                        # 파라미터는 그라드를 요구하지만
    assert not z.requires_grad and not codec.decode(z).requires_grad   # 출력은 아니다


def test_input_cast_to_vae_dtype_and_batch_independence():
    vae = FakeVae().to(torch.bfloat16)
    codec = Z.VaeCodec(vae, scaling_factor=0.5, shift_factor=0.25)
    x = block_image(b=3)
    z = codec.encode(x)                                 # float32 입력이 bf16 VAE로 캐스트되어 들어간다
    assert z.dtype == torch.float32
    z1 = codec.encode(x[1:2])
    assert torch.equal(z[1:2], z1)                      # 배치 위치와 무관


@pytest.mark.parametrize("shape", [(1, 3, 12, 16), (1, 3, 16, 12), (1, 4, 16, 16), (3, 16, 16), (1, 1, 3, 16, 16)])
def test_encode_rejects_bad_shapes(shape):
    _, codec = _codec()
    with pytest.raises(ValueError):
        codec.encode(torch.zeros(shape))


def test_encode_rejects_nonfinite_input_and_decode_rejects_bad_latents():
    _, codec = _codec()
    x = block_image()
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        codec.encode(x)
    with pytest.raises(ValueError):
        codec.decode(torch.zeros(1, 15, 4, 4))                    # 채널 불일치
    with pytest.raises(ValueError):
        codec.decode(torch.zeros(16, 4, 4))                       # ndim
    z = torch.zeros(1, 16, 4, 4)
    z[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        codec.decode(z)


def test_nonfinite_decode_output_raises_numeric_error():
    vae = FakeVae(nan_decode=True)
    codec = Z.VaeCodec(vae, scaling_factor=0.5, shift_factor=0.25)
    with pytest.raises(Z.TeacherNumericError):
        codec.decode(torch.zeros(1, 16, 4, 4))

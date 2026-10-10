"""GPU 부록(E0, 미실행): T4/CUDA + tensorrt/nunchaku + 교사 가중치가 있어야 한다."""
import pytest
import torch

from buc.teacher import quant as Q

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_trt_int8_engine_builds(tmp_path):
    if not Q.tensorrt_available():
        pytest.skip("tensorrt 없음")
    from tch10_toys import MLP, calib
    m = MLP().eval()
    onnx = tmp_path / "mlp.onnx"
    torch.onnx.export(m, (calib(4).batches(1)[0],), str(onnx))
    h = Q.build_trt_engine(onnx, tmp_path / "mlp.plan", calib=calib(16))
    assert isinstance(h, str) and len(h) == 64 and (tmp_path / "mlp.plan").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_w4a4_emulation_runs_on_cuda():
    from tch10_toys import MLP, calib, mlp_inputs, rel
    m = MLP()
    ref = m(mlp_inputs(2, 5)).detach()
    Q.quantize_model(m, Q.QuantSpec("w4a4"), calib(8), weight_prov={"ENGINE"})
    y = m.cuda()(mlp_inputs(2, 5).cuda()).cpu()
    assert rel(y, ref) <= 0.12

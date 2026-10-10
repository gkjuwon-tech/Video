"""RD-05 GPU 테스트: RD-07 출력처럼 CUDA 텐서로 온 defocus·깊이를 받아들인다. 이 환경에서는 실행하지 못했다."""
import pytest

import rd05_helpers as H


@pytest.mark.gpu
def test_cuda_tensor_inputs_match_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.read import blur_est as B
    c, z = H.scene(3.0, 2.0, seed=40, noise=0.03)
    a = B.estimate_defocus(c, z, f_px=H.F_PX, width_px=H.W)
    tc = torch.from_numpy(c).float().cuda()[None, None]
    tz = torch.from_numpy(z).float().cuda()[None, None]
    b = B.read_blur(defocus=tc, depth_m=tz, f_px=H.F_PX, width_px=H.W).defocus
    assert abs(b.z_f_m / a.z_f_m - 1) <= 1e-3 and abs(b.n_eq / a.n_eq - 1) <= 1e-3

// RDR-03 TSQ (Textured Soft Quad) CUDA rasterizer: forward + backward, float32.
//
// Contract: ops/harness/contracts/RDR-03.md. Loaded lazily by buc/render/tsq.py::load_extension() via
// torch.utils.cpp_extension.load. The Python side culls, sorts (camera z ascending, stable) and computes the
// per-TSQ camera-space features G (differentiable in torch); this kernel does the per-pixel work and returns
//   out (P, 8) = [rgb_acc (3), alpha, dnum, nsum (3)]
// where rgb_acc = sum w_i c_i, alpha = sum w_i, dnum = sum w_i z_i, nsum = sum w_i nw_i, w_i = A_i prod_{j<i}(1-A_j).
// Background, depth = dnum/alpha and normal normalisation are done in torch.
//
// G layout (M, 22): 0-2 t_u, 3-5 t_v, 6-8 n (camera frame), 9 numer = n.p, 10 t_u.p, 11 t_v.p, 12 s_u, 13 s_v,
// 14 opacity, 15 edge n (>= 2), 16 cu, 17 cv (centre projection, px), 18 zc, 19-21 world normal (camera facing).
// pages (M, 64, 3): page[j*8 + i] = row j (v), column i (u). Software bilinear, clamp addressing.
//
// Semantics (same as the torch reference):
//   D = ((x - cx - skew*yd)/fx, yd = (y - cy)/fy, 1), Dn = n.D, ok = |Dn| > 1e-6 && t = numer/Dn > 0
//   u = (t * t_u.D - t_u.p)/s_u, v likewise; E = ok ? 0.5(|u|^n + |v|^n) : inf; E_lp = |x - c|^2
//   A = min(0.999, o exp(-min(E, E_lp))), A < 1/255 -> 0; colour = bilinear(page, ok ? (u, v) : (0, 0))
//   z = ok ? t : zc.
// |u|^n = exp(min(n log|u|, 80)), 0 at u = 0 (gradient 0 there).
//
// Backward uses a forward-order second pass per pixel: with F = g . out (total), the suffix sum
//   S_after_i = F - sum_{k<=i} w_k f_k, dL/dA_i = T_i f_i - S_after_i / (1 - A_i).
// Gradients are accumulated with atomicAdd into gG (M, 22) and gpages (M, 64, 3).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TSQ_NG 22
#define TSQ_PAGE 8
#define TSQ_ALPHA_MAX 0.999f
#define TSQ_ALPHA_MIN (1.0f / 255.0f)
#define TSQ_DENOM_EPS 1e-6f
#define TSQ_LOGPOW_MAX 80.0f

struct Cam {
  float fx, fy, skew, cx, cy;
};

// |x|^n with clamp; also returns d/dx and d/dn (0 where clamped or x == 0).
__device__ __forceinline__ void spow_grad(float x, float n, float &p, float &dpdx, float &dpdn) {
  float a = fabsf(x);
  if (a > 0.f) {
    float la = logf(a);
    float L = n * la;
    if (L < TSQ_LOGPOW_MAX) {
      p = expf(L);
      dpdx = p * n / a * (x > 0.f ? 1.f : -1.f);
      dpdn = p * la;
    } else {
      p = expf(TSQ_LOGPOW_MAX);
      dpdx = 0.f;
      dpdn = 0.f;
    }
  } else {
    p = 0.f;
    dpdx = 0.f;
    dpdn = 0.f;
  }
}

__device__ __forceinline__ float spow(float x, float n) {
  float a = fabsf(x);
  if (!(a > 0.f)) return 0.f;
  return expf(fminf(n * logf(a), TSQ_LOGPOW_MAX));
}

// Bilinear setup: texel coordinate, base texel, fractions, and whether the coordinate is inside the clamp range.
struct Bil {
  int i00;
  float fx, fy;
  bool inx, iny;
};

__device__ __forceinline__ Bil bil_setup(float u, float v) {
  Bil b;
  float x = 4.f * (u + 1.f) - 0.5f;
  float y = 4.f * (v + 1.f) - 0.5f;
  b.inx = (x >= 0.f) && (x <= (float)(TSQ_PAGE - 1));
  b.iny = (y >= 0.f) && (y <= (float)(TSQ_PAGE - 1));
  x = fminf(fmaxf(x, 0.f), (float)(TSQ_PAGE - 1));
  y = fminf(fmaxf(y, 0.f), (float)(TSQ_PAGE - 1));
  float x0 = fminf(floorf(x), (float)(TSQ_PAGE - 2));
  float y0 = fminf(floorf(y), (float)(TSQ_PAGE - 2));
  b.fx = x - x0;
  b.fy = y - y0;
  b.i00 = (int)y0 * TSQ_PAGE + (int)x0;
  return b;
}

struct Frag {
  bool ok, useE, clamped;
  float A, t, u, v, E, d2, ex, z;
  float pu, pv, dpu, dpv, dnu, dnv;  // |u|^n, |v|^n and their derivatives
  float tuD, tvD, Dn;
};

// Evaluate TSQ i at pixel (px, py) with ray direction D. Returns false when the fragment does not contribute.
__device__ __forceinline__ bool eval_frag(const float *__restrict__ g, float px, float py, float Dx, float Dy,
                                          bool need_grad, Frag &f) {
  const float Dz = 1.f;
  f.tuD = g[0] * Dx + g[1] * Dy + g[2] * Dz;
  f.tvD = g[3] * Dx + g[4] * Dy + g[5] * Dz;
  f.Dn = g[6] * Dx + g[7] * Dy + g[8] * Dz;
  f.ok = fabsf(f.Dn) > TSQ_DENOM_EPS;
  f.t = g[9] / (f.ok ? f.Dn : 1.f);
  f.ok = f.ok && (f.t > 0.f);
  f.u = (f.t * f.tuD - g[10]) / g[12];
  f.v = (f.t * f.tvD - g[11]) / g[13];
  float dx = px - g[16], dy = py - g[17];
  f.d2 = dx * dx + dy * dy;
  float n = g[15];
  if (f.ok) {
    if (need_grad) {
      spow_grad(f.u, n, f.pu, f.dpu, f.dnu);
      spow_grad(f.v, n, f.pv, f.dpv, f.dnv);
    } else {
      f.pu = spow(f.u, n);
      f.pv = spow(f.v, n);
    }
    f.E = 0.5f * (f.pu + f.pv);
  } else {
    f.E = INFINITY;
  }
  f.useE = f.ok && (f.E < f.d2);
  float q = f.useE ? f.E : f.d2;
  f.ex = expf(-q);
  float a = g[14] * f.ex;
  f.clamped = a > TSQ_ALPHA_MAX;
  f.A = f.clamped ? TSQ_ALPHA_MAX : a;
  if (f.A < TSQ_ALPHA_MIN) return false;
  f.z = f.ok ? f.t : g[18];
  return true;
}

__device__ __forceinline__ void ray_dir(int pix, int W, const Cam &c, float &px, float &py, float &Dx, float &Dy) {
  px = (float)(pix % W);
  py = (float)(pix / W);
  Dy = (py - c.cy) / c.fy;
  Dx = (px - c.cx - c.skew * Dy) / c.fx;
}

__global__ void tsq_forward_kernel(const float *__restrict__ G, const float *__restrict__ pages, int M, int W,
                                   int H, Cam cam, float *__restrict__ out) {
  int pix = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= W * H) return;
  float px, py, Dx, Dy;
  ray_dir(pix, W, cam, px, py, Dx, Dy);
  float T = 1.f;
  float r = 0.f, gg = 0.f, b = 0.f, al = 0.f, dn = 0.f, n0 = 0.f, n1 = 0.f, n2 = 0.f;
  for (int i = 0; i < M; ++i) {
    const float *g = G + (size_t)i * TSQ_NG;
    Frag f;
    if (!eval_frag(g, px, py, Dx, Dy, false, f)) continue;
    Bil bl = bil_setup(f.ok ? f.u : 0.f, f.ok ? f.v : 0.f);
    const float *pg = pages + (size_t)i * TSQ_PAGE * TSQ_PAGE * 3;
    float w00 = (1.f - bl.fx) * (1.f - bl.fy), w01 = bl.fx * (1.f - bl.fy);
    float w10 = (1.f - bl.fx) * bl.fy, w11 = bl.fx * bl.fy;
    const float *c00 = pg + 3 * bl.i00, *c01 = c00 + 3, *c10 = c00 + 3 * TSQ_PAGE, *c11 = c10 + 3;
    float w = f.A * T;
    r += w * (w00 * c00[0] + w01 * c01[0] + w10 * c10[0] + w11 * c11[0]);
    gg += w * (w00 * c00[1] + w01 * c01[1] + w10 * c10[1] + w11 * c11[1]);
    b += w * (w00 * c00[2] + w01 * c01[2] + w10 * c10[2] + w11 * c11[2]);
    al += w;
    dn += w * f.z;
    n0 += w * g[19];
    n1 += w * g[20];
    n2 += w * g[21];
    T *= (1.f - f.A);
  }
  float *o = out + (size_t)pix * 8;
  o[0] = r;
  o[1] = gg;
  o[2] = b;
  o[3] = al;
  o[4] = dn;
  o[5] = n0;
  o[6] = n1;
  o[7] = n2;
}

__global__ void tsq_backward_kernel(const float *__restrict__ G, const float *__restrict__ pages, int M, int W,
                                    int H, Cam cam, const float *__restrict__ out, const float *__restrict__ gout,
                                    float *__restrict__ gG, float *__restrict__ gP) {
  int pix = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= W * H) return;
  float px, py, Dx, Dy;
  ray_dir(pix, W, cam, px, py, Dx, Dy);
  const float *go = gout + (size_t)pix * 8;
  const float *o = out + (size_t)pix * 8;
  float gr = go[0], ggn = go[1], gb = go[2], ga = go[3], gd = go[4], gn0 = go[5], gn1 = go[6], gn2 = go[7];
  float Ftot = gr * o[0] + ggn * o[1] + gb * o[2] + ga * o[3] + gd * o[4] + gn0 * o[5] + gn1 * o[6] + gn2 * o[7];
  if (gr == 0.f && ggn == 0.f && gb == 0.f && ga == 0.f && gd == 0.f && gn0 == 0.f && gn1 == 0.f && gn2 == 0.f)
    return;
  float T = 1.f;
  float Sup = 0.f;
  for (int i = 0; i < M; ++i) {
    const float *g = G + (size_t)i * TSQ_NG;
    Frag f;
    if (!eval_frag(g, px, py, Dx, Dy, true, f)) continue;
    float uu = f.ok ? f.u : 0.f, vv = f.ok ? f.v : 0.f;
    Bil bl = bil_setup(uu, vv);
    const float *pg = pages + (size_t)i * TSQ_PAGE * TSQ_PAGE * 3;
    float w00 = (1.f - bl.fx) * (1.f - bl.fy), w01 = bl.fx * (1.f - bl.fy);
    float w10 = (1.f - bl.fx) * bl.fy, w11 = bl.fx * bl.fy;
    const float *c00 = pg + 3 * bl.i00, *c01 = c00 + 3, *c10 = c00 + 3 * TSQ_PAGE, *c11 = c10 + 3;
    float col[3], dcdx[3], dcdy[3];
    for (int k = 0; k < 3; ++k) {
      col[k] = w00 * c00[k] + w01 * c01[k] + w10 * c10[k] + w11 * c11[k];
      dcdx[k] = (1.f - bl.fy) * (c01[k] - c00[k]) + bl.fy * (c11[k] - c10[k]);
      dcdy[k] = (1.f - bl.fx) * (c10[k] - c00[k]) + bl.fx * (c11[k] - c01[k]);
    }
    float w = f.A * T;
    float fi = gr * col[0] + ggn * col[1] + gb * col[2] + ga + gd * f.z + gn0 * g[19] + gn1 * g[20] + gn2 * g[21];
    Sup += w * fi;
    float Safter = Ftot - Sup;
    float dA = T * fi - Safter / (1.f - f.A);
    T *= (1.f - f.A);

    float *gg_i = gG + (size_t)i * TSQ_NG;
    // colour -> pages
    float gc[3] = {w * gr, w * ggn, w * gb};
    float *gp = gP + (size_t)i * TSQ_PAGE * TSQ_PAGE * 3;
    for (int k = 0; k < 3; ++k) {
      atomicAdd(gp + 3 * bl.i00 + k, gc[k] * w00);
      atomicAdd(gp + 3 * (bl.i00 + 1) + k, gc[k] * w01);
      atomicAdd(gp + 3 * (bl.i00 + TSQ_PAGE) + k, gc[k] * w10);
      atomicAdd(gp + 3 * (bl.i00 + TSQ_PAGE + 1) + k, gc[k] * w11);
    }
    // normals
    atomicAdd(gg_i + 19, w * gn0);
    atomicAdd(gg_i + 20, w * gn1);
    atomicAdd(gg_i + 21, w * gn2);
    // alpha
    float du = 0.f, dv = 0.f, dt = 0.f;
    if (!f.clamped) {
      atomicAdd(gg_i + 14, dA * f.ex);
      float dq = -dA * f.A;
      if (f.useE) {
        // E = 0.5(|u|^n + |v|^n)
        du += 0.5f * dq * f.dpu;
        dv += 0.5f * dq * f.dpv;
        atomicAdd(gg_i + 15, 0.5f * dq * (f.dnu + f.dnv));
      } else {
        float dx = px - g[16], dy = py - g[17];
        atomicAdd(gg_i + 16, -2.f * dx * dq);
        atomicAdd(gg_i + 17, -2.f * dy * dq);
      }
    }
    if (f.ok) {
      // colour -> (u, v): x = 4(u+1) - 0.5
      float gcx = gc[0] * dcdx[0] + gc[1] * dcdx[1] + gc[2] * dcdx[2];
      float gcy = gc[0] * dcdy[0] + gc[1] * dcdy[1] + gc[2] * dcdy[2];
      if (bl.inx) du += 4.f * gcx;
      if (bl.iny) dv += 4.f * gcy;
      dt += w * gd;
      // u = (t tuD - tu_p)/su, v = (t tvD - tv_p)/sv
      float isu = 1.f / g[12], isv = 1.f / g[13];
      dt += du * f.tuD * isu + dv * f.tvD * isv;
      float a_u = du * f.t * isu, a_v = dv * f.t * isv;
      atomicAdd(gg_i + 0, a_u * Dx);
      atomicAdd(gg_i + 1, a_u * Dy);
      atomicAdd(gg_i + 2, a_u);
      atomicAdd(gg_i + 3, a_v * Dx);
      atomicAdd(gg_i + 4, a_v * Dy);
      atomicAdd(gg_i + 5, a_v);
      atomicAdd(gg_i + 10, -du * isu);
      atomicAdd(gg_i + 11, -dv * isv);
      atomicAdd(gg_i + 12, -du * f.u * isu);
      atomicAdd(gg_i + 13, -dv * f.v * isv);
      // t = numer / Dn
      float iDn = 1.f / f.Dn;
      atomicAdd(gg_i + 9, dt * iDn);
      float a_n = -dt * f.t * iDn;
      atomicAdd(gg_i + 6, a_n * Dx);
      atomicAdd(gg_i + 7, a_n * Dy);
      atomicAdd(gg_i + 8, a_n);
    } else {
      atomicAdd(gg_i + 18, w * gd);
    }
  }
}

static Cam make_cam(double fx, double fy, double skew, double cx, double cy) {
  Cam c;
  c.fx = (float)fx;
  c.fy = (float)fy;
  c.skew = (float)skew;
  c.cx = (float)cx;
  c.cy = (float)cy;
  return c;
}

static void check_inputs(const torch::Tensor &G, const torch::Tensor &pages) {
  TORCH_CHECK(G.is_cuda() && pages.is_cuda(), "tsq: CUDA tensors required");
  TORCH_CHECK(G.scalar_type() == torch::kFloat32 && pages.scalar_type() == torch::kFloat32, "tsq: float32 only");
  TORCH_CHECK(G.dim() == 2 && G.size(1) == TSQ_NG, "tsq: G must be (M, 22)");
  TORCH_CHECK(pages.dim() == 3 && pages.size(0) == G.size(0) && pages.size(1) == TSQ_PAGE * TSQ_PAGE &&
                  pages.size(2) == 3,
              "tsq: pages must be (M, 64, 3)");
}

std::vector<torch::Tensor> tsq_forward(torch::Tensor G, torch::Tensor pages, int64_t W, int64_t H, double fx,
                                       double fy, double skew, double cx, double cy) {
  check_inputs(G, pages);
  G = G.contiguous();
  pages = pages.contiguous();
  const int M = (int)G.size(0);
  const int P = (int)(W * H);
  auto out = torch::zeros({P, 8}, G.options());
  if (P > 0) {
    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tsq_forward_kernel<<<blocks, threads, 0, stream>>>(G.data_ptr<float>(), pages.data_ptr<float>(), M, (int)W,
                                                       (int)H, make_cam(fx, fy, skew, cx, cy),
                                                       out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out};
}

std::vector<torch::Tensor> tsq_backward(torch::Tensor G, torch::Tensor pages, torch::Tensor out,
                                        torch::Tensor gout, int64_t W, int64_t H, double fx, double fy,
                                        double skew, double cx, double cy) {
  check_inputs(G, pages);
  G = G.contiguous();
  pages = pages.contiguous();
  out = out.contiguous();
  gout = gout.contiguous();
  const int M = (int)G.size(0);
  const int P = (int)(W * H);
  TORCH_CHECK(out.size(0) == P && out.size(1) == 8 && gout.size(0) == P && gout.size(1) == 8,
              "tsq: out/gout must be (P, 8)");
  auto gG = torch::zeros_like(G);
  auto gP = torch::zeros_like(pages);
  if (P > 0 && M > 0) {
    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tsq_backward_kernel<<<blocks, threads, 0, stream>>>(
        G.data_ptr<float>(), pages.data_ptr<float>(), M, (int)W, (int)H, make_cam(fx, fy, skew, cx, cy),
        out.data_ptr<float>(), gout.data_ptr<float>(), gG.data_ptr<float>(), gP.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gG, gP};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tsq_forward, "TSQ rasterizer forward (CUDA, float32)");
  m.def("backward", &tsq_backward, "TSQ rasterizer backward (CUDA, float32)");
}

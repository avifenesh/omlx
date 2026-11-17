"""Prototype: Splash's q8q prefill attention design, ported to an MLX JIT kernel over oMLX's affine8 KV cache.

Design credit: Splash (https://github.com/incoai/splash, Apache-2.0), runtime/metal/kernels/common/paged_attention_tile.h
(splash_paged_attention_tile, splash_attention_page_softmax) and kernels/prefill/paged_attention.metal (prefill_attention_q8q_split).
This is an independent re-implementation of that design for oMLX's layout; no Splash code is compiled or linked.

One threadgroup (256 threads, 8 simdgroups) owns one KV head and ROWS=8 query rows: M = 8 rows x G=6 query heads of the
GQA group = 48 fused rows, one contiguous [M, D] block (queries are re-laid out KV-head-major [Hk][qL][G][D]).
Per 32-key page: S = Qi8 . Ki8 (Metal 4 MPP matmul2d, int8 x int8 -> int32, exact), stored to threadgroup memory;
four lanes per fused row apply key scale x query scale x softmax scale, run the online softmax (rescale of the
running output only when some row max grew), fold the value scale into P and store P as bf16; then
O += P(bf16) . Vi8 (MPP, fp32 accumulator, relaxed precision) with V read in place from the cache.
One history split (2048-row chunks), output normalized in the kernel (fp32 [Hk][qL][G][D]).

oMLX affine8 cache: Hadamard-rotated, signed int8 codes (uint32 words viewed as int8), one scale per (token, KV head),
token-major K and V [Hk][T][D]. PVT=True takes a dim-major V copy [Hk][D][T] (Splash's layout; an engine layout change).
"""
import sys
from functools import lru_cache

import mlx.core as mx

HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

namespace q8q {

template <int G, int ROWS, int D, int N, bool QK8, bool PVT, bool SKIP>
METAL_FUNC void tile(
    const device int8_t* qc, const device bfloat* qb, const device float* qs,
    const device int8_t* kc, const device float* ks,
    const device int8_t* vc, const device float* vs,
    device float* out, int qL, int T, int Tpad, float sm_scale,
    threadgroup float* scores_raw, threadgroup bfloat* probs,
    threadgroup float* row_max, threadgroup float* row_sum, threadgroup float* prev,
    threadgroup atomic_uint* rescale, uint kvh, uint tidx, uint tid) {
  constexpr int M = ROWS * G;
  constexpr int LPR = 4;   // lanes per fused row
  constexpr int TPL = 8;   // tokens per lane
  static_assert(N == LPR * TPL, "four lanes of eight tokens span a page");
  static_assert(LPR * M <= 256, "softmax lanes fit the threadgroup");
  using Score = metal::conditional_t<QK8, int, float>;
  using QE = metal::conditional_t<QK8, int8_t, bfloat>;
  threadgroup Score* scores = reinterpret_cast<threadgroup Score*>(scores_raw);

  const int committed = T - qL;                 // the chunk is the last qL keys
  const int first_pos = committed + int(tidx) * ROWS;
  const int visible = first_pos + ROWS;
  const int pages = (visible + N - 1) / N;
  const ulong qoff = (ulong(kvh) * ulong(qL) + ulong(tidx) * ROWS) * G;

  // Softmax lane geometry and its row's score scale (query scale folded in).
  const uint fr = tid / LPR;
  const uint column = (tid % LPR) * TPL;
  const bool lane = tid < uint(LPR * M);
  float score_scale = sm_scale;
  int limit = 0;
  if (lane) {
    if (QK8) score_scale *= qs[qoff + fr];
    limit = first_pos + int(fr / G) + 1;        // causal end of this row
  }
  if (tid < uint(M)) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device QE* qptr;
  if constexpr (QK8) qptr = (device QE*)(qc + qoff * D);
  else qptr = (device QE*)(qb + qoff * D);
  using QT = tensor<device QE, dextents<int, 2>, tensor_inline>;
  using KT = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
  using ST = tensor<threadgroup Score, dextents<int, 2>, tensor_inline>;
  using PT = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
  using OT = tensor<device float, dextents<int, 2>, tensor_inline>;
  QT qt(qptr, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  KT kt((device int8_t*)(kc + ulong(kvh) * ulong(Tpad) * D), dextents<int, 2>{D, Tpad}, array<int, 2>{1, D});
  KT vt((device int8_t*)(vc + ulong(kvh) * ulong(Tpad) * D), dextents<int, 2>{PVT ? Tpad : D, PVT ? D : Tpad},
        array<int, 2>{1, PVT ? Tpad : D});
  ST st(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  PT pt(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});

  constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, PVT, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<8>> qk;
  matmul2d<pv_desc, execution_simdgroups<8>> pv;
  constexpr int VE0 = PVT ? N : D;
  constexpr int VE1 = PVT ? D : N;
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vt.template slice<VE0, VE1>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool full = uint(running.get_capacity()) * 256u == uint(M * D);
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (full || running.is_valid_element(i)) running[i] = 0.0f;

  const device float* ksp = ks + ulong(kvh) * ulong(Tpad);
  const device float* vsp = vs + ulong(kvh) * ulong(Tpad);
  for (int page = 0; page < pages; ++page) {
    const int t0 = page * N;
    auto kp = kt.template slice<D, N>(0, t0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), Score>();
    qk.run(q0, kp, sc);
    sc.store(st.template slice<N, M>(0, 0));
    if (tid == 0) atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane) {
      const int token = t0 + int(column);
      threadgroup const vec<Score, 4>* s4 = reinterpret_cast<threadgroup const vec<Score, 4>*>(scores + fr * N + column);
      float4 lo = float4(s4[0]), hi = float4(s4[1]);
      const device float4* k4 = reinterpret_cast<const device float4*>(ksp + token);
      lo *= k4[0]; hi *= k4[1];
      lo *= score_scale; hi *= score_scale;
      float s[TPL] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float lmax = -INFINITY;
      for (int j = 0; j < TPL; ++j) { s[j] = token + j < limit ? s[j] : -INFINITY; lmax = max(lmax, s[j]); }
      lmax = max(lmax, simd_shuffle_xor(lmax, 1));
      lmax = max(lmax, simd_shuffle_xor(lmax, 2));
      const float pmax = row_max[fr];
      const float nmax = max(pmax, lmax);
      float p[TPL]; float lsum = 0.0f;
      for (int j = 0; j < TPL; ++j) { p[j] = token + j < limit ? fast::exp(s[j] - nmax) : 0.0f; lsum += p[j]; }
      lsum += simd_shuffle_xor(lsum, 1);
      lsum += simd_shuffle_xor(lsum, 2);
      if (column == 0) {
        const float scale = (nmax == -INFINITY || nmax == pmax) ? 1.0f : fast::exp(pmax - nmax);
        prev[fr] = scale;
        row_sum[fr] = row_sum[fr] * scale + lsum;
        row_max[fr] = nmax;
        if (scale != 1.0f) atomic_store_explicit(rescale, 1u, memory_order_relaxed);
      }
      const device float4* v4 = reinterpret_cast<const device float4*>(vsp + token);
      const float4 vl = v4[0], vh = v4[1];
      const float4 a(token + 0 < limit ? p[0] * vl.x : 0.0f, token + 1 < limit ? p[1] * vl.y : 0.0f,
                     token + 2 < limit ? p[2] * vl.z : 0.0f, token + 3 < limit ? p[3] * vl.w : 0.0f);
      const float4 b(token + 4 < limit ? p[4] * vh.x : 0.0f, token + 5 < limit ? p[5] * vh.y : 0.0f,
                     token + 6 < limit ? p[6] * vh.z : 0.0f, token + 7 < limit ? p[7] * vh.w : 0.0f);
      threadgroup bfloat4* p4 = reinterpret_cast<threadgroup bfloat4*>(probs + fr * N + column);
      p4[0] = bfloat4(a); p4[1] = bfloat4(b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!SKIP || atomic_load_explicit(rescale, memory_order_relaxed)) {
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!full && !running.is_valid_element(i)) continue;
        auto c = running.get_multidimensional_index(i);
        running[i] *= prev[c[1]];
      }
    }
    auto vp = vt.template slice<VE0, VE1>(PVT ? t0 : 0, PVT ? 0 : t0);
    pv.run(p0, vp, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!full && !running.is_valid_element(i)) continue;
    auto c = running.get_multidimensional_index(i);
    running[i] /= row_sum[c[1]];
  }
  OT ot(out + qoff * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(ot.template slice<D, M>(0, 0));
}
}  // namespace q8q
"""

SOURCE = r"""
  constexpr int M = {ROWS} * {G};
  alignas(16) threadgroup float scores[M * 32];
  alignas(16) threadgroup bfloat probs[M * 32];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float prev[M];
  threadgroup atomic_uint rescale;
  q8q::tile<{G}, {ROWS}, 256, 32, {QK8}, {PVT}, {SKIP}>(
      qc, reinterpret_cast<const device bfloat*>(qb), qs, kc, ks, vc, vs, out,
      prm[0], prm[1], prm[2], scl[0], scores, probs, row_max, row_sum, prev, &rescale,
      threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, thread_index_in_threadgroup);
"""

INPUTS = ["qc", "qb", "qs", "kc", "ks", "vc", "vs", "prm", "scl"]


def _b(x):
    return "true" if x else "false"


@lru_cache(maxsize=None)
def kernel(g=6, rows=8, qk8=True, pvt=False, skip=True):
    src = SOURCE
    for k, v in dict(G=g, ROWS=rows, QK8=_b(qk8), PVT=_b(pvt), SKIP=_b(skip)).items():
        src = src.replace("{" + k + "}", str(v))
    name = f"q8q_v1_g{g}_r{rows}_qk8{int(qk8)}_pvt{int(pvt)}_skip{int(skip)}"
    return mx.fast.metal_kernel(name=name, input_names=INPUTS, output_names=["out"], header=HEADER,
                                source=src, ensure_row_contiguous=True)


_PH = {}


def _ph(dtype):
    if dtype not in _PH:
        _PH[dtype] = mx.zeros((64,), dtype=dtype)
    return _PH[dtype]


def prepare_kv(codes_u32, scales, pvt=False):
    """codes_u32 [1, Hk, T, D/4] uint32 (oMLX affine8 indices), scales [1, Hk, T] -> kernel operands
    (int8 codes [Hk, Tpad, D] or dim-major [Hk, D, Tpad] when pvt, fp32 scales [Hk, Tpad])."""
    _, Hk, T, W = codes_u32.shape
    D = W * 4
    c = codes_u32.view(mx.int8).reshape(Hk, T, D)
    s = scales.astype(mx.float32).reshape(Hk, T)
    Tpad = (T + 31) // 32 * 32
    if Tpad != T:
        c = mx.pad(c, [(0, 0), (0, Tpad - T), (0, 0)])
        s = mx.pad(s, [(0, 0), (0, Tpad - T)])
    if pvt:
        c = mx.contiguous(c.transpose(0, 2, 1))
    return c, s, T


def quant_queries(rq, Hk):
    """Rotated queries [1, H, qL, D] bf16 -> KV-head-major int8 codes [Hk, qL, G, D] + fp32 scales [Hk, qL, G]."""
    _, H, qL, D = rq.shape
    G = H // Hk
    x = rq.reshape(Hk, G, qL, D).transpose(0, 2, 1, 3).astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(x), axis=-1) / 127.0, 1e-30)
    c = mx.clip(mx.round(x / s[..., None]), -127, 127).astype(mx.int8)
    return c, s


def run(rq, k, v, scale, *, qk8=True, pvt=False, skip=True, rows=8):
    """rq [1, H, qL, D] rotated bf16 queries (the chunk = the last qL keys); k, v from prepare_kv (v with the same pvt).
    Returns the attention output in the rotated space, [1, H, qL, D] fp32 (inverse rotation is the caller's)."""
    _, H, qL, D = rq.shape
    kc, ks, T = k
    vc, vs, _ = v
    Hk = kc.shape[0]
    Tpad = ks.shape[1]
    G = H // Hk
    assert D == 256 and qL % rows == 0 and qL <= T
    if qk8:
        qc, qs = quant_queries(rq, Hk)
        qb = _ph(mx.bfloat16)
    else:
        qb = mx.contiguous(rq.reshape(Hk, G, qL, D).transpose(0, 2, 1, 3))
        qc, qs = _ph(mx.int8), _ph(mx.float32)
    prm = mx.array([qL, T, Tpad, 0, 0, 0, 0, 0], dtype=mx.int32)
    scl = mx.array([scale] * 8, dtype=mx.float32)
    (out,) = kernel(G, rows, qk8, pvt, skip)(
        inputs=[qc, qb, qs, kc, ks, vc, vs, prm, scl],
        grid=(256 * Hk, qL // rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(Hk, qL, G, D)], output_dtypes=[mx.float32])
    return out.transpose(0, 2, 1, 3).reshape(1, H, qL, D)

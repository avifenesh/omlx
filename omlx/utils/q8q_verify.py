# SPDX-License-Identifier: Apache-2.0
"""Speculative verify attention over the affine8 KV cache: a few causal rows at the end of a long history.

Design credit: Splash (https://github.com/incoai/splash, Apache-2.0), runtime/metal/kernels/common/paged_attention_tile.h
(the paged attention tile on the tensor ops). This is an independent re-implementation of that design for oMLX's
layout, cut to the verify shape; no Splash code is compiled or linked. Batch 1, n <= 8 query rows per tile (the last
n keys), head dim 256, the whole GQA group of a KV head fused into one tile of M = n x G rows (padded to a multiple
of 8; M <= 48 keeps the accumulator in registers). More rows run as more tiles.

The few rows leave the GPU idle unless the history is split, so one threadgroup (256 threads, 8 simdgroups) owns one
(KV head, history split). Per 64-key page: S = Q(bf16) . Ki8 (MPP matmul2d, fp32), key scale and softmax scale folded,
online softmax, P x value scale stored as bf16, O += P . Vi8 (MPP, fp32). K and V are read in place from the cache
buffers (the strided views update_and_fetch returns). Each split writes an unnormalized fp32 partial with its
(max, sum); a combine kernel merges the splits into fp32 [1, H, n, D] in the rotated space. The caller rotates the
output back before casting, which keeps the error at the level of the dequantize routes.
"""

from functools import cache

import mlx.core as mx

HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

PAGE = 64
TILE_ROWS = 8

MAIN = r"""
  constexpr int G = {G};
  constexpr int M = {M};
  constexpr int D = 256;
  constexpr int N = {N};
  constexpr int LPR = {LPR};
  constexpr int TPL = N / LPR;
  static_assert(TPL % 4 == 0 && LPR * M <= 256, "lanes");
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probs[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float prev[M];
  threadgroup atomic_uint rescale;

  const uint tid = thread_index_in_threadgroup;
  const int split = int(threadgroup_position_in_grid.x);
  const uint kvh = threadgroup_position_in_grid.y;
  const int n = prm[0];
  const int T = prm[1];
  const int S = prm[2];
  const int shift = prm[3];        // rows after this tile (tiles of a longer verify)
  const float sm_scale = scl[0];
  const int visible = T - shift;
  const int pages = (visible + N - 1) / N;
  const int lo = (pages * split) / S;
  const int hi = (pages * (split + 1)) / S;

  const device int8_t* kbase = reinterpret_cast<const device int8_t*>(kc) + ulong(kvh) * ulong(kc_strides[1]) * 4;
  const device int8_t* vbase = reinterpret_cast<const device int8_t*>(vc) + ulong(kvh) * ulong(vc_strides[1]) * 4;
  const device {NT}* ksp = ks + ulong(kvh) * ulong(ks_strides[1]);
  const device {NT}* vsp = vs + ulong(kvh) * ulong(vs_strides[1]);
  const ulong qoff = ulong(kvh) * M;
  // Token strides from the views (4 int8 codes per uint32 word); norms may be strided too.
  const int kts = int(kc_strides[2]) * 4, vts = int(vc_strides[2]) * 4;
  const int kns = int(ks_strides[2]), vns = int(vs_strides[2]);

  const uint fr = tid / LPR;
  const uint column = (tid % LPR) * TPL;
  const bool lane = tid < uint(LPR * M);
  int limit = 0;
  if (lane) {
    const int r = int(fr) / G;
    limit = r < n ? visible - n + r + 1 : 0;
  }
  if (tid < uint(M)) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using QT = tensor<device bfloat, dextents<int, 2>, tensor_inline>;
  using KT = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
  using ST = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
  using PT = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
  using OT = tensor<device float, dextents<int, 2>, tensor_inline>;
  QT qt((device bfloat*)(qc + qoff * D), dextents<int, 2>{D, M}, array<int, 2>{1, D});
  KT kt((device int8_t*)kbase, dextents<int, 2>{D, T}, array<int, 2>{1, kts});
  KT vt((device int8_t*)vbase, dextents<int, 2>{D, T}, array<int, 2>{1, vts});
  ST st(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  PT pt(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<8>> qk;
  matmul2d<pv_desc, execution_simdgroups<8>> pv;
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vt.template slice<D, N>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool full = uint(running.get_capacity()) * 256u == uint(M * D);
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (full || running.is_valid_element(i)) running[i] = 0.0f;

  for (int page = lo; page < hi; ++page) {
    const int t0 = page * N;
    auto kp = kt.template slice<D, N>(0, t0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    qk.run(q0, kp, sc);
    sc.store(st.template slice<N, M>(0, 0));
    if (tid == 0) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane) {
      const int token = t0 + int(column);
      threadgroup const float4* s4 = reinterpret_cast<threadgroup const float4*>(scores + fr * N + column);
      float s[TPL];
      for (int q = 0; q < TPL / 4; ++q) {
        float4 x = s4[q] * sm_scale;
        s[4 * q] = x.x; s[4 * q + 1] = x.y; s[4 * q + 2] = x.z; s[4 * q + 3] = x.w;
      }
      float lmax = -INFINITY;
      for (int j = 0; j < TPL; ++j) {
        s[j] = token + j < limit ? s[j] * float(ksp[(token + j) * kns]) : -INFINITY;
        lmax = max(lmax, s[j]);
      }
      for (int o = 1; o < LPR; o <<= 1) lmax = max(lmax, simd_shuffle_xor(lmax, ushort(o)));
      const float pmax = row_max[fr];
      const float nmax = max(pmax, lmax);
      float p[TPL]; float lsum = 0.0f;
      for (int j = 0; j < TPL; ++j) { p[j] = token + j < limit ? fast::exp(s[j] - nmax) : 0.0f; lsum += p[j]; }
      for (int o = 1; o < LPR; o <<= 1) lsum += simd_shuffle_xor(lsum, ushort(o));
      if (column == 0) {
        const float sc2 = (nmax == -INFINITY || nmax == pmax) ? 1.0f : fast::exp(pmax - nmax);
        prev[fr] = sc2;
        row_sum[fr] = row_sum[fr] * sc2 + lsum;
        row_max[fr] = nmax;
        if (sc2 != 1.0f) atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
      }
      threadgroup vec<bfloat, 4>* p4 = reinterpret_cast<threadgroup vec<bfloat, 4>*>(probs + fr * N + column);
      for (int q = 0; q < TPL / 4; ++q) {
        const int b = 4 * q;
        float4 pp;
        for (int e = 0; e < 4; ++e) pp[e] = token + b + e < limit ? p[b + e] * float(vsp[(token + b + e) * vns]) : 0.0f;
        p4[q] = vec<bfloat, 4>(pp);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!full && !running.is_valid_element(i)) continue;
        auto c = running.get_multidimensional_index(i);
        running[i] *= prev[c[1]];
      }
    }
    auto vp = vt.template slice<D, N>(0, t0);
    pv.run(p0, vp, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong base = (ulong(split) * {HK} + kvh) * M;
  OT ot(out + base * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(ot.template slice<D, M>(0, 0));
  if (tid < uint(M)) { sm[base + tid] = row_max[tid]; sl[base + tid] = row_sum[tid]; }
"""

# Re-lay rotated bf16 queries [1, H, n, D] out KV-head-major [Hk][M][D] (fused row r * G + g);
# rows >= n are zeros. One simdgroup per (fused row, KV head).
LAYOUT = r"""
  constexpr int G = {G};
  constexpr int M = {M};
  constexpr int D = 256;
  const uint lane = thread_index_in_simdgroup;
  const uint fr = threadgroup_position_in_grid.y;
  const uint kvh = threadgroup_position_in_grid.z;
  const int n = prm[0];
  const int r = int(fr) / G, g = int(fr) % G;
  device bfloat* dst = qc + (ulong(kvh) * M + fr) * D + lane * 8;
  if (r >= n) {
    for (int j = 0; j < 8; ++j) dst[j] = bfloat(0.0f);
    return;
  }
  const device bfloat* src = q + (ulong(int(kvh) * G + g) * n + r) * D + lane * 8;
  for (int j = 0; j < 8; ++j) dst[j] = src[j];
"""

# Merge the splits: thread (d, fused row, KV head) -> fp32 [1, H, n, D].
COMBINE = r"""
  constexpr int G = {G};
  constexpr int M = {M};
  constexpr int D = 256;
  const uint d = thread_position_in_grid.x;
  const uint fr = thread_position_in_grid.y;
  const uint kvh = thread_position_in_grid.z;
  const int n = prm[0];
  const int S = prm[2];
  const int r = int(fr) / G, g = int(fr) % G;
  if (r >= n) return;
  float top = -INFINITY;
  for (int s = 0; s < S; ++s) top = max(top, sm[(ulong(s) * {HK} + kvh) * M + fr]);
  float acc = 0.0f, den = 0.0f;
  for (int s = 0; s < S; ++s) {
    const ulong b = (ulong(s) * {HK} + kvh) * M + fr;
    const float w = sl[b] > 0.0f ? fast::exp(sm[b] - top) : 0.0f;
    den += w * sl[b];
    acc += w * part[b * D + d];
  }
  y[(ulong(int(kvh) * G + g) * n + r) * D + d] = acc / den;
"""


def _fill(src, **kw):
    for k, v in kw.items():
        src = src.replace("{" + k + "}", str(v))
    return src


@cache
def _main_kernel(g, m, hk, nt):
    lpr = 4 if m * 4 <= 256 else 2
    return mx.fast.metal_kernel(
        name=f"omlx_q8v_main_g{g}_m{m}_h{hk}_{nt}",
        input_names=["qc", "kc", "ks", "vc", "vs", "prm", "scl"],
        output_names=["out", "sm", "sl"],
        header=HEADER,
        source=_fill(MAIN, G=g, M=m, HK=hk, NT=nt, N=PAGE, LPR=lpr),
        ensure_row_contiguous=False,
    )


@cache
def _layout_kernel(g, m):
    return mx.fast.metal_kernel(
        name=f"omlx_q8v_layout_g{g}_m{m}",
        input_names=["q", "prm"],
        output_names=["qc"],
        source=_fill(LAYOUT, G=g, M=m),
    )


@cache
def _combine_kernel(g, m, hk):
    return mx.fast.metal_kernel(
        name=f"omlx_q8v_comb_g{g}_m{m}_h{hk}",
        input_names=["part", "sm", "sl", "prm"],
        output_names=["y"],
        source=_fill(COMBINE, G=g, M=m, HK=hk),
    )


def splits_for(tokens: int) -> int:
    """About 60 pages per threadgroup: 32 splits at 128K keys, 64 at 245K (measured best on M5 Pro)."""
    return max(1, min(64, (tokens + PAGE - 1) // PAGE // 60))


def _tile(rq, kidx, knorm, vidx, vnorm, scale, split, shift):
    _, heads, n, dim = rq.shape
    hk = int(kidx.shape[1])
    tokens = int(kidx.shape[2])
    g = heads // hk
    m = (n * g + 7) // 8 * 8
    nt = "half" if knorm.dtype == mx.float16 else "float"
    prm = mx.array([n, tokens, split, shift], dtype=mx.int32)
    (qc,) = _layout_kernel(g, m)(
        inputs=[rq, prm],
        grid=(32, m, hk),
        threadgroup=(32, 1, 1),
        output_shapes=[(hk * m * dim,)],
        output_dtypes=[mx.bfloat16],
    )
    part, sm, sl = _main_kernel(g, m, hk, nt)(
        inputs=[qc, kidx, knorm, vidx, vnorm, prm, mx.array([scale], dtype=mx.float32)],
        grid=(256 * split, hk, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(split * hk * m * dim,), (split * hk * m,), (split * hk * m,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (y,) = _combine_kernel(g, m, hk)(
        inputs=[part, sm, sl, prm],
        grid=(dim, m, hk),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, n, dim)],
        output_dtypes=[mx.float32],
    )
    return y


def run(rotated_queries, keys, values, scale, split=None):
    """rotated_queries: [1, H, n, 256] bf16, the last n positions of the history, causal.
    keys / values: affine8 states (indices [1, Hk, T, 64] uint32, norms [1, Hk, T]), views allowed.
    Returns fp32 [1, H, n, 256] in the rotated space. More than 8 rows run as tiles of 8.
    """
    n = int(rotated_queries.shape[2])
    tokens = int(keys.indices.shape[2])
    split = split or splits_for(tokens)
    tiles = [
        _tile(
            rotated_queries[:, :, r0 : r0 + TILE_ROWS],
            keys.indices,
            keys.norms,
            values.indices,
            values.norms,
            scale,
            split,
            n - min(n, r0 + TILE_ROWS),
        )
        for r0 in range(0, n, TILE_ROWS)
    ]
    return tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=2)

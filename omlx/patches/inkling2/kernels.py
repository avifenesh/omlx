# SPDX-License-Identifier: Apache-2.0
"""Attention and attention-residual primitives for Inkling 2."""

from __future__ import annotations

import logging
import os

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available
from ...utils import nax_attention
from ..qwen35_verify_qmm import is_row_exact_armed
from . import verify_attention

logger = logging.getLogger(__name__)

# Upper bound on one additive mask block (elements) in the general attention
# path; queries are split into sub-blocks so the mask stays below it.
_MASK_BUDGET = 1 << 29


_ATTN_RES_TG = 256
_attn_res_kernels: dict = {}
# Fused kernels whose threadgroup this GPU accepts, by variant. A pipeline's
# thread limit depends on its register use, so it is found by launching each
# variant once (M1 Pro accepts 576 threads for some attn_res variants and only
# 384-512 for others).
_launch_ok: dict = {}


def _fits_threadgroup(key, launch) -> bool:
    """Run ``launch`` (a tiny launch of the variant) once per ``key``; False
    when Metal rejects its threadgroup size, so the caller falls back."""
    ok = _launch_ok.get(key)
    if ok is None:
        try:
            mx.eval(launch())
            ok = True
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            logger.info(
                "Inkling 2 fused kernel %s exceeds this GPU's threadgroup "
                "limit; using the MLX ops",
                key,
            )
            ok = False
        _launch_ok[key] = ok
    return ok


def _attn_res_kernel(blocks: int, queries: int):
    """Fused depth mixing for ``blocks`` completed blocks plus the partial sum.

    One threadgroup per token row: dot products of every source with every
    query, softmax over sources, then the weighted sum, without stacking the
    sources.
    """
    key = (blocks, queries)
    kernel = _attn_res_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)] + ["p"]
    loads = "\n".join(
        f"        float s{i} = {name}[base + h];" for i, name in enumerate(srcs)
    )
    dots = "\n".join(
        f"        acc[{i} * Q + qi] += s{i} * qv;" for i in range(len(srcs))
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(len(srcs)))
    source = f"""
    constexpr uint NS = {len(srcs)};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{dots}
        }}
    }}
    threadgroup float red[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_{blocks}_{queries}",
        input_names=[*srcs, "query", "scale"],
        output_names=["out"],
        source=source,
    )
    _attn_res_kernels[key] = kernel
    return kernel


def attn_res_mix(blocks: list, partial_n: mx.array, query: mx.array, scale: float):
    """Mix unit-RMS depth sources with learned queries.

    ``blocks`` holds [B, L, H] FP32 sources, ``partial_n`` is the current
    partial sum (also unit-RMS) and ``query`` is [Q, H]. Returns [B, L, Q, H].
    """
    if mx.default_device() != mx.gpu:
        src = mx.stack([*blocks, partial_n], axis=-2)
        logits = (src @ query.T) * scale
        w = mx.softmax(logits, axis=-2, precise=True)
        return w.swapaxes(-1, -2) @ src
    b, length, hidden = partial_n.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq)
    out = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial_n.astype(mx.float32),
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out


# The fused kernels below repeat the arithmetic of the pinned MLX's rms_norm
# and reduce kernels, so outputs keep their bits. Rerun
# test_inkling2_fused_layer_kernels_keep_the_mlx_bits after an MLX pin change.
_RMS_INV_HEADER = """
inline float omlx_inkling2_rms_inv(
    float acc,
    threadgroup float* local_sums,
    threadgroup float* inv_mean,
    uint lane,
    uint sg,
    float width,
    float eps) {
    acc = simd_sum(acc);
    if (sg == 0) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) {
            inv_mean[0] = metal::precise::rsqrt(acc / width + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv_mean[0];
}
"""
_attn_res_fused_kernels: dict = {}


def _rms_threads(hidden: int) -> int:
    """Threadgroup size of MLX's single-row rms_norm: 4 values per thread."""
    return (((hidden + 3) // 4 + 31) // 32) * 32


def _attn_res_fused_kernel(
    blocks: int, queries: int, norm: bool, f32: bool, pout: bool
):
    """``attn_res_mix`` with its input unit-RMS and, given ``norm``, its output
    RMSNorm and cast in one kernel."""
    key = (blocks, queries, norm, f32, pout)
    kernel = _attn_res_fused_kernels.get(key)
    if kernel is not None:
        return kernel
    nsrc = blocks + 1
    srcs = [f"b{i}" for i in range(blocks)]

    def loads(index: str) -> str:
        lines = [
            f"        float s{i} = {name}[base + {index}];"
            for i, name in enumerate(srcs)
        ]
        lines.append(f"        float s{blocks} = ps[{index}];")
        return "\n".join(lines)

    f32_store = (
        "                    out32[((size_t)qi * gsize + row) * H + h] = float(o);"
        if f32
        else ""
    )
    p_store = "                    pn[base + h] = pv;" if pout else ""
    dots = "\n".join(
        f"            acc[{i} * Q + qi] += s{i} * qv;" for i in range(nsrc)
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(nsrc))
    if norm:
        output = f"""
    for (uint qi = 0; qi < Q; ++qi) {{
        float mv[4];
        float macc = 0.0f;
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
{loads("h")}
                    mv[i] = {mix};
                }} else {{
                    mv[i] = 0.0f;
                }}
                macc = metal::fma(mv[i], mv[i], macc);
            }}
        }}
        const float inv = omlx_inkling2_rms_inv(macc, local_sums, inv_mean, lane, sg, width, eps[1]);
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
                    const T o = T(norm_w[qi * H + h] * (mv[i] * inv));
                    out[((size_t)qi * gsize + row) * H + h] = o;
{f32_store}
                }}
            }}
        }}
    }}
"""
    else:
        output = f"""
    for (uint h = tid; h < H; h += threads_per_threadgroup.x) {{
{loads("h")}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    source = f"""
    constexpr uint NS = {nsrc};
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t base = (size_t)row * H;
    const size_t gsize = (size_t)partial_shape[0] * partial_shape[1];
    // Runtime width: rms_norm divides by its axis size argument.
    const float width = float(partial_shape[partial_ndim - 1]);
    threadgroup float ps[H];
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    {{
        float xv[4];
        float acc = 0.0f;
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                xv[i] = h < H ? float(partial[base + h]) : 0.0f;
                acc = metal::fma(xv[i], xv[i], acc);
            }}
        }}
        const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, eps[0]);
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
                    const float pv = 1.0f * (xv[i] * inv);
                    ps[h] = pv;
{p_store}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    if (tid < TGM) {{
        for (uint h = tid; h < H; h += TGM) {{
{loads("h")}
            for (uint qi = 0; qi < Q; ++qi) {{
                float qv = query[qi * H + h];
{dots}
            }}
        }}
    }}
    threadgroup float red[NS * Q][TGM / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0 && sg < TGM / 32) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TGM / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
{output}
"""
    inputs = [*srcs, "partial", "query", "scale", "eps"]
    if norm:
        inputs.append("norm_w")
    outputs = ["out"]
    if norm and f32:
        outputs.append("out32")
    if pout:
        outputs.append("pn")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_fused_{blocks}_{queries}_{int(norm)}{int(f32)}{int(pout)}",
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_RMS_INV_HEADER,
    )
    _attn_res_fused_kernels[key] = kernel
    return kernel


def attn_res_fused(
    blocks: list,
    partial: mx.array,
    query: mx.array,
    scale: float,
    *,
    unit_eps: float,
    norm_weight: mx.array | None = None,
    norm_eps: float = 0.0,
    out_dtype=None,
    want_f32: bool = False,
    want_partial: bool = False,
    _probe: bool = False,
):
    """``attn_res_mix(blocks, unit_rms(partial), ...)``, then per query
    ``rms_norm(mix, norm_weight).astype(out_dtype)`` when ``norm_weight`` is
    given, as one kernel; None when unsupported. Returns the mix ([B, L, Q, H]
    FP32, or [Q, B, L, H] when normalized), its FP32 copy if ``want_f32`` and
    the unit-RMS partial if ``want_partial``."""
    b, length, hidden = partial.shape
    nq = query.shape[0]
    norm = norm_weight is not None
    tgr = _rms_threads(hidden)
    threads = max(tgr, _ATTN_RES_TG)
    if (
        mx.default_device() != mx.gpu
        or threads > 1024
        or hidden * 4 > 16384
        or any(x.dtype != mx.float32 for x in blocks)
        or query.dtype != mx.float32
        or (norm and norm_weight.dtype != mx.float32)
    ):
        return None
    f32 = norm and want_f32
    variant = ("attn_res", len(blocks), nq, norm, f32, want_partial, hidden)
    variant += (partial.dtype, out_dtype)
    if not _probe and not _fits_threadgroup(
        variant,
        lambda: attn_res_fused(
            [mx.zeros((1, 1, hidden), dtype=mx.float32) for _ in blocks],
            mx.zeros((1, 1, hidden), dtype=partial.dtype),
            mx.zeros(query.shape, dtype=mx.float32),
            scale,
            unit_eps=unit_eps,
            norm_weight=mx.ones(norm_weight.shape, dtype=mx.float32) if norm else None,
            norm_eps=norm_eps,
            out_dtype=out_dtype,
            want_f32=want_f32,
            want_partial=want_partial,
            _probe=True,
        ),
    ):
        return None
    kernel = _attn_res_fused_kernel(len(blocks), nq, norm, f32, want_partial)
    rows = b * length
    inputs = [
        *blocks,
        partial,
        query,
        mx.array([scale], dtype=mx.float32),
        mx.array([unit_eps, norm_eps], dtype=mx.float32),
    ]
    out_type = (out_dtype or partial.dtype) if norm else mx.float32
    shape = (nq, b, length, hidden) if norm else (b, length, nq, hidden)
    shapes = [shape]
    dtypes = [out_type]
    if norm:
        inputs.append(norm_weight)
    if f32:
        shapes.append(shape)
        dtypes.append(mx.float32)
    if want_partial:
        shapes.append((b, length, hidden))
        dtypes.append(mx.float32)
    template = [
        ("H", hidden),
        ("Q", nq),
        ("TGM", _ATTN_RES_TG),
        ("TGR", tgr),
    ]
    if norm:
        template.append(("T", out_type))
    return tuple(
        kernel(
            inputs=inputs,
            template=template,
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=shapes,
            output_dtypes=dtypes,
        )
    )


# Elementwise steps keep one statement per op, as the compiled graph does, so
# nothing is contracted into a fused multiply-add.


_LOOP_IN_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint j = threadgroup_position_in_grid.y;
    const uint i = j / 2;
    const int rows = x_shape[0] * x_shape[1];
    const float width = float(x_shape[3]);
    const device float* src = (j % 2 == 0)
        ? x + ((size_t)row * x_shape[2] + (XQ ? i : 0)) * H
        : m + ((size_t)row * 4 + M0 + i) * H;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    float xv[4];
    float acc = 0.0f;
    if (tid < TGR) {
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            xv[e] = h < H ? src[h] : 0.0f;
            acc = metal::fma(xv[e], xv[e], acc);
        }
    }
    const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, 1e-6f);
    if (tid < TGR) {
        device float* dst = z + ((size_t)j * rows + row) * H;
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const float u = 1.0f * (xv[e] * inv);
                dst[h] = float(bfloat16_t(u));
            }
        }
    }
"""

# MLX's small column reduce adds zero to each value, then the second to the first.
_LOOP_LOW_SOURCE = """
    const uint idx = thread_position_in_grid.x;
    const uint n = low_shape[0] / 2;
    const uint per = low_shape[1] * low_shape[2];
    if (idx >= n * per) {
        return;
    }
    const uint i = idx / per;
    const uint rest = idx % per;
    const float a0 = low[(size_t)(2 * i) * per + rest] + 0.0f;
    const float a1 = low[(size_t)(2 * i + 1) * per + rest] + 0.0f;
    const float s = a1 + a0;
    out[idx] = float(bfloat16_t(s));
"""

_LOOP_OUT_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint i = threadgroup_position_in_grid.y;
    const int rows = x_shape[0] * x_shape[1];
    const float width = float(x_shape[3]);
    const device float* xs = x + ((size_t)row * x_shape[2] + (XQ ? i : 0)) * H;
    const device float* ms = m + ((size_t)row * 4 + M0 + i) * H;
    const device float* gs = gates + ((size_t)i * rows + row) * 2 * H;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    float ov[4];
    float acc = 0.0f;
    if (tid < TGR) {
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const float gain = metal::precise::tanh(gs[h]);
                const float forget = metal::precise::tanh(gs[H + h]);
                const float keep = 1.0f - forget;
                const float kept = keep * xs[h];
                const float added = gain * ms[h];
                ov[e] = kept + added;
            } else {
                ov[e] = 0.0f;
            }
            acc = metal::fma(ov[e], ov[e], acc);
        }
    }
    const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, eps[0]);
    if (tid < TGR) {
        const device float* wr = norm_w + (NW > 1 ? i : 0) * H;
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const T o = T(wr[h] * (ov[e] * inv));
                out[((size_t)i * rows + row) * H + h] = o;
{f32_store}
            }
        }
    }
"""

# Fused BF16 loop-controller matmuls for short decode/verify blocks: checkpoint
# weights are BF16 widened to FP32 at load, so streaming them as BF16 halves
# controller traffic while keeping FP32 accumulation and one BF16 round.
# Fusing unit-RMS normalization into the down projection eliminates the
# separate loop_message_inputs launch and intermediate z buffer.
# Unlike the kernels above, these two do not repeat MLX's reduction order, so
# their bits differ from the FP32 matmul path. A threadgroup serves RB rows
# (and RPT ranks) with each row's sums in the same order whatever RB and RPT
# are, so every row has the bits of its serial decode step.
_LOOP_DOWN_BF16_SOURCE = """
    constexpr uint VECS = H / 4;
    constexpr uint STEPS = (VECS + 127) / 128;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint r0 = threadgroup_position_in_grid.x * RPT;
    const uint i = threadgroup_position_in_grid.y;
    const uint row0 = threadgroup_position_in_grid.z * RB;
    const uint rows = (uint)x_shape[0] * (uint)x_shape[1];
    const uint qx = (uint)x_shape[2];
    const uint nb = min(uint(RB), rows - row0);

    const device float4* xp[RB];
    const device float4* mp[RB];
    threadgroup float sgs[RB * 8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < RB; ++j) {
        const uint row = row0 + min(j, nb - 1);
        xp[j] = (const device float4*)(x + ((size_t)row * qx + (XQ ? i : 0)) * H);
        mp[j] = (const device float4*)(m + ((size_t)row * 4 + M0 + i) * H);
        float ax = 0.0f;
        float am = 0.0f;
        for (uint s = 0; s < STEPS; ++s) {
            uint idx = s * 128 + tid;
            if (idx < VECS) {
                float4 xw = xp[j][idx];
                float4 mw = mp[j][idx];
                ax = fma(xw.x, xw.x, fma(xw.y, xw.y, fma(xw.z, xw.z, fma(xw.w, xw.w, ax))));
                am = fma(mw.x, mw.x, fma(mw.y, mw.y, fma(mw.z, mw.z, fma(mw.w, mw.w, am))));
            }
        }
        float sx = simd_sum(ax);
        float sm = simd_sum(am);
        if (lane == 0) {
            sgs[j * 8 + sg] = sx;
            sgs[j * 8 + 4 + sg] = sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_x[RB];
    float inv_m[RB];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < RB; ++j) {
        const threadgroup float* q = sgs + j * 8;
        inv_x[j] = metal::rsqrt((q[0] + q[1] + q[2] + q[3]) / float(H) + 1e-6f);
        inv_m[j] = metal::rsqrt((q[4] + q[5] + q[6] + q[7]) / float(H) + 1e-6f);
    }

    // acc[j][k]: row j, rank k; each sums its products in the order of s.
    float acc[RB][RPT];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < RB; ++j) {
        #pragma clang loop unroll(full)
        for (uint k = 0; k < RPT; ++k) {
            acc[j][k] = 0.0f;
        }
    }
    for (uint s = 0; s < STEPS; ++s) {
        uint idx = s * 128 + tid;
        if (idx < VECS) {
            float4 xn[RB];
            float4 mn[RB];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < RB; ++j) {
                float4 xw = xp[j][idx];
                float4 mw = mp[j][idx];
                xn[j] = float4(float(bfloat16_t(xw.x * inv_x[j])), float(bfloat16_t(xw.y * inv_x[j])),
                               float(bfloat16_t(xw.z * inv_x[j])), float(bfloat16_t(xw.w * inv_x[j])));
                mn[j] = float4(float(bfloat16_t(mw.x * inv_m[j])), float(bfloat16_t(mw.y * inv_m[j])),
                               float(bfloat16_t(mw.z * inv_m[j])), float(bfloat16_t(mw.w * inv_m[j])));
            }
            #pragma clang loop unroll(full)
            for (uint k = 0; k < RPT; ++k) {
                const size_t r = r0 + k;
                uint2 dx = ((const device uint2*)(down_t + (((size_t)i * 2 + 0) * R + r) * H))[idx];
                uint2 dm = ((const device uint2*)(down_t + (((size_t)i * 2 + 1) * R + r) * H))[idx];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < RB; ++j) {
                    float a = acc[j][k];
                    a = fma(xn[j].x, as_type<float>(dx.x << 16), a);
                    a = fma(xn[j].y, as_type<float>(dx.x & 0xFFFF0000u), a);
                    a = fma(xn[j].z, as_type<float>(dx.y << 16), a);
                    a = fma(xn[j].w, as_type<float>(dx.y & 0xFFFF0000u), a);
                    a = fma(mn[j].x, as_type<float>(dm.x << 16), a);
                    a = fma(mn[j].y, as_type<float>(dm.x & 0xFFFF0000u), a);
                    a = fma(mn[j].z, as_type<float>(dm.y << 16), a);
                    a = fma(mn[j].w, as_type<float>(dm.y & 0xFFFF0000u), a);
                    acc[j][k] = a;
                }
            }
        }
    }
    threadgroup float sgs_a[RB * RPT * 4];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < RB; ++j) {
        #pragma clang loop unroll(full)
        for (uint k = 0; k < RPT; ++k) {
            float sa = simd_sum(acc[j][k]);
            if (lane == 0) {
                sgs_a[(j * RPT + k) * 4 + sg] = sa;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < nb * RPT) {
        const uint j = tid / RPT;
        const uint k = tid % RPT;
        const threadgroup float* q = sgs_a + (j * RPT + k) * 4;
        low[((size_t)i * rows + row0 + j) * R + r0 + k] = float(
            bfloat16_t(q[0] + q[1] + q[2] + q[3]));
    }
"""

_LOOP_GATES_BF16_SOURCE = """
    constexpr uint R_WORDS = R / 2;
    const uint rows = low_shape[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint h = thread_position_in_grid.x;
    const uint n = threadgroup_position_in_grid.y;
    const uint row0 = threadgroup_position_in_grid.z * RB;
    const uint nb = min(uint(RB), rows - row0);
    threadgroup float lows[RB * R];
    for (uint e = tid; e < nb * R; e += 256) {
        lows[e] = low[((size_t)n * rows + row0) * R + e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (h >= H) {
        return;
    }
    // One thread sums each (row, column) over the ranks in order.
    const device uint* gp = (const device uint*)(gf_t + ((size_t)n * H + h) * 2 * R);
    uint gw[R_WORDS];
    uint fw[R_WORDS];
    #pragma clang loop unroll(full)
    for (uint w = 0; w < R_WORDS; ++w) {
        gw[w] = gp[w];
        fw[w] = gp[R_WORDS + w];
    }
    for (uint j = 0; j < nb; ++j) {
        const threadgroup float* l = lows + j * R;
        float g = 0.0f;
        float f = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < R_WORDS; ++w) {
            g = fma(l[2 * w], as_type<float>(gw[w] << 16), g);
            g = fma(l[2 * w + 1], as_type<float>(gw[w] & 0xFFFF0000u), g);
            f = fma(l[2 * w], as_type<float>(fw[w] << 16), f);
            f = fma(l[2 * w + 1], as_type<float>(fw[w] & 0xFFFF0000u), f);
        }
        size_t nr = (size_t)n * rows + row0 + j;
        gates[nr * (2 * H) + h] = g;
        gates[nr * (2 * H) + H + h] = f;
    }
"""
_loop_kernels: dict = {}


def _loop_kernel(kind: str, f32: bool = False):
    key = (kind, f32)
    kernel = _loop_kernels.get(key)
    if kernel is not None:
        return kernel
    if kind == "in":
        names, outs, source = ["x", "m"], ["z"], _LOOP_IN_SOURCE
    elif kind == "low":
        names, outs, source = ["low"], ["out"], _LOOP_LOW_SOURCE
    elif kind == "down_bf16":
        names, outs, source = ["x", "m", "down_t"], ["low"], _LOOP_DOWN_BF16_SOURCE
    elif kind == "gates_bf16":
        names, outs, source = ["low", "gf_t"], ["gates"], _LOOP_GATES_BF16_SOURCE
    else:
        store = (
            "                out32[((size_t)i * rows + row) * H + h] = float(o);"
            if f32
            else ""
        )
        names = ["gates", "x", "m", "norm_w", "eps"]
        outs = ["out", "out32"] if f32 else ["out"]
        source = _LOOP_OUT_SOURCE.replace("{f32_store}", store)
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_loop_{kind}{int(f32)}",
        input_names=names,
        output_names=outs,
        source=source,
        header=_RMS_INV_HEADER if kind in ("in", "out") else "",
    )
    _loop_kernels[key] = kernel
    return kernel


def loop_bf16_supported(hidden: int, rank: int) -> bool:
    return (
        mx.default_device() == mx.gpu
        and 0 < hidden <= 4096
        and hidden % 128 == 0
        and rank > 0
        and rank % 64 == 0
    )


# Rows and ranks per threadgroup of the BF16 controller kernels. They split
# the work only: a row's sums do not depend on them or on the row count.
_LOOP_ROWS_PER_GROUP = 4
_LOOP_RANKS_PER_GROUP = 8
_LOOP_GATE_ROWS_PER_GROUP = 8


def loop_message_down_bf16(
    mixed: mx.array, messages: mx.array, down_t: mx.array, m0: int
) -> mx.array:
    """Fused unit-RMS + two-half down projection from FP32 ``mixed``/``messages``
    and BF16 ``down_t`` [n, 2, r, H] to BF16-rounded FP32 [n, rows, r]."""
    b, length, qx, hidden = mixed.shape
    rows = b * length
    n, rank = down_t.shape[0], down_t.shape[2]
    rb, rpt = _LOOP_ROWS_PER_GROUP, _LOOP_RANKS_PER_GROUP
    return _loop_kernel("down_bf16")(
        inputs=[mixed, messages, down_t],
        template=[
            ("H", hidden), ("R", rank), ("M0", m0), ("XQ", qx > 1), ("RB", rb), ("RPT", rpt)
        ],
        grid=(rank // rpt * 128, n, -(-rows // rb)),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, rows, rank)],
        output_dtypes=[mx.float32],
    )[0]


def loop_message_gates_bf16(low: mx.array, gf_t: mx.array) -> mx.array:
    """Up projection from FP32 ``low`` [n, rows, r] and BF16 ``gf_t``
    [n, H, 2, r] to FP32 ``gates`` [n, rows, 2H]."""
    n, rows, rank = low.shape
    hidden = gf_t.shape[1]
    rb = _LOOP_GATE_ROWS_PER_GROUP
    return _loop_kernel("gates_bf16")(
        inputs=[low, gf_t],
        template=[("H", hidden), ("R", rank), ("RB", rb)],
        grid=(-(-hidden // 256) * 256, n, -(-rows // rb)),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, rows, 2 * hidden)],
        output_dtypes=[mx.float32],
    )[0]


def loop_messages_supported(mixed: mx.array, messages: mx.array) -> bool:
    hidden = mixed.shape[-1]
    return (
        mx.default_device() == mx.gpu
        and _rms_threads(hidden) <= 1024
        and mixed.dtype == mx.float32
        and messages.dtype == mx.float32
        and messages.shape[2] == 4
    )


def loop_message_inputs(mixed, messages, m0: int, n: int):
    """[2n, rows, H] FP32: for each controlled input i, the BF16-rounded unit
    RMS of ``mixed[:, :, i or 0]`` then of ``messages[:, :, m0 + i]``."""
    b, length, qx, hidden = mixed.shape
    rows = b * length
    tgr = _rms_threads(hidden)
    return _loop_kernel("in")(
        inputs=[mixed, messages],
        template=[("H", hidden), ("TGR", tgr), ("M0", m0), ("XQ", qx > 1)],
        grid=(rows * tgr, 2 * n, 1),
        threadgroup=(tgr, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def loop_message_low(low: mx.array):
    """``low.reshape(n, 2, rows, r).sum(1)`` rounded to BF16, as FP32 [n, rows, r]."""
    two_n, rows, rank = low.shape
    total = two_n // 2 * rows * rank
    return _loop_kernel("low")(
        inputs=[low],
        grid=(total, 1, 1),
        threadgroup=(min(total, 256), 1, 1),
        output_shapes=[(two_n // 2, rows, rank)],
        output_dtypes=[mx.float32],
    )[0]


def loop_message_outputs(
    gates, mixed, messages, m0, n, norm_w, norm_eps, dtype, want_f32
):
    """The controlled inputs ``(1 - tanh(f)) * x + tanh(g) * message`` after
    their RMSNorm (FP32 weights [NW, H]) and cast to ``dtype``, [n, B, L, H];
    with ``want_f32`` also their FP32 copy."""
    b, length, qx, hidden = mixed.shape
    rows = b * length
    tgr = _rms_threads(hidden)
    shapes = [(n, b, length, hidden)]
    dtypes = [dtype]
    if want_f32:
        shapes.append((n, b, length, hidden))
        dtypes.append(mx.float32)
    return _loop_kernel("out", want_f32)(
        inputs=[gates, mixed, messages, norm_w, mx.array([norm_eps], dtype=mx.float32)],
        template=[
            ("T", dtype),
            ("H", hidden),
            ("TGR", tgr),
            ("M0", m0),
            ("XQ", qx > 1),
            ("NW", norm_w.shape[0]),
        ],
        grid=(rows * tgr, n, 1),
        threadgroup=(tgr, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


def _decode_attention(q, k, v, rel, sink, scale, window):
    s = k.shape[2]
    lo = max(0, s - window) if window else 0
    if lo:
        k = k[:, :, lo:]
        v = v[:, :, lo:]
    sv = s - lo
    r = rel.shape[-1]
    n = min(r, sv)
    hq = q.shape[1]
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < sv:
        tail = mx.concatenate([mx.zeros((hq, sv - n), dtype=tail.dtype), tail], -1)
    mask = tail.reshape(1, hq, 1, sv).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)
    )


# Flash attention with the relative-bias band and per-query sinks, for
# prefill-sized BF16 or FP16 query blocks (FP16 after the M1/M2 activation
# recast). One threadgroup runs 32 queries of one head
# in four simdgroups (8 rows each) over 16-key tiles staged in half. Scores
# live in the log2 domain; each row's online softmax starts from its sink
# logit, so the zero-value sink is exact without an extra key.
_FLASH_SOURCE = """
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    // The fixed-trip loops over head-dim and key fragments are unrolled in
    // full: left to the compiler, the simdgroup matrix arrays they index
    // spill on M1 and the kernel runs 20-25% slower.
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    // Q, K and V are FP16, or BF16 values within half's range (L2-clipped
    // q/k, unit-RMS v), so staging them in half is exact; scores accumulate
    // in FP32 and the softmax scale is applied there. K is stored transposed
    // and both tiles are padded by 16 bytes per row against bank conflicts.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // 8 BF16 or FP16 values (16 bytes) per load.
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                if (IS_BF16) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                        kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                        vd[2 * j] = half(as_type<float>(vw[j] << 16));
                        vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                    }
                } else {
                    // FP16 activations (M1/M2 recast): the words already hold halves.
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        const half2 k2 = as_type<half2>(kw[j]);
                        const half2 v2 = as_type<half2>(vw[j]);
                        kd[(2 * j) * LDK] = k2.x;
                        kd[(2 * j + 1) * LDK] = k2.y;
                        vd[2 * j] = v2.x;
                        vd[2 * j + 1] = v2.y;
                    }
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        // Full-attention tiles that every row sees whole and past the bias
        // band need neither the mask nor the band: only the scale applies.
        const bool plain = WINDOW == 0 && q0 + BQ <= L && k0 >= lp
            && k0 + BK <= S && a_first - (k0 + BK - 1) >= R;
        if (plain) {
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    const float s = Sf[kb].thread_elements()[e] * qscale;
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
_flash_kernel = None
# Shorter query blocks take the decode kernel when batched or left-padded,
# else SDPA.
_FLASH_MIN_QUERIES = 32


# FlashDecoding for batched or left-padded decode-sized query blocks (decode
# steps and verify): the bias band, the window and the sink are applied in the kernel,
# so no mask is built and K/V are read in place. One threadgroup reduces one
# chunk of CHUNK backward distances of one (row, position, KV head) to a
# partial, and a second kernel folds the partials into the sink, nearest chunk
# first. The tiling depends only on how many keys a query sees, so a batched
# verify row has the bits of the same token's batched decode step.
_DECODE_PART_SOURCE = """
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    // Fixed-trip fragment loops are unrolled in full, as in _FLASH_SOURCE.
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int chunk = threadgroup_position_in_grid.x;
    const int gy = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int lp = dims[3 + b];
    const int pos = gy % L;
    const int hk = gy / L;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    // The G query heads of this KV head are the rows of the tile.
    const int g = int(sg) * 8 + fm;
    const bool row_ok = g < G;
    const int h = hk * G + (row_ok ? g : 0);
    // Slots count back from utop. Full-attention rows put it a multiple of
    // CHUNK above the newest real key, so tiles sit on absolute key positions
    // and a verify row is tiled exactly as the decode step of its token.
    // Windowed rows count from their own key: their cache drops old keys.
    const int a = S - L + pos;
    const int ua = a - lp;
    const int utop = WINDOW > 0 ? ua + 1 : ((max(S - 1 - lp, 0)) / CHUNK + 1) * CHUNK;
    const int shift = utop - 1 - ua;
    const int e_lo = utop - (S - lp);
    const int ulo = WINDOW > 0 ? max(0, ua - WINDOW + 1) : 0;
    const int row_end = utop - ulo;
    const int d_begin = chunk * CHUNK;
    const int d_end = ua < 0 ? d_begin : min(d_begin + CHUNK, row_end);
    const float qscale = scale[0] * LOG2E;

    // As in the prefill kernel: tiles staged in half (exact for FP16 and for
    // the BF16 values attention sees), K transposed, both padded.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = 0.0f;
    float l = 0.0f;
    const device T* relp = rel + ((size_t)(b * L + pos) * HQ + h) * R;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    // The vector load reads 16 bytes at a 16-byte address: a view whose rows
    // or base sit elsewhere takes the scalar loop.
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] & 7) == 0 && (v_strides[2] & 7) == 0
        && (uintptr_t(kb_ptr) & 15) == 0 && (uintptr_t(vb_ptr) & 15) == 0;

    // Slot r of a tile is the key at distance d0 + r.
    for (int d0 = d_begin; d0 < d_end; d0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (d0 + r < d_end && d0 + r >= e_lo) {
                    const int key = lp + utop - 1 - d0 - r;
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                if (IS_BF16) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                        kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                        vd[2 * j] = half(as_type<float>(vw[j] << 16));
                        vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        const half2 k2 = as_type<half2>(kw[j]);
                        const half2 v2 = as_type<half2>(vw[j]);
                        kd[(2 * j) * LDK] = k2.x;
                        kd[(2 * j + 1) * LDK] = k2.y;
                        vd[2 * j] = v2.x;
                        vd[2 * j + 1] = v2.y;
                    }
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                float kv = 0.0f;
                float vv = 0.0f;
                if (d0 + r < d_end && d0 + r >= e_lo) {
                    const int key = lp + utop - 1 - d0 - r;
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(sg) * 8 >= G) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        // A tile may hold no key this row sees; rmax then stays -FLT_MAX and
        // the tile leaves m, l and O unchanged.
        float sc[BK / 4];
        float rmax = -FLT_MAX;
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const int dd = d0 + kb * 8 + fn + e;
                const int d = dd - shift;
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (d >= 0 && d < R) {
                    s += float(relp[d]) * LOG2E;
                }
                sc[kb * 2 + e] = s;
                if (dd < row_end && d >= 0 && (WINDOW == 0 || d < WINDOW)) {
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const bool first = d0 == d_begin;
        const float m_new = first ? rmax : max(m, rmax);
        const float factor = first ? 0.0f : precise::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const int dd = d0 + kb * 8 + fn + e;
                const int d = dd - shift;
                const float p = (dd < row_end && d >= 0 && (WINDOW == 0 || d < WINDOW))
                    ? precise::exp2(sc[kb * 2 + e] - m_new) : 0.0f;
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        // A chunk past the query's keys leaves l == 0 and is skipped.
        const size_t slot = (((size_t)b * HQ + h) * L + pos) * NS + chunk;
        if (fn == 0) {
            part_m[slot] = m;
            part_l[slot] = l;
        }
        device float* op = part_o + slot * D;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = Of[db].thread_elements()[0];
            op[db * 8 + fn + 1] = Of[db].thread_elements()[1];
        }
    }
"""

# One simdgroup per query: the online softmax over its chunk partials,
# starting from the zero-value sink. L and NS are read at run time, so new
# block lengths and cache sizes do not compile new kernels.
_DECODE_FOLD_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const int L = fdims[0];
    const int NS = fdims[1];
    const uint lane = thread_index_in_simdgroup;
    const int row = threadgroup_position_in_grid.x;
    const int pos = row % L;
    const int h = (row / L) % HQ;
    const int b = row / (L * HQ);
    constexpr int PER = (D + 31) / 32;
    float m = float(sink[b * sink_strides[0] + pos * sink_strides[1]
                         + h * sink_strides[2]]) * LOG2E;
    float l = 1.0f;
    float o[PER];
    for (int i = 0; i < PER; ++i) {
        o[i] = 0.0f;
    }
    const size_t base = (size_t)row * NS;
    for (int c = 0; c < NS; ++c) {
        const float lc = part_l[base + c];
        if (!(lc > 0.0f)) {
            continue;
        }
        const float mc = part_m[base + c];
        const float m_new = max(m, mc);
        const float f0 = precise::exp2(m - m_new);
        const float f1 = precise::exp2(mc - m_new);
        const float l0 = l * f0;
        const float l1 = lc * f1;
        l = l0 + l1;
        m = m_new;
        const device float* po = part_o + (base + c) * D;
        for (int i = 0; i < PER; ++i) {
            const int e = int(lane) + 32 * i;
            if (e < D) {
                const float o0 = o[i] * f0;
                const float o1 = po[e] * f1;
                o[i] = o0 + o1;
            }
        }
    }
    const float inv = 1.0f / l;
    for (int i = 0; i < PER; ++i) {
        const int e = int(lane) + 32 * i;
        if (e < D) {
            out[(size_t)row * D + e] = T(o[i] * inv);
        }
    }
"""
_decode_kernels = None
# Distances per chunk for full and sliding layers. They stay fixed per layer
# because a row's bits depend on its chunk layout.
_DECODE_CHUNK = 1024
_DECODE_SLIDING_CHUNK = 128
# Simdgroups staging one tile (the first ones also run the queries).
_DECODE_STAGERS = 2


def _decode_chunks(s: int, window: int, chunk: int) -> int:
    """Chunks of the part kernel grid over the keys a row sees."""
    return max(1, -(-(min(s, window) if window else s) // chunk))


def _use_flash_decode(q, k) -> bool:
    """BF16/FP16 queries on the GPU whose head layout fits the decode kernel."""
    return (
        q.dtype in (mx.bfloat16, mx.float16)
        and mx.default_device() == mx.gpu
        and q.shape[-1] % 8 == 0
        # Bytes of the staged K and V tiles, within Metal's 32 KB of
        # threadgroup memory (head dimensions up to 400).
        and 80 * q.shape[-1] + 256 <= 32768
        and q.shape[1] % k.shape[1] == 0
    )


def decode_reads_row_ends(length: int, dtype) -> bool:
    """Whether ``banded_sink_attention`` takes ``row_ends`` for these query blocks."""
    return (
        length < _FLASH_MIN_QUERIES
        and dtype in (mx.bfloat16, mx.float16)
        and mx.default_device() == mx.gpu
    )


def gqa_flash_decode_attention(
    q, k, v, rel, sink, *, scale, window, left_padding=None, row_ends=None
):
    """``banded_sink_attention`` for batched or left-padded decode-sized
    BF16/FP16 query blocks. Each row has the bits of the same row run alone;
    ``row_ends`` lists each row's own end in ``k`` when rows stop short."""
    global _decode_kernels
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    g = hq // hkv
    if _decode_kernels is None:
        _decode_kernels = (
            mx.fast.metal_kernel(
                name="omlx_inkling2_flash_decode_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["part_o", "part_m", "part_l"],
                source=_row_end_source(_DECODE_PART_SOURCE),
                ensure_row_contiguous=False,
            ),
            mx.fast.metal_kernel(
                name="omlx_inkling2_flash_decode_fold",
                input_names=["part_o", "part_m", "part_l", "sink", "fdims"],
                output_names=["out"],
                source=_DECODE_FOLD_SOURCE,
                ensure_row_contiguous=False,
            ),
        )
    part, fold = _decode_kernels
    # Verify blocks share each staged tile across their positions.
    shared = length > 1 and _Q8_SHARED_TILES and (
        (not window and length * g <= 8 * _Q8_MAX_ROW_GROUPS)
        or (window and length <= _WINDOW_ROWS_MAX and g <= 8)
    )
    if shared and window:
        global _decode_window_rows_part
        if _decode_window_rows_part is None:
            _decode_window_rows_part = mx.fast.metal_kernel(
                name="omlx_inkling2_flash_verify_window_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["part_o", "part_m", "part_l"],
                source=_row_end_source(_window_rows_part_source(_DECODE_PART_SOURCE)),
                ensure_row_contiguous=False,
            )
        part = _decode_window_rows_part
    elif shared:
        global _decode_rows_part
        if _decode_rows_part is None:
            _decode_rows_part = mx.fast.metal_kernel(
                name="omlx_inkling2_flash_verify_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["part_o", "part_m", "part_l"],
                source=_row_end_source(_rows_part_source(_DECODE_PART_SOURCE)),
                ensure_row_contiguous=False,
            )
        part = _decode_rows_part
    chunk = _DECODE_SLIDING_CHUNK if window else _DECODE_CHUNK
    ns = _decode_chunks(s, window, chunk)
    if shared and window:
        nsg = max(length, _DECODE_STAGERS)
    else:
        nsg = max(-(-(length * g if shared else g) // 8), _DECODE_STAGERS)
    ends = [s] * b if row_ends is None else [int(e) for e in row_ends]
    dims = mx.array([length, s, ns] + [0] * b + ends, dtype=mx.int32)
    if left_padding is not None:
        dims = mx.concatenate([dims[:3], left_padding.astype(mx.int32), dims[3 + b :]])
    part_o, part_m, part_l = part(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("IS_BF16", q.dtype == mx.bfloat16),
            ("D", d),
            ("HQ", hq),
            ("G", g),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
            ("CHUNK", chunk),
            ("NSG", nsg),
        ],
        grid=(ns * nsg * 32, hkv * (1 if shared else length), b),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[
            (b, hq, length, ns, d),
            (b, hq, length, ns),
            (b, hq, length, ns),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return fold(
        inputs=[part_o, part_m, part_l, sink, mx.array([length, ns], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", d), ("HQ", hq)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


def _row_end_source(src: str) -> str:
    """A q8 part kernel source whose rows end at their own written column,
    read from ``dims`` after the left padding of every row."""
    old = "    const int S = dims[1];\n"
    if src.count(old) != 1:
        raise RuntimeError("q8 kernel: row end anchor not found once")
    new = "    const int S = dims[3 + int(threadgroups_per_grid.z) + b];\n"
    return src.replace(old, new)


# The decode kernel over an 8-bit affine cache (OMLX_INKLING_KV8): the same
# kernel with K and V staged from int8 codes (four per uint32, low byte first,
# sign-extended) times one scale per vector, instead of a BF16 copy of the
# whole cache. Everything after the staging is unchanged.
def _q8_part_source() -> str:
    src = _DECODE_PART_SOURCE
    head = src.index("    const device T* kb_ptr")
    tail = src.index(
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        if (int(sg) * 8 >= G) {"
    )
    staging = """    const device uint* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device uint* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const device SC* ksb = ks + b * ks_strides[0] + hk * ks_strides[1];
    const device SC* vsb = vs + b * vs_strides[0] + hk * vs_strides[1];
    // Vector loads need contiguous 16-byte aligned code rows.
    const bool vec_q8 = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] & 3) == 0 && (v_strides[2] & 3) == 0
        && (uintptr_t(kb_ptr) & 15) == 0 && (uintptr_t(vb_ptr) & 15) == 0;
    #define KEY_OF(d0_, r_) (lp + utop - 1 - (d0_) - (r_))

    // Slot r of a tile is the key at distance d0 + r.
    for (int d0 = d_begin; d0 < d_end; d0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_q8) {
            // 16 codes per 16-byte load; one scale read per key and slot.
            for (int c = tid; c < BK * (D / 16); c += NT) {
                const int r = c / (D / 16);
                const int w4 = c % (D / 16);
                uint4 kw = uint4(0u);
                uint4 vw = uint4(0u);
                float ksc = 0.0f;
                float vsc = 0.0f;
                if (d0 + r < d_end && d0 + r >= e_lo) {
                    const int key = KEY_OF(d0, r);
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + w4 * 4);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + w4 * 4);
                    ksc = float(ksb[key * ks_strides[2]]);
                    vsc = float(vsb[key * vs_strides[2]]);
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        const int col = w4 * 16 + i * 4 + j;
                        const float kc = float(int(((kw[i] >> (8 * j)) & 0xFFu) ^ 0x80u) - 128);
                        const float vc = float(int(((vw[i] >> (8 * j)) & 0xFFu) ^ 0x80u) - 128);
                        Ks[col * LDK + r] = half(kc * ksc);
                        Vs[r * LDV + col] = half(vc * vsc);
                    }
                }
            }
        } else
        for (int c = tid; c < BK * (D / 4); c += NT) {
            const int r = c / (D / 4);
            const int w = c % (D / 4);
            uint kw = 0u;
            uint vw = 0u;
            float ksc = 0.0f;
            float vsc = 0.0f;
            if (d0 + r < d_end && d0 + r >= e_lo) {
                const int key = KEY_OF(d0, r);
                kw = kb_ptr[key * k_strides[2] + w * k_strides[3]];
                vw = vb_ptr[key * v_strides[2] + w * v_strides[3]];
                ksc = float(ksb[key * ks_strides[2]]);
                vsc = float(vsb[key * vs_strides[2]]);
            }
            for (int j = 0; j < 4; ++j) {
                const int col = w * 4 + j;
                const float kc = float(int(((kw >> (8 * j)) & 0xFFu) ^ 0x80u) - 128);
                const float vc = float(int(((vw >> (8 * j)) & 0xFFu) ^ 0x80u) - 128);
                Ks[col * LDK + r] = half(kc * ksc);
                Vs[r * LDV + col] = half(vc * vsc);
            }
        }
"""
    return _row_end_source(src[:head] + staging + src[tail:])


_decode_q8_part = None


# Verify blocks (L > 1 queries) over the 8-bit cache: one threadgroup per KV
# head serves every query position, so each K/V tile is staged once instead of
# once per position.
def _q8_rows_part_source() -> str:
    return _rows_part_source(_q8_part_source())


def _rows_part_source(src: str) -> str:
    """The shared-tile verify variant of a decode part kernel source."""
    edits = [
        (
            "    const int pos = gy % L;\n    const int hk = gy / L;\n",
            "    const int hk = gy;\n",
        ),
        (
            "    // The G query heads of this KV head are the rows of the tile.\n"
            "    const int g = int(sg) * 8 + fm;\n"
            "    const bool row_ok = g < G;\n"
            "    const int h = hk * G + (row_ok ? g : 0);\n",
            "    // Rows of the tile are (position, head) pairs of this KV head.\n"
            "    const int rr = int(sg) * 8 + fm;\n"
            "    const bool row_ok = rr < L * G;\n"
            "    const int pos = row_ok ? rr / G : 0;\n"
            "    const int g = row_ok ? rr % G : 0;\n"
            "    const int h = hk * G + g;\n",
        ),
        (
            "        if (int(sg) * 8 >= G) {\n",
            "        if (int(sg) * 8 >= L * G) {\n",
        ),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f"verify kernel: anchor not found once: {old[:60]!r}")
        src = src.replace(old, new)
    return src


def _window_rows_part_source(src: str) -> str:
    """The shared-tile verify variant for windowed rows: one simdgroup per
    position keeps the decode step's own-key tiles, and each staged tile step
    holds BK + L - 1 keys so every position reads its 16 at its own offset."""
    edits = [
        (
            "    const int pos = gy % L;\n    const int hk = gy / L;\n",
            "    const int hk = gy;\n    const int pos = min(int(sg), L - 1);\n",
        ),
        (
            "    const int g = int(sg) * 8 + fm;\n    const bool row_ok = g < G;\n",
            "    const int g = fm;\n    const bool row_ok = g < G && int(sg) < L;\n",
        ),
        (
            "    const int d_end = ua < 0 ? d_begin : min(d_begin + CHUNK, row_end);\n",
            "    const int lag = L - 1 - pos;\n"
            "    const int d_end = S - lp <= 0 ? d_begin : min(d_begin + CHUNK, min(S - lp, WINDOW));\n",
        ),
        ("    constexpr int LDK = BK + 8;\n", "    constexpr int LDK = BK + 16;\n"),
        ("    threadgroup half Vs[BK * LDV];\n", "    threadgroup half Vs[(BK + 7) * LDV];\n"),
        (
            "            for (int c = tid; c < BK * D / 8; c += NT) {\n",
            "            for (int c = tid; c < (BK + L - 1) * D / 8; c += NT) {\n",
        ),
        (
            "                if (d0 + r < d_end && d0 + r >= e_lo) {\n"
            "                    const int key = lp + utop - 1 - d0 - r;\n"
            "                    kw = ",
            "                if (S - 1 - d0 - r >= lp) {\n"
            "                    const int key = S - 1 - d0 - r;\n"
            "                    kw = ",
        ),
        (
            "            for (int c = tid; c < BK * D; c += NT) {\n",
            "            for (int c = tid; c < (BK + L - 1) * D; c += NT) {\n",
        ),
        (
            "                if (d0 + r < d_end && d0 + r >= e_lo) {\n"
            "                    const int key = lp + utop - 1 - d0 - r;\n"
            "                    kv = ",
            "                if (S - 1 - d0 - r >= lp) {\n"
            "                    const int key = S - 1 - d0 - r;\n"
            "                    kv = ",
        ),
        ("        if (int(sg) * 8 >= G) {\n", "        if (int(sg) >= L) {\n"),
        (
            "                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);\n",
            "                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8 + lag, LDK);\n",
        ),
        (
            "                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);\n",
            "                simdgroup_load(Vm, Vs + (kb * 8 + lag) * LDV + db * 8, LDV);\n",
        ),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f"windowed verify kernel: anchor not found once: {old[:60]!r}")
        src = src.replace(old, new)
    return src


_decode_window_rows_part = None
_decode_q8_rows_part = None
_decode_rows_part = None
# Off: OMLX_INKLING_Q8_SHARED=0 (verify blocks re-stage tiles per position).
_Q8_SHARED_TILES = os.environ.get("OMLX_INKLING_Q8_SHARED", "1") != "0"
# Simdgroups of 8 (position, head) rows per threadgroup (1024 threads max).
_Q8_MAX_ROW_GROUPS = 32
# Positions of a windowed verify block (staged tile steps hold BK + 7 keys).
_WINDOW_ROWS_MAX = 8


_MPP_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# The q8 part kernel on the tensor units (D = 128, full-attention rows), for
# decode and verify alike: one simdgroup per 16 (position, head) rows, 32-key
# tiles at the same absolute key positions as _DECODE_PART_SOURCE. A lane's 4
# fragment columns are one code word; the key scale multiplies the score and
# the value scale P, so tiles enter matmul2d as raw codes.
_Q8_NAX_PART_SOURCE = """
    constexpr int BK = 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int chunk = threadgroup_position_in_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int lp = dims[3 + b];
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    const float qscale = scale[0] * LOG2E;
    const int utop = ((max(S - 1 - lp, 0)) / CHUNK + 1) * CHUNK;
    const int e_lo = utop - (S - lp);
    const int d_begin = chunk * CHUNK;
    const int d_end = min(d_begin + CHUNK, utop);

    bool rok[2];
    int rpos[2];
    int rh[2];
    int shift[2];
    const device T* relp[2];
    for (short i = 0; i < 2; ++i) {
        const int rr = int(sg) * 16 + fm + i * 8;
        rok[i] = rr < L * G;
        rpos[i] = rok[i] ? rr / G : 0;
        rh[i] = hk * G + (rok[i] ? rr % G : 0);
        shift[i] = rok[i] ? utop - 1 - (S - L + rpos[i] - lp) : (1 << 30);
        relp[i] = rel + ((size_t)(b * L + rpos[i]) * HQ + rh[i]) * R;
    }

    constexpr int LDQ = D + 8;
    threadgroup half Qs[NF * 16 * LDQ];
    {
        threadgroup half* qd = Qs + int(sg) * 16 * LDQ;
        for (int c = lane; c < 16 * D; c += 32) {
            const int r = c / D;
            const int col = c % D;
            const int rr = int(sg) * 16 + r;
            float qv = 0.0f;
            if (rr < L * G) {
                qv = float(q[b * q_strides[0] + (hk * G + rr % G) * q_strides[1]
                             + (rr / G) * q_strides[2] + col * q_strides[3]]);
            }
            qd[r * LDQ + col] = half(qv);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup half* q0 = Qs + (int(sg) * 16 + fm) * LDQ + fn;
    const threadgroup half* q1 = q0 + 8 * LDQ;

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;
    auto pa = pv_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<half, half, float>();
    using PA = metal::remove_addrspace_t<decltype(pa)>;
    using PB = metal::remove_addrspace_t<decltype(pb)>;
    auto o0 = pv_op.template get_destination_cooperative_tensor<PA, PB, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<PA, PB, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<PA, PB, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<PA, PB, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f; o1[e] = 0.0f; o2[e] = 0.0f; o3[e] = 0.0f;
    }
    auto qa = qk_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto kb = qk_op.template get_right_input_cooperative_tensor<half, half, float>();
    using QA = metal::remove_addrspace_t<decltype(qa)>;
    using KB = metal::remove_addrspace_t<decltype(kb)>;
    auto sc = qk_op.template get_destination_cooperative_tensor<QA, KB, float>();

    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};
    const device uint* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device uint* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const device SC* ksb = ks + b * ks_strides[0] + hk * ks_strides[1];
    const device SC* vsb = vs + b * vs_strides[0] + hk * vs_strides[1];
    const int kw = k_strides[3];
    const int vw = v_strides[3];
    const int top = lp + utop - 1;
    const bool sg_live = int(sg) * 16 < L * G;

    for (int d0 = d_begin; d0 < d_end && sg_live; d0 += BK) {
        const device uint* kr[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                const int key = clamp(top - d0 - (tn * 16 + fm + i * 8), 0, S - 1);
                kr[tn][i] = kb_ptr + key * k_strides[2] + (fn / 4) * kw;
            }
        }
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        #pragma clang loop unroll(full)
        for (short kk = 0; kk < D / 16; ++kk) {
            const half4 qv0 = *(const threadgroup half4*)(q0 + kk * 16);
            const half4 qv1 = *(const threadgroup half4*)(q1 + kk * 16);
            for (short j = 0; j < 4; ++j) {
                qa[j] = qv0[j];
                qa[4 + j] = qv1[j];
            }
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const half4 kv4 = half4(as_type<char4>(kr[tn][i][kk * 4 * kw]));
                    for (short j = 0; j < 4; ++j) {
                        kb[tn * 8 + i * 4 + j] = kv4[j];
                    }
                }
            }
            qk_op.run(qa, kb, sc);
        }

        float s[2][2][4];
        bool ok[2][2][4];
        float vsc[2][4];
        float rmax[2] = {-INFINITY, -INFINITY};
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const int dd = d0 + tn * 16 + fn + j;
                const bool kvalid = dd < d_end && dd >= e_lo;
                const int key = clamp(top - dd, 0, S - 1);
                const float ksc = float(ksb[key * ks_strides[2]]) * qscale;
                vsc[tn][j] = kvalid ? float(vsb[key * vs_strides[2]]) : 0.0f;
                for (short i = 0; i < 2; ++i) {
                    const int d = dd - shift[i];
                    float x = float(sc[tn * 8 + i * 4 + j]) * ksc;
                    if (d >= 0 && d < R) {
                        x += float(relp[i][d]) * LOG2E;
                    }
                    const bool vis = kvalid && d >= 0;
                    s[tn][i][j] = x;
                    ok[tn][i][j] = vis;
                    rmax[i] = vis ? max(rmax[i], x) : rmax[i];
                }
            }
        }
        float factor[2];
        float mnew[2];
        float rs[2] = {0.0f, 0.0f};
        for (short i = 0; i < 2; ++i) {
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(1)));
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(8)));
            mnew[i] = max(m_run[i], rmax[i]);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f; o1[e] *= f; o2[e] *= f; o3[e] *= f;
            }
        }

        for (short ks16 = 0; ks16 < 2; ++ks16) {
            for (short i = 0; i < 2; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const float p = ok[ks16][i][j] ? exp2(s[ks16][i][j] - mnew[i]) : 0.0f;
                    rs[i] += p;
                    pa[i * 4 + j] = half(p * vsc[ks16][j]);
                }
            }
            const int key0 = clamp(top - d0 - (ks16 * 16 + fm), 0, S - 1);
            const int key1 = clamp(top - d0 - (ks16 * 16 + fm + 8), 0, S - 1);
            const device uint* v0 = vb_ptr + key0 * v_strides[2] + (fn / 4) * vw;
            const device uint* v1 = vb_ptr + key1 * v_strides[2] + (fn / 4) * vw;
            #pragma clang loop unroll(full)
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    const half4 a = half4(as_type<char4>(v0[((np * 32 + tn * 16) / 4) * vw]));
                    const half4 c = half4(as_type<char4>(v1[((np * 32 + tn * 16) / 4) * vw]));
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = a[j];
                        pb[tn * 8 + 4 + j] = c[j];
                    }
                }
                if (np == 0) pv_op.run(pa, pb, o0);
                else if (np == 1) pv_op.run(pa, pb, o1);
                else if (np == 2) pv_op.run(pa, pb, o2);
                else pv_op.run(pa, pb, o3);
            }
        }
        for (short i = 0; i < 2; ++i) {
            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
            l_run[i] = l_run[i] * factor[i] + rs[i];
        }
    }

    for (short i = 0; i < 2; ++i) {
        if (!rok[i]) {
            continue;
        }
        // A chunk past the row's keys leaves l == 0 and the fold skips it.
        const size_t slot = (((size_t)b * HQ + rh[i]) * L + rpos[i]) * NS + chunk;
        if (fn == 0) {
            part_m[slot] = m_run[i] == -INFINITY ? 0.0f : m_run[i];
            part_l[slot] = l_run[i];
        }
        device float* op = part_o + slot * D + fn;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                op[tn * 16 + j] = o0[e];
                op[32 + tn * 16 + j] = o1[e];
                op[64 + tn * 16 + j] = o2[e];
                op[96 + tn * 16 + j] = o3[e];
            }
        }
    }
"""
_decode_q8_nax_part = None


def q8_decode_supported(q, k_state, kv) -> bool:
    """Decode-sized blocks over an 8-bit affine cache the q8 kernel can read."""
    indices = getattr(k_state, "indices", None)
    norms = getattr(k_state, "norms", None)
    return (
        getattr(getattr(kv, "key_codec", None), "bits", None) == 8
        and getattr(getattr(kv, "value_codec", None), "bits", None) == 8
        and indices is not None
        and indices.dtype == mx.uint32
        and norms is not None
        and norms.ndim == 3
        and q.shape[2] < _FLASH_MIN_QUERIES
        and q.shape[-1] % 8 == 0
        and q.shape[-1] == indices.shape[-1] * 4
        and q.dtype in (mx.bfloat16, mx.float16)
        and mx.default_device() == mx.gpu
        and 80 * q.shape[-1] + 256 <= 32768
        and q.shape[1] % indices.shape[1] == 0
    )


def gqa_flash_decode_attention_q8(
    q, k_state, v_state, rel, sink, *, scale, window, left_padding=None, row_ends=None
):
    """``gqa_flash_decode_attention`` reading an 8-bit affine cache in place.

    ``q`` is already in the key rotation; ``k_state``/``v_state`` hold
    ``indices`` [B, Hkv, S, D/4] uint32 and ``norms`` [B, Hkv, S] scales; the
    output is in the value rotation. S is the cached length the caller passes
    through the state's ``shape``; ``row_ends`` lists each row's own written
    end when rows stop short of S.
    """
    global _decode_q8_part, _decode_q8_rows_part, _decode_q8_nax_part, _decode_kernels
    b, hq, length, d = q.shape
    k_idx, v_idx = k_state.indices, v_state.indices
    k_sc, v_sc = k_state.norms, v_state.norms
    hkv = k_idx.shape[1]
    s = int(k_state.shape[2])
    g = hq // hkv
    if _decode_q8_part is None:
        _decode_q8_part = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_decode_part_q8",
            input_names=["q", "k", "v", "ks", "vs", "rel", "dims", "scale"],
            output_names=["part_o", "part_m", "part_l"],
            source=_q8_part_source(),
            ensure_row_contiguous=False,
        )
    if _decode_kernels is None:
        # Builds the shared fold kernel.
        gqa_flash_decode_attention(
            q[:1, :, :1], mx.zeros((1, hkv, 1, d), q.dtype), mx.zeros((1, hkv, 1, d), q.dtype),
            rel[:1, :1], sink[:1, :1], scale=scale, window=window,
        )
    fold = _decode_kernels[1]
    # Verify blocks share each staged tile across their positions.
    shared = (
        length > 1 and not window and _Q8_SHARED_TILES and length * g <= 8 * _Q8_MAX_ROW_GROUPS
    )
    if shared and _decode_q8_rows_part is None:
        _decode_q8_rows_part = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_verify_part_q8",
            input_names=["q", "k", "v", "ks", "vs", "rel", "dims", "scale"],
            output_names=["part_o", "part_m", "part_l"],
            source=_q8_rows_part_source(),
            ensure_row_contiguous=False,
        )
    chunk = _DECODE_SLIDING_CHUNK if window else _DECODE_CHUNK
    ns = _decode_chunks(s, window, chunk)
    ends = [s] * b if row_ends is None else [int(e) for e in row_ends]
    dims = mx.array([length, s, ns] + [0] * b + ends, dtype=mx.int32)
    if left_padding is not None:
        dims = mx.concatenate([dims[:3], left_padding.astype(mx.int32), dims[3 + b :]])
    inputs = [
        q, k_idx, v_idx, k_sc, v_sc, mx.contiguous(rel), dims,
        mx.array([scale], dtype=mx.float32),
    ]
    # Q of up to 7 16-row fragments fits the 32 KB of threadgroup memory.
    nax = not window and d == 128 and length * g <= 7 * 16 and (length == 1 or shared)
    if nax and is_nax_available():
        if _decode_q8_nax_part is None:
            _decode_q8_nax_part = mx.fast.metal_kernel(
                name="omlx_inkling2_flash_decode_part_q8_nax",
                input_names=["q", "k", "v", "ks", "vs", "rel", "dims", "scale"],
                output_names=["part_o", "part_m", "part_l"],
                header=_MPP_HEADER,
                source=_row_end_source(_Q8_NAX_PART_SOURCE),
                ensure_row_contiguous=False,
            )
        nf = -(-(length * g) // 16)
        kernel = _decode_q8_nax_part
        template = [
            ("T", q.dtype), ("SC", k_sc.dtype), ("D", d), ("HQ", hq), ("G", g),
            ("R", rel.shape[-1]), ("CHUNK", chunk), ("NF", nf),
        ]
        grid = (ns * nf * 32, hkv, b)
        threadgroup = (nf * 32, 1, 1)
    else:
        rows = length * g if shared else g
        nsg = max(-(-rows // 8), _DECODE_STAGERS)
        kernel = _decode_q8_rows_part if shared else _decode_q8_part
        template = [
            ("T", q.dtype),
            ("SC", k_sc.dtype),
            ("IS_BF16", q.dtype == mx.bfloat16),
            ("D", d),
            ("HQ", hq),
            ("G", g),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
            ("CHUNK", chunk),
            ("NSG", nsg),
        ]
        grid = (ns * nsg * 32, hkv * (1 if shared else length), b)
        threadgroup = (nsg * 32, 1, 1)
    part_o, part_m, part_l = kernel(
        inputs=inputs,
        template=template,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[
            (b, hq, length, ns, d),
            (b, hq, length, ns),
            (b, hq, length, ns),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return fold(
        inputs=[part_o, part_m, part_l, sink, mx.array([length, ns], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", d), ("HQ", hq)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


def _use_flash(length: int, window: int) -> bool:
    if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:
        return False
    return bool(window) or not is_nax_available()


def prefill_mask_elements() -> int | None:
    """Mask elements one full-attention prefill call holds, None on flash."""
    return None if _use_flash(_FLASH_MIN_QUERIES, 0) else _MASK_BUDGET


def flash_banded_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for BF16/FP16 prefill blocks; same arguments and result."""
    global _flash_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            ensure_row_contiguous=False,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    out = _flash_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sink.astype(mx.float32)),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("IS_BF16", q.dtype == mx.bfloat16),
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
        ],
        grid=(((length + 31) // 32) * 128, hq, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
    row_ends=None,
    int8_keys=None,
) -> mx.array:
    """Attention with a relative bias band and per-query sink logits.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D] (queries are the last L keys);
    rel: [B, L, Hq, R] bias by backward distance d (applied for d < R);
    sink: [B, L, Hq] logits of a zero-value sink. ``window`` > 0 limits
    keys to d < window (sliding layers use window == R). ``row_ends`` lists
    each row's own end in ``k`` when rows stop short of the last key.
    ``int8_keys`` (fp32 queries, int8 key codes, key scales), when ``k`` is
    their dequantization, lets full-attention prefill blocks run Q.K^T in int8.
    """
    b, length = q.shape[0], q.shape[2]
    if row_ends is not None:
        return gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=scale, window=window,
            left_padding=left_padding, row_ends=row_ends,
        )
    if (
        (
            window
            or b > 1
            or left_padding is not None
            or (1 < length <= 8 and _Q8_SHARED_TILES and not is_row_exact_armed())
        )
        and length < _FLASH_MIN_QUERIES
        and _use_flash_decode(q, k)
    ):
        # Batches, a row left alone in a batch cache, and every windowed row:
        # the kernel reads left_padding itself instead of masking every cached
        # key, and a windowed row has the same bits alone as in a batch.
        return gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    if b == 1 and length == 1 and left_padding is None:
        return _decode_attention(q, k, v, rel, sink, scale, window)
    if b == 1 and left_padding is None and is_row_exact_armed():
        # Row-exact verify: each row is the decode step of its token, over the
        # keys up to its own position.
        fused = verify_attention.rows_attention(
            q, k, v, rel, sink, scale=scale, window=window
        )
        if fused is not None:
            return fused
        # The rows are joined as [B, L, H, D], so the output projection reads
        # them without a transpose copy.
        s = k.shape[2]
        rows = [
            _decode_attention(
                q[:, :, r : r + 1],
                k[:, :, : s - length + 1 + r],
                v[:, :, : s - length + 1 + r],
                rel[:, r : r + 1],
                sink[:, r : r + 1],
                scale,
                window,
            ).transpose(0, 2, 1, 3)
            for r in range(length)
        ]
        return mx.concatenate(rows, axis=1).transpose(0, 2, 1, 3)
    if int8_keys is not None:
        out = nax_attention.nax_banded_int8_key_attention(
            *int8_keys, v, rel, sink, scale=scale, left_padding=left_padding
        )
        if out is not None:
            return out
    out = nax_attention.nax_banded_sink_attention(
        q, k, v, rel, sink, scale=scale, left_padding=left_padding, window=window
    )
    if out is not None:
        return out
    if q.dtype in (mx.bfloat16, mx.float16) and _use_flash(length, window):
        return flash_banded_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)


def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):
    """MLX SDPA over an additive mask that carries the bias band and sinks."""
    b, hq, length, d = q.shape
    s = k.shape[2]
    r = rel.shape[-1]
    if window:
        assert window == r, "sliding layers expect rel_extent == window"
    lo = max(0, s - length - window + 1) if window else 0
    hkv = k.shape[1]
    # A zero key/value row carries the sink: its logit comes from the mask.
    zero = mx.zeros((b, hkv, 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    lp_ok = None
    if left_padding is not None:
        lp_ok = (lo + mx.arange(nk))[None, :] >= left_padding[:, None]
        lp_ok = mx.concatenate([mx.ones((b, 1), dtype=mx.bool_), lp_ok], axis=1)
        lp_ok = lp_ok[:, None, None, :]

    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
        bias = mx.take_along_axis(relq, gidx, axis=-1)
        vis = dist >= 0
        if window:
            vis = vis & (dist < window)
        near_mask = mx.where(vis, mx.where(dist < r, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        if lp_ok is not None:
            mask = mx.where(lp_ok, mask, neg)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        if qb < length:
            # A lazy graph would hold every block's mask at once (12 GiB at
            # 64K keys); evaluated block by block, one mask is alive.
            mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


# One head of one token per 32-thread threadgroup. Clip sums add rounded
# squares (MLX squares in a separate kernel), 4 adjacent values per lane then
# simd_sum, as MLX's row reduce; d_rel rows are summed in order. v follows
# rms_norm, whose squares are fused multiply-adds.
_QKV_POST_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint task = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const int L = q_shape[1];
    const int b = row / L;
    const int t = row % L;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    if (task < HQ + HKV) {
        const bool is_q = task < HQ;
        const int head = is_q ? task : task - HQ;
        const int col = head * D + lane * 4;
        float xf[4];
        if (is_q) {
            const device T* src = q + b * q_strides[0] + t * q_strides[1];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * q_strides[2]]) * float(q_scale[col + i]);
            }
        } else {
            const device T* src = k + b * k_strides[0] + t * k_strides[1];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * k_strides[2]]);
            }
        }
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float sq = xf[i] * xf[i];
            acc = sq + acc;
        }
        const float norm = metal::precise::sqrt(simd_sum(acc));
        const float f = metal::min(qk_clip[0] / (norm + 1e-6f), 1.0f);
        const int heads = is_q ? HQ : HKV;
        device T* dst = (is_q ? qo : ko) + ((size_t)row * heads + head) * D + lane * 4;
        for (int i = 0; i < 4; ++i) {
            T o = T(xf[i] * f);
            if (HAS_TAU && is_q) {
                o = T(float(o) * tau[b * tau_strides[0] + t * tau_strides[1]]);
            }
            dst[i] = o;
        }
    } else if (task < HQ + 2 * HKV) {
        const int head = task - HQ - HKV;
        const int col = head * D + lane * 4;
        const device float* src = v + b * v_strides[0] + t * v_strides[1];
        float xv[4];
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            xv[i] = src[(col + i) * v_strides[2]];
            acc = metal::fma(xv[i], xv[i], acc);
        }
        const float inv = omlx_inkling2_rms_inv(
            acc, local_sums, inv_mean, lane, 0, float(D), 1e-6f);
        device T* dst = vo + ((size_t)row * HKV + head) * D + lane * 4;
        for (int i = 0; i < 4; ++i) {
            dst[i] = T(1.0f * (xv[i] * inv));
        }
    } else {
        const int head = (task - HQ - 2 * HKV) * 32 + lane;
        if (head >= HQ) {
            return;
        }
        const device T* src = r + b * r_strides[0] + t * r_strides[1];
        float xf[R];
        float acc = 0.0f;
        for (int j = 0; j < R; ++j) {
            xf[j] = float(src[(head * R + j) * r_strides[2]]) * float(r_scale[head * R + j]);
            const float sq = xf[j] * xf[j];
            acc = sq + acc;
        }
        const float norm = metal::precise::sqrt(acc);
        const float f = metal::min(r_clip[0] / (norm + 1e-6f), 1.0f);
        device T* dst = ro + ((size_t)row * HQ + head) * R;
        for (int j = 0; j < R; ++j) {
            T o = T(xf[j] * f);
            if (HAS_TAU) {
                o = T(float(o) * tau[b * tau_strides[0] + t * tau_strides[1]]);
            }
            dst[j] = o;
        }
    }
"""
_qkv_post_kernel = None


def qkv_post(q, k, v, r, q_scale, r_scale, qk_clip, r_clip, tau, heads, kv_heads):
    """``Inkling2Attention.project``'s clips, v unit-RMS and log scaling as one
    kernel, or None when unsupported. Returns q, k, v as [B, L, H, D] and r as
    [B, L, Hq, R] in q's dtype."""
    global _qkv_post_kernel
    b, length, dq = q.shape
    dkv = k.shape[-1]
    head_dim = dq // heads
    d_rel = r.shape[-1] // heads
    if (
        mx.default_device() != mx.gpu
        or head_dim != 128
        or dkv != kv_heads * head_dim
        or not 0 < d_rel <= 64
        or k.dtype != q.dtype
        or r.dtype != q.dtype
        or v.dtype != mx.float32
    ):
        return None
    if _qkv_post_kernel is None:
        _qkv_post_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_qkv_post",
            input_names=[
                "q",
                "k",
                "v",
                "r",
                "q_scale",
                "r_scale",
                "qk_clip",
                "r_clip",
                "tau",
            ],
            output_names=["qo", "ko", "vo", "ro"],
            source=_QKV_POST_SOURCE,
            header=_RMS_INV_HEADER,
            ensure_row_contiguous=False,
        )
    has_tau = tau is not None
    # 0-d arrays would be bound by value; the kernel indexes them.
    qk_clip = qk_clip.reshape(1)
    r_clip = r_clip.reshape(1)
    if has_tau:
        tau = mx.broadcast_to(tau.reshape(tau.shape[0], length), (b, length))
    else:
        tau = qk_clip.reshape(1, 1)
    dtype = q.dtype
    tasks = heads + 2 * kv_heads + (heads + 31) // 32
    return _qkv_post_kernel(
        inputs=[q, k, v, r, q_scale, r_scale, qk_clip, r_clip, tau],
        template=[
            ("T", dtype),
            ("HQ", heads),
            ("HKV", kv_heads),
            ("D", head_dim),
            ("R", d_rel),
            ("HAS_TAU", has_tau),
        ],
        grid=(tasks * 32, b * length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (b, length, heads, head_dim),
            (b, length, kv_heads, head_dim),
            (b, length, kv_heads, head_dim),
            (b, length, heads, d_rel),
        ],
        output_dtypes=[dtype] * 4,
    )


_ROUTER_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"""
_router_kernel = None


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """Sigmoid scores, top-k by ``score + bias``, renormalized unbiased weights.

    ``logits``: [N, E] FP32. Returns ([N, K] uint32 indices, [N, K] FP32).
    """
    global _router_kernel
    rows, experts = logits.shape
    if mx.default_device() != mx.gpu or experts % 32:
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale
        return inds, w
    if _router_kernel is None:
        _router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights"],
            source=_ROUTER_SOURCE,
        )
    inds, weights = _router_kernel(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights


_SHORT_CONV_HEADER = """
// Row i of the padded window [state, x], x rounded to T; the state has p rows.
template <typename T, typename S, typename X>
inline float omlx_inkling2_padded(
    const device S* srow,
    int64_t ss,
    const device X* xrow,
    int64_t xs,
    int p,
    int i) {
    return i < p ? float(srow[i * ss]) : float(T(xrow[(i - p) * xs]));
}

// x is the experts' weighted sum, added as MLX's small column reduce does.
// The weights may be bound in the constant address space (fewer than 8).
template <typename T, typename S, typename Z, int KE, typename WP>
inline float omlx_inkling2_padded_moe(
    const device S* srow,
    int64_t ss,
    const device Z* zcol,
    WP wt,
    int C,
    int p,
    int i) {
    if (i < p) {
        return float(srow[i * ss]);
    }
    const int t = i - p;
    const device Z* z = zcol + (size_t)t * KE * C;
    const auto w = wt + (size_t)t * KE;
    float acc = 0.0f;
    for (int e = 0; e < KE; ++e) {
        const float prod = float(z[e * C]) * w[e];
        const float term = prod + 0.0f;
        acc = e == 0 ? term : term + acc;
    }
    return float(T(acc));
}
"""

# One thread per (channel, padded row, batch row) of the window [state, x].
# Rows below K - 1 only copy the state; the others also compute their token.
# Each product and sum is its own statement, in the order and precision of
# short_conv_reference: a fused multiply-add would change the bits.
_SHORT_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int y = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device X* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];
    const float own = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y);
    xp[((size_t)b * (P + L) + y) * C + c] = own;
    if (y < P) {
        return;
    }
    const device W* wrow = w + c * w_strides[0];
    const int t = y - P;
    float acc = float(wrow[0]) * omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, t);
    for (int j = 1; j < K; ++j) {
        const float tap = float(wrow[j * w_strides[1]]);
        const float term = tap * omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, t + j);
        acc = acc + term;
    }
    // Rounded to the weight dtype before the residual add, as the reference.
    const float conv = float(W(acc));
    const float res = conv + own;
    const float o = res * scale[0];
    // The result in T, then the caller's residual add and cast.
    float v = float(T(o));
    if (HAS_RESIDUAL) {
        v = float(residual[b * residual_strides[0] + t * residual_strides[1]
                           + c * residual_strides[2]]) + v;
    }
    out[((size_t)b * L + t) * C + c] = O(v);
"""
_short_conv_kernel = None


def short_conv_reference(x, w, state, scale, residual=None, out_dtype=None, dtype=None):
    """``short_conv`` as MLX ops (CPU path and test reference)."""
    x = x.astype(dtype or x.dtype)
    xf = x.astype(mx.float32)
    k = w.shape[-1]
    xp = mx.concatenate([state, xf], axis=1)
    length = xf.shape[1]
    out = w[:, 0] * xp[:, 0:length]
    for j in range(1, k):
        out = out + w[:, j] * xp[:, j : j + length]
    # The reference rounds the depthwise conv to the weight dtype before the
    # residual add.
    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale
    out = out.astype(x.dtype)
    if residual is not None:
        out = residual.astype(mx.float32) + out.astype(mx.float32)
    return out.astype(out_dtype or x.dtype), xp


def short_conv(
    x: mx.array,
    w: mx.array,
    state: mx.array,
    scale: float,
    residual: mx.array | None = None,
    out_dtype=None,
    dtype=None,
):
    """Depthwise causal conv with residual (taps: ``w[:, K-1]`` is the current
    token), as one kernel.

    ``x``: [B, L, C], read as ``dtype`` (the activation dtype, default
    ``x.dtype``); ``w``: [C, K]; ``state``: [B, K-1, C], the previous K-1
    inputs. The output is ``(x + conv) * scale`` in ``dtype``; given a
    ``residual`` [B, L, C], it is then added to the residual in FP32. Either
    way it is returned cast to ``out_dtype`` (default ``dtype``), with the
    FP32 padded window ``[state, x]``, whose last K-1 rows are the next
    state. Strided inputs are read in place.
    """
    global _short_conv_kernel
    dtype = dtype or x.dtype
    out_dtype = out_dtype or dtype
    if mx.default_device() != mx.gpu:
        return short_conv_reference(x, w, state, scale, residual, out_dtype, dtype)
    if _short_conv_kernel is None:
        _short_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_short_conv",
            input_names=["x", "w", "state", "residual", "scale"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_SHORT_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    k = w.shape[-1]
    has_residual = residual is not None
    # Without a residual, x is bound in its place and never read.
    residual = residual if has_residual else x
    out, xp = _short_conv_kernel(
        inputs=[x, w, state, residual, mx.array([scale], dtype=mx.float32)],
        template=[
            ("T", dtype),
            ("X", x.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", out_dtype),
            ("K", k),
            ("HAS_RESIDUAL", has_residual),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    return out, xp


_SHORT_CONV_MOE_SOURCE = (
    _SHORT_CONV_SOURCE.replace("const int C = x_shape[2];", "const int C = x_shape[3];")
    .replace(
        "const device X* xrow = x + b * x_strides[0] + c * x_strides[2];",
        "const device X* zcol = x + (size_t)b * L * KE * C + c;",
    )
    .replace(
        "const int64_t xs = x_strides[1];",
        "const auto wt = moe_w + (size_t)b * L * KE;",
    )
    .replace(
        "omlx_inkling2_padded<T>(srow, ss, xrow, xs, P,",
        "omlx_inkling2_padded_moe<T, S, X, KE>(srow, ss, zcol, wt, C, P,",
    )
)
_short_conv_moe_kernel = None
# The weighted sum is MLX's small column reduce only while one threadgroup row
# holds each expert.
_MOE_SUM_MAX_EXPERTS = 8


def moe_short_conv(
    z: mx.array,
    weights: mx.array,
    w: mx.array,
    state: mx.array,
    scale: float,
    residual: mx.array,
    dtype,
):
    """``short_conv`` of the expert sum ``moe_sum(z, weights)`` with the sum done
    in the kernel, or None when unsupported."""
    global _short_conv_moe_kernel
    b, length, experts, channels = z.shape
    if mx.default_device() != mx.gpu or experts > _MOE_SUM_MAX_EXPERTS:
        return None
    if weights.dtype != mx.float32:
        return None
    if _short_conv_moe_kernel is None:
        _short_conv_moe_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_short_conv_moe",
            input_names=["x", "w", "state", "residual", "scale", "moe_w"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_SHORT_CONV_MOE_SOURCE,
            ensure_row_contiguous=False,
        )
    k = w.shape[-1]
    return _short_conv_moe_kernel(
        inputs=[
            mx.contiguous(z),
            w,
            state,
            residual,
            mx.array([scale], dtype=mx.float32),
            mx.contiguous(weights),
        ],
        template=[
            ("T", dtype),
            ("X", z.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", dtype),
            ("K", k),
            ("KE", experts),
            ("HAS_RESIDUAL", True),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[dtype, mx.float32],
    )


# The full-attention k/v conv: per-token taps static_w[lag] + dyn[head, lag],
# laid out like _SHORT_CONV_SOURCE, with the sum over lags newest first and
# no rounding before the residual add, as dynamic_conv_reference.
_DYNAMIC_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int y = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device T* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];
    const float own = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y);
    xp[((size_t)b * (P + L) + y) * C + c] = own;
    if (y < P) {
        return;
    }
    const int t = y - P;
    const int head = c / (C / H);
    const auto drow = dyn + b * dyn_strides[0] + t * dyn_strides[1]
        + head * K * dyn_strides[2];
    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const float sw = float(static_w[lag * static_w_strides[0]
                                        + c * static_w_strides[1]]);
        const float tap = sw + float(drow[lag * dyn_strides[2]]);
        const float term = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y - lag)
            * tap;
        acc = lag == 0 ? term : acc + term;
    }
    const float res = acc + own;
    const float o = res * scale[0];
    out[((size_t)b * L + t) * C + c] = O(float(T(o)));
"""
_dynamic_conv_kernel = None


def dynamic_conv_reference(x, dyn, static_w, state, heads, scale, out_dtype=None):
    """``dynamic_conv`` as MLX ops (CPU path and test reference)."""
    xf = x.astype(mx.float32)
    b, length, c = xf.shape
    k = static_w.shape[0]
    xp = mx.concatenate([state, xf], axis=1)
    dyn = dyn.astype(mx.float32).reshape(b, length, heads, 1, k)
    out = None
    for lag in range(k):
        start = k - 1 - lag
        window = xp[:, start : start + length].reshape(b, length, heads, c // heads)
        # A slice, not static_w[lag]: integer indexing is a gather kernel.
        tap = static_w[lag : lag + 1].reshape(heads, c // heads) + dyn[..., lag]
        term = window * tap
        out = term if out is None else out + term
    out = (out.reshape(b, length, c) + xf) * scale
    return out.astype(x.dtype).astype(out_dtype or x.dtype), xp


def dynamic_conv(
    x: mx.array,
    dyn: mx.array,
    static_w: mx.array,
    state: mx.array,
    heads: int,
    scale: float,
    out_dtype=None,
):
    """Depthwise causal conv with per-token taps ``static_w[lag] + dyn[head,
    lag]`` and residual, as one kernel (full-attention k and v).

    ``x``: [B, L, C]; ``dyn``: [B, L, heads * K]; ``static_w``: [K, C], lag 0
    the current token; ``state``: [B, K-1, C]. Returns ``(x + conv) * scale``
    rounded to ``x``'s dtype, cast to ``out_dtype`` (default ``x.dtype``),
    and the FP32 padded window ``[state, x]``.
    """
    global _dynamic_conv_kernel
    out_dtype = out_dtype or x.dtype
    if mx.default_device() != mx.gpu:
        return dynamic_conv_reference(x, dyn, static_w, state, heads, scale, out_dtype)
    if _dynamic_conv_kernel is None:
        _dynamic_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dynamic_conv",
            input_names=["x", "dyn", "static_w", "state", "scale"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_DYNAMIC_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    k = static_w.shape[0]
    out, xp = _dynamic_conv_kernel(
        inputs=[x, dyn, static_w, state, mx.array([scale], dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("S", state.dtype),
            ("O", out_dtype),
            ("K", k),
            ("H", heads),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    return out, xp

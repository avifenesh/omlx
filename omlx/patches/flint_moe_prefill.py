# SPDX-License-Identifier: Apache-2.0
"""Flint MoE prefill: consume the sorted expert rows with the native weighted sum.

Flint's ``SparseMoeBlock`` (mlx-vlm ``models/flint``) runs the stock
``SwitchGLU``: it sorts the routed rows, runs the experts, scatters the result
back to ``[B, T, top_k, D]``, then upcasts it to float32, scales it by the
router weights and sums over ``top_k``. On a 2,052-token prefill chunk that
scatter and float32 combine cost about 2 ms per layer on an M5 Pro, as much as
a third of the down projection.

For prefill chunks this keeps Flint's router (sigmoid, selection bias, FP32
normalization) and hands the sorted expert output straight to the
``moe_weighted_sum`` kernel the Qwen MoE prefill path already uses
(``qwen35_moe_weighted_sum``), which also takes the fused gate/up activation
and the routed A8 hooks of that path. The shared expert is added as before.
Decode and verify rows keep their own path (``flint_decode``).

``OMLX_FLINT_MOE_PREFILL=0`` turns it off.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("OMLX_FLINT_MOE_PREFILL", "1") == "1"
_MIN_TOKENS = 1024


def apply_flint_moe_prefill_patch() -> bool:
    if not _ENABLED or not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.flint import language as flint

        from .qwen35_moe_weighted_sum import (
            _native_switch_weighted_sum,
            _native_weighted_sum,
        )
    except ImportError:
        return False
    weighted_sum = _native_weighted_sum()
    if weighted_sum is None:
        return False
    cls = flint.SparseMoeBlock
    if getattr(cls, "_omlx_flint_moe_prefill", False):
        return True
    orig_call = cls.__call__

    def patched_call(self, x):
        if (
            x.ndim != 3
            or x.shape[-2] < _MIN_TOKENS
            or x.dtype not in (mx.bfloat16, mx.float16)
            or self.top_k not in (6, 8, 10)
        ):
            return orig_call(self, x)
        try:
            indices, weights = self.route(x)
            out = _native_switch_weighted_sum(
                self.switch_mlp, x, indices, weights, weighted_sum
            ).astype(x.dtype)
        except Exception:
            logger.warning("Flint MoE prefill fast path failed", exc_info=True)
            return orig_call(self, x)
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
        return out

    cls.__call__ = patched_call
    cls._omlx_flint_moe_prefill = True
    cls._omlx_flint_moe_prefill_original_call = orig_call
    logger.info("Flint MoE prefill weighted-sum patch applied (min_tokens=%d)", _MIN_TOKENS)
    return True


__all__ = ["apply_flint_moe_prefill_patch"]

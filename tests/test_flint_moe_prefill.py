# SPDX-License-Identifier: Apache-2.0
"""Flint MoE prefill weighted sum against the stock SparseMoeBlock body."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

flint = pytest.importorskip("mlx_vlm.models.flint.language")

from omlx.patches import flint_moe_prefill  # noqa: E402


def _block(experts=32, bits=6):
    args = SimpleNamespace(
        hidden_size=2048,
        num_experts_per_tok=6,
        num_experts=experts,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=1024,
    )
    mx.random.seed(3)
    block = flint.SparseMoeBlock(args)
    block.expert_bias = mx.random.normal((experts,)) * 0.1
    nn.quantize(block, group_size=64, bits=bits)
    mx.eval(block.parameters())
    return block


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("rows", [1024, 2052])
def test_prefill_matches_the_stock_body(rows):
    block = _block()
    x = (mx.random.normal((1, rows, 2048)) * 0.5).astype(mx.bfloat16)
    cls = flint.SparseMoeBlock
    orig = getattr(cls, "_omlx_flint_moe_prefill_original_call", cls.__call__)
    ref = orig(block, x).astype(mx.float32)
    if not flint_moe_prefill.apply_flint_moe_prefill_patch():
        pytest.skip("moe_weighted_sum kernel unavailable")
    out = block(x).astype(mx.float32)
    assert out.shape == ref.shape == (1, rows, 2048)
    diff = mx.abs(out - ref)
    assert mx.max(diff).item() <= 0.02 * mx.max(mx.abs(ref)).item() + 1e-2
    assert mx.mean(diff).item() < 2e-3


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_short_rows_keep_the_stock_body():
    block = _block()
    assert flint_moe_prefill.apply_flint_moe_prefill_patch() or True
    cls = flint.SparseMoeBlock
    orig = getattr(cls, "_omlx_flint_moe_prefill_original_call", cls.__call__)
    x = mx.random.normal((1, 64, 2048)).astype(mx.bfloat16)
    assert mx.array_equal(block(x), orig(block, x)).item()

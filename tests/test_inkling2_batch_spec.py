# SPDX-License-Identifier: Apache-2.0
"""Batched speculative verify for Inkling 2 (shared accept count).

Rows of a batched verify all keep the fewest drafts any row accepted, so the
rollback trims every row by one count. Each row must then decode exactly as
it would alone, and the packet clamp must keep each row's output tokens valid.
"""

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp.fused_batch import _clamp_to_shared_accept, _uniform_accept
from tests.test_inkling_vlm_mtp import _TINY_LOOP, _inkling2_tiny


def _join(caches):
    joined = []
    for parts in zip(*caches):
        if hasattr(parts[0], "offset"):
            joined.append(type(parts[0]).merge(list(parts)))
        else:
            parts[0].extend(parts[1])
            joined.append(parts[0])
    return joined


@pytest.mark.parametrize("accepted", [0, 3, 7])
@pytest.mark.parametrize("dims", [{}, _TINY_LOOP], ids=["plain", "looped"])
def test_batched_verify_shared_rollback_matches_each_row_alone(dims, accepted):
    model = _inkling2_tiny(sliding_window_size=16, rel_extent=16, **dims)
    assert model._omlx_drafter_single_row is False and _uniform_accept(model)
    drafts = 7
    ids = mx.random.randint(0, 300, (2, 80))
    lengths = (37, 12)
    alone, expected = [], []
    for row, n in enumerate(lengths):
        cache = model.make_cache()
        model(ids[row : row + 1, :n], cache=cache)
        alone.append(cache)
        # Reference: the accepted prefix decoded token by token, then one step.
        reference = model.make_cache()
        model(ids[row : row + 1, :n], cache=reference)
        for t in range(accepted + 1):
            model(ids[row : row + 1, n + t : n + t + 1], cache=reference)
        nxt = n + accepted + 1
        expected.append(model(ids[row : row + 1, nxt : nxt + 1], cache=reference))
    joined = _join(alone)
    block = mx.stack([ids[r, n : n + drafts + 1] for r, n in enumerate(lengths)])
    out = model(block, cache=joined, return_hidden=True, capture_layer_ids=[5])
    mx.eval(out.logits)
    assert model.rollback_speculative_cache(joined, None, [accepted, accepted], drafts + 1) == accepted
    nxt = mx.stack(
        [ids[r, n + accepted + 1 : n + accepted + 2] for r, n in enumerate(lengths)]
    )
    got = model(nxt, cache=joined)
    for row in range(2):
        diff = mx.abs(got[row : row + 1, :, :300] - expected[row][..., :300]).max().item()
        assert diff < 1e-3, (row, diff)


def test_unequal_row_counts_are_rejected():
    model = _inkling2_tiny()
    with pytest.raises(RuntimeError):
        model.rollback_speculative_cache(model.make_cache(), None, [1, 2], 8)


def test_clamp_greedy_keeps_the_target_tokens():
    depth = 3
    # [accepted, targets(depth+1), drafts(depth)]
    a = [3, 5, 6, 7, 9, 5, 6, 7]
    b = [1, 5, 8, 2, 4, 5, 3, 1]
    _clamp_to_shared_accept([a, b], None, depth)
    assert a[0] == b[0] == 1
    # Row a's token after one accepted draft is its target there, which is its draft.
    assert a[1 + 1] == a[1 + depth + 1 + 1] == 6


def test_clamp_stochastic_emits_the_accepted_draft():
    depth = 3
    # [accepted, drafts(depth), corrections(depth), bonus]
    a = [3, 11, 12, 13, 21, 22, 23, 99]
    b = [1, 31, 32, 33, 41, 42, 43, 98]
    _clamp_to_shared_accept(None, [a, b], depth)
    assert a[0] == 1 and a[1 + depth + 1] == 12  # its accepted second draft
    assert b == [1, 31, 32, 33, 41, 42, 43, 98]  # the shortest row is untouched


@pytest.mark.parametrize("dims", [{}, _TINY_LOOP], ids=["plain", "looped"])
def test_rows_with_different_counts_roll_back_alone_and_rejoin(dims):
    # A row can lower its own count after the shared clamp (stop sequence,
    # token limit): each row is split off with its verify windows, rolled
    # back alone, and the batch is rebuilt from the rows.
    model = _inkling2_tiny(sliding_window_size=16, rel_extent=16, **dims)
    drafts, counts = 7, (2, 5)
    ids = mx.random.randint(0, 300, (2, 80))
    lengths = (37, 12)
    alone, expected = [], []
    for row, n in enumerate(lengths):
        cache = model.make_cache()
        model(ids[row : row + 1, :n], cache=cache)
        alone.append(cache)
        reference = model.make_cache()
        model(ids[row : row + 1, : n + counts[row] + 1], cache=reference)
        nxt = n + counts[row] + 1
        expected.append(model(ids[row : row + 1, nxt : nxt + 1], cache=reference))
    joined = _join(alone)
    block = mx.stack([ids[r, n : n + drafts + 1] for r, n in enumerate(lengths)])
    mx.eval(model(block, cache=joined, return_hidden=True, capture_layer_ids=[5]).logits)
    rows = []
    for i, accepted in enumerate(counts):
        row = model.extract_verify_row(joined, i)
        assert model.mtp_partial_rollback(row, accepted, drafts)
        rows.append(row)
    rejoined = _join(rows)
    nxt = mx.stack(
        [ids[r, n + counts[r] + 1 : n + counts[r] + 2] for r, n in enumerate(lengths)]
    )
    got = model(nxt, cache=rejoined)
    for row in range(2):
        diff = mx.abs(got[row : row + 1, :, :300] - expected[row][..., :300]).max().item()
        assert diff < 1e-3, (row, diff)

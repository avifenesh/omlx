# SPDX-License-Identifier: Apache-2.0
"""Batched speculative verify for Inkling 2.

Over dense caches, rows of a batched verify all keep the fewest drafts any row
accepted, so the rollback trims every row by one count. Each row must then
decode exactly as it would alone, and the packet clamp must keep each row's
output tokens valid. Over the 8-bit affine cache each row keeps its own count.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache

from omlx.affine4 import Affine8KVCache
from omlx.patches.inkling2 import loopcd
from omlx.patches.mlx_lm_mtp.fused_batch import (
    _clamp_to_shared_accept,
    _ragged_trim_rows,
    _rows_trim_alone,
    _uniform_accept,
)
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed
from tests.test_inkling_vlm_mtp import (
    _FIRST_CONTROL,
    _OPEN_THINKING,
    _TINY_LOOP,
    _TINY_LOOPCD,
    _TINY_PLAIN,
    _bf16,
    _inkling2_tiny,
)


def _join(caches):
    joined = []
    for parts in zip(*caches):
        if hasattr(parts[0], "offset"):
            joined.append(type(parts[0]).merge(list(parts)))
        else:
            for part in parts[1:]:
                parts[0].extend(part)
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


def _kv8_cache(model):
    return [
        Affine8KVCache(bits=8, seed=3) if type(c) is KVCache else c
        for c in model.make_cache()
    ]


def _prefilled(model, ids, lengths):
    rows = []
    for row, n in enumerate(lengths):
        cache = _kv8_cache(model)
        model(ids[row : row + 1, :n], cache=cache)
        mx.eval([c.state for c in cache])
        rows.append(cache)
    return _join(rows)


def test_rows_trim_alone_over_the_8bit_cache_only():
    model = _inkling2_tiny()
    ids = mx.random.randint(0, 300, (2, 20))
    dense = []
    for row in range(2):
        cache = model.make_cache()
        model(ids[row : row + 1, : 10 + row], cache=cache)
        dense.append(cache)
    assert not _rows_trim_alone(_join(dense))
    assert _rows_trim_alone(_prefilled(model, ids, (10, 11)))


# Accepted drafts per row and cycle (7 drafts a cycle).
_ACCEPTS = [(5, 1, 3), (0, 7, 2), (7, 0, 4), (2, 6, 7), (3, 3, 0), (6, 2, 5), (1, 4, 7)]


def _verify_cycles(model, ids, lengths, tokens, *, realign=False):
    """Logits of each row's kept verify positions, cycle by cycle, then of one
    decode step; rejected drafts open thinking."""
    drafts, vocab = 7, 300
    cache = _prefilled(model, ids, lengths)
    done, kept = [0, 0, 0], []
    for want in _ACCEPTS:
        block = [
            tokens[r][done[r] : done[r] + 1 + want[r]]
            + [_OPEN_THINKING] * (drafts - want[r])
            for r in range(3)
        ]
        set_verify_qmm_armed(True)
        try:
            logits = model(mx.array(block), cache=cache)[..., :vocab]
        finally:
            set_verify_qmm_armed(False)
        kept.append([logits[r, : want[r] + 1] for r in range(3)])
        accepted = list(want)
        with _ragged_trim_rows(cache, accepted, drafts):
            rolled = model.rollback_speculative_cache(cache, None, accepted, drafts + 1)
        assert rolled == accepted
        if realign:
            for entry in cache:
                if getattr(type(entry), "_omlx_ragged_trim", False):
                    entry._align_row_ends()
        done = [d + a + 1 for d, a in zip(done, accepted)]
        if model.model._caches_thinking_flags:
            history = [
                ids[r, :n].tolist() + tokens[r][: done[r]]
                for r, n in enumerate(lengths)
            ]
            last = [[t for t in h if t >= _FIRST_CONTROL][-1:] for h in history]
            assert cache[-1][0][:, 0].tolist() == [c == [_OPEN_THINKING] for c in last]
    assert min(n + d for n, d in zip(lengths, done)) > 1024
    step = model(mx.array([[tokens[r][done[r]]] for r in range(3)]), cache=cache)
    kept.append([step[r, :, :vocab] for r in range(3)])
    return kept


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize(
    "dims", [_TINY_PLAIN, _TINY_LOOP, _TINY_LOOPCD], ids=["plain", "looped", "loopcd"]
)
def test_rows_keep_their_own_accepted_drafts(dims, head_dim, monkeypatch):
    # Batched verify over the 8-bit cache, each row keeping its own accepted
    # drafts across window recuts and the first 1024-key chunk boundary: every
    # kept verify row has the bits of the batch's greedy decode step for its
    # token. The loop controllers' verify matmuls round apart from the decode
    # ones, so looped rows are held to the same cycles over realigned caches.
    if mx.default_device() != mx.gpu:
        pytest.skip("q8 attention kernels")
    monkeypatch.setattr(loopcd, "FIRST_CONTROL_TOKEN", _FIRST_CONTROL)
    monkeypatch.setattr(loopcd, "THINKING_TOKEN", _OPEN_THINKING)
    model = _bf16(
        _inkling2_tiny(sliding_window_size=16, rel_extent=16, head_dim=head_dim, **dims)
    )
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    lengths = (1003, 998, 1012)
    ids = mx.random.randint(0, _FIRST_CONTROL, (3, max(lengths)))
    steps = sum(max(a) + 1 for a in _ACCEPTS) + 8

    cache = _prefilled(model, ids, lengths)
    first = mx.stack(
        [
            model(ids[r : r + 1, :n], cache=_kv8_cache(model))[0, -1]
            for r, n in enumerate(lengths)
        ]
    )
    tokens = [mx.argmax(first[:, :300], axis=-1)]
    reference = []
    for _ in range(steps):
        logits = model(tokens[-1][:, None], cache=cache)[:, -1, :300]
        reference.append(logits)
        tokens.append(mx.argmax(logits, axis=-1))
    reference = mx.stack(reference, axis=1)
    tokens = mx.stack(tokens, axis=1).tolist()

    got = _verify_cycles(model, ids, lengths, tokens)
    if "loop_config" in dims:
        expected = _verify_cycles(model, ids, lengths, tokens, realign=True)
    else:
        done, expected = [0, 0, 0], []
        for want in _ACCEPTS + [(0, 0, 0)]:
            expected.append(
                [reference[r, done[r] : done[r] + want[r] + 1] for r in range(3)]
            )
            done = [d + a + 1 for d, a in zip(done, want)]
    for cycle, (rows, refs) in enumerate(zip(got, expected)):
        for r, (a, b) in enumerate(zip(rows, refs)):
            assert mx.array_equal(a, b).item(), (cycle, r)

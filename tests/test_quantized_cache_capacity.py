# SPDX-License-Identifier: Apache-2.0
"""Joining quantized caches: no whole-state copies beyond the batch itself.

A restored or joined quantized cache used to be exactly as long as its tokens,
so the first token appended to it copied the whole packed state. Merging
several requests also kept every request's cache alive next to the batch
until the last layer was merged.
"""

import importlib

import mlx.core as mx
import pytest
from mlx_vlm.turboquant import _state_length

import omlx.scheduler  # noqa: F401  (applies BatchGenerator cache patches)
from omlx.affine4 import Affine8KVCache, BatchAffine8KVCache
from omlx.turboquant_kv import _MIN_SPARE_TOKENS, _growth_step, _spare_capacity

HEADS, DIM = 2, 64


def _cache(length: int, seed: int = 0) -> Affine8KVCache:
    cache = Affine8KVCache(seed=7)
    keys = mx.random.normal((1, HEADS, length, DIM), key=mx.random.key(seed))
    cache.update_and_fetch(keys.astype(mx.float16), keys.astype(mx.float16))
    mx.eval(cache.keys, cache.values)
    return cache


def _append(cache, tokens: int = 1) -> None:
    rows = cache.keys.norms.shape[0]
    new = mx.random.normal((rows, HEADS, tokens, DIM)).astype(mx.float16)
    cache.update_and_fetch(new, new)
    mx.eval(cache.keys, cache.values)


def _row(state, row: int, start: int, end: int):
    return [part[row : row + 1, :, start:end] for part in state]


def _equal(left, right) -> bool:
    return all(mx.array_equal(a, b).item() for a, b in zip(left, right))


def _written(cache):
    return [part for state in cache.state for part in getattr(state, "_state", state)]


def test_growth_step_scales_with_long_caches():
    assert _growth_step(0) == 256
    assert _growth_step(16_000) == 256
    assert _growth_step(65_536) == 1024
    assert _growth_step(245_000) == 3584
    # A built state leaves at least 64 tokens and at most one step more of room.
    for tokens in (1, 255, 256, 4000, 65_536, 245_000):
        step = _growth_step(tokens)
        assert _MIN_SPARE_TOKENS <= _spare_capacity(tokens) - tokens < step + _MIN_SPARE_TOKENS


def test_merge_writes_rows_end_aligned_with_room_to_grow():
    long, short = _cache(700, seed=1), _cache(300, seed=2)
    expected_long, expected_short = _written(long), _written(short)

    batch = BatchAffine8KVCache.merge([long, short])

    assert batch.offset.tolist() == [700, 300]
    assert batch.left_padding.tolist() == [0, 400]
    assert _state_length(batch.keys) == _spare_capacity(700)
    keys, values = batch.state
    assert _state_length(keys) == 700
    merged = [part for state in (keys, values) for part in state]
    assert _equal(_row(merged, 0, 0, 700), expected_long)
    assert _equal(_row(merged, 1, 400, 700), expected_short)

    # The next tokens are written in place: no new buffer, earlier rows intact.
    capacity = _state_length(batch.keys)
    _append(batch, 3)
    assert _state_length(batch.keys) == capacity
    keys, values = batch.state
    merged = [part for state in (keys, values) for part in state]
    assert _equal(_row(merged, 0, 0, 700), expected_long)
    assert _equal(_row(merged, 1, 400, 700), expected_short)


def test_extend_joins_into_one_buffer_with_room_to_grow():
    gen = importlib.import_module("mlx_lm.generate")
    running, joining = _cache(300, seed=3), _cache(700, seed=4)
    expected_running, expected_joining = _written(running), _written(joining)
    cache_a, cache_b = [running], [joining]

    extended = gen._extend_cache(cache_a, cache_b)
    batch = extended[0]

    assert isinstance(batch, BatchAffine8KVCache)
    # Both donors were let go: the joining layer, and the singleton that was
    # copied into the batch.
    assert cache_a == [None] and cache_b == [None]
    assert batch.offset.tolist() == [300, 700]
    assert batch.left_padding.tolist() == [400, 0]
    keys, values = batch.state
    merged = [part for state in (keys, values) for part in state]
    assert _equal(_row(merged, 0, 400, 700), expected_running)
    assert _equal(_row(merged, 1, 0, 700), expected_joining)

    capacity = _state_length(batch.keys)
    assert capacity == _spare_capacity(700)
    _append(batch)
    assert _state_length(batch.keys) == capacity


def test_filter_keeps_the_spare_columns():
    batch = BatchAffine8KVCache.merge([_cache(700, seed=5), _cache(300, seed=6)])
    expected = _row(_written(batch), 1, 400, 700)

    batch.filter([1])
    mx.eval(batch.keys, batch.values)

    assert batch.left_padding.tolist() == [0]
    assert _equal(_row(_written(batch), 0, 0, 300), expected)
    capacity = _state_length(batch.keys)
    assert capacity > batch._phys_end == 300
    _append(batch)
    assert _state_length(batch.keys) == capacity
    assert _equal(_row(_written(batch), 0, 0, 300), expected)


def test_merge_of_several_requests_lets_go_of_each_donor_layer():
    gen = importlib.import_module("mlx_lm.generate")
    layers = 4
    first = [_cache(600, seed=10 + i) for i in range(layers)]
    second = [_cache(600, seed=20 + i) for i in range(layers)]
    expected = [(_written(a), _written(b)) for a, b in zip(first, second)]

    merged = gen._merge_caches([first, second])

    assert first == [None] * layers and second == [None] * layers
    for batch, (row0, row1) in zip(merged, expected):
        assert isinstance(batch, BatchAffine8KVCache)
        assert _equal(_row(_written(batch), 0, 0, 600), row0)
        assert _equal(_row(_written(batch), 1, 0, 600), row1)


def test_merge_of_several_requests_peaks_below_twice_the_batch():
    """Peak while merging stays near one batch, not batch plus all donors."""
    gen = importlib.import_module("mlx_lm.generate")
    layers, tokens = 8, 4096
    caches = [
        [_cache(tokens, seed=30 + r * layers + i) for i in range(layers)]
        for r in range(2)
    ]
    donors = sum(
        part.nbytes
        for request in caches
        for cache in request
        for part in _written(cache)
    )
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()

    merged = gen._merge_caches(caches)
    mx.eval([part for batch in merged for part in _written(batch)])

    extra = mx.get_peak_memory() - baseline
    # Stock merge: every donor stays alive until the end, about 1.0 x donors
    # on top of them. Layer by layer: two layers of room at most.
    assert (
        extra < 0.6 * donors
    ), f"merge peaked {extra / donors:.2f} x the donors above them"


@pytest.mark.parametrize("empty_first", [False, True])
def test_merge_keeps_an_empty_request_as_a_zero_row(empty_first):
    full, empty = _cache(40, seed=7), Affine8KVCache(seed=7)
    expected = _written(full)
    caches = [empty, full] if empty_first else [full, empty]

    batch = BatchAffine8KVCache.merge(caches)

    row = 1 if empty_first else 0
    assert batch.keys.norms.shape[0] == 2
    assert _equal(_row(_written(batch), row, 0, 40), expected)
    assert not mx.any(batch.keys.norms[1 - row]).item()


@pytest.mark.parametrize("stays", [0, 1])
def test_a_row_leaving_is_copied_out_layer_by_layer(monkeypatch, stays):
    """The rows that stay are copied at the next token one layer at a time.

    Each layer's old buffer goes before the next layer copies, so the peak
    does not hold the old batch on top of the rows that stay. The copy runs
    inside the scheduler's deferred clear window after a finish (#435), so it
    must not clear the pool itself.
    """
    import omlx.cache.prefix_cache as prefix_cache

    flushes = []
    monkeypatch.setattr(
        prefix_cache.BlockAwarePrefixCache,
        "_flush_restore_pool_if_over",
        staticmethod(lambda flush_bytes: flushes.append(flush_bytes) or False),
    )
    layers, tokens = 8, 4096
    batches = [
        BatchAffine8KVCache.merge(
            [_cache(tokens, seed=40 + i), _cache(tokens, seed=60 + i)]
        )
        for i in range(layers)
    ]
    expected = [
        [mx.contiguous(part) for part in _row(_written(batch), stays, 0, tokens)]
        for batch in batches
    ]
    mx.eval(expected)
    kept = sum(part.nbytes for parts in expected for part in parts)
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()

    for batch in batches:
        batch.filter([stays])
    # One decode step: every layer appends, evaluated together at the end.
    new = mx.random.normal((1, HEADS, 1, DIM)).astype(mx.float16)
    for batch in batches:
        batch.update_and_fetch(new, new)
    mx.eval([part for batch in batches for part in _written(batch)])

    extra = mx.get_peak_memory() - before
    assert extra < 0.5 * kept, f"leave peaked {extra / kept:.2f} x the kept rows"
    # The leaving row's half of the old buffer is no longer live.
    assert mx.get_active_memory() - before < -0.8 * kept
    assert flushes == []
    for batch, parts in zip(batches, expected):
        assert batch.left_padding.tolist() == [0]
        assert _equal(_row(_written(batch), 0, 0, tokens), parts)


def test_filter_of_scattered_rows_still_gathers_them():
    rows = [_cache(200 + 50 * i, seed=80 + i) for i in range(3)]
    batch = BatchAffine8KVCache.merge(rows)
    expected = [_row(_written(batch), r, 100 - 50 * r, 300) for r in (0, 2)]
    expected = [[mx.contiguous(part) for part in parts] for parts in expected]
    mx.eval(expected)

    batch.filter([0, 2])
    mx.eval(batch.keys, batch.values)

    assert batch.offset.tolist() == [200, 300]
    assert batch.left_padding.tolist() == [100, 0]
    assert _equal(_row(_written(batch), 0, 100, 300), expected[0])
    assert _equal(_row(_written(batch), 1, 0, 300), expected[1])
    _append(batch)
    assert _equal(_row(_written(batch), 1, 0, 300), expected[1])

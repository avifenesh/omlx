# SPDX-License-Identifier: Apache-2.0
"""q8q verify attention (a few causal rows over a long affine8 history) against
the existing attention route on the same cache, read through the
update_and_fetch views."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import omlx.affine4 as a4
from tests.test_affine4 import require_tensor_ops


def _cache(keys, rows, seed, cache=None):
    mx.random.seed(seed)
    cache = cache or a4.Affine8KVCache()
    done = 0
    while done < keys - rows:
        n = min(4096, keys - rows - done)
        cache.update_and_fetch(
            mx.random.normal((1, 4, n, 256)).astype(mx.bfloat16),
            mx.random.normal((1, 4, n, 256)).astype(mx.bfloat16),
        )
        mx.eval(cache.state)
        done += n
    k, v = cache.update_and_fetch(
        mx.random.normal((1, 4, rows, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, rows, 256)).astype(mx.bfloat16),
    )
    q = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16)
    mx.eval(k, v, q)
    return cache, k, v, q


@pytest.mark.parametrize("rows", [2, 3, 5, 7, 8, 9, 12])
@pytest.mark.parametrize("keys", [4096, 5000])
def test_verify_matches_the_existing_route(monkeypatch, keys, rows):
    require_tensor_ops()
    monkeypatch.setattr(a4, "_Q8V_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TWO_TILE_MIN_KEYS", 1024)
    cache, k, v, q = _cache(keys, rows, seed=rows)

    def run(on):
        monkeypatch.setattr(a4, "_Q8V_ATTENTION", on)
        out = cache.attention(
            q, keys_state=k, values_state=v, scale=256**-0.5, mask="causal"
        )
        return out.astype(mx.float32)

    assert a4._q8q_verify_attention(cache, q, k, v, 256**-0.5, "causal") is not None
    ref, out = run(False), run(True)
    diff = mx.abs(out - ref)
    assert out.shape == ref.shape == (1, 24, rows, 256)
    assert mx.max(diff).item() < 0.01 * mx.max(mx.abs(ref)).item() + 1e-3
    assert mx.mean(diff).item() < 1e-3


@pytest.mark.parametrize(
    "rows,mask,keys",
    [
        (1, "causal", 4096),  # one row: the decode kernels
        (17, "causal", 4096),  # past the tile limit
        (8, None, 4096),  # not causal
        (8, "causal", 512),  # short history
    ],
)
def test_verify_falls_through(monkeypatch, rows, mask, keys):
    require_tensor_ops()
    monkeypatch.setattr(a4, "_Q8V_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TWO_TILE_MIN_KEYS", 1024)
    cache, k, v, q = _cache(keys, rows, seed=1)
    assert a4._q8q_verify_attention(cache, q, k, v, 256**-0.5, mask) is None


@pytest.mark.parametrize("padding", [0, 3])
def test_verify_on_a_batch_cache_needs_no_left_padding(monkeypatch, padding):
    require_tensor_ops()
    monkeypatch.setattr(a4, "_Q8V_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TWO_TILE_MIN_KEYS", 1024)
    cache, k, v, q = _cache(4096, 8, seed=2, cache=a4.BatchAffine8KVCache([padding]))
    out = a4._q8q_verify_attention(cache, q, k, v, 256**-0.5, "causal")
    assert (out is None) == bool(padding)


@pytest.mark.parametrize("split", [3, 7, 128])
@pytest.mark.parametrize("rows", [8, 16])
def test_verify_splits_and_a_tight_buffer(monkeypatch, split, rows):
    """Uneven, empty (more splits than pages) and long-run split counts, and
    states copied to exact-size buffers, against the views."""
    require_tensor_ops()
    from omlx.utils import q8q_verify as q8v

    monkeypatch.setattr(a4, "_Q8V_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TWO_TILE_MIN_KEYS", 1024)
    cache, k, v, q = _cache(5000, rows, seed=split)
    monkeypatch.setattr(a4, "_Q8V_ATTENTION", False)
    ref = cache.attention(
        q, keys_state=k, values_state=v, scale=256**-0.5, mask="causal"
    )
    ks, vs = cache._unwrap(k), cache._unwrap(v)
    tight = [
        SimpleNamespace(indices=mx.array(s.indices) + 0, norms=mx.array(s.norms) + 0)
        for s in (ks, vs)
    ]
    rq = cache.key_codec.prepare_queries(q).astype(mx.bfloat16)
    for keys_state, values_state in ((ks, vs), tight):
        out = q8v.run(rq, keys_state, values_state, 256**-0.5, split=split)
        out = cache.value_codec._rotate_inverse(out).astype(mx.float32)
        diff = mx.abs(out - ref.astype(mx.float32))
        assert mx.max(diff).item() < 0.01 * mx.max(mx.abs(ref)).item() + 1e-3
        assert mx.mean(diff).item() < 1e-3


@pytest.mark.parametrize("rows", [17, 30, 40])
def test_tail_rows_match_the_existing_route(monkeypatch, rows):
    """A new question's tail (17-40 rows) runs as tiles of 8 past the tail threshold."""
    require_tensor_ops()
    monkeypatch.setattr(a4, "_Q8V_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TWO_TILE_MIN_KEYS", 1024)
    monkeypatch.setattr(a4, "_Q8V_TAIL_MIN_KEYS", 4096)
    cache, k, v, q = _cache(5000, rows, seed=rows)
    assert a4._q8q_verify_attention(cache, q, k, v, 256**-0.5, "causal") is not None
    monkeypatch.setattr(a4, "_Q8V_ATTENTION", False)
    ref = cache.attention(q, keys_state=k, values_state=v, scale=256**-0.5, mask="causal")
    monkeypatch.setattr(a4, "_Q8V_ATTENTION", True)
    out = cache.attention(q, keys_state=k, values_state=v, scale=256**-0.5, mask="causal")
    diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
    assert mx.max(diff).item() < 0.01 * mx.max(mx.abs(ref)).item() + 1e-3
    assert mx.mean(diff).item() < 1e-3
    monkeypatch.setattr(a4, "_Q8V_TAIL", False)
    assert a4._q8q_verify_attention(cache, q, k, v, 256**-0.5, "causal") is None

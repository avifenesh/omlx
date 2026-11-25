# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 attention over an 8-bit affine KV cache (OMLX_INKLING_KV8).

``attend`` runs the dense kernels in the cache's rotated space: rotated query,
keys and values as codes * scale, output rotated back. That must equal dense
attention over the cache's own dequantized keys and values, and stay close to
attention over the original bf16 keys and values.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.affine4 import Affine8KVCache
from omlx.patches.inkling2 import kernels
from omlx.patches.inkling2.model import Inkling2Attention


def _inputs(b=1, hq=24, hkv=8, length=1, s=300, d=128, r=16, seed=0):
    keys = mx.random.split(mx.random.key(seed), 6)
    dt = mx.bfloat16
    q = (mx.random.normal((b, hq, length, d), key=keys[0]) * 0.3).astype(dt)
    k = (mx.random.normal((b, hkv, s, d), key=keys[1]) * 0.3).astype(dt)
    v = (mx.random.normal((b, hkv, s, d), key=keys[2]) * 0.3).astype(dt)
    rel = (mx.random.normal((b, length, hq, r), key=keys[3]) * 0.1).astype(dt)
    sink = (mx.random.normal((b, length, hq), key=keys[4]) * 0.1).astype(dt)
    return q, k, v, rel, sink


def _attn(window=0):
    return SimpleNamespace(scale=128**-0.5, window=window)


def _rel_err(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


@pytest.mark.parametrize("length", [1, 8])
def test_rotated_space_attention_matches_dequantized_dense(length):
    q, k, v, rel, sink = _inputs(length=length)
    cache = Affine8KVCache(bits=8, seed=3)
    out = Inkling2Attention.attend(_attn(), q, k, v, rel, sink, cache)
    mx.eval(out)
    ks, vs = cache.state
    kd = cache.key_codec.dequantize(ks).astype(mx.bfloat16)
    vd = cache.value_codec.dequantize(vs).astype(mx.bfloat16)
    ref = kernels.banded_sink_attention(q, kd, vd, rel, sink, scale=128**-0.5, window=0)
    assert out.shape == ref.shape and out.dtype == q.dtype
    assert _rel_err(out, ref) < 2e-2
    # And close to attention over the original bf16 keys and values.
    dense = kernels.banded_sink_attention(q, k, v, rel, sink, scale=128**-0.5, window=0)
    assert _rel_err(out, dense) < 3e-2


def test_dense_cache_path_is_unchanged():
    from mlx_lm.models.cache import KVCache

    q, k, v, rel, sink = _inputs()
    out = Inkling2Attention.attend(_attn(), q, k, v, rel, sink, KVCache())
    ref = kernels.banded_sink_attention(q, k, v, rel, sink, scale=128**-0.5, window=0)
    assert mx.array_equal(out, ref).item()


def test_flag_controls_dense_kv_attention(monkeypatch):
    import importlib

    import omlx.patches.inkling2.model as m

    monkeypatch.setenv("OMLX_INKLING_KV8", "1")
    assert importlib.reload(m).Model.dense_kv_attention is False
    monkeypatch.delenv("OMLX_INKLING_KV8")
    assert importlib.reload(m).Model.dense_kv_attention is True


@pytest.mark.parametrize("length", [1, 8])
def test_q8_decode_kernel_matches_the_dequantized_path(length, monkeypatch):
    import omlx.patches.inkling2.model as m

    q, k, v, rel, sink = _inputs(length=length, s=1500)
    on, off = Affine8KVCache(bits=8, seed=3), Affine8KVCache(bits=8, seed=3)
    monkeypatch.setattr(m, "_Q8_DECODE", True)
    got = Inkling2Attention.attend(_attn(), q, k, v, rel, sink, on)
    monkeypatch.setattr(m, "_Q8_DECODE", False)
    ref = Inkling2Attention.attend(_attn(), q, k, v, rel, sink, off)
    assert _rel_err(got, ref) < 1e-2


def test_q8_decode_kernel_reads_left_padded_batch_rows(monkeypatch):
    import omlx.patches.inkling2.model as m
    from omlx.patches.inkling2 import kernels as kn

    q, k, v, rel, sink = _inputs(b=2, length=1, s=700)
    cache = Affine8KVCache(bits=8, seed=3)
    cache.update_and_fetch(k, v)
    ks, vs = cache.state
    qr = cache.key_codec.prepare_queries(q).astype(q.dtype)
    pad = mx.array([0, 250], dtype=mx.int32)

    class _S:
        def __init__(self, st, n):
            self.indices, self.norms, self.shape = st.indices, st.norms, (2, 8, n, 0)

    got = kn.gqa_flash_decode_attention_q8(
        qr, _S(ks, 700), _S(vs, 700), rel, sink, scale=128**-0.5, window=0, left_padding=pad
    )
    kd = cache.key_codec.dequantize_rotated(ks, mx.bfloat16)
    vd = cache.value_codec.dequantize_rotated(vs, mx.bfloat16)
    ref = kn.gqa_flash_decode_attention(
        qr, kd, vd, rel, sink, scale=128**-0.5, window=0, left_padding=pad
    )
    assert _rel_err(got, ref) < 1e-2
    assert monkeypatch is not None and m is not None

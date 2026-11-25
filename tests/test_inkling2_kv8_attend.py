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


@pytest.mark.parametrize("b,length,pad", [(1, 2, None), (1, 8, None), (2, 8, [0, 300]), (3, 16, [5, 0, 900])])
def test_q8_verify_shared_tiles_match_per_position_kernel(b, length, pad, monkeypatch):
    from omlx.patches.inkling2 import kernels as kn

    q, k, v, rel, sink = _inputs(b=b, length=length, s=2100)
    cache = Affine8KVCache(bits=8, seed=3)
    cache.update_and_fetch(k, v)
    ks, vs = cache.state
    qr = cache.key_codec.prepare_queries(q).astype(q.dtype)
    lp = None if pad is None else mx.array(pad, dtype=mx.int32)

    class _S:
        def __init__(self, st):
            self.indices, self.norms, self.shape = st.indices, st.norms, (b, 8, 2100, 0)

    out = {}
    for shared in (True, False):
        monkeypatch.setattr(kn, "_Q8_SHARED_TILES", shared)
        out[shared] = kn.gqa_flash_decode_attention_q8(
            qr, _S(ks), _S(vs), rel, sink, scale=128**-0.5, window=0, left_padding=lp
        )
    # Tiles and chunks group keys differently per row, so the two kernels
    # round differently; both must be as close to an FP32 reference.
    kd = cache.key_codec.dequantize_rotated(ks, mx.float32)
    vd = cache.value_codec.dequantize_rotated(vs, mx.float32)
    ref = kn._masked_attention(
        qr.astype(mx.float32), kd, vd, rel.astype(mx.float32), sink.astype(mx.float32),
        128**-0.5, 0, lp,
    )
    shared_err, per_pos_err = _rel_err(out[True], ref), _rel_err(out[False], ref)
    assert shared_err < 1e-2 and shared_err <= 1.25 * per_pos_err + 1e-3, (shared_err, per_pos_err)


@pytest.mark.parametrize(
    "b,length,pad,window",
    [(1, 8, None, 0), (3, 8, [0, 300, 7], 0), (2, 16, None, 0), (2, 8, [0, 3], 64)],
)
def test_dense_verify_shared_tiles_match_reference(b, length, pad, window, monkeypatch):
    from omlx.patches.inkling2 import kernels as kn

    s = 120 if window else 2100
    q, k, v, rel, sink = _inputs(b=b, length=length, s=s, r=64 if window else 16)
    lp = None if pad is None else mx.array(pad, dtype=mx.int32)
    out = {}
    for shared in (True, False):
        monkeypatch.setattr(kn, "_Q8_SHARED_TILES", shared)
        out[shared] = kn.gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=128**-0.5, window=window, left_padding=lp
        )
    ref = kn._masked_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        rel.astype(mx.float32), sink.astype(mx.float32), 128**-0.5, window, lp,
    )
    shared_err, per_pos_err = _rel_err(out[True], ref), _rel_err(out[False], ref)
    assert shared_err < 1e-2 and shared_err <= 1.25 * per_pos_err + 1e-3, (shared_err, per_pos_err)
    # A lone row now takes the shared kernel through the dispatcher too.
    monkeypatch.setattr(kn, "_Q8_SHARED_TILES", True)
    routed = kn.banded_sink_attention(q, k, v, rel, sink, scale=128**-0.5, window=window, left_padding=lp)
    assert _rel_err(routed, ref) < 1e-2

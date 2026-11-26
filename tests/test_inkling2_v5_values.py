# SPDX-License-Identifier: Apache-2.0
"""affine8 caches with 5-bit values (OMLX_AFFINE8_VALUE_BITS=5, read at import).

Run in a subprocess so the module-level value width is set before import:
the codec packs a dense 5-bit stream on GPU and CPU alike, the cache keeps
8-bit keys and 5-bit values through merge and extract, and the in-place
decode and shared-tile verify kernels match attention over the cache's own
dequantized keys and values.
"""

import os
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

SCRIPT = textwrap.dedent(
    """
    import mlx.core as mx
    from types import SimpleNamespace
    import omlx.affine4 as a
    import omlx.patches.inkling2.model as m
    from omlx.patches.inkling2 import kernels as kn
    assert a._VALUE_BITS == 5
    c5 = a.Affine5Codec(128, seed=4)
    x = (mx.random.normal((2, 8, 70, 128)) * 0.4).astype(mx.bfloat16)
    st = c5.quantize(x)
    assert st.indices.shape[-1] == 20
    with mx.stream(mx.cpu):
        assert mx.array_equal(c5.quantize(x).indices, st.indices).item()
    used = set()
    orig = kn._q8_kernel
    kn._q8_kernel = lambda v5, shared: (used.add((v5, shared)), orig(v5, shared))[1]
    def err(x, y):
        x, y = x.astype(mx.float32), y.astype(mx.float32)
        return (mx.abs(x - y).max() / mx.abs(y).max()).item()
    for b, L in ((1, 1), (3, 8)):
        ks = mx.random.split(mx.random.key(b * 10 + L), 6)
        q = (mx.random.normal((b, 24, L, 128), key=ks[0]) * 0.3).astype(mx.bfloat16)
        k = (mx.random.normal((b, 8, 900, 128), key=ks[1]) * 0.3).astype(mx.bfloat16)
        v = (mx.random.normal((b, 8, 900, 128), key=ks[2]) * 0.3).astype(mx.bfloat16)
        rel = (mx.random.normal((b, L, 24, 16), key=ks[3]) * 0.1).astype(mx.bfloat16)
        sink = (mx.random.normal((b, L, 24), key=ks[4]) * 0.1).astype(mx.bfloat16)
        cache = a.Affine8KVCache(bits=8, seed=3)
        out = m.Inkling2Attention.attend(
            SimpleNamespace(scale=128**-0.5, window=0), q, k, v, rel, sink, cache)
        kst, vst = cache.state
        assert (cache.key_codec.bits, cache.value_codec.bits) == (8, 5)
        ref = kn._masked_attention(
            q.astype(mx.float32), cache.key_codec.dequantize(kst).astype(mx.float32),
            cache.value_codec.dequantize(vst).astype(mx.float32),
            rel.astype(mx.float32), sink.astype(mx.float32), 128**-0.5, 0, None)
        assert err(out, ref) < 1e-2, err(out, ref)
        if b == 3:
            row = a.Affine8KVCache.merge([cache]).extract(0)
            assert row.value_codec.bits == 5 and row.state[1].indices.shape[-1] == 20
    assert used == {(True, False), (True, True)}, used
    print("ok")
    """
)


@pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels")
def test_five_bit_values_pack_merge_and_attend():
    env = dict(os.environ, OMLX_AFFINE8_VALUE_BITS="5")
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stderr[-2000:]

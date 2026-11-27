# SPDX-License-Identifier: Apache-2.0
"""Bounded attention state for Inkling 2's sliding layer visits.

A sliding visit attends to its newest ``window`` keys only, so older keys are
never read again. Its keys and values are therefore state of bounded size, and
they live beside the visit's conv windows in one ``ArraysCache``:

    slot 0, 1   keys and values, [B, heads, n, dim], oldest first
    slot 2-5    the four conv windows
    slot 6      left padding of each row, [B] int32

A full-attention visit keeps a ``KVCache`` and a plain four-slot
``ArraysCache``.

The state holds the ``window - 1`` keys the next token can still reach.
Appending writes into a buffer with room to grow, so a decode step copies
nothing; the buffer is re-cut once per ``window`` tokens. Keys are only ever
dropped in whole attention tiles, so a single row's prefill tiles sit where a
cache of every key would have them and its output is bit-identical to it.

Rows of a batched verify that keep fewer drafts end short of the newest
column ("lag"); appends write each row at its own end and the decode kernels
read each row's end. The re-cut, or any read of slots 0, 1 or 6 from outside
this module, first moves such rows right behind more left padding.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache as _ArraysCache

from .kernels import decode_reads_row_ends

KEYS, VALUES, CONV, PADDING = 0, 1, 2, 6
SLOTS = 7

# Keys per tile of the flash prefill kernel (BK in kernels._FLASH_SOURCE).
_TILE = 16


class ArraysCache(_ArraysCache):
    """``ArraysCache`` of a sliding visit, joinable across rows.

    Rows hold key windows of different lengths, so joining left-pads the
    shorter ones and slot 6 says by how much. The class keeps mlx-lm's name
    because oMLX selects its cache handlers by class name; a restored state
    arrives as the plain class and ``adopt`` turns it into this one.
    """

    # A batched verify rolls every row back by one count (shared accept).
    _omlx_mtp_batch_rollback_cache = True

    def __init__(self, size: int = SLOTS, left_padding=None, window: int = 0):
        super().__init__(size, left_padding)
        self.window = window

    # Every reader outside this module sees rows that end at the last column.
    @property
    def cache(self):
        _settle(self)
        return self._slots

    @cache.setter
    def cache(self, value):
        self._slots = value

    def __getitem__(self, idx):
        if idx in (KEYS, VALUES, PADDING):
            _settle(self)
        return self._slots[idx]

    def __setitem__(self, idx, value):
        if idx in (KEYS, VALUES, PADDING):
            _settle(self)
        self._slots[idx] = value

    @classmethod
    def adopt(cls, entry, window: int | None = None):
        """The state behind ``entry`` (oMLX wraps restored states), as this class."""
        state = entry
        while not isinstance(state, _ArraysCache):
            state = state._inner
        if type(state) is _ArraysCache:
            state._slots = state.__dict__.pop("cache")
            state.__class__ = cls
            state.window = 0
        if window is not None:
            state.window = window
        return state

    def extract(self, idx):
        row = ArraysCache(len(self.cache), window=self.window)
        row.cache = [None if c is None else c[idx : idx + 1] for c in self.cache]
        pad = _padding(self)[idx]
        if pad and row.cache[KEYS] is not None:
            row.cache[KEYS] = row.cache[KEYS][:, :, pad:]
            row.cache[VALUES] = row.cache[VALUES][:, :, pad:]
        if row.cache[KEYS] is not None:
            row.cache[PADDING] = mx.zeros((1,), dtype=mx.int32)
        return row

    def filter(self, batch_indices):
        super().filter(batch_indices)
        pads = _padding(self)
        drop = min(pads, default=0)
        if drop:
            # Whole padding columns only.
            self.cache[KEYS] = self.cache[KEYS][:, :, drop:]
            self.cache[VALUES] = self.cache[VALUES][:, :, drop:]
            self.cache[PADDING] = mx.array([p - drop for p in pads], dtype=mx.int32)

    def extend(self, other):
        joined = _join([self, other])
        self.cache = joined.cache
        self.left_padding = None
        self.lengths = None

    @classmethod
    def merge(cls, caches):
        return _join(caches)


def _padding(state) -> list[int]:
    keys = state.cache[KEYS]
    if keys is None:
        return []
    pad = state.cache[PADDING]
    if pad is None:
        return [0] * keys.shape[0]
    return [int(p) for p in pad.tolist()]


def _join(states) -> ArraysCache:
    """One batched state from ``states``; windows right-aligned."""
    joined = ArraysCache(len(states[0].cache), window=states[0].window)
    widths = [0 if s.cache[KEYS] is None else s.cache[KEYS].shape[2] for s in states]
    width = max(widths)
    for slot in range(len(joined.cache)):
        parts = [s.cache[slot] for s in states]
        template = next((p for p in parts if p is not None), None)
        if slot == PADDING or template is None:
            continue
        rows = []
        for state, part, own in zip(states, parts, widths):
            if part is None:
                shape = (state.batch_size, *template.shape[1:])
                part = mx.zeros(shape, dtype=template.dtype)
                if slot in (KEYS, VALUES):
                    part = part[:, :, :0]
            if slot in (KEYS, VALUES) and own < width:
                fill = (part.shape[0], part.shape[1], width - own, part.shape[3])
                part = mx.concatenate(
                    [mx.zeros(fill, dtype=template.dtype), part], axis=2
                )
            rows.append(part)
        joined.cache[slot] = mx.concatenate(rows, axis=0)
    if width:
        pads = []
        for state, own in zip(states, widths):
            if state.cache[KEYS] is None:
                pads.extend([width] * state.batch_size)
            else:
                pads.extend(p + width - own for p in _padding(state))
        joined.cache[PADDING] = mx.array(pads, dtype=mx.int32)
    return joined


class _Buffer:
    """Growth buffers of one state; row ``b`` holds keys up to ``n - lags[b]``."""

    __slots__ = ("keys", "values", "n", "view", "pads", "shown", "lags")

    def __init__(self):
        self.keys = self.values = self.view = None
        self.n = 0
        # Left padding of each row, in buffer rows and as last published.
        self.pads: list[int] = []
        self.shown: list[int] | None = None
        self.lags: list[int] = []

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]


def _buffer(state) -> _Buffer:
    buffer = state.__dict__.get("_omlx_window")
    keys = state._slots[KEYS]
    if buffer is not None and buffer.view is keys:
        return buffer
    # A new, restored, joined or filtered state: start from what it holds.
    buffer = _Buffer()
    if keys is not None:
        buffer.keys, buffer.values = keys, state._slots[VALUES]
        buffer.view = keys
        buffer.n = keys.shape[2]
        pad = state._slots[PADDING]
        buffer.pads = [0] * keys.shape[0] if pad is None else pad.tolist()
        buffer.shown = list(buffer.pads)
        buffer.lags = [0] * keys.shape[0]
    state.__dict__["_omlx_window"] = buffer
    return buffer


def _settle(state) -> None:
    """Move rows that end short of the newest column right, behind more left
    padding, so every row ends at the last column of the state slots."""
    buffer = state.__dict__.get("_omlx_window")
    if buffer is None or not any(buffer.lags):
        return
    if buffer.view is not state._slots[KEYS]:
        return
    start = _floor_tile(buffer.n - (state.window - 1))
    state._slots[KEYS] = state._slots[VALUES] = buffer.view = None
    moves = []
    for row, shift in enumerate(buffer.lags):
        if not shift:
            continue
        lo = max(buffer.pads[row] + shift, start)
        for data in (buffer.keys, buffer.values):
            moved = data[row : row + 1, :, lo - shift : buffer.n - shift] * 1
            moves.append((data, row, lo, moved))
        buffer.pads[row] += shift
    # The copies run first and let go of the buffers, so the writes are in place.
    mx.async_eval([moved for *_, moved in moves])
    for data, row, lo, moved in moves:
        data[row : row + 1, :, lo : buffer.n] = moved
    buffer.lags = [0] * len(buffer.lags)
    _publish(state, buffer)


def _floor_tile(n: int) -> int:
    return max(0, n) // _TILE * _TILE


def _first(state, buffer: _Buffer) -> int:
    """The first column any row's next step can still reach."""
    return _floor_tile(buffer.n - max(buffer.lags, default=0) - (state.window - 1))


def _publish(state, buffer: _Buffer) -> None:
    """Set the state slots to the keys a later step can still reach."""
    start = _first(state, buffer)
    slots = state._slots
    slots[KEYS] = buffer.keys[:, :, start : buffer.n]
    slots[VALUES] = buffer.values[:, :, start : buffer.n]
    buffer.view = slots[KEYS]
    pads = [max(0, p - start) for p in buffer.pads]
    if pads != buffer.shown or slots[PADDING] is None:
        slots[PADDING] = mx.array(pads, dtype=mx.int32)
        buffer.shown = pads


def append(state, keys: mx.array, values: mx.array):
    """Add a chunk's keys and values to a sliding visit's state.

    Returns the keys and values the chunk attends to, the chunk's own rows
    last, the rows' left padding (None when no row is padded) and each row's
    end in those keys (None when every row ends at the last one).
    """
    buffer = _buffer(state)
    window = state.window
    if window <= 0:
        raise ValueError("A sliding state needs its layer's window size")
    length = keys.shape[2]
    if any(buffer.lags) and not decode_reads_row_ends(length, keys.dtype):
        _settle(state)
        buffer = _buffer(state)
    first = _first(state, buffer)
    if buffer.n + length > buffer.capacity:
        # Re-cut: the reachable keys move to the front of a buffer with room
        # for this chunk, or for the next ``window`` decode steps. Rows that
        # end short move right on the way, behind more left padding.
        kept = buffer.n - first
        b, heads, _, dim = keys.shape
        shape = (b, heads, kept + max(length, window), dim)
        new_keys = mx.zeros(shape, dtype=keys.dtype)
        new_values = mx.zeros(shape, dtype=values.dtype)
        if kept and not any(buffer.lags):
            new_keys[:, :, :kept] = buffer.keys[:, :, first : buffer.n]
            new_values[:, :, :kept] = buffer.values[:, :, first : buffer.n]
        elif kept:
            for row, lag in enumerate(buffer.lags):
                end = buffer.n - lag
                rows = slice(row, row + 1)
                new_keys[rows, :, lag:kept] = buffer.keys[rows, :, first:end]
                new_values[rows, :, lag:kept] = buffer.values[rows, :, first:end]
        buffer.keys, buffer.values, buffer.n = new_keys, new_values, kept
        buffer.pads = [
            max(0, p - first) + lag for p, lag in zip(buffer.pads, buffer.lags)
        ]
        buffer.lags = [0] * b
        first = 0
    if not buffer.pads:
        buffer.pads = [0] * keys.shape[0]
        buffer.lags = [0] * keys.shape[0]
    # The state slots are views of the buffers; dropping them first lets the
    # writes below happen in place.
    state._slots[KEYS] = state._slots[VALUES] = buffer.view = None
    if any(buffer.lags):
        for row, lag in enumerate(buffer.lags):
            end = buffer.n - lag
            rows = slice(row, row + 1)
            buffer.keys[rows, :, end : end + length] = keys[rows]
            buffer.values[rows, :, end : end + length] = values[rows]
    else:
        buffer.keys[:, :, buffer.n : buffer.n + length] = keys
        buffer.values[:, :, buffer.n : buffer.n + length] = values
    buffer.n += length
    _publish(state, buffer)
    left_padding = row_ends = None
    if any(buffer.pads):
        pads = [max(0, p - first) for p in buffer.pads]
        left_padding = mx.array(pads, dtype=mx.int32)
    if any(buffer.lags):
        row_ends = [buffer.n - lag - first for lag in buffer.lags]
    return (
        buffer.keys[:, :, first : buffer.n],
        buffer.values[:, :, first : buffer.n],
        left_padding,
        row_ends,
    )


def trim(state, count: int) -> bool:
    """Drop the ``count`` newest keys: the rejected drafts of the verify
    block just appended, which the buffer still holds the window for."""
    buffer = _buffer(state)
    if buffer.keys is None or not 0 <= count <= buffer.n - max(buffer.lags, default=0):
        return False
    state._slots[KEYS] = state._slots[VALUES] = buffer.view = None
    buffer.n -= count
    _publish(state, buffer)
    return True


def trim_rows(state, counts) -> bool:
    """``trim`` by ``counts[b]`` keys in row ``b``. Rows that drop more than
    the fewest end short of the newest column until they are moved."""
    buffer = _buffer(state)
    if buffer.keys is None or min(counts) < 0:
        return False
    ends = [buffer.n - lag - count for lag, count in zip(buffer.lags, counts)]
    if min(ends) < 0:
        return False
    state._slots[KEYS] = state._slots[VALUES] = buffer.view = None
    buffer.n = max(ends)
    buffer.lags = [buffer.n - end for end in ends]
    _publish(state, buffer)
    return True


def newest(state, count: int):
    """The ``count`` newest keys and values, e.g. the chunk just appended."""
    buffer = _buffer(state)
    return (
        buffer.keys[:, :, buffer.n - count : buffer.n],
        buffer.values[:, :, buffer.n - count : buffer.n],
    )


def visits(cache: list) -> list[tuple]:
    """``(KVCache or None, state)`` of each layer visit, read off the entries."""
    out, entry = [], 0
    while entry < len(cache):
        if isinstance(getattr(cache[entry], "cache", None), list):
            out.append((None, ArraysCache.adopt(cache[entry])))
            entry += 1
        else:
            out.append((cache[entry], cache[entry + 1]))
            entry += 2
    return out

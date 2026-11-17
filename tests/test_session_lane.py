# SPDX-License-Identifier: Apache-2.0
"""Session lane: the scheduler keeps a finished request's affine attention
caches for the next request on the same prompt prefix."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.affine4 import Affine8KVCache
from omlx.scheduler import Scheduler


def _live(length, seed=7):
    cache = Affine8KVCache(bits=8, seed=seed)
    cache.update_and_fetch(
        mx.random.normal((1, 2, length, 16)).astype(mx.float16),
        mx.random.normal((1, 2, length, 16)).astype(mx.float16),
    )
    mx.eval(cache.state)
    return cache


def _scheduler(enabled=True):
    sched = SimpleNamespace(
        config=SimpleNamespace(session_lane=enabled), _session_lane=None
    )
    sched._drop_session_lane = lambda: Scheduler._drop_session_lane(sched)
    return sched


def _request(tokens, rid="r"):
    return SimpleNamespace(prompt_token_ids=list(tokens), request_id=rid)


def test_lane_serves_a_request_on_the_same_prefix():
    sched = _scheduler()
    live = _live(12)
    Scheduler._keep_session_lane(sched, _request(range(10)), [live, object()])
    layers = Scheduler._take_session_lane(sched, _request(list(range(8)) + [5, 6]), 8)
    assert layers == {0: live}
    assert sched._session_lane is None


@pytest.mark.parametrize(
    "tokens,cached",
    [
        (list(range(7)) + [42, 1], 8),  # diverges before the hit
        (list(range(10)) + [1, 2, 3, 4], 13),  # hit past the cached tokens
    ],
)
def test_lane_misses_are_dropped(tokens, cached):
    sched = _scheduler()
    Scheduler._keep_session_lane(sched, _request(range(10)), [_live(12)])
    assert Scheduler._take_session_lane(sched, _request(tokens), cached) is None
    assert sched._session_lane is None


def test_lane_needs_the_setting_and_live_state():
    sched = _scheduler(enabled=False)
    Scheduler._keep_session_lane(sched, _request(range(10)), [_live(12)])
    assert sched._session_lane is None
    sched = _scheduler()
    emptied = _live(12)
    emptied.filter([])
    Scheduler._keep_session_lane(sched, _request(range(10)), [emptied])
    assert sched._session_lane is None


def test_lane_is_not_kept_without_memory_headroom():
    sched = _scheduler()
    sched._admission_limit_bytes = lambda: 1
    Scheduler._keep_session_lane(sched, _request(range(10)), [_live(12)])
    assert sched._session_lane is None

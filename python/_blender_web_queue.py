# SPDX-License-Identifier: GPL-2.0-or-later

"""Shared browser Worker queues for multiprocessing's spawn context."""

from multiprocessing import context, reduction
import operator
from queue import Empty, Full
from types import GenericAlias
import weakref

import _blender_web_process as _worker


_QUEUE_ROLE = 2
_ForkingPickler = reduction.ForkingPickler


def _timeout(block, timeout):
    if not block:
        return 0.0
    if timeout is None:
        return -1.0
    return max(timeout, 0.0)


class _ChannelQueue:
    def _create(self, maxsize):
        channel = _worker.channel_create(maxsize)
        _worker.channel_retain(channel, _QUEUE_ROLE)
        self._attach(channel, maxsize)

    def _attach(self, channel, maxsize):
        self._channel = channel
        self._maxsize = maxsize
        try:
            self._release = weakref.finalize(
                self, _worker.channel_release, channel, _QUEUE_ROLE
            )
        except BaseException:
            _worker.channel_release(channel, _QUEUE_ROLE)
            raise

    def __getstate__(self):
        context.assert_spawning(self)
        channel = self._open_channel()
        ticket = _worker.channel_export(channel, _QUEUE_ROLE)
        try:
            context.get_spawning_popen()._channel_tickets.append(ticket)
        except BaseException:
            _worker.channel_cancel(ticket)
            raise
        return ticket, self._maxsize

    def __setstate__(self, state):
        ticket, maxsize = state
        self._attach(_worker.channel_import(ticket), maxsize)

    def _open_channel(self):
        if not self._release.alive:
            raise ValueError(f"Queue {self!r} is closed")
        return self._channel

    def _put(self, obj, block=True, timeout=None):
        channel = self._open_channel()
        seconds = _timeout(block, timeout)
        payload = bytes(_ForkingPickler.dumps(obj))
        if not _worker.channel_put(channel, payload, seconds):
            raise Full

    def _get(self, block=True, timeout=None):
        channel = self._open_channel()
        payload = _worker.channel_get(channel, _timeout(block, timeout))
        if payload is None:
            raise Empty
        return _ForkingPickler.loads(payload)

    def empty(self):
        return _worker.channel_qsize(self._open_channel()) == 0

    def close(self):
        self._release()

    __class_getitem__ = classmethod(GenericAlias)


class Queue(_ChannelQueue):
    def __init__(self, maxsize=0, *, ctx=None):
        self._create(max(operator.index(maxsize), 0))

    def put(self, obj, block=True, timeout=None):
        self._put(obj, block, timeout)

    def get(self, block=True, timeout=None):
        return self._get(block, timeout)

    def put_nowait(self, obj):
        self.put(obj, False)

    def get_nowait(self):
        return self.get(False)

    def qsize(self):
        return _worker.channel_qsize(self._open_channel())

    def full(self):
        size = self.qsize()
        return self._maxsize > 0 and size >= self._maxsize

    def join_thread(self):
        # Each put hands the payload to the shared channel before it returns.
        assert not self._release.alive, f"Queue {self!r} not closed"

    def cancel_join_thread(self):
        pass


class SimpleQueue(_ChannelQueue):
    def __init__(self, *, ctx=None):
        self._create(0)

    def put(self, obj):
        self._put(obj)

    def get(self):
        return self._get()

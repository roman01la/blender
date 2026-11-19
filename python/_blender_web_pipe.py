# SPDX-License-Identifier: GPL-2.0-or-later

"""Message-oriented multiprocessing connections backed by browser Worker channels."""

from contextlib import ExitStack
import io
from multiprocessing import context
from multiprocessing.connection import Connection as _Connection

import _blender_web_process as _worker


_READER = 0
_WRITER = 1


class Connection(_Connection):
    def __init__(self, reader, writer):
        super().__init__(reader if reader is not None else writer,
                         reader is not None, writer is not None)
        self._reader = reader
        self._writer = writer

    def fileno(self):
        self._check_closed()
        raise OSError("browser pipe connections have no file descriptor")

    def _release_reader(self):
        reader, self._reader = self._reader, None
        if reader is not None:
            _worker.channel_release(reader, _READER)

    def _close(self):
        try:
            self._release_reader()
        finally:
            writer, self._writer = self._writer, None
            if writer is not None:
                _worker.channel_release(writer, _WRITER)

    def _bad_message_length(self):
        if self._writable:
            self._release_reader()
        super()._bad_message_length()

    def _send_bytes(self, buffer):
        _worker.channel_put(self._writer, bytes(buffer), -1.0)

    def _recv_bytes(self, maxsize=None):
        payload = _worker.channel_get(self._reader, -1.0)
        if maxsize is not None and len(payload) > maxsize:
            return None
        message = io.BytesIO(payload)
        message.seek(0, io.SEEK_END)
        return message

    def _poll(self, timeout):
        seconds = -1.0 if timeout is None else max(float(timeout), 0.0)
        return _worker.channel_poll(self._reader, seconds)

    def __reduce__(self):
        self._check_closed()
        context.assert_spawning(self)
        tickets = context.get_spawning_popen()._channel_tickets
        reader_ticket = self._export(self._reader, _READER, tickets)
        writer_ticket = self._export(self._writer, _WRITER, tickets)
        return _rebuild_connection, (reader_ticket, writer_ticket)

    @staticmethod
    def _export(channel, role, tickets):
        if channel is None:
            return None
        ticket = _worker.channel_export(channel, role)
        try:
            tickets.append(ticket)
        except BaseException:
            _worker.channel_cancel(ticket)
            raise
        return ticket


def _claim_connection_channel(ticket, role, rollback):
    if ticket is None:
        return None
    channel = _worker.channel_import(ticket)
    rollback.callback(_worker.channel_release, channel, role)
    return channel


def _rebuild_connection(reader_ticket, writer_ticket):
    with ExitStack() as rollback:
        reader = _claim_connection_channel(reader_ticket, _READER, rollback)
        writer = _claim_connection_channel(writer_ticket, _WRITER, rollback)
        connection = Connection(reader, writer)
        rollback.pop_all()
    return connection


def _pipe_channel(rollback):
    channel = _worker.channel_create(0)
    try:
        _worker.channel_retain(channel, _READER)
    except BaseException:
        _worker.channel_close(channel)
        raise
    rollback.callback(_worker.channel_release, channel, _READER)
    _worker.channel_retain(channel, _WRITER)
    rollback.callback(_worker.channel_release, channel, _WRITER)
    return channel


def Pipe(duplex=True):
    """Return connected endpoints. If duplex is false, return reader, writer."""
    with ExitStack() as rollback:
        first = _pipe_channel(rollback)
        second = _pipe_channel(rollback) if duplex else None
        endpoints = Connection(first, second), Connection(second, first)
        rollback.pop_all()
    return endpoints

# SPDX-License-Identifier: GPL-2.0-or-later

"""A SyncManager whose server runs in an isolated browser Python Worker."""

from collections import deque
from itertools import count
import math
import multiprocessing
from multiprocessing import managers as _native, reduction
import operator
import os
from queue import Empty, Full
import threading
import time
import weakref

import _blender_web_process as _worker


_READER = 0
_QUEUE = 2
_CLIENT = os.urandom(16)
_PICKLE = reduction.ForkingPickler
_LOCAL_SERVER = None
_QUEUE_METHODS = (
    "empty", "full", "get", "get_nowait", "join", "put", "put_nowait",
    "qsize", "task_done",
)
_FACTORIES = (
    "list", "dict", "Namespace", "Value", "Array", "Queue", "JoinableQueue",
    "Event", "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Barrier",
)
_REGISTRY = _native.SyncManager._registry.copy()


def _deadline(timeout):
    if timeout is None:
        return None
    if math.isnan(timeout):
        raise ValueError("timeout cannot be NaN")
    return time.monotonic() + timeout


def _shared_memory():
    try:
        from multiprocessing import shared_memory
    except ImportError as error:
        raise NotImplementedError("browser shared memory is unavailable") from error
    return shared_memory


def _unlink_segments(names):
    if not names:
        return
    shared_memory = _shared_memory()
    for name in tuple(names):
        try:
            segment = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            pass
        else:
            try:
                segment.unlink()
            except FileNotFoundError:
                pass
            finally:
                segment.close()
        names.remove(name)


def _release_endpoint(endpoint):
    requests, liveness = endpoint
    try:
        _worker.channel_release(requests, _QUEUE)
    finally:
        _worker.channel_release(liveness, _READER)


def _rpc(endpoint, object_id, method, args=(), kwds=None, *, timeout=None):
    requests, liveness = endpoint
    if _worker.channel_poll(liveness, 0):
        raise EOFError("manager server has stopped")
    reply = _worker.channel_create(1)
    _worker.channel_retain(reply, _QUEUE)
    try:
        request = (reply, (_CLIENT, threading.get_ident()), object_id, method, args, kwds or {})
        _worker.channel_put(requests, bytes(_PICKLE.dumps(request)), -1)
        deadline = _deadline(timeout)
        while True:
            seconds = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
            response = _worker.channel_get(reply, seconds)
            if response is not None:
                break
            if _worker.channel_poll(liveness, 0):
                raise EOFError("manager server has stopped")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("manager server did not respond")
    finally:
        _worker.channel_release(reply, _QUEUE)
    kind, value = _PICKLE.loads(response)
    if kind == "error":
        raise value
    if kind == "proxy":
        return _restore_proxy(endpoint, *value)
    return value


class _WorkerProxy:
    def _callmethod(self, methodname, args=(), kwds=None):
        if _LOCAL_SERVER is not None and self._endpoint == _LOCAL_SERVER.endpoint:
            result = _LOCAL_SERVER.invoke((_CLIENT, threading.get_ident()), self._id,
                                          methodname, args, kwds or {})
            if isinstance(result, _Wait):
                raise RuntimeError("a manager referent cannot block its own server")
            return result
        return _rpc(self._endpoint, self._id, methodname, args, kwds)

    def __reduce__(self):
        return _restore_proxy, (self._endpoint, self._token.typeid, self._id)


def _make_proxy_class(kind):
    proxy = _REGISTRY[kind][3]
    if proxy is _native.AutoProxy:
        proxy = _native.MakeProxyType(f"AutoProxy[{kind}]", _QUEUE_METHODS)
    return type(proxy.__name__, (_WorkerProxy, proxy), {"__module__": __name__})


_PROXY_CLASSES = {kind: _make_proxy_class(kind) for kind in (*_FACTORIES, "Iterator")}


def _restore_proxy(endpoint, kind, object_id):
    proxy_type = _PROXY_CLASSES[kind]
    proxy = proxy_type.__new__(proxy_type)
    proxy._endpoint = endpoint
    proxy._id = object_id
    proxy._token = _native.Token(kind, endpoint, object_id)
    if _LOCAL_SERVER is None or endpoint != _LOCAL_SERVER.endpoint:
        requests, liveness = endpoint
        _worker.channel_retain(requests, _QUEUE)
        try:
            _worker.channel_retain(liveness, _READER)
        except BaseException:
            _worker.channel_release(requests, _QUEUE)
            raise
        try:
            proxy._close = weakref.finalize(proxy, _release_endpoint, endpoint)
        except BaseException:
            _release_endpoint(endpoint)
            raise
    return proxy


class _Wait:
    def __init__(self, attempt, timeout=None, on_timeout=False):
        self.attempt = attempt
        self.deadline = _deadline(timeout)
        self.on_timeout = on_timeout

    def step(self):
        ready, value = self.attempt()
        if ready:
            return value
        if self.deadline is not None and time.monotonic() >= self.deadline:
            if isinstance(self.on_timeout, type) and issubclass(self.on_timeout, Exception):
                raise self.on_timeout
            return self.on_timeout
        return self

    def cancel(self):
        pass


def _try_queue(method, args, unavailable):
    try:
        return True, method(*args)
    except unavailable:
        return False, None


def _queue_operation(obj, method, *args, **kwds):
    if method == "join":
        return _Wait(lambda: (not obj.unfinished_tasks, None))

    def put(item, block=True, timeout=None):
        if not block or timeout is not None and timeout < 0:
            return obj.put(item, block, timeout)
        return _Wait(lambda: _try_queue(obj.put_nowait, (item,), Full), timeout, Full)

    def get(block=True, timeout=None):
        if not block or timeout is not None and timeout < 0:
            return obj.get(block, timeout)
        return _Wait(lambda: _try_queue(obj.get_nowait, (), Empty), timeout, Empty)

    operation = put if method == "put" else get
    return operation(*args, **kwds)


class _RecursiveLock:
    def __init__(self):
        self.owner = None
        self.depth = 0

    def try_acquire(self, client):
        if self.owner is None or self.owner == client:
            self.owner = client
            self.depth += 1
            return True
        return False

    def release(self, client):
        if self.owner != client:
            raise RuntimeError("cannot release un-acquired lock")
        self.depth -= 1
        if self.depth == 0:
            self.owner = None


def _acquire(kind, obj, client, blocking=True, timeout=None):
    if kind in {"Lock", "RLock"}:
        if timeout == -1:
            timeout = None
        if timeout is not None and timeout < 0:
            raise ValueError("timeout value must be positive")
        if timeout is not None and timeout > threading.TIMEOUT_MAX:
            raise OverflowError("timeout value is too large")
    if timeout is not None and not blocking:
        raise ValueError("can't specify a timeout for a non-blocking call")
    attempt = (lambda: obj.try_acquire(client)) if kind == "RLock" else (lambda: obj.acquire(False))
    if not blocking:
        return attempt()
    return _Wait(lambda: (attempt(), True), timeout)


class _ConditionWait(_Wait):
    def __init__(self, condition, client, depth, timeout):
        self.condition = condition
        self.client = client
        self.depth = depth
        self.deadline = _deadline(timeout)
        self.notified = None

    def step(self):
        if self.notified is None and self.deadline is not None and time.monotonic() >= self.deadline:
            self.condition.waiting.remove(self)
            self.notified = False
            self.deadline = None
        if self.notified is not None and self.condition.try_acquire(self.client):
            if self.condition.kind == "RLock":
                self.condition.lock.depth = self.depth
            return self.notified
        return self

    def cancel(self):
        if self.notified is None:
            self.condition.waiting.remove(self)


class _Condition:
    def __init__(self, kind, lock):
        self.kind = kind
        self.lock = lock
        self.waiting = deque()

    def try_acquire(self, client):
        return (self.lock.try_acquire(client) if self.kind == "RLock"
                else self.lock.acquire(False))

    def acquire(self, client, *args, **kwds):
        return _acquire(self.kind, self.lock, client, *args, **kwds)

    def release(self, client):
        if self.kind == "RLock":
            self.lock.release(client)
        else:
            self.lock.release()

    def owned(self, client):
        return self.lock.owner == client if self.kind == "RLock" else self.lock.locked()

    def wait(self, client, timeout=None):
        if not self.owned(client):
            raise RuntimeError("cannot wait on un-acquired lock")
        depth = self.lock.depth if self.kind == "RLock" else 1
        operation = _ConditionWait(self, client, depth, timeout)
        if self.kind == "RLock":
            self.lock.depth = 1
        self.release(client)
        self.waiting.append(operation)
        return operation

    def notify(self, client, n=1):
        if not self.owned(client):
            raise RuntimeError("cannot notify on un-acquired lock")
        while self.waiting and n > 0:
            operation = self.waiting.popleft()
            operation.notified = True
            operation.deadline = None
            n -= 1

    def notify_all(self, client):
        self.notify(client, len(self.waiting))


class _BarrierWait(_Wait):
    def __init__(self, barrier, index, timeout):
        self.barrier = barrier
        self.index = index
        self.deadline = _deadline(timeout)
        self.released = False
        self.failed = False

    def step(self):
        if self.failed:
            raise threading.BrokenBarrierError
        if self.released:
            return self.index
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.barrier.abort()
            raise threading.BrokenBarrierError
        return self

    def cancel(self):
        if not self.released and not self.failed:
            self.barrier.abort()


class _Barrier:
    def __init__(self, parties, action=None, timeout=None):
        self.parties = operator.index(parties)
        self.action = action
        self.timeout = timeout
        self.waiting = []
        self.broken = False

    @property
    def n_waiting(self):
        return len(self.waiting)

    def wait(self, timeout=None):
        if self.broken:
            raise threading.BrokenBarrierError
        operation = _BarrierWait(self, self.n_waiting, self.timeout if timeout is None else timeout)
        self.waiting.append(operation)
        if self.n_waiting == self.parties:
            try:
                if self.action:
                    self.action()
            except BaseException:
                self.abort()
                raise
            for waiter in self.waiting:
                waiter.released = True
            self.waiting.clear()
        return operation

    def reset(self):
        for waiter in self.waiting:
            waiter.failed = True
        self.waiting.clear()
        self.broken = False

    def abort(self):
        self.reset()
        self.broken = True


class _ProxyResult:
    def __init__(self, kind, object_id):
        self.value = kind, object_id


class _Server:
    def __init__(self, endpoint, track_segments):
        self.endpoint = endpoint
        self.objects = {}
        self.pending = {}
        self.ids = count(1)
        self.segments = set() if track_segments else None

    def register(self, kind, obj):
        object_id = next(self.ids)
        self.objects[object_id] = kind, obj
        return _ProxyResult(kind, object_id)

    def condition(self, lock=None):
        if lock is None:
            return _Condition("RLock", _RecursiveLock())
        if not isinstance(lock, _WorkerProxy) or lock._endpoint != self.endpoint:
            raise TypeError("condition lock must belong to this manager")
        kind, referent = self.objects[lock._id]
        if kind not in {"Lock", "RLock"}:
            raise TypeError("condition requires a Lock or RLock proxy")
        return _Condition(kind, referent)

    def invoke(self, client, object_id, method, args, kwds):
        if method == "#create":
            kind, args, kwds = args
            factory = {
                "RLock": _RecursiveLock, "Condition": self.condition, "Barrier": _Barrier,
            }.get(kind, _REGISTRY[kind][0])
            return self.register(kind, factory(*args, **kwds))
        if method == "#ping":
            return None
        if method == "#track_segment" and object_id is None and self.segments is not None:
            self.segments.add(args[0])
            return None
        kind, obj = self.objects[object_id]
        if method == "#GETVALUE":
            return obj
        proxy_type = _PROXY_CLASSES[kind]
        if method not in proxy_type._exposed_ and method not in {"__repr__", "__str__"}:
            raise AttributeError(f"{kind} has no exposed method {method!r}")
        if kind in {"Queue", "JoinableQueue"} and method in {"get", "put", "join"}:
            return _queue_operation(obj, method, *args, **kwds)
        if kind == "Event" and method == "wait":
            timeout, = args
            return _Wait(lambda: (obj.is_set(), True), timeout)
        if kind in {"Lock", "RLock", "Semaphore", "BoundedSemaphore"} and method == "acquire":
            return _acquire(kind, obj, client, *args, **kwds)
        if kind == "RLock" and method == "release":
            return obj.release(client)
        if kind == "Condition" and method in proxy_type._exposed_:
            return getattr(obj, method)(client, *args, **kwds)
        value = getattr(obj, method)(*args, **kwds)
        result_kind = getattr(proxy_type, "_method_to_typeid_", {}).get(method)
        return self.register(result_kind, value) if result_kind is not None else value

    @staticmethod
    def retain_reply(reply):
        try:
            _worker.channel_retain(reply, _QUEUE)
        except ValueError:
            return False
        return True

    def send(self, reply, kind, value, *, retained=False):
        if not retained and not self.retain_reply(reply):
            return
        try:
            try:
                payload = bytes(_PICKLE.dumps((kind, value)))
            except Exception as error:
                payload = bytes(_PICKLE.dumps(("error", _native.RemoteError(str(error)))))
            _worker.channel_put(reply, payload, 0)
        finally:
            if not retained:
                _worker.channel_release(reply, _QUEUE)

    def respond(self, reply, result, *, retained=False):
        if isinstance(result, _ProxyResult):
            self.send(reply, "proxy", result.value, retained=retained)
        else:
            self.send(reply, "value", result, retained=retained)

    def step(self, reply, operation):
        if not self.retain_reply(reply):
            operation.cancel()
            return True
        try:
            try:
                result = operation.step()
            except Exception as error:
                self.send(reply, "error", error, retained=True)
                return True
            if result is operation:
                return False
            self.respond(reply, result, retained=True)
            return True
        finally:
            _worker.channel_release(reply, _QUEUE)

    def dispatch(self, request):
        reply, client, object_id, method, args, kwds = request
        if method == "#shutdown":
            self.respond(reply, None)
            return False
        try:
            result = self.invoke(client, object_id, method, args, kwds)
        except Exception as error:
            self.send(reply, "error", error)
        else:
            if isinstance(result, _Wait):
                if not self.step(reply, result):
                    self.pending[reply] = result
            else:
                self.respond(reply, result)
        for waiting, operation in tuple(self.pending.items()):
            if self.step(waiting, operation):
                del self.pending[waiting]
        return True

    def serve(self, requests):
        while True:
            deadlines = [operation.deadline for operation in self.pending.values()
                         if operation.deadline is not None]
            timeout = max(0, min(deadlines) - time.monotonic()) if deadlines else None
            try:
                request = requests.get(timeout=timeout)
            except Empty:
                for waiting, operation in tuple(self.pending.items()):
                    if self.step(waiting, operation):
                        del self.pending[waiting]
            else:
                if not self.dispatch(request):
                    return


def _run_server(requests, writer, endpoint, track_segments):
    global _LOCAL_SERVER
    _LOCAL_SERVER = _Server(endpoint, track_segments)
    try:
        _LOCAL_SERVER.serve(requests)
    finally:
        try:
            _unlink_segments(_LOCAL_SERVER.segments)
        finally:
            _LOCAL_SERVER = None
            writer.close()
            requests.close()


def _shutdown(process, requests, reader, endpoint, timeout, segments):
    try:
        if process.is_alive():
            try:
                _rpc(endpoint, None, "#shutdown", timeout=timeout)
            except (EOFError, OSError, TimeoutError, ValueError):
                pass
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()
    finally:
        try:
            _worker.channel_close(endpoint[0])
            while True:
                try:
                    _worker.channel_get(endpoint[0], 0)
                except EOFError:
                    break
        finally:
            reader.close()
            requests.close()
            _unlink_segments(segments)


class SyncManager(_native.SyncManager):
    def __init__(self, address=None, authkey=None, serializer="pickle",
                 ctx=None, *, shutdown_timeout=1.0):
        if address is not None or authkey is not None or serializer != "pickle":
            raise NotImplementedError("browser managers use a local pickle transport")
        self._ctx = ctx or multiprocessing.get_context("spawn")
        self._shutdown_timeout = shutdown_timeout
        self._address = None
        self._state = _native.State()
        self._state.value = _native.State.INITIAL
        self._segments = None

    def start(self):
        if self._state.value != _native.State.INITIAL:
            raise RuntimeError("manager has already started")
        requests = self._ctx.Queue()
        reader, writer = self._ctx.Pipe(False)
        endpoint = requests._open_channel(), reader._reader
        process = self._ctx.Process(
            target=_run_server, args=(requests, writer, endpoint, self._segments is not None),
            name=type(self).__name__,
        )
        process.daemon = True
        try:
            process.start()
        except BaseException:
            reader.close()
            requests.close()
            raise
        finally:
            writer.close()
        self._process = process
        self._address = endpoint
        self._cleanup = weakref.finalize(
            self, _shutdown, process, requests, reader, endpoint, self._shutdown_timeout,
            self._segments,
        )
        try:
            _rpc(endpoint, None, "#ping")
        except BaseException:
            self.shutdown()
            raise
        self._state.value = _native.State.STARTED

    def shutdown(self):
        if self._state.value != _native.State.SHUTDOWN and hasattr(self, "_cleanup"):
            self._state.value = _native.State.SHUTDOWN
            self._cleanup()

    def _create_proxy(self, kind, *args, **kwds):
        if self._state.value != _native.State.STARTED:
            raise RuntimeError("manager server is not running")
        return _rpc(self._address, None, "#create", (kind, args, kwds))

    def Pool(self, *args, **kwds):
        raise NotImplementedError("a browser manager Worker cannot launch a pool of workers")

    def get_server(self):
        raise NotImplementedError("browser managers have no network server")

    def connect(self):
        raise NotImplementedError("browser managers cannot connect to a network address")

    @classmethod
    def register(cls, *args, **kwds):
        raise NotImplementedError("browser managers support the built-in proxy types")


def _factory(kind):
    def create(self, *args, **kwds):
        return self._create_proxy(kind, *args, **kwds)
    create.__name__ = kind
    return create


for _kind in _FACTORIES:
    setattr(SyncManager, _kind, _factory(_kind))


class SharedMemoryManager(SyncManager):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._segments = set()

    def _track_segment(self, segment):
        name = segment.name
        try:
            self._segments.add(name)
            _rpc(self._address, None, "#track_segment", (name,))
        except BaseException:
            self._segments.discard(name)
            try:
                segment.unlink()
            finally:
                segment.close()
            raise

    def _memory_module(self):
        if self._state.value != _native.State.STARTED:
            raise RuntimeError("manager server is not running")
        return _shared_memory()

    def SharedMemory(self, size):
        segment = self._memory_module().SharedMemory(create=True, size=size, track=False)
        self._track_segment(segment)
        return segment

    def ShareableList(self, sequence):
        shared = self._memory_module().ShareableList(sequence)
        self._track_segment(shared.shm)
        return shared


def Manager(context=None):
    manager = SyncManager(ctx=context)
    manager.start()
    return manager

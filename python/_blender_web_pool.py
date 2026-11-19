# SPDX-License-Identifier: GPL-2.0-or-later

"""Multiprocessing pools with browser-thread result and callback delivery."""

from collections import deque
import io
import multiprocessing
from multiprocessing import TimeoutError, pool as _native, reduction
import operator
import os
from pathlib import Path
from queue import Empty
import sys
import time
import traceback
import types
import warnings
import weakref

if sys.platform == "emscripten":
    import _blender_web_process as _events
else:
    _events = None


_live_pools = weakref.WeakValueDictionary()


def _dispatch():
    while channel := _events.next_pool_event():
        pool = _live_pools.get(channel)
        if pool is not None:
            pool._pump()


class _SourcePickler(reduction.ForkingPickler):
    """Collect the Python modules named by a spawn pickle, including its arguments."""

    def __init__(self, buffer):
        super().__init__(buffer)
        self.modules = set()

    def reducer_override(self, obj):
        name = type(obj).__module__
        if callable(obj):
            name = getattr(obj, "__module__", name)
        if isinstance(name, str) and name and name.partition(".")[0] not in sys.stdlib_module_names:
            self.modules.add(name)
        return NotImplemented


def _encode_call(call, sources):
    buffer = io.BytesIO()
    pickler = _SourcePickler(buffer)
    pickler.dump(call)
    manifest = {}
    if sys.platform == "emscripten":
        for name in sorted(pickler.modules):
            parts = name.split(".")
            if not all(part.isidentifier() for part in parts):
                raise ValueError(f"Invalid spawn module name {name!r}")
            for index in range(1, len(parts) + 1):
                module_name = ".".join(parts[:index])
                if module_name not in sources:
                    module = sys.modules.get(module_name)
                    filename = getattr(module, "__file__", None)
                    source = None
                    if filename and filename.endswith(".py"):
                        path = Path(filename)
                        loader = getattr(getattr(module, "__spec__", None), "loader", None)
                        read = getattr(loader, "get_data", None)
                        source = (read(filename) if read else path.read_bytes(),
                                  path.name == "__init__.py")
                    sources[module_name] = source
                if sources[module_name] is not None:
                    manifest[module_name] = sources[module_name]
    return manifest, buffer.getvalue()


def _decode_call(payload):
    manifest, pickled = payload
    if manifest:
        root = Path("/process/modules")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        main_source = None
        for name, (source, package) in manifest.items():
            if name == "__main__":
                main_source = source
                continue
            path = root.joinpath(*name.split("."))
            path = path / "__init__.py" if package else path.with_suffix(".py")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source)
        if main_source is not None and sys.modules["__main__"].__name__ != "__mp_main__":
            main = types.ModuleType("__mp_main__")
            main.__file__ = str(root / "__mp_main__.py")
            sys.modules["__main__"] = sys.modules["__mp_main__"] = main
            exec(compile(main_source, main.__file__, "exec"), main.__dict__)
    return reduction.ForkingPickler.loads(pickled)


class _Initializer:
    def __init__(self, func, args, sources):
        self.call = (func, args, {})
        self.sources = sources

    def __reduce__(self):
        # Serialize shared Queue and Pipe arguments inside Process.start(),
        # where the spawn context can export ownership to this worker.
        return _decode_call, (_encode_call(self.call, self.sources),)


class _WorkerInput:
    def __init__(self, tasks):
        self.tasks = tasks

    def get(self):
        task = self.tasks.get()
        if task is None:
            return None
        job, index, payload = task
        try:
            func, args, kwds = _decode_call(payload)
        except Exception as error:
            return job, index, _raise_decode_error, (error,), {}
        return job, index, func, args, kwds


def _raise_decode_error(error):
    raise error


def _pool_worker(tasks, results, token, initialize, max_tasks):
    initializer, initargs, _ = initialize
    _native.worker(
        _WorkerInput(tasks), results, initializer, initargs, max_tasks, True
    )
    results.put((None, token, None))


def _stop_workers(workers, tasks, results, channel):
    if channel is not None:
        _live_pools.pop(channel, None)
        _events.watch_pool(channel, False)
    for worker in workers.values():
        if worker.is_alive():
            worker.terminate()
    tasks.close()
    results.close()


class _Result:
    def __init__(self, pool, callback=None, error_callback=None):
        self._pool = pool
        self._job = next(_native.job_counter)
        self._done = False
        self._callback = callback
        self._error_callback = error_callback
        pool._cache[self._job] = self

    def ready(self):
        if not self._done:
            self._pool._pump()
        return self._done

    def successful(self):
        if not self.ready():
            raise ValueError(f"{self!r} not ready")
        return self._success

    def wait(self, timeout=None):
        if not self._done:
            self._pool._wait(lambda: self._done, timeout)

    def get(self, timeout=None):
        self.wait(timeout)
        if not self._done:
            raise TimeoutError
        if not self._success:
            raise self._value
        return self._value

    def _finish(self, success, value, notify=True):
        self._done = True
        self._success = success
        self._value = value
        self._pool._cache.pop(self._job)
        self._pool = None
        callback = self._callback if success else self._error_callback
        self._callback = self._error_callback = None
        if notify and callback is not None:
            try:
                callback(value)
            except BaseException:
                traceback.print_exc()

    def _set(self, index, result):
        self._finish(*result)

    def _fail(self, error):
        self._finish(False, error, notify=False)


class _MapResult(_Result):
    def __init__(self, pool, length, chunksize, callback, error_callback):
        super().__init__(pool, callback, error_callback)
        self._chunksize = chunksize
        self._remaining = 0
        self._submitting = True
        self._success = True
        self._value = [None] * length

    def _submit(self, index, func, args, kwds):
        self._remaining += 1
        self._pool._submit(self, index, func, args, kwds)

    def _seal(self, empty):
        self._submitting = False
        if not self._remaining:
            self._finish(self._success, self._value, notify=not empty)

    def _set(self, index, result):
        success, value = result
        if self._success:
            if success:
                start = index * self._chunksize
                self._value[start:start + self._chunksize] = value
            else:
                self._success = False
                self._value = value
        self._remaining -= 1
        if not self._remaining and not self._submitting:
            self._finish(self._success, self._value)


class _IMapResult:
    def __init__(self, pool, tasks, ordered):
        self._pool = pool
        self._job = next(_native.job_counter)
        self._tasks = iter(tasks)
        self._ordered = ordered
        self._items = deque()
        self._unsorted = {}
        self._submitted = self._received = self._inflight = 0
        self._exhausted = False
        pool._cache[self._job] = self

    def __iter__(self):
        return self

    def next(self, timeout=None):
        if not self._items and self._pool is not None:
            self._pool._wait(lambda: bool(self._items) or self._pool is None, timeout)
        if not self._items:
            if self._pool is None:
                raise StopIteration
            raise TimeoutError
        success, value = self._items.popleft()
        if self._pool is not None:
            self._fill()
        if not success:
            raise value
        return value

    __next__ = next

    def _fill(self, draining=False):
        if self._exhausted:
            return
        pool = self._pool
        limit = 2 * pool._processes
        for _ in range(limit):
            buffered = 0 if draining else len(self._items) + len(self._unsorted)
            if self._inflight + buffered >= limit:
                break
            try:
                func, args = next(self._tasks)
            except StopIteration:
                self._exhausted = True
                break
            except Exception as error:
                self._inflight += 1
                self._set(self._submitted, (False, error))
                self._exhausted = True
                break
            index = self._submitted
            self._submitted += 1
            self._inflight += 1
            pool._submit(self, index, func, args, {})
        self._finish_if_drained()

    def _finish_if_drained(self):
        if self._pool is not None and self._exhausted and not self._inflight:
            self._pool._cache.pop(self._job)
            self._pool = None

    def _set(self, index, result):
        self._inflight -= 1
        if self._ordered:
            self._unsorted[index] = result
            while self._received in self._unsorted:
                self._items.append(self._unsorted.pop(self._received))
                self._received += 1
        else:
            self._items.append(result)
        self._finish_if_drained()

    def _fail(self, error):
        self._items.clear()
        self._unsorted.clear()
        self._items.append((False, error))
        self._pool._cache.pop(self._job)
        self._pool = None


class Pool:
    """Fixed-size Pool for browser spawn processes with no service threads."""

    Process = staticmethod(_native.Pool.Process)
    _check_running = _native.Pool._check_running
    __reduce__ = _native.Pool.__reduce__
    __enter__ = _native.Pool.__enter__
    __exit__ = _native.Pool.__exit__
    apply = _native.Pool.apply
    map = _native.Pool.map
    starmap = _native.Pool.starmap
    map_async = _native.Pool.map_async
    starmap_async = _native.Pool.starmap_async

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        self._state = _native.INIT
        if processes is None:
            processes = os.process_cpu_count() or 1
        processes = operator.index(processes)
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")
        if maxtasksperchild is not None:
            if not isinstance(maxtasksperchild, int) or maxtasksperchild <= 0:
                raise ValueError("maxtasksperchild must be a positive int or None")
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._ctx = context or multiprocessing.get_context("spawn")
        self._processes = processes
        self._max_tasks = maxtasksperchild
        self._workers = {}
        self._retired = set()
        self._next_worker = 0
        self._stopping = False
        self._cache = {}
        self._sources = {}
        self._initialize = _Initializer(initializer, initargs, self._sources)
        self._tasks = self._ctx.Queue()
        try:
            self._results = self._ctx.Queue()
        except BaseException:
            self._tasks.close()
            raise
        self._event_channel = self._results._channel if _events is not None else None
        self._cleanup = weakref.finalize(
            self, _stop_workers, self._workers, self._tasks, self._results, self._event_channel
        )
        try:
            if self._event_channel is not None:
                _events.watch_pool(self._event_channel, True)
                _live_pools[self._event_channel] = self
            self._start_workers()
        except BaseException:
            self.terminate()
            raise
        self._state = _native.RUN

    def __del__(self):
        if self._state == _native.RUN:
            warnings.warn(f"unclosed running multiprocessing pool {self!r}",
                          ResourceWarning, source=self)

    def _start_workers(self):
        while len(self._workers) < self._processes:
            token = self._next_worker
            self._next_worker += 1
            worker = self.Process(
                self._ctx, target=_pool_worker,
                args=(self._tasks, self._results, token, self._initialize, self._max_tasks)
            )
            worker.name = worker.name.replace("Process", "PoolWorker")
            worker.daemon = True
            worker.start()
            self._workers[token] = worker
            if self._event_channel is not None:
                _events.watch_pool_worker(worker.pid, self._event_channel)

    def _submit(self, result, index, func, args, kwds):
        try:
            payload = _encode_call((func, args, kwds), self._sources)
            self._tasks.put((result._job, index, payload))
        except Exception as error:
            result._set(index, (False, error))

    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        self._pump()
        self._check_running()
        result = _Result(self, callback, error_callback)
        self._submit(result, 0, func, args, kwds)
        return result

    def _map_async(self, func, iterable, mapper, chunksize=None, callback=None,
                   error_callback=None):
        self._pump()
        self._check_running()
        if not hasattr(iterable, "__len__"):
            iterable = list(iterable)
        length = len(iterable)
        if chunksize is None:
            parts = 4 * self._processes
            chunksize = max(1, (length + parts - 1) // parts)
        else:
            chunksize = operator.index(chunksize)
            if chunksize < 1:
                raise ValueError("Chunksize must be at least 1")
        result = _MapResult(self, length, chunksize, callback, error_callback)
        if not length:
            result._seal(empty=True)
            return result
        batches = _native.Pool._get_tasks(func, iterable, chunksize)
        for job, index, function, args, kwargs in _native.Pool._guarded_task_generation(
            self, result._job, mapper, batches
        ):
            if job not in self._cache:
                break
            result._submit(index, function, args, kwargs)
        result._seal(empty=False)
        return result

    def imap(self, func, iterable, chunksize=1):
        return self._imap(func, iterable, chunksize, ordered=True)

    def imap_unordered(self, func, iterable, chunksize=1):
        return self._imap(func, iterable, chunksize, ordered=False)

    def _imap(self, func, iterable, chunksize, ordered):
        self._pump()
        self._check_running()
        chunksize = operator.index(chunksize)
        if chunksize < 1:
            raise ValueError("Chunksize must be at least 1")
        if chunksize == 1:
            tasks = ((func, (value,)) for value in iterable)
        else:
            batches = _native.Pool._get_tasks(func, iterable, chunksize)
            tasks = ((_native.mapstar, (batch,)) for batch in batches)
        result = _IMapResult(self, tasks, ordered)
        result._fill()
        return result if chunksize == 1 else (item for batch in result for item in batch)

    def _handle_result(self, item):
        job, index, value = item
        if job is None:
            self._retired.add(index)
        elif job in self._cache:
            self._cache[job]._set(index, value)

    def _read_available(self):
        while self._cleanup.alive and self._state != _native.TERMINATE:
            try:
                self._handle_result(self._results.get_nowait())
            except Empty:
                return
            except (OSError, EOFError) as error:
                self._break_pool(error)
                return

    def _reap_workers(self):
        for token, worker in tuple(self._workers.items()):
            if token not in self._workers:
                continue
            code = worker.exitcode
            if code is None:
                continue
            worker.join()
            worker.close()
            del self._workers[token]
            if code == 0 and token not in self._retired:
                # Retirement can arrive after the previous result drain.
                self._read_available()
                if self._state == _native.TERMINATE:
                    return
            if code or token not in self._retired:
                self._break_pool(BrokenPipeError(f"Pool worker exited unexpectedly ({code})"))
                return
            self._retired.remove(token)
        if self._state == _native.RUN or (self._state == _native.CLOSE and self._cache):
            try:
                self._start_workers()
            except Exception as error:
                self._break_pool(error)

    def _pump(self, timeout=0):
        if not self._cleanup.alive or self._state == _native.TERMINATE:
            return
        if timeout:
            try:
                self._handle_result(self._results.get(timeout=timeout))
            except Empty:
                pass
            except (OSError, EOFError) as error:
                self._break_pool(error)
                return
        self._read_available()
        if self._cleanup.alive and self._state != _native.TERMINATE:
            self._reap_workers()
            if self._state == _native.CLOSE:
                for result in tuple(self._cache.values()):
                    if isinstance(result, _IMapResult):
                        result._fill(draining=True)
            self._send_stop_if_drained()
            if self._stopping and not self._workers:
                self._cleanup()

    def _wait(self, ready, timeout=None):
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0)
        self._pump()
        while not ready():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            if self._event_channel is not None:
                if not _events.wait_pool_event(self._event_channel,
                                                -1 if remaining is None else remaining):
                    return
                self._pump()
            else:
                self._pump(0.05 if remaining is None else min(0.05, remaining))

    def _send_stop_if_drained(self):
        if self._state == _native.CLOSE and not self._cache and not self._stopping:
            self._stopping = True
            for _ in self._workers:
                self._tasks.put(None)

    def close(self):
        if self._state == _native.RUN:
            self._state = _native.CLOSE
            self._pump()

    def _break_pool(self, error):
        self._state = _native.TERMINATE
        for result in tuple(self._cache.values()):
            result._fail(error)
        self._cleanup()

    def terminate(self):
        if self._state != _native.TERMINATE:
            self._break_pool(BrokenPipeError("Pool was terminated"))
        self._join_workers()

    def _join_workers(self):
        for worker in self._workers.values():
            worker.join()
            worker.close()
        self._workers.clear()
        self._cleanup()

    def join(self):
        if self._state == _native.RUN:
            raise ValueError("Pool is still running")
        if self._state not in (_native.CLOSE, _native.TERMINATE):
            raise ValueError("In unknown state")
        if self._state == _native.CLOSE:
            self._wait(lambda: not self._cache)
        self._join_workers()

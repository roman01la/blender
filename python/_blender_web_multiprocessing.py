# SPDX-License-Identifier: GPL-2.0-or-later

"""Browser Worker-backed spawn context for Blender's embedded Python."""

import multiprocessing
from multiprocessing import connection, context, managers, process, reduction
from pathlib import Path
import signal
import struct
import sys
import weakref

import _blender_web_process as _worker


_CHILD_BOOTSTRAP = """
from pathlib import Path
import pickle
import sys
import types

import _blender_web_process
if hasattr(_blender_web_process, 'shared_memory_available'):
    from _blender_web_shared_memory import install
    install()

sources, job = pickle.load(sys.stdin.buffer)
root = Path('/process/modules')
sys.path.insert(0, str(root))
main_source = None
for name, source, package in sources:
    if name == '__main__':
        main_source = source
        continue
    path = root.joinpath(*name.split('.'))
    path = path / '__init__.py' if package else path.with_suffix('.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
if main_source is not None:
    main = types.ModuleType('__mp_main__')
    main.__file__ = '/process/modules/__mp_main__.py'
    sys.modules['__main__'] = sys.modules['__mp_main__'] = main
    exec(compile(main_source, main.__file__, 'exec'), main.__dict__)
target, args, kwargs = pickle.loads(job)
if target is not None:
    target(*args, **kwargs)
"""


def _module_sources(target):
    if target is None:
        return []
    # Attribute lookups (including builtins) remain ordinary spawn pickles.
    module_name = getattr(target, "__module__", None)
    if not module_name:
        return []
    names = module_name.split(".")
    if not all(part.isidentifier() for part in names):
        raise ValueError(f"Invalid spawn module name {module_name!r}")
    sources = []
    for index in range(len(names)):
        name = ".".join(names[:index + 1])
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename is None or not filename.endswith(".py"):
            continue
        source = Path(filename)
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        get_data = getattr(loader, "get_data", None)
        contents = get_data(filename) if get_data is not None else source.read_bytes()
        sources.append((name, contents, source.name == "__init__.py"))
    return sources


class _WorkerPopen:
    # The browser transport does not provide a POSIX file descriptor.
    sentinel = None

    def __init__(self, child):
        self._channel_tickets = []
        try:
            job = (child._target, child._args, child._kwargs)
            previous = context.get_spawning_popen()
            context.set_spawning_popen(self)
            try:
                encoded_job = bytes(reduction.ForkingPickler.dumps(job))
            finally:
                context.set_spawning_popen(previous)
            payload = reduction.ForkingPickler.dumps((_module_sources(child._target), encoded_job))
            tickets = struct.pack(f"<{len(self._channel_tickets)}i", *self._channel_tickets)
            self.pid = _worker.start(_CHILD_BOOTSTRAP, bytes(payload), tickets)
        except BaseException:
            for ticket in self._channel_tickets:
                _worker.channel_cancel(ticket)
            raise
        self.finalizer = weakref.finalize(self, _worker.close, self.pid)

    def poll(self):
        return _worker.poll(self.pid)

    def wait(self, timeout=None):
        return _worker.wait(self.pid, -1 if timeout is None else max(float(timeout), 0.0))

    def terminate(self):
        _worker.stop(self.pid, signal.SIGTERM)

    def kill(self):
        _worker.stop(self.pid, signal.SIGKILL)

    def close(self):
        self.finalizer()


class SpawnProcess(process.BaseProcess):
    _start_method = "spawn"

    @staticmethod
    def _Popen(child):
        if type(child) is not SpawnProcess:
            raise NotImplementedError("Subclasses of Process are unsupported by browser spawn")
        return _WorkerPopen(child)

    @staticmethod
    def _after_fork():
        pass

    @property
    def sentinel(self):
        raise NotImplementedError("Browser processes do not expose POSIX sentinel handles")


class SpawnContext(context.BaseContext):
    _name = "spawn"
    Process = SpawnProcess

    def Queue(self, maxsize=0):
        from _blender_web_queue import Queue
        return Queue(maxsize)

    def SimpleQueue(self):
        from _blender_web_queue import SimpleQueue
        return SimpleQueue()

    def Pipe(self, duplex=True):
        from _blender_web_pipe import Pipe
        return Pipe(duplex)

    def Pool(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None):
        from _blender_web_pool import Pool
        return Pool(processes, initializer, initargs, maxtasksperchild, context=self)

    def Manager(self):
        from _blender_web_managers import Manager
        return Manager(self)


def install():
    from _blender_web_managers import SyncManager
    from _blender_web_shared_memory import install as install_shared_memory

    spawn = SpawnContext()
    context._concrete_contexts.clear()
    context._concrete_contexts["spawn"] = spawn
    default = context._default_context
    default._default_context = spawn
    default._actual_context = None
    default.Process = SpawnProcess
    multiprocessing.Process = SpawnProcess
    for name in ("Queue", "SimpleQueue", "Pipe", "Pool", "Manager"):
        setattr(multiprocessing, name, getattr(spawn, name))
    connection.Pipe = spawn.Pipe
    managers.SyncManager = SyncManager
    multiprocessing.get_all_start_methods = lambda: ["spawn"]
    install_shared_memory()
    if _worker.shared_memory_available():
        from _blender_web_managers import SharedMemoryManager
        managers.SharedMemoryManager = SharedMemoryManager
        managers.HAS_SHMEM = True
        if "SharedMemoryManager" not in managers.__all__:
            managers.__all__.append("SharedMemoryManager")

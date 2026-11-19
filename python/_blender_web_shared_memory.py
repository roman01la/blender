# SPDX-License-Identifier: GPL-2.0-or-later

"""CPython shared memory backed by the editor and workers' common Wasm memory.

The browser session owns the name registry and releases all segments when it
ends. The ``track`` argument is accepted for compatibility. There is no separate
OS resource tracker, and names remain available until unlinked or the session
ends. Existing handles remain valid until closed, including after unlink.
"""

from functools import partial
import importlib.util
import multiprocessing
import operator
import secrets
import struct
import sys
import types

import _blender_web_process as _worker


class SharedMemory:
    """Create or attach to a named, writable browser shared memory segment."""

    def __init__(self, name=None, create=False, size=0, *, track=True):
        self._map = None
        self._buf = None
        size = operator.index(size)
        if size < 0 or (create and size == 0):
            raise ValueError("'size' must be a positive integer when creating shared memory")
        if name is None and not create:
            raise ValueError("'name' can only be None if create=True")
        self._track = bool(track)
        if name is None:
            while True:
                name = "psm_" + secrets.token_hex(8)
                try:
                    self._map = _worker.shared_memory_open(name, True, size, self._track)
                    break
                except FileExistsError:
                    continue
        else:
            self._map = _worker.shared_memory_open(name, create, size, self._track)
        self._name = name
        try:
            self._buf = memoryview(self._map)
        except BaseException:
            if create:
                _worker.shared_memory_unlink(name)
            self.close()
            raise
        self._size = len(self._buf)

    @property
    def buf(self):
        return self._buf

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    def close(self):
        """Close this handle, following mmap's exported-buffer rules."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._map is not None:
            self._map.close()
            self._map = None

    def unlink(self):
        """Remove the name from the registry, retaining active buffer views."""
        _worker.shared_memory_unlink(self._name)

    def __del__(self):
        try:
            self.close()
        except (BufferError, OSError):
            # The native owner is retained by any surviving memoryview.
            pass

    def __reduce__(self):
        return partial(type(self), track=self._track), (self.name, False, self.size)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, size={self.size})"


def install():
    """Install the browser backend and reuse the bundled CPython ShareableList."""
    module_name = "multiprocessing.shared_memory"
    if getattr(sys.modules.get(module_name), "SharedMemory", None) is SharedMemory:
        return

    spec = importlib.util.find_spec(module_name)
    source = spec.loader.get_source(module_name)
    class_start = source.index("\n_encoding = ") + 1
    class_source = source[class_start:]
    if "\nclass ShareableList:" not in class_source:
        raise ImportError("The bundled CPython SharedMemory module has an unexpected layout")

    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(
        __all__=["SharedMemory", "ShareableList"],
        __doc__=__doc__,
        SharedMemory=SharedMemory,
        partial=partial,
        struct=struct,
        types=types,
    )
    class_source = "\n" * source.count("\n", 0, class_start) + class_source
    exec(compile(class_source, spec.origin, "exec"), module.__dict__)
    SharedMemory.__module__ = module_name
    sys.modules[module_name] = module
    multiprocessing.shared_memory = module

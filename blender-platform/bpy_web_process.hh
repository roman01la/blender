/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <Python.h>
#include <emscripten.h>
#include <emscripten/threading.h>

#include <cmath>
#include <limits>

#include "channel_api.h"
#include "shared_memory_api.h"

// clang-format off
EM_JS(int, blender_web_channel_request,
      (int operation, int id, int argument, const char *input, size_t input_size,
       double seconds, char **output, size_t *output_size), {
  if (ENVIRONMENT_IS_PTHREAD) return -4;
  const store = (value) => {
    if (!(value instanceof Uint8Array)) return value;
    const buffer = value.length ? _malloc(BigInt(value.length)) : 0n;
    if (value.length && !buffer) return -7;
    HEAPU8.set(value, Number(buffer));
    HEAPU64[Number(output) / 8] = BigInt(buffer);
    HEAPU64[Number(output_size) / 8] = BigInt(value.length);
    return 0;
  };
  // During unwind, Asyncify may return an earlier import's pointer-sized value.
  const pending = Asyncify.handleSleep(wake => {
    const bytes = operation === 1 ? HEAPU8.slice(Number(input), Number(input + input_size)) : null;
    const result = Module["blenderPythonProcessBridge"].channel(operation, id, argument, bytes, seconds);
    if (result instanceof Promise) result.then(value => wake(store(value)), () => wake(-4));
    else wake(store(result));
  });
  return pending instanceof Promise ? pending : Number(pending);
});

EM_JS(int, blender_web_shared_memory_available, (), {
  if (ENVIRONMENT_IS_PTHREAD) return 0;
  const bridge = Module["blenderPythonProcessBridge"];
  if (!bridge) return 0;
  const allocate = size => BigInt(_malloc(size));
  return bridge.configureSharedMemory({
    memory: wasmMemory,
    allocate,
    allocateZeroed(size) {
      const pointer = allocate(size);
      if (pointer) HEAPU8.fill(0, Number(pointer), Number(pointer + size));
      return pointer;
    },
    free: pointer => _free(pointer),
  });
});

EM_JS(int, blender_web_shared_memory_request,
      (int operation, const char *name, size_t requested_size, int track, uint32_t lease,
       void **pointer, size_t *actual_size, uint32_t *out_lease), {
  if (ENVIRONMENT_IS_PTHREAD) return -4;
  const result = Module["blenderPythonProcessBridge"].sharedMemoryRequest(
    operation, name ? UTF8ToString(Number(name)) : null, Number(requested_size), Boolean(track), lease >>> 0);
  if (result.status === 0 && (operation === 0 || operation === 1)) {
    HEAPU64[Number(pointer) / 8] = result.pointer;
    HEAPU64[Number(actual_size) / 8] = BigInt(result.size);
    HEAPU32[Number(out_lease) / 4] = result.lease;
  }
  return result.status;
});
// clang-format on

namespace blender::python::web_process {

constexpr int running = std::numeric_limits<int>::min();

// clang-format off
EM_JS(int, blender_web_process_available, (), {
  return Module["blenderPythonProcessBridge"] ? 1 : 0;
});

EM_JS(int, blender_web_process_launch_job,
      (const char *source, const char *input, size_t size, const char *tickets, size_t ticket_size), {
  return Module["blenderPythonProcessBridge"].start(
      UTF8ToString(Number(source)), HEAPU8.slice(Number(input), Number(input + size)),
      new Int32Array(HEAPU8.slice(Number(tickets), Number(tickets + ticket_size)).buffer));
});

EM_JS(int, blender_web_process_poll_job, (int id), {
  return Module["blenderPythonProcessBridge"].poll(id);
});

EM_JS(int, blender_web_process_wait_job, (int id, double seconds), {
  const pending = Asyncify.handleAsync(() => Module["blenderPythonProcessBridge"].wait(id, seconds));
  return pending instanceof Promise ? pending : Number(pending);
});

EM_JS(void, blender_web_process_stop_job, (int id, int signal), {
  Module["blenderPythonProcessBridge"].stop(id, signal);
});

EM_JS(void, blender_web_process_forget_job, (int id), {
  Module["blenderPythonProcessBridge"].close(id);
});

EM_JS(void, blender_web_watch_pool, (int channel, int enabled), {
  const bridge = Module["blenderPythonProcessBridge"];
  if (!Module["blenderPoolDispatchInstalled"]) {
    let queued = false;
    const wake = () => {
      if (queued || ABORT || !bridge.hasPoolEvents()) return;
      queued = true;
      queueMicrotask(() => {
        queued = false;
        // Resume the active editor entry before running another Python callback.
        if (ABORT || Asyncify.currData || Module["blenderWasmPending"] || !bridge.hasPoolEvents()) return;
        try {
          const pending = _blender_web_dispatch_pool_events();
          if (pending instanceof Promise) {
            Module["blenderWasmPending"] = pending;
            pending.then(() => {
              Module["blenderWasmPending"] = null;
              wake();
            }, handleException);
          }
        } catch (error) {
          handleException(error);
        }
      });
    };
    Asyncify.sleepCallbacks?.push(wake);
    Module["blenderPoolWake"] = wake;
    bridge.setPoolWake(wake);
    Module["blenderPoolDispatchInstalled"] = true;
  }
  bridge.watchPool(channel, Boolean(enabled));
});

EM_JS(void, blender_web_watch_pool_worker, (int pid, int channel), {
  Module["blenderPythonProcessBridge"].watchPoolWorker(pid, channel);
});

EM_JS(int, blender_web_next_pool_event, (), {
  return Module["blenderPythonProcessBridge"].nextPoolEvent();
});

EM_JS(int, blender_web_wait_pool_event, (int channel, double seconds), {
  const pending = Asyncify.handleSleep(wake => {
    const result = Module["blenderPythonProcessBridge"].waitPoolEvent(channel, seconds);
    if (result instanceof Promise) result.then(wake);
    else wake(result);
  });
  return pending instanceof Promise ? pending : Number(pending);
});
// clang-format on

extern "C" EMSCRIPTEN_KEEPALIVE void blender_web_dispatch_pool_events()
{
  if (!Py_IsInitialized() || Py_IsFinalizing()) {
    return;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *pool = PyDict_GetItemString(PyImport_GetModuleDict(), "_blender_web_pool");
  if (pool) {
    PyObject *result = PyObject_CallMethod(pool, "_dispatch", nullptr);
    if (!result) {
      PyErr_WriteUnraisable(pool);
    }
    Py_XDECREF(result);
  }
  PyGILState_Release(gil);
}

static PyObject *exit_code(const int value)
{
  if (value == running) {
    Py_RETURN_NONE;
  }
  return PyLong_FromLong(value);
}

static bool require_runtime_thread()
{
  if (emscripten_is_main_runtime_thread()) {
    return true;
  }
  PyErr_SetString(PyExc_RuntimeError, "Python workers must be managed on the Blender runtime thread");
  return false;
}

static PyObject *start(PyObject *, PyObject *args)
{
  const char *source;
  const char *payload;
  const char *tickets = nullptr;
  Py_ssize_t size;
  Py_ssize_t ticket_size = 0;
  if (!PyArg_ParseTuple(args, "sy#|y#:start", &source, &payload, &size, &tickets, &ticket_size) ||
      !require_runtime_thread()) {
    return nullptr;
  }
  if (ticket_size % sizeof(int32_t) != 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid browser spawn channel tickets");
    return nullptr;
  }
  const int id = blender_web_process_launch_job(source, payload, size_t(size), tickets, size_t(ticket_size));
  if (id == 0) {
    PyErr_SetString(PyExc_OSError, "Could not start the browser Python worker");
    return nullptr;
  }
  return PyLong_FromLong(id);
}

static PyObject *poll(PyObject *, PyObject *args)
{
  int id;
  if (!PyArg_ParseTuple(args, "i:poll", &id) || !require_runtime_thread()) {
    return nullptr;
  }
  return exit_code(blender_web_process_poll_job(id));
}

static PyObject *wait(PyObject *, PyObject *args)
{
  int id;
  double seconds;
  if (!PyArg_ParseTuple(args, "id:wait", &id, &seconds) || !require_runtime_thread()) {
    return nullptr;
  }
  int code = blender_web_process_poll_job(id);
  if (code == running && seconds != 0.0) {
    if (std::isnan(seconds)) {
      PyErr_SetString(PyExc_ValueError, "timeout cannot be NaN");
      return nullptr;
    }
    PyThreadState *thread_state = PyEval_SaveThread();
    code = blender_web_process_wait_job(id, seconds);
    PyEval_RestoreThread(thread_state);
  }
  return exit_code(code);
}

static PyObject *stop(PyObject *, PyObject *args)
{
  int id;
  int signal;
  if (!PyArg_ParseTuple(args, "ii:stop", &id, &signal) || !require_runtime_thread()) {
    return nullptr;
  }
  blender_web_process_stop_job(id, signal);
  Py_RETURN_NONE;
}

static PyObject *close(PyObject *, PyObject *args)
{
  int id;
  if (!PyArg_ParseTuple(args, "i:close", &id) || !require_runtime_thread()) {
    return nullptr;
  }
  blender_web_process_forget_job(id);
  Py_RETURN_NONE;
}

static PyObject *watch_pool(PyObject *, PyObject *args)
{
  int channel;
  int enabled;
  if (!PyArg_ParseTuple(args, "ip:watch_pool", &channel, &enabled) || !require_runtime_thread()) {
    return nullptr;
  }
  blender_web_watch_pool(channel, enabled);
  Py_RETURN_NONE;
}

static PyObject *watch_pool_worker(PyObject *, PyObject *args)
{
  int pid;
  int channel;
  if (!PyArg_ParseTuple(args, "ii:watch_pool_worker", &pid, &channel) || !require_runtime_thread()) {
    return nullptr;
  }
  blender_web_watch_pool_worker(pid, channel);
  Py_RETURN_NONE;
}

static PyObject *next_pool_event(PyObject *, PyObject *)
{
  return PyLong_FromLong(blender_web_next_pool_event());
}

static PyObject *wait_pool_event(PyObject *, PyObject *args)
{
  int channel;
  double seconds;
  if (!PyArg_ParseTuple(args, "id:wait_pool_event", &channel, &seconds) ||
      !require_runtime_thread())
  {
    return nullptr;
  }
  if (std::isnan(seconds)) {
    PyErr_SetString(PyExc_ValueError, "timeout cannot be NaN");
    return nullptr;
  }
  PyThreadState *thread_state = PyEval_SaveThread();
  const int ready = blender_web_wait_pool_event(channel, seconds);
  PyEval_RestoreThread(thread_state);
  return PyBool_FromLong(ready);
}

static PyObject *init()
{
  static PyMethodDef methods[] = {{"start", start, METH_VARARGS, nullptr},
                                  {"poll", poll, METH_VARARGS, nullptr},
                                  {"wait", wait, METH_VARARGS, nullptr},
                                  {"stop", stop, METH_VARARGS, nullptr},
                                  {"close", close, METH_VARARGS, nullptr},
                                  {"watch_pool", watch_pool, METH_VARARGS, nullptr},
                                  {"watch_pool_worker", watch_pool_worker, METH_VARARGS, nullptr},
                                  {"next_pool_event", next_pool_event, METH_NOARGS, nullptr},
                                  {"wait_pool_event", wait_pool_event, METH_VARARGS, nullptr},
                                  BLENDER_WEB_CHANNEL_METHODS
                                  BLENDER_WEB_SHARED_MEMORY_METHODS
                                  {nullptr, nullptr, 0, nullptr}};
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_blender_web_process", nullptr, -1, methods};
  PyObject *result = PyModule_Create(&module);
  if (result && blender_web_shared_memory_init(result) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  blender_web_shared_memory_available();
  return result;
}

}  // namespace blender::python::web_process

// SPDX-License-Identifier: GPL-2.0-or-later

#include <Python.h>
#include <emscripten.h>
#include <stdio.h>

#include "channel_api.h"
#include "shared_memory_api.h"

EM_JS(int, blender_web_channel_request,
      (int operation, int id, int argument, const char *input, size_t input_size,
       double seconds, char **output, size_t *output_size), {
  if (ENVIRONMENT_IS_PTHREAD) return -4;
  const bytes = operation === 1 ? HEAPU8.slice(Number(input), Number(input + input_size)) : null;
  const result = Module["channelRequest"](operation, id, argument, bytes, seconds);
  if (!(result instanceof Uint8Array)) return result;
  const buffer = result.length ? _malloc(BigInt(result.length)) : 0n;
  if (result.length && !buffer) return -7;
  HEAPU8.set(result, Number(buffer));
  HEAPU64[Number(output) / 8] = BigInt(buffer);
  HEAPU64[Number(output_size) / 8] = BigInt(result.length);
  return 0;
});

EM_JS(int, blender_web_shared_memory_available, (void), {
  return !ENVIRONMENT_IS_PTHREAD && typeof Module["sharedMemoryRequest"] === "function";
});

EM_JS(int,
      blender_web_shared_memory_request,
      (int operation, const char *name, size_t requested_size, int track, uint32_t lease,
       void **pointer, size_t *actual_size, uint32_t *out_lease), {
        if (ENVIRONMENT_IS_PTHREAD || !Module["sharedMemoryRequest"]) return -4;
        const result = Module["sharedMemoryRequest"](
            operation, name ? UTF8ToString(Number(name)) : null, Number(requested_size), track,
            lease >>> 0);
        if (result.status === 0 && operation < 2) {
          const view = new DataView(wasmMemory.buffer);
          view.setBigUint64(Number(pointer), result.pointer, true);
          view.setBigUint64(Number(actual_size), BigInt(result.size), true);
          view.setUint32(Number(out_lease), result.lease, true);
        }
        return result.status;
      });

static PyObject *init_channels(void)
{
  static PyMethodDef methods[] = {
      BLENDER_WEB_CHANNEL_METHODS
      BLENDER_WEB_SHARED_MEMORY_METHODS
      {NULL, NULL, 0, NULL},
  };
  static struct PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_blender_web_process", NULL, -1, methods};
  PyObject *result = PyModule_Create(&module);
  if (result && blender_web_shared_memory_init(result) < 0) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}

EMSCRIPTEN_KEEPALIVE int blender_python_child_run(void)
{
  if (!freopen("/process/stdin", "rb", stdin) ||
      !freopen("/process/stdout", "wb", stdout) ||
      !freopen("/process/stderr", "wb", stderr))
  {
    return 125;
  }
  if (PyImport_AppendInittab("_blender_web_process", init_channels) < 0) {
    return 125;
  }

  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);
  config.parse_argv = 1;
  config.site_import = 0;
  config.write_bytecode = 0;
  config.module_search_paths_set = 1;

  char *args[] = {"/blender-python", "/process/job.py"};
  PyStatus status = PyConfig_SetBytesArgv(&config, 2, args);
  if (!PyStatus_Exception(status)) {
    status = PyWideStringList_Append(&config.module_search_paths, L"/lib/python313.zip");
  }
  if (!PyStatus_Exception(status)) {
    status = Py_InitializeFromConfig(&config);
  }
  /* Keep the status message alive until it has been reported. */
  if (PyStatus_Exception(status) && !PyStatus_IsExit(status)) {
    fprintf(stderr, "%s\n", status.err_msg);
  }
  PyConfig_Clear(&config);

  const int result = PyStatus_Exception(status) ?
                         (PyStatus_IsExit(status) ? status.exitcode : 125) :
                         Py_RunMain();
  fflush(stdout);
  fflush(stderr);
  return result;
}

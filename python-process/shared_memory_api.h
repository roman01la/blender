/* SPDX-License-Identifier: GPL-2.0-or-later */

#ifndef BLENDER_WEB_SHARED_MEMORY_API_H
#define BLENDER_WEB_SHARED_MEMORY_API_H

#include <Python.h>
#include <stdint.h>
#include <string.h>

enum BlenderWebSharedMemoryOperation {
  BLENDER_WEB_SHARED_MEMORY_CREATE = 0,
  BLENDER_WEB_SHARED_MEMORY_ATTACH = 1,
  BLENDER_WEB_SHARED_MEMORY_RELEASE = 2,
  BLENDER_WEB_SHARED_MEMORY_UNLINK = 3,
};

#ifdef __cplusplus
extern "C" {
#endif
int blender_web_shared_memory_available(void);
int blender_web_shared_memory_request(int operation,
                                       const char *name,
                                       size_t requested_size,
                                       int track,
                                       uint32_t lease,
                                       void **pointer,
                                       size_t *actual_size,
                                       uint32_t *out_lease);
#ifdef __cplusplus
}
#endif

typedef struct BlenderWebMemoryMap {
  PyObject_HEAD
  void *address;
  Py_ssize_t size;
  Py_ssize_t exports;
  uint32_t lease;
} BlenderWebMemoryMap;

static PyObject *blender_web_shared_memory_error(int status, const char *name)
{
  switch (status) {
    case -2:
      PyErr_Format(PyExc_FileNotFoundError, "Shared memory %s does not exist", name);
      break;
    case -3:
      PyErr_Format(PyExc_FileExistsError, "Shared memory %s already exists", name);
      break;
    case -5:
      PyErr_SetString(PyExc_ValueError, "Invalid shared memory name or size");
      break;
    case -7:
      return PyErr_NoMemory();
    default:
      PyErr_Format(PyExc_RuntimeError, "Browser shared memory request failed (%d)", status);
      break;
  }
  return NULL;
}

static int blender_web_memory_map_release(BlenderWebMemoryMap *self)
{
  if (!self->lease) {
    return 0;
  }
  const int status = blender_web_shared_memory_request(
      BLENDER_WEB_SHARED_MEMORY_RELEASE, NULL, 0, 0, self->lease, NULL, NULL, NULL);
  if (status == 0) {
    self->lease = 0;
    self->address = NULL;
  }
  return status;
}

static PyObject *blender_web_memory_map_close(PyObject *object, PyObject *Py_UNUSED(args))
{
  BlenderWebMemoryMap *self = (BlenderWebMemoryMap *)object;
  if (self->exports) {
    PyErr_SetString(PyExc_BufferError, "Cannot close shared memory with exported buffers");
    return NULL;
  }
  const int status = blender_web_memory_map_release(self);
  if (status < 0) {
    return blender_web_shared_memory_error(status, "");
  }
  Py_RETURN_NONE;
}

static void blender_web_memory_map_dealloc(PyObject *object)
{
  BlenderWebMemoryMap *self = (BlenderWebMemoryMap *)object;
  blender_web_memory_map_release(self);
  PyTypeObject *type = Py_TYPE(object);
  type->tp_free(object);
  Py_DECREF(type);
}

static int blender_web_memory_map_getbuffer(PyObject *object, Py_buffer *view, int flags)
{
  BlenderWebMemoryMap *self = (BlenderWebMemoryMap *)object;
  if (!self->lease) {
    PyErr_SetString(PyExc_ValueError, "Shared memory is closed");
    return -1;
  }
  if (PyBuffer_FillInfo(view, object, self->address, self->size, 0, flags) < 0) {
    return -1;
  }
  self->exports++;
  return 0;
}

static void blender_web_memory_map_releasebuffer(PyObject *object, Py_buffer *Py_UNUSED(view))
{
  ((BlenderWebMemoryMap *)object)->exports--;
}

static PyObject *blender_web_shared_memory_is_available(PyObject *Py_UNUSED(module),
                                                        PyObject *Py_UNUSED(args))
{
  return PyBool_FromLong(blender_web_shared_memory_available());
}

static PyObject *blender_web_shared_memory_open(PyObject *module, PyObject *args)
{
  const char *name;
  int create;
  Py_ssize_t size;
  int track;
  if (!PyArg_ParseTuple(args, "spnp:shared_memory_open", &name, &create, &size, &track)) {
    return NULL;
  }
  if (!blender_web_shared_memory_available()) {
    PyErr_SetString(PyExc_NotImplementedError, "Shared Wasm Python worker memory is unavailable");
    return NULL;
  }
  if (size < 0 || !name[0] || (create && size == 0)) {
    return blender_web_shared_memory_error(-5, name);
  }

  PyObject *type = PyObject_GetAttrString(module, "_SharedMemoryMap");
  if (!type) {
    return NULL;
  }
  if (!PyType_Check(type) || ((PyTypeObject *)type)->tp_dealloc != blender_web_memory_map_dealloc ||
      ((PyTypeObject *)type)->tp_basicsize != sizeof(BlenderWebMemoryMap))
  {
    Py_DECREF(type);
    PyErr_SetString(PyExc_RuntimeError, "Invalid browser shared memory buffer type");
    return NULL;
  }
  BlenderWebMemoryMap *map = (BlenderWebMemoryMap *)((PyTypeObject *)type)->tp_alloc(
      (PyTypeObject *)type, 0);
  Py_DECREF(type);
  if (!map) {
    return NULL;
  }

  size_t actual_size = 0;
  const int operation = create ? BLENDER_WEB_SHARED_MEMORY_CREATE : BLENDER_WEB_SHARED_MEMORY_ATTACH;
  const int status = blender_web_shared_memory_request(
      operation, name, (size_t)size, track, 0, &map->address, &actual_size, &map->lease);
  if (status < 0 || !map->address || !map->lease || actual_size == 0 ||
      actual_size > (size_t)PY_SSIZE_T_MAX)
  {
    Py_DECREF((PyObject *)map);
    if (status == 0 && create) {
      blender_web_shared_memory_request(
          BLENDER_WEB_SHARED_MEMORY_UNLINK, name, 0, 0, 0, NULL, NULL, NULL);
    }
    return blender_web_shared_memory_error(status < 0 ? status : -4, name);
  }
  map->size = (Py_ssize_t)actual_size;
  return (PyObject *)map;
}

static PyObject *blender_web_shared_memory_unlink(PyObject *Py_UNUSED(module), PyObject *argument)
{
  Py_ssize_t size;
  const char *name = PyUnicode_AsUTF8AndSize(argument, &size);
  if (!name) {
    return NULL;
  }
  if (!blender_web_shared_memory_available()) {
    PyErr_SetString(PyExc_NotImplementedError, "Shared Wasm Python worker memory is unavailable");
    return NULL;
  }
  if (!size || memchr(name, '\0', (size_t)size)) {
    return blender_web_shared_memory_error(-5, name);
  }
  const int status = blender_web_shared_memory_request(
      BLENDER_WEB_SHARED_MEMORY_UNLINK, name, 0, 0, 0, NULL, NULL, NULL);
  if (status < 0) {
    return blender_web_shared_memory_error(status, name);
  }
  Py_RETURN_NONE;
}

static int blender_web_shared_memory_init(PyObject *module)
{
  static PyMethodDef map_methods[] = {
      {"close", blender_web_memory_map_close, METH_NOARGS, NULL},
      {NULL, NULL, 0, NULL},
  };
  static PyType_Slot map_slots[] = {
      {Py_tp_dealloc, (void *)blender_web_memory_map_dealloc},
      {Py_tp_methods, (void *)map_methods},
      {Py_bf_getbuffer, (void *)blender_web_memory_map_getbuffer},
      {Py_bf_releasebuffer, (void *)blender_web_memory_map_releasebuffer},
      {0, NULL},
  };
  static PyType_Spec map_spec = {
      "_blender_web_process._SharedMemoryMap",
      sizeof(BlenderWebMemoryMap),
      0,
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
      map_slots,
  };
  PyObject *type = PyType_FromSpec(&map_spec);
  if (!type) {
    return -1;
  }
  const int status = PyModule_AddObjectRef(module, "_SharedMemoryMap", type);
  Py_DECREF(type);
  return status;
}

#define BLENDER_WEB_SHARED_MEMORY_METHODS \
  {"shared_memory_available", blender_web_shared_memory_is_available, METH_NOARGS, NULL}, \
      {"shared_memory_open", blender_web_shared_memory_open, METH_VARARGS, NULL}, \
      {"shared_memory_unlink", blender_web_shared_memory_unlink, METH_O, NULL},

#endif /* BLENDER_WEB_SHARED_MEMORY_API_H */

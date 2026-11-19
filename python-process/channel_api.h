/* SPDX-License-Identifier: GPL-2.0-or-later */

#ifndef BLENDER_WEB_CHANNEL_API_H
#define BLENDER_WEB_CHANNEL_API_H

#include <Python.h>
#include <stdlib.h>

enum BlenderWebChannelOperation {
  BLENDER_WEB_CHANNEL_CREATE = 0,
  BLENDER_WEB_CHANNEL_PUT = 1,
  BLENDER_WEB_CHANNEL_GET = 2,
  BLENDER_WEB_CHANNEL_POLL = 3,
  BLENDER_WEB_CHANNEL_QSIZE = 4,
  BLENDER_WEB_CHANNEL_CLOSE = 5,
  BLENDER_WEB_CHANNEL_RETAIN = 6,
  BLENDER_WEB_CHANNEL_RELEASE = 7,
  BLENDER_WEB_CHANNEL_EXPORT = 8,
  BLENDER_WEB_CHANNEL_IMPORT = 9,
  BLENDER_WEB_CHANNEL_CANCEL = 10,
};

#ifdef __cplusplus
extern "C" {
#endif
int blender_web_channel_request(int operation,
                                int id,
                                int argument,
                                const char *input,
                                size_t input_size,
                                double seconds,
                                char **output,
                                size_t *output_size);
#ifdef __cplusplus
}
#endif

static PyObject *blender_web_channel_error(int status)
{
  switch (status) {
    case -1:
      PyErr_SetNone(PyExc_TimeoutError);
      break;
    case -2:
      PyErr_SetNone(PyExc_EOFError);
      break;
    case -3:
      PyErr_SetNone(PyExc_BrokenPipeError);
      break;
    case -4:
      PyErr_SetString(PyExc_ValueError, "Invalid or released browser channel handle");
      break;
    case -5:
      PyErr_SetString(PyExc_ValueError, "Invalid browser channel timeout");
      break;
    case -6:
      PyErr_SetString(PyExc_ValueError, "Invalid browser channel capacity or lease role");
      break;
    case -7:
      return PyErr_NoMemory();
    default:
      PyErr_Format(PyExc_RuntimeError, "Unexpected browser channel status %d", status);
      break;
  }
  return NULL;
}

static PyObject *blender_web_channel_control(int operation, int id, int argument, int returns_int)
{
  const int status = blender_web_channel_request(operation, id, argument, NULL, 0, 0, NULL, NULL);
  if (status < 0) {
    return blender_web_channel_error(status);
  }
  if (returns_int) {
    return PyLong_FromLong(status);
  }
  Py_RETURN_NONE;
}

static PyObject *blender_web_channel_id_control(int operation, PyObject *arg, int returns_int)
{
  const int id = PyLong_AsInt(arg);
  if (id == -1 && PyErr_Occurred()) {
    return NULL;
  }
  return blender_web_channel_control(operation, id, 0, returns_int);
}

static PyObject *blender_web_channel_role_control(int operation, PyObject *args, int returns_int)
{
  int id;
  int role;
  if (!PyArg_ParseTuple(args, "ii", &id, &role)) {
    return NULL;
  }
  return blender_web_channel_control(operation, id, role, returns_int);
}

static PyObject *blender_web_channel_create(PyObject *Py_UNUSED(self), PyObject *args)
{
  int maxsize = 0;
  if (!PyArg_ParseTuple(args, "|i:channel_create", &maxsize)) {
    return NULL;
  }
  return blender_web_channel_control(BLENDER_WEB_CHANNEL_CREATE, 0, maxsize, 1);
}

static PyObject *blender_web_channel_put(PyObject *Py_UNUSED(self), PyObject *args)
{
  int id;
  PyObject *payload;
  double seconds;
  if (!PyArg_ParseTuple(args, "iSd:channel_put", &id, &payload, &seconds)) {
    return NULL;
  }

  const char *input = PyBytes_AS_STRING(payload);
  const size_t input_size = (size_t)PyBytes_GET_SIZE(payload);
  int status;
  Py_BEGIN_ALLOW_THREADS
  status = blender_web_channel_request(
      BLENDER_WEB_CHANNEL_PUT, id, 0, input, input_size, seconds, NULL, NULL);
  Py_END_ALLOW_THREADS
  if (status < -1) {
    return blender_web_channel_error(status);
  }
  return PyBool_FromLong(status > 0);
}

static PyObject *blender_web_channel_get(PyObject *Py_UNUSED(self), PyObject *args)
{
  int id;
  double seconds;
  if (!PyArg_ParseTuple(args, "id:channel_get", &id, &seconds)) {
    return NULL;
  }

  char *output = NULL;
  size_t output_size = 0;
  int status;
  Py_BEGIN_ALLOW_THREADS
  status = blender_web_channel_request(
      BLENDER_WEB_CHANNEL_GET, id, 0, NULL, 0, seconds, &output, &output_size);
  Py_END_ALLOW_THREADS

  PyObject *result;
  if (status == -1) {
    result = Py_NewRef(Py_None);
  }
  else if (status < 0) {
    result = blender_web_channel_error(status);
  }
  else if (output_size > (size_t)PY_SSIZE_T_MAX) {
    PyErr_SetString(PyExc_OverflowError, "Browser channel response is too large");
    result = NULL;
  }
  else if (!output && output_size > 0) {
    PyErr_SetString(PyExc_RuntimeError, "Missing browser channel response payload");
    result = NULL;
  }
  else {
    result = PyBytes_FromStringAndSize(output, (Py_ssize_t)output_size);
  }
  free(output);
  return result;
}

static PyObject *blender_web_channel_poll(PyObject *Py_UNUSED(self), PyObject *args)
{
  int id;
  double seconds;
  if (!PyArg_ParseTuple(args, "id:channel_poll", &id, &seconds)) {
    return NULL;
  }

  int status;
  Py_BEGIN_ALLOW_THREADS
  status = blender_web_channel_request(
      BLENDER_WEB_CHANNEL_POLL, id, 0, NULL, 0, seconds, NULL, NULL);
  Py_END_ALLOW_THREADS
  if (status < -1) {
    return blender_web_channel_error(status);
  }
  return PyBool_FromLong(status > 0);
}

static PyObject *blender_web_channel_qsize(PyObject *Py_UNUSED(self), PyObject *arg)
{
  return blender_web_channel_id_control(BLENDER_WEB_CHANNEL_QSIZE, arg, 1);
}

static PyObject *blender_web_channel_close(PyObject *Py_UNUSED(self), PyObject *arg)
{
  return blender_web_channel_id_control(BLENDER_WEB_CHANNEL_CLOSE, arg, 0);
}

static PyObject *blender_web_channel_retain(PyObject *Py_UNUSED(self), PyObject *args)
{
  return blender_web_channel_role_control(BLENDER_WEB_CHANNEL_RETAIN, args, 0);
}

static PyObject *blender_web_channel_release(PyObject *Py_UNUSED(self), PyObject *args)
{
  return blender_web_channel_role_control(BLENDER_WEB_CHANNEL_RELEASE, args, 0);
}

static PyObject *blender_web_channel_export(PyObject *Py_UNUSED(self), PyObject *args)
{
  return blender_web_channel_role_control(BLENDER_WEB_CHANNEL_EXPORT, args, 1);
}

static PyObject *blender_web_channel_import(PyObject *Py_UNUSED(self), PyObject *arg)
{
  return blender_web_channel_id_control(BLENDER_WEB_CHANNEL_IMPORT, arg, 1);
}

static PyObject *blender_web_channel_cancel(PyObject *Py_UNUSED(self), PyObject *arg)
{
  return blender_web_channel_id_control(BLENDER_WEB_CHANNEL_CANCEL, arg, 0);
}

#define BLENDER_WEB_CHANNEL_METHODS \
  {"channel_create", blender_web_channel_create, METH_VARARGS, NULL}, \
      {"channel_put", blender_web_channel_put, METH_VARARGS, NULL}, \
      {"channel_get", blender_web_channel_get, METH_VARARGS, NULL}, \
      {"channel_poll", blender_web_channel_poll, METH_VARARGS, NULL}, \
      {"channel_qsize", blender_web_channel_qsize, METH_O, NULL}, \
      {"channel_close", blender_web_channel_close, METH_O, NULL}, \
      {"channel_retain", blender_web_channel_retain, METH_VARARGS, NULL}, \
      {"channel_release", blender_web_channel_release, METH_VARARGS, NULL}, \
      {"channel_export", blender_web_channel_export, METH_VARARGS, NULL}, \
      {"channel_import", blender_web_channel_import, METH_O, NULL}, \
      {"channel_cancel", blender_web_channel_cancel, METH_O, NULL},

#endif /* BLENDER_WEB_CHANNEL_API_H */

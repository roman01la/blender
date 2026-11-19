/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <emscripten.h>
#include <errno.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

extern char __heap_base;

static _Atomic uintptr_t program_break;
static uintptr_t arena_end;

EMSCRIPTEN_KEEPALIVE void blender_web_configure_child_arena(uintptr_t end)
{
  arena_end = end;
  atomic_store(&program_break, (uintptr_t)&__heap_base);
}

uintptr_t *emscripten_get_sbrk_ptr(void)
{
  return (uintptr_t *)&program_break;
}

void *_sbrk64(int64_t increment)
{
  const uintptr_t heap_base = (uintptr_t)&__heap_base;
  const uintptr_t mask = _Alignof(max_align_t) - 1;
  uintptr_t current = atomic_load(&program_break);
  for (;;) {
    uintptr_t next;
    if (increment < 0) {
      const uint64_t distance = (0 - (uint64_t)increment) & ~mask;
      if (distance > current - heap_base) {
        break;
      }
      next = current - distance;
    }
    else {
      const uint64_t distance = ((uint64_t)increment + mask) & ~mask;
      if (distance > arena_end - current) {
        break;
      }
      next = current + distance;
    }
    if (atomic_compare_exchange_weak(&program_break, &current, next)) {
      return (void *)current;
    }
  }
  errno = ENOMEM;
  return (void *)-1;
}

void *sbrk(intptr_t increment)
{
  return _sbrk64(increment);
}

int brk(void *address)
{
  (void)address;
  errno = ENOSYS;
  return -1;
}

/* The pthread runtime has weak references to the separate Wasm Workers API. */
int emscripten_wasm_worker_self_id(void)
{
  return 0;
}

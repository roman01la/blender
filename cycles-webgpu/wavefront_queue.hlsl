/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __CYCLES_WEBGPU_WAVEFRONT_QUEUE__
#define __CYCLES_WEBGPU_WAVEFRONT_QUEUE__

/* The compact queue occupies the tail of the ray buffer, immediately after
 * its count and fixed-size ray slots. Every stage keeps original slot IDs. */
bool cycles_queue_slot(uint dispatch_index, out uint lane)
{
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  const uint base = 1 + 12 * ray_words[0];
  if (dispatch_index >= ray_words[base]) {
    lane = 0;
    return false;
  }
  lane = ray_words[base + 1 + dispatch_index];
  return true;
#else
  lane = dispatch_index;
  return lane < ray_words[0];
#endif
}

#endif

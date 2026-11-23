/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

using Float4 = float __attribute__((ext_vector_type(4)));
using UInt4 = unsigned int __attribute__((ext_vector_type(4)));

UInt4 cycles_work_pixel(unsigned int tile_x,
                        unsigned int tile_y,
                        unsigned int tile_width,
                        unsigned int tile_height,
                        unsigned int start_sample,
                        unsigned int num_samples,
                        unsigned int work_index,
                        float scrambling_distance);

void cycles_camera_ray(UInt4 pixel,
                       Float4 projection_x,
                       Float4 projection_y,
                       Float4 projection_z,
                       Float4 projection_w,
                       Float4 world_x,
                       Float4 world_y,
                       Float4 world_z,
                       Float4 clip,
                       Float4 lens,
                       Float4 aperture,
                       Float4 dx,
                       Float4 dy,
                       Float4 &origin,
                       Float4 &direction,
                       unsigned int &rng_pixel,
                       float &dP,
                       float &dD);

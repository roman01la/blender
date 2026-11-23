/* SPDX-License-Identifier: Apache-2.0 */

#include <array>
#include <bit>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "camera_kernels.hh"

#ifdef __aarch64__
#  include <arm_neon.h>
#endif
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

static std::vector<unsigned int> sampling{~0u};

unsigned int cycles_camera_sample_word(unsigned int i)
{
  return sampling[i];
}

static void set_sampling(const unsigned int pattern, const bool fixed_jitter)
{
  const unsigned int sequence_size = 2;
  sampling = {pattern, 0x89327aec, 0xe0000000, 7, sequence_size,
              std::bit_cast<unsigned int>(fixed_jitter ? -0.234f : FLT_MAX),
              std::bit_cast<unsigned int>(fixed_jitter ? -0.722f : FLT_MAX)};
  for (int i = 0; i < FILTER_TABLE_SIZE; ++i) {
    sampling.push_back(std::bit_cast<unsigned int>(float(i) / (FILTER_TABLE_SIZE - 1)));
  }
  if (pattern == ccl::SAMPLING_PATTERN_TABULATED_SOBOL) {
    const unsigned int entries = sequence_size * NUM_TAB_SOBOL_PATTERNS * NUM_TAB_SOBOL_DIMENSIONS;
    for (unsigned int i = 0; i < entries; ++i) {
      sampling.push_back(std::bit_cast<unsigned int>(float(i) / float(entries)));
    }
  }
}

struct TestCase {
  std::array<unsigned int, 6> tile;
  std::array<Float4, 12> camera;
  unsigned int visibility = 1;
};

template<typename Range> void write_range(const Range &range)
{
  std::cout << '[';
  bool first = true;
  for (const auto &value : range) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << value;
  }
  std::cout << ']';
}

int main(int argc, char **argv)
{
  const TestCase camera_cases[] = {
      {{2, 3, 11, 7, 6, 3},
       {Float4{0.22f, 0, 0, -2}, Float4{0, 0.19f, 0, -1.4f}, Float4{0, 0, 0, 1},
        Float4{0, 0, 0, 1}, Float4{1, 0, 0, 0}, Float4{0, 1, 0, 0}, Float4{0, 0, 1, 0},
        Float4{0.12f, 42, 0.2f, 0}}},
      {{13, 17, 8, 5, 17, 4},
       {Float4{0.1f, 0, 0, -1.5f}, Float4{0, 0.11f, 0, -2}, Float4{0, 0, 0, 1.7f},
        Float4{0.01f, -0.002f, 0, 0.9f},
        Float4{0, 0, 1, 19}, Float4{0, 1, 0, 35}, Float4{-1, 0, 0, -21},
        Float4{0.043f, 501, 1.1f, 0}}},
      {{0, 0, 1, 1, 0, 1},
       {Float4{1, 0, 0, -0.5f}, Float4{0, 1, 0, -0.5f}, Float4{0, 0, 0, 1},
        Float4{0, 0, 0, 1}, Float4{1, 0, 0, 0}, Float4{0, 1, 0, 0}, Float4{0, 0, 1, 0},
        Float4{0.1f, 10, 0.9f, 0}}},
  };
  // View the packed CPU BVH fixture from above. The cameras cover all three
  // instances and the gaps between them, with both work tile orderings.
  const std::array<Float4, 12> overhead{
      Float4{0.0577f, 0, 0, -1.401f}, Float4{0, 0.0713f, 0, -0.611f},
      Float4{0, 0, 0, 1}, Float4{0, 0, 0, 1},
      Float4{1, 0, 0, 0.132f}, Float4{0, -1, 0, -0.172f}, Float4{0, 0, -1, 17.7f},
      Float4{0.13f, 48, 0.2f, 0}};
  auto diffuse_view = overhead;
  diffuse_view[7].z = 1.1f;
  auto clipped_view = overhead;
  clipped_view[7].y = 4.5f;
  auto below_view = overhead;
  below_view[5] = {0, 1, 0, 0.132f};
  below_view[6] = {0, 0, 1, -15.2f};
  const TestCase primary_cases[] = {
      {{3, 2, 43, 13, 4, 2}, overhead, 1},
      {{3, 2, 43, 13, 7, 2}, diffuse_view, 4},
      {{15, 4, 21, 9, 0, 1}, clipped_view, 1},
      {{4, 3, 37, 13, 9, 2},
       {Float4{0.063f, 0, 0, -1.361f}, Float4{0, 0.0721f, 0, -0.659f},
        Float4{0, 0, 0, 1}, Float4{0.0011f, -0.0003f, 0, 0.99f},
        Float4{0.961f, 0.276f, 0, 2.1f}, Float4{0.276f, -0.961f, 0, -0.37f},
        Float4{0, 0, -1, 16.1f}, Float4{0.082f, 45, 1.9f, 0}},
       1},
      {{5, 3, 37, 12, 10, 1}, below_view, 1},
  };
  const bool sampled = argc == 2 && std::string_view(argv[1]) == "--sampled-primary";
  const bool primary = sampled || (argc == 2 && std::string_view(argv[1]) == "--primary");
  if (argc != 1 && !primary) {
    return 2;
  }
  auto perspective_dof = overhead;
  perspective_dof[8] = {0, 0.26f, 12.0f, 1.0f};
  auto perspective_polygon = perspective_dof;
  perspective_polygon[8].w = 0.67f;
  perspective_polygon[9] = {5, 0.21f, 0, 0};

  auto orthographic = overhead;
  orthographic[7].x = 0.25f;
  orthographic[2] = {};
  orthographic[8] = {float(ccl::CAMERA_ORTHOGRAPHIC), 0, 15.0f, 1.0f};
  auto orthographic_dof = orthographic;
  orthographic_dof[8].y = 0.46f;
  orthographic_dof[8].w = 1.4f;
  orthographic_dof[9] = {7, -0.15f, 0, 0};

  const TestCase sampled_cases[] = {
      {{3, 2, 43, 13, 0, 3}, overhead, 1},
      {{3, 2, 43, 13, 2, 3}, diffuse_view, 4},
      {{5, 3, 37, 12, 2, 3}, overhead, 1},
      {{5, 3, 37, 12, 0, 1}, below_view, 1},
      {{18, 5, 13, 8, 0, 4}, perspective_dof, 1},
      {{18, 5, 13, 8, 0, 4}, perspective_polygon, 1},
      {{9, 3, 10, 7, 0, 3}, orthographic, 1},
      {{9, 3, 10, 7, 0, 5}, orthographic_dof, 1},
  };
  const unsigned int patterns[] = {ccl::SAMPLING_PATTERN_BLUE_NOISE_PURE,
                                   ccl::SAMPLING_PATTERN_SOBOL_BURLEY,
                                   ccl::SAMPLING_PATTERN_TABULATED_SOBOL,
                                   ccl::SAMPLING_PATTERN_BLUE_NOISE_FIRST};
  const auto cases = sampled ? std::span<const TestCase>{sampled_cases} :
                     primary ? std::span<const TestCase>{primary_cases} :
                               std::span<const TestCase>{camera_cases};

  std::cout << std::setprecision(9) << '[';
  for (size_t i = 0; i < std::size(cases); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    const auto &[tile, camera, visibility] = cases[i];
    const unsigned int count = tile[2] * tile[3] * tile[5];
    std::cout << "{\"tile\":[";
    for (auto value : tile) {
      std::cout << value << ',';
    }
    std::cout << count;
    if (primary) {
      std::cout << ',' << visibility;
    }
    std::cout << "],\"camera\":[";
    for (size_t row = 0; row < camera.size(); ++row) {
      if (row != 0) {
        std::cout << ',';
      }
      const auto &r = camera[row];
      write_range(std::array{r.x, r.y, r.z, r.w});
    }
    if (sampled) {
      const size_t pattern_index = i % std::size(patterns);
      set_sampling(patterns[pattern_index], pattern_index == 3);
      std::cout << "],\"sampling\":";
      write_range(sampling);
      std::cout << ",\"expected\":[";
    }
    else {
      std::cout << "],\"expected\":[";
    }
    for (unsigned int index = 0; index < count; ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      const auto pixel = cycles_work_pixel(tile[0], tile[1], tile[2], tile[3], tile[4], tile[5],
                                           index, camera[7].z);
      Float4 origin, direction;
      unsigned int rng_pixel;
      float dP, dD;
      cycles_camera_ray(pixel, camera[0], camera[1], camera[2], camera[3], camera[4], camera[5],
                        camera[6], camera[7], camera[8], camera[9], camera[10], camera[11],
                        origin, direction, rng_pixel, dP, dD);
      std::cout << '[';
      write_range(std::array{pixel.x, pixel.y, pixel.z, pixel.w});
      std::cout << ',';
      write_range(std::array{origin.x, origin.y, origin.z, origin.w, direction.x, direction.y,
                             direction.z, direction.w});
      std::cout << ']';
    }
    std::cout << "]}";
  }
  std::cout << "]\n";
}

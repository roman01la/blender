/* SPDX-License-Identifier: Apache-2.0 */

#include <array>
#include <bit>
#include <iostream>
#include <vector>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

void cycles_subsurface_walk(Float4 entry, Float4 direction, Float4 normal, Float4 albedo,
                            Float4 radius, Float4 weight, UInt4 rng, UInt4 surface, UInt4 &hit,
                            Float4 &coordinates, Float4 &exit_direction, Float4 &throughput);

namespace {

std::vector<unsigned int> scene(10, 0);
const std::array surface{1u, unsigned(ccl::SD_OBJECT_TRANSFORM_APPLIED), 0u, 0u, 0u, 0u, 0u,
                         0u, 0u, 0u, 0u, 0u, 0u, 0u, unsigned(ccl::SD_OBJECT_TRANSFORM_APPLIED)};
const std::array sampling{unsigned(ccl::SAMPLING_PATTERN_SOBOL_BURLEY), 0u, 0u, 0u, 0u, 0u, 0u};

template<typename Range> void print_array(const Range &values)
{
  bool first = true;
  std::cout << '[';
  for (const auto value : values) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << value;
  }
  std::cout << ']';
}

void append(std::vector<unsigned int> &words, const UInt4 v)
{
  words.insert(words.end(), {v.x, v.y, v.z, v.w});
}

}  // namespace

unsigned int cycles_scene_word(unsigned int index)
{
  return scene.at(index);
}

unsigned int cycles_surface_word(unsigned int index)
{
  return surface.at(index);
}

unsigned int cycles_camera_sample_word(unsigned int index)
{
  return sampling.at(index);
}

int main()
{
  /* One flattened leaf containing two slabs of the local object and a blocker
   * on a different object. The local traversal must ignore the blocker. */
  const std::array vertices{
      -10.0f, -10.0f, 0.0f, 10.0f, -10.0f, 0.0f, 0.0f, 10.0f, 0.0f,
      -10.0f, -10.0f, 0.3f, 10.0f, -10.0f, 0.3f, 0.0f, 10.0f, 0.3f,
      -10.0f, -10.0f, 0.1f, 10.0f, -10.0f, 0.1f, 0.0f, 10.0f, 0.1f,
  };
  scene[0] = 0xffffffffu;
  const auto section = [](unsigned int slot, std::initializer_list<unsigned int> words) {
    scene[slot] = scene.size();
    scene.insert(scene.end(), words);
  };
  section(1, {});
  section(2, {0, 3, 0, ccl::PRIMITIVE_TRIANGLE});
  section(3, {0, 0, 0});
  section(4, {0, 1, 2});
  section(5, {0, 0, 1});
  section(6, {0xffffffffu, 0xffffffffu});
  scene[7] = scene.size();
  scene.resize(scene.size() + 26);
  section(8, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  scene[9] = scene.size();
  for (const float value : vertices) {
    scene.push_back(std::bit_cast<unsigned int>(value));
  }

  constexpr unsigned int count = 32;
  std::vector<unsigned int> inputs, output;
  append(inputs, UInt4{count, 0, 0, 0});
  unsigned int hits = 0;
  for (unsigned int i = 0; i < count; ++i) {
    const Float4 entry{0, 0, 0, 0};
    const Float4 direction{0, 0, 1, 0};
    const bool skin = i % 2;
    const Float4 normal{0, 0, -1, skin ? 1.0f : 0.0f};
    const Float4 albedo{0.8f, 0.6f, 0.4f, 1.4f};
    const Float4 radius{0.7f, 0.5f, 0.3f, skin ? 2.15f : 0.0f};
    const Float4 weight{0.4f, 0.3f, 0.2f, skin ? 1.0f : 0.25f};
    const UInt4 rng{42, 48, i, std::bit_cast<unsigned int>(1.0f)};
    const UInt4 shading_hit{0, 0, 0, 0};
    for (const Float4 value : {entry, direction, normal, albedo, radius, weight}) {
      append(inputs, std::bit_cast<UInt4>(value));
    }
    append(inputs, rng);
    append(inputs, shading_hit);
    UInt4 hit;
    Float4 coordinates, exit_direction, throughput;
    cycles_subsurface_walk(entry, direction, normal, albedo, radius, weight, rng, shading_hit,
                           hit, coordinates, exit_direction, throughput);
    if (hit.x) {
      ++hits;
      if (hit.z != 0 || hit.y > 1) {
        std::cerr << "Subsurface walk hit a different object\n";
        return 1;
      }
    }
    append(output, hit);
    append(output, std::bit_cast<UInt4>(coordinates));
    append(output, std::bit_cast<UInt4>(exit_direction));
    append(output, std::bit_cast<UInt4>(throughput));
  }
  if (hits == 0) {
    std::cerr << "Subsurface fixture did not find an exit\n";
    return 1;
  }
  std::cout << "{\"scene\":";
  print_array(scene);
  std::cout << ",\"surface\":";
  print_array(surface);
  std::cout << ",\"sampling\":";
  print_array(sampling);
  std::cout << ",\"entries\":";
  print_array(inputs);
  std::cout << ",\"expected\":";
  print_array(output);
  std::cout << "}\n";
}

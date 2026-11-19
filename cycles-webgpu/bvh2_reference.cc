/* SPDX-License-Identifier: Apache-2.0 */

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
void cycles_bvh2_trace(
    UInt4 origin, UInt4 direction, UInt4 options, UInt4 &hit, UInt4 &coordinates);

static std::vector<uint32_t> scene_words;

unsigned int cycles_scene_word(unsigned int index)
{
  return scene_words.at(index);
}

static std::vector<uint32_t> read_words(const char *path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (size <= 0 || size % 4 || !input) {
    throw std::runtime_error(std::string("Invalid word buffer: ") + path);
  }
  std::vector<uint32_t> words(size / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), size);
  if (!input) {
    throw std::runtime_error(std::string("Unable to read word buffer: ") + path);
  }
  return words;
}

int main(int argc, char **argv)
{
  static_assert(std::endian::native == std::endian::little);
  if (argc != 4) {
    std::cerr << "Usage: bvh2-reference scene.bin rays.bin hits.bin\n";
    return 1;
  }
  try {
    scene_words = read_words(argv[1]);
    const auto ray_words = read_words(argv[2]);
    const uint32_t count = ray_words[0];
    if (scene_words.size() < 10 || count == 0 || ray_words.size() != 1 + size_t(count) * 12) {
      throw std::runtime_error("Invalid BVH fixture layout");
    }
    std::ofstream output(argv[3], std::ios::binary);
    for (uint32_t i = 0; i < count; i++) {
      const auto *ray = ray_words.data() + 1 + size_t(i) * 12;
      const UInt4 origin{ray[0], ray[1], ray[2], ray[3]};
      const UInt4 direction{ray[4], ray[5], ray[6], ray[7]};
      const UInt4 options{ray[8], ray[9], ray[10], ray[11]};
      UInt4 hit, coordinates;
      cycles_bvh2_trace(origin, direction, options, hit, coordinates);
      const std::array words{
          hit.x, hit.y, hit.z, hit.w, coordinates.x, coordinates.y, coordinates.z, coordinates.w};
      output.write(reinterpret_cast<const char *>(words.data()), sizeof(words));
    }
    if (!output) {
      throw std::runtime_error("Unable to write hit buffer");
    }
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

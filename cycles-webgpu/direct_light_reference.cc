/* SPDX-License-Identifier: Apache-2.0 */

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

void cycles_triangle_surface(
    UInt4 hit, UInt4 coordinates, UInt4 direction, UInt4 &position, UInt4 &normal);
unsigned int cycles_direct_light(UInt4 position, UInt4 normal, UInt4 primary_hit, Float4 &result);

static std::vector<uint32_t> scene_words;
static std::vector<uint32_t> surface_words;
static std::vector<uint32_t> light_words;

unsigned int cycles_scene_word(unsigned int index)
{
  return scene_words.at(index);
}

unsigned int cycles_surface_word(unsigned int index)
{
  return surface_words.at(index);
}

unsigned int cycles_light_word(unsigned int index)
{
  return light_words.at(index);
}

static std::vector<uint32_t> read_words(const char *path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (size <= 0 || size % 4) {
    throw std::runtime_error(std::string("Invalid word buffer: ") + path);
  }
  std::vector<uint32_t> words(size / 4);
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(words.data()), size)) {
    throw std::runtime_error(std::string("Unable to read word buffer: ") + path);
  }
  return words;
}

static UInt4 read_record(const uint32_t *words)
{
  return {words[0], words[1], words[2], words[3]};
}

int main(int argc, char **argv)
{
  static_assert(std::endian::native == std::endian::little);
  if (argc != 7) {
    std::cerr << "Usage: direct-light-reference scene.bin rays.bin hits.bin surface.bin "
                 "lighting.bin samples.bin\n";
    return 1;
  }

  try {
    scene_words = read_words(argv[1]);
    const auto rays = read_words(argv[2]);
    const auto hits = read_words(argv[3]);
    surface_words = read_words(argv[4]);
    light_words = read_words(argv[5]);

    const size_t count = rays[0];
    if (scene_words.size() < 10 || count == 0 || rays.size() != 1 + count * 12 ||
        hits.size() != count * 8 || surface_words.size() < 2 || light_words.size() < 12 ||
        light_words.size() != 12 + size_t(light_words[3]) * 4)
    {
      throw std::runtime_error("Invalid direct light fixture layout");
    }

    std::ofstream output(argv[6], std::ios::binary);
    std::array<size_t, 5> status_counts{};
    for (size_t index = 0; index < count; ++index) {
      const UInt4 direction = read_record(rays.data() + 5 + index * 12);
      const UInt4 hit = read_record(hits.data() + index * 8);
      const UInt4 coordinates = read_record(hits.data() + index * 8 + 4);
      UInt4 position, normal;
      cycles_triangle_surface(hit, coordinates, direction, position, normal);
      Float4 sample;
      const unsigned int status = cycles_direct_light(position, normal, hit, sample);
      ++status_counts.at(status);
      output.write(reinterpret_cast<const char *>(&sample), sizeof(sample));
    }
    if (!output) {
      throw std::runtime_error("Unable to write sample buffer");
    }

    std::cout << "{\"misses\":" << status_counts[0] << ",\"unsupported\":" << status_counts[1]
              << ",\"backlit\":" << status_counts[2] << ",\"shadowed\":" << status_counts[3]
              << ",\"lit\":" << status_counts[4] << "}\n";
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

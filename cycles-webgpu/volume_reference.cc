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
#include "kernel/svm/node_types.h"
#include "volume_kernels.hh"

namespace {

using namespace ccl;

SVMInputFloat literal(const float value)
{
  return {std::bit_cast<uint>(value)};
}

SVMInputFloat3 literal(const float r, const float g, const float b)
{
  return {literal(r), literal(g), literal(b)};
}

template<typename Node> std::array<VolumeUInt4, 5> data(const Node &node)
{
  std::array<VolumeUInt4, 5> words{};
  static_assert(sizeof(Node) <= sizeof(words));
  __builtin_memcpy(words.data(), &node, sizeof(node));
  return words;
}

void append(std::vector<uint> &words, const VolumeUInt4 record)
{
  words.insert(words.end(), {record.x, record.y, record.z, record.w});
}

void print_array(const std::vector<uint> &values)
{
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      std::cout << ',';
    }
    std::cout << values[i];
  }
  std::cout << ']';
}

struct Case {
  ShaderNodeType type;
  std::array<VolumeUInt4, 5> words;
  int phases;
};

}  // namespace

int main()
{
  std::vector<Case> cases;
  cases.push_back({NODE_CLOSURE_VOLUME,
                   data(SVMNodeClosureVolume{.closure_type = CLOSURE_VOLUME_ABSORPTION_ID,
                                             .density = literal(0.7f),
                                             .mix_weight_offset = SVM_STACK_INVALID}),
                   0});
  struct Phase {
    ClosureType closure;
    float param1;
    float param_extra;
  };
  const std::array phase_types{
      Phase{CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, 0.3f, 0.0f},
      Phase{CLOSURE_VOLUME_FOURNIER_FORAND_ID, 1.33f, 0.05f},
      Phase{CLOSURE_VOLUME_DRAINE_ID, -0.24f, 0.6f},
      Phase{CLOSURE_VOLUME_RAYLEIGH_ID, 0.0f, 0.0f},
      Phase{CLOSURE_VOLUME_MIE_ID, 3.0f, 0.0f},
  };
  for (const auto &phase : phase_types) {
    const int phases = phase.closure == CLOSURE_VOLUME_MIE_ID ? 2 : 1;
    cases.push_back({NODE_CLOSURE_VOLUME,
                     data(SVMNodeClosureVolume{.closure_type = phase.closure,
                                               .density = literal(0.7f),
                                               .param1 = literal(phase.param1),
                                               .param_extra = literal(phase.param_extra),
                                               .mix_weight_offset = SVM_STACK_INVALID}),
                     phases});
    cases.push_back({NODE_VOLUME_COEFFICIENTS,
                     data(SVMNodeVolumeCoefficients{.closure_type = phase.closure,
                                                   .absorption_coeffs = literal(0.1f, 0.0f, 0.9f),
                                                   .emission_coeffs = literal(0.1f, 0.2f, 0.0f),
                                                   .param1 = literal(phase.param1),
                                                   .param_extra = literal(phase.param_extra),
                                                   .mix_weight_offset = SVM_STACK_INVALID}),
                     phases});
  }
  cases.push_back({NODE_PRINCIPLED_VOLUME,
                   data(SVMNodePrincipledVolume{.absorption_color = literal(0.7f, 0.8f, 0.1f),
                                                .emission_color = literal(0.4f, 0.3f, 0.8f),
                                                .blackbody_tint = literal(0.2f, 0.3f, 0.4f),
                                                .density = literal(0.9f),
                                                .anisotropy = literal(-0.2f),
                                                .emission = literal(0.03f),
                                                .blackbody = literal(0.25f),
                                                .temperature = literal(2980.0f),
                                                .mix_weight_offset = SVM_STACK_INVALID}),
                   1});

  constexpr uint sample_count = 16;
  std::vector<uint> entries, expected;
  append(entries, VolumeUInt4{uint(cases.size()) * sample_count, 0, 0, 0});
  for (const auto &test : cases) {
    for (uint i = 0; i < sample_count; ++i) {
      const bool shadow = i % 4 == 0;
      const bool extinction_only = i % 4 == 1;
      const uint visibility = shadow ? PATH_RAY_VISIBILITY_SHADOW : PATH_RAY_VISIBILITY_CAMERA;
      const uint flags = extinction_only ? PATH_RAY_EXTINCTION : 0u;
      const VolumeUInt4 options{test.type, visibility, flags, 0u};
      const VolumeFloat4 weight{0.25f, 0.6f, 0.8f, (i % 2) ? 1.0f : 1.25f};
      const VolumeFloat4 incoming{0, 0, 1, 0};
      const VolumeFloat4 random{(i + 0.41f) / sample_count, (i + 1.0f) / (sample_count + 1),
                               i * 0.23f, 0.0f};
      append(entries, options);
      append(entries, std::bit_cast<VolumeUInt4>(weight));
      for (const auto &word : test.words) {
        append(entries, word);
      }
      append(entries, std::bit_cast<VolumeUInt4>(incoming));
      append(entries, std::bit_cast<VolumeUInt4>(random));
      VolumeFloat4 extinction, scattering, transmittance, outgoing, emission;
      cycles_volume_coefficients(options, weight, test.words[0], test.words[1], test.words[2],
                                  test.words[3], test.words[4], incoming, random, extinction,
                                  scattering, transmittance, outgoing, emission);
      const int phases = shadow || extinction_only ? 0 : test.phases;
      if (extinction.w != phases ||
          (phases > 0 && (scattering.w != float(LABEL_VOLUME_SCATTER) || !(transmittance.w > 0.0f))) ||
          ((shadow || extinction_only) && !is_zero(make_float3(emission.x, emission.y, emission.z))))
      {
        std::cerr << "Unexpected stock volume result for node " << test.type << ", closure "
                  << test.words[0].x << ", sample " << i << ", phases " << extinction.w
                  << " / " << phases << ", label " << scattering.w << ", pdf " << transmittance.w
                  << '\n';
        return 1;
      }
      for (const auto result : {extinction, scattering, transmittance, outgoing, emission}) {
        append(expected, std::bit_cast<VolumeUInt4>(result));
      }
    }
  }
  std::cout << "{\"entries\":";
  print_array(entries);
  std::cout << ",\"expected\":";
  print_array(expected);
  std::cout << "}\n";
}

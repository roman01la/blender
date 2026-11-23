/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

CCL_NAMESPACE_BEGIN

/* The stock ShaderNodeType enum supplies these indices. Keep the mask out of
 * the packed GPU scene ABI: it is only needed during pipeline selection. */
using WebGPUSvmNodeMask = std::array<uint32_t, 8>;

struct WebGPUSvmNodeUsage {
  WebGPUSvmNodeMask surface;
  WebGPUSvmNodeMask shadow;
  WebGPUSvmNodeMask emission;
};

/* This versioned trailer belongs to the host transfer, not the GPU scene ABI.
 * Strip it before applying the adapter's storage binding limit or accessing any
 * shader payload. The two tags frame the masks separately from the SVM words. */
constexpr uint32_t webgpu_svm_node_trailer_tag = 0x53564d31u; /* SVM1 */
constexpr size_t webgpu_svm_node_trailer_words = 2 + 3 * WebGPUSvmNodeMask{}.size();

inline void webgpu_append_svm_node_usage(std::vector<uint32_t> &packed,
                                         const WebGPUSvmNodeUsage &usage)
{
  packed.push_back(webgpu_svm_node_trailer_tag);
  for (const auto *mask : {&usage.surface, &usage.shadow, &usage.emission}) {
    packed.insert(packed.end(), mask->begin(), mask->end());
  }
  packed.push_back(~webgpu_svm_node_trailer_tag);
}

inline WebGPUSvmNodeUsage webgpu_remove_svm_node_usage(std::span<const uint32_t> &packed)
{
  WebGPUSvmNodeUsage usage{};
  if (packed.size() < webgpu_svm_node_trailer_words) {
    return usage;
  }
  const auto trailer = packed.last(webgpu_svm_node_trailer_words);
  if (trailer.front() != webgpu_svm_node_trailer_tag ||
      trailer.back() != ~webgpu_svm_node_trailer_tag)
  {
    return usage;
  }
  size_t offset = 1;
  for (auto *mask : {&usage.surface, &usage.shadow, &usage.emission}) {
    std::copy_n(trailer.data() + offset, mask->size(), mask->begin());
    offset += mask->size();
  }
  packed = packed.first(packed.size() - trailer.size());
  return usage;
}

inline std::string webgpu_svm_mask_nodes(const WebGPUSvmNodeMask &mask)
{
  std::string result;
  for (uint32_t node = 0; node < mask.size() * 32; ++node) {
    if (mask[node / 32] & (1u << (node % 32))) {
      if (!result.empty()) {
        result += ',';
      }
      result += std::to_string(node);
    }
  }
  return result;
}

inline bool webgpu_svm_specialization_covers(const std::string_view source,
                                            const WebGPUSvmNodeMask &needs)
{
  constexpr std::string_view marker = "\n// Cycles WebGPU specialized SVM nodes: ";
  const size_t start = source.find(marker);
  if (start == source.npos) {
    return false;
  }
  const auto tail = source.substr(start + marker.size());
  auto list = tail.substr(0, tail.find('\n'));
  WebGPUSvmNodeMask provides{};
  while (!list.empty()) {
    const auto item = list.substr(0, list.find(','));
    uint32_t node;
    const auto result = std::from_chars(item.data(), item.data() + item.size(), node);
    if (result.ec != std::errc{} || result.ptr != item.data() + item.size() ||
        node >= provides.size() * 32)
    {
      return false;
    }
    provides[node / 32] |= 1u << (node % 32);
    if (item.size() == list.size()) {
      break;
    }
    list.remove_prefix(item.size() + 1);
  }
  bool used = false;
  for (size_t i = 0; i < needs.size(); ++i) {
    if (needs[i] & ~provides[i]) {
      return false;
    }
    used |= needs[i] != 0;
  }
  return used;
}

CCL_NAMESPACE_END

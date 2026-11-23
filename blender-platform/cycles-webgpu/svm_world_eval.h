/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "svm_material_eval.h"

CCL_NAMESPACE_BEGIN

/* The world fast path has no surface state or closures besides a single
 * Background. Interpret the synchronized stock SVM records using the same
 * math, color, texture, and mapping helpers as the material shader. */
template<bool Validate = true, typename Fetch, typename FetchImage>
ccl_device_inline bool webgpu_svm_eval_world(Fetch fetch,
                                             FetchImage image,
                                             uint offset,
                                             const uint end,
                                             const float3 direction,
                                             const float3 rgb_to_y,
                                             Spectrum &radiance)
{
  float stack[SVM_STACK_SIZE];
  WebGPUSvmStackUsage written;
  const auto read = [&]<typename Node>(Node &node) {
    if constexpr (Validate) {
      return webgpu_svm_read(fetch, offset, end, node);
    }
    else {
      constexpr uint words = sizeof(Node) / sizeof(uint);
      node = webgpu_svm_decode<Node>(fetch, offset, std::make_index_sequence<words>{});
      offset += words;
      return true;
    }
  };
  bool has_image = false;
  bool has_weight = false;
  bool has_background = false;
  Spectrum weight = zero_spectrum();
  radiance = zero_spectrum();

  while (offset < end) {
    const uint type = fetch(offset++);
    switch (type) {
      case NODE_END:
        return !Validate || (offset == end && has_image && has_background);
      case NODE_GEOMETRY: {
        SVMNodeGeometry node;
        if (!read(node) ||
            (Validate && (node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
                          uint(node.out_offset) + 2 >= SVM_STACK_SIZE)))
        {
          return false;
        }
        const float3 value = node.geom_type == NODE_GEOM_P ? direction : -direction;
        if (Validate && node.geom_type != NODE_GEOM_P && node.geom_type != NODE_GEOM_I &&
            node.geom_type != NODE_GEOM_N && node.geom_type != NODE_GEOM_Ng)
        {
          return false;
        }
        stack_store_float3(stack, node.out_offset, value);
        webgpu_svm_stack_mark3(written, node.out_offset);
        break;
      }
      case NODE_VALUE_F: {
        SVMNodeValueF node;
        if (!read(node) || (Validate && (node.out_offset >= SVM_STACK_SIZE))) {
          return false;
        }
        svm_node_value_f<float>(stack, node);
        webgpu_svm_stack_mark(written, node.out_offset);
        break;
      }
      case NODE_VALUE_V: {
        SVMNodeValueV node;
        if (!read(node) || (Validate && (uint(node.out_offset) + 2 >= SVM_STACK_SIZE))) {
          return false;
        }
        stack_store_float(stack, node.out_offset, node.value.x);
        stack_store_float(stack, uint(node.out_offset) + 1, node.value.y);
        stack_store_float(stack, uint(node.out_offset) + 2, node.value.z);
        webgpu_svm_stack_mark3(written, node.out_offset);
        break;
      }
      case NODE_MAPPING: {
        SVMNodeMapping node;
        if (!read(node) ||
            (Validate && (node.mapping_type > NODE_MAPPING_TYPE_NORMAL ||
                          !webgpu_svm_inputs_valid(
                              written, node.vector, node.location, node.rotation, node.scale) ||
                          uint(node.result_offset) + 2 >= SVM_STACK_SIZE)))
        {
          return false;
        }
        svm_node_mapping<float3>(stack, node);
        webgpu_svm_stack_mark3(written, node.result_offset);
        break;
      }
      case NODE_TEXTURE_MAPPING: {
        SVMNodeTextureMapping node;
        if (!read(node) || (Validate && (!webgpu_svm_stack_has3(written, node.vec_offset) ||
                                         uint(node.out_offset) + 2 >= SVM_STACK_SIZE)))
        {
          return false;
        }
        svm_node_texture_mapping<float3>(stack, node);
        webgpu_svm_stack_mark3(written, node.out_offset);
        break;
      }
      case NODE_TEX_ENVIRONMENT: {
        SVMNodeTexEnvironment node;
        if (!read(node)) {
          return false;
        }
        if constexpr (Validate) {
          WebGPUSvmStackUsage varying;
          WebGPUSvmNormalSlots normals;
          if (has_image ||
              (node.projection != NODE_ENVIRONMENT_EQUIRECTANGULAR &&
               node.projection != NODE_ENVIRONMENT_MIRROR_BALL) ||
              !stack_valid(node.out_offset) || stack_valid(node.alpha_offset) ||
              (node.flags & NODE_IMAGE_ALPHA_UNASSOCIATE) ||
              !webgpu_svm_eval_image(
                  image, node, WebGPUSvmNoSurface{}, stack, written, varying, normals))
          {
            return false;
          }
        }
        else {
          const float2 uv = webgpu_svm_environment_coordinates(stack_load_float3(stack, node.co),
                                                               node.projection);
          float4 color;
          if (!webgpu_svm_fetch_image(image, node, uv, color)) {
            return false;
          }
          stack_store_float3(stack, node.out_offset, make_float3(color));
        }
        has_image = true;
        break;
      }
      case NODE_MATH: {
        SVMNodeMath node;
        if (!read(node) ||
            (Validate &&
             (node.math_type > NODE_MATH_FLOORED_MODULO || node.result_offset >= SVM_STACK_SIZE ||
              !webgpu_svm_inputs_valid(written, node.value1, node.value2, node.value3))))
        {
          return false;
        }
        svm_node_math(stack, node);
        webgpu_svm_stack_mark(written, node.result_offset);
        break;
      }
      case NODE_CLAMP: {
        SVMNodeClamp node;
        if (!read(node) ||
            (Validate &&
             (node.clamp_type > NODE_CLAMP_RANGE || node.result_offset >= SVM_STACK_SIZE ||
              !webgpu_svm_inputs_valid(written, node.value, node.min, node.max))))
        {
          return false;
        }
        svm_node_clamp(stack, node);
        webgpu_svm_stack_mark(written, node.result_offset);
        break;
      }
      case NODE_MIX_FLOAT: {
        SVMNodeMixFloat node;
        if (!read(node) ||
            (Validate && (node.result_offset >= SVM_STACK_SIZE ||
                          !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac))))
        {
          return false;
        }
        svm_node_mix_float(stack, node);
        webgpu_svm_stack_mark(written, node.result_offset);
        break;
      }
      case NODE_MIX: {
        SVMNodeMix node;
        if (!read(node) ||
            (Validate &&
             (node.mix_type > NODE_MIX_CLAMP || uint(node.result_offset) + 2 >= SVM_STACK_SIZE ||
              !webgpu_svm_inputs_valid(written, node.c1, node.c2, node.fac))))
        {
          return false;
        }
        svm_node_mix(stack, node);
        webgpu_svm_stack_mark3(written, node.result_offset);
        break;
      }
      case NODE_MIX_COLOR: {
        SVMNodeMixColor node;
        if (!read(node) ||
            (Validate && (node.blend_type > NODE_MIX_EXCLUSION ||
                          uint(node.result_offset) + 2 >= SVM_STACK_SIZE ||
                          !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac))))
        {
          return false;
        }
        svm_node_mix_color(stack, node);
        webgpu_svm_stack_mark3(written, node.result_offset);
        break;
      }
      case NODE_SEPARATE_COLOR: {
        SVMNodeSeparateColor node;
        if (!read(node) ||
            (Validate && (node.color_type > NODE_COMBSEP_COLOR_HSL ||
                          !webgpu_svm_input_valid(written, node.color) ||
                          !webgpu_svm_outputs_valid(node.red_offset, SVM_STACK_INVALID) ||
                          !webgpu_svm_outputs_valid(node.green_offset, SVM_STACK_INVALID) ||
                          !webgpu_svm_outputs_valid(node.blue_offset, SVM_STACK_INVALID))))
        {
          return false;
        }
        svm_node_separate_color(stack, node);
        for (const uint channel : {node.red_offset, node.green_offset, node.blue_offset}) {
          if (stack_valid(channel)) {
            webgpu_svm_stack_mark(written, channel);
          }
        }
        break;
      }
      case NODE_COMBINE_COLOR: {
        SVMNodeCombineColor node;
        if (!read(node) ||
            (Validate && (node.color_type > NODE_COMBSEP_COLOR_HSL ||
                          uint(node.color_offset) + 2 >= SVM_STACK_SIZE ||
                          !webgpu_svm_inputs_valid(written, node.red, node.green, node.blue))))
        {
          return false;
        }
        svm_node_combine_color(stack, node);
        webgpu_svm_stack_mark3(written, node.color_offset);
        break;
      }
      case NODE_CONVERT: {
        SVMNodeConvert node;
        if (!read(node)) {
          return false;
        }
        if (node.convert_type == NODE_CONVERT_FV) {
          if (Validate && (uint(node.to_offset) + 2 >= SVM_STACK_SIZE ||
                           !webgpu_svm_stack_has(written, node.from_offset)))
          {
            return false;
          }
          const float value = stack_load_float(stack, node.from_offset);
          stack_store_float3(stack, node.to_offset, make_float3(value));
          webgpu_svm_stack_mark3(written, node.to_offset);
        }
        else {
          if (Validate &&
              ((node.convert_type != NODE_CONVERT_CF && node.convert_type != NODE_CONVERT_VF) ||
               node.to_offset >= SVM_STACK_SIZE ||
               !webgpu_svm_stack_has3(written, node.from_offset)))
          {
            return false;
          }
          const float3 value = stack_load_float3(stack, node.from_offset);
          const float scalar = node.convert_type == NODE_CONVERT_CF ? dot(value, rgb_to_y) :
                                                                      average(value);
          stack_store_float(stack, node.to_offset, scalar);
          webgpu_svm_stack_mark(written, node.to_offset);
        }
        break;
      }
      case NODE_EMISSION_WEIGHT: {
        SVMNodeEmissionWeight node;
        if (!read(node) || (Validate && (has_weight || !webgpu_svm_inputs_valid(
                                                           written, node.color, node.strength))))
        {
          return false;
        }
        weight = rgb_to_spectrum(stack_load(stack, node.color)) * stack_load(stack, node.strength);
        has_weight = true;
        break;
      }
      case NODE_CLOSURE_BACKGROUND: {
        SVMNodeClosureBackground node;
        if (!read(node) ||
            (Validate && (!has_weight || has_background || stack_valid(node.mix_weight_offset))))
        {
          return false;
        }
        radiance = weight;
        has_background = true;
        break;
      }
      default:
        return false;
    }
  }
  return false;
}

CCL_NAMESPACE_END

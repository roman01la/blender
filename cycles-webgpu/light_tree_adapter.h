/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

namespace ccl {

ccl_device_inline float light_tree_word_float(const uint word)
{
  return std::bit_cast<float>(cycles_light_word(word));
}

ccl_device_inline packed_float3 light_tree_word_vector(const uint word)
{
  return make_float3(light_tree_word_float(word),
                     light_tree_word_float(word + 1),
                     light_tree_word_float(word + 2));
}

ccl_device_inline uint light_tree_lamp_word(KernelGlobals kg, const uint lamp)
{
  return kg->tree_lamp_offset + lamp * CYCLES_WEBGPU_TREE_LAMP_WORDS;
}

ccl_device_inline KernelLight light_fetch(KernelGlobals kg, lights_tag, uint index)
{
  const uint word = light_tree_lamp_word(kg, index);
  KernelLight lamp{};
  lamp.type = int(cycles_light_word(word + CYCLES_WEBGPU_TREE_LAMP_TYPE));
  lamp.co = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_LAMP_POSITION);
  lamp.shader_id = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_LAMP_SHADER));
  lamp.max_bounces = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_MAX_BOUNCES);
  lamp.strength[0] = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_STRENGTH);
  lamp.strength[1] = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_STRENGTH + 1);
  lamp.strength[2] = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_STRENGTH + 2);
  if (lamp.type == LIGHT_AREA) {
    lamp.area.axis_u = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_LAMP_AXIS_U);
    lamp.area.len_u = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_AXIS_U + 3);
    lamp.area.axis_v = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_LAMP_AXIS_V);
    lamp.area.len_v = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_AXIS_V + 3);
    lamp.area.dir = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_LAMP_DIRECTION);
    lamp.area.invarea = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_DIRECTION + 3);
    lamp.area.tan_half_spread = light_tree_word_float(word + CYCLES_WEBGPU_TREE_LAMP_SPREAD);
    lamp.area.normalize_spread = light_tree_word_float(word +
                                                       CYCLES_WEBGPU_TREE_LAMP_NORMALIZE_SPREAD);
  }
  else if (lamp.type == LIGHT_POINT) {
    const uint point_word = word + CYCLES_WEBGPU_TREE_LAMP_AXIS_U;
    lamp.spot.radius = light_tree_word_float(point_word);
    lamp.spot.eval_fac = light_tree_word_float(point_word + 1);
    /* Older scene records leave the point-specific fields zero. */
    if (lamp.spot.radius == 0.0f && lamp.spot.eval_fac == 0.0f) {
      lamp.spot.eval_fac = 0.25f * M_1_PI_F;
    }
    lamp.spot.is_sphere = cycles_light_word(point_word + 2) != 0;
  }
  return lamp;
}

ccl_device_inline KernelLightTreeNode light_fetch(KernelGlobals kg, light_tree_nodes_tag, uint index)
{
  const uint word = kg->tree_node_offset + index * CYCLES_WEBGPU_TREE_NODE_WORDS;
  KernelLightTreeNode node{};
  node.bbox.min = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_NODE_MIN);
  node.bbox.max = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_NODE_MAX);
  node.bcone.axis = light_tree_word_vector(word + CYCLES_WEBGPU_TREE_NODE_AXIS);
  node.bcone.theta_o = light_tree_word_float(word + CYCLES_WEBGPU_TREE_NODE_THETA_O);
  node.bcone.theta_e = light_tree_word_float(word + CYCLES_WEBGPU_TREE_NODE_THETA_E);
  node.energy = light_tree_word_float(word + CYCLES_WEBGPU_TREE_NODE_ENERGY);
  node.type = LightTreeNodeType(cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_TYPE));
  node.num_emitters = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_EMITTER_COUNT));
  node.inner.left_child = std::bit_cast<int>(
      cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_LEFT_OR_FIRST));
  node.inner.right_child = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_RIGHT));
  node.bit_trail = cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_BIT_TRAIL);
  node.bit_skip = uint8_t(cycles_light_word(word + CYCLES_WEBGPU_TREE_NODE_BIT_SKIP));
  return node;
}

ccl_device_inline KernelLightTreeEmitter light_fetch(KernelGlobals kg, light_tree_emitters_tag, uint index)
{
  const uint word = kg->tree_emitter_offset + index * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  KernelLightTreeEmitter emitter{};
  emitter.theta_o = light_tree_word_float(word + CYCLES_WEBGPU_TREE_EMITTER_THETA_O);
  emitter.theta_e = light_tree_word_float(word + CYCLES_WEBGPU_TREE_EMITTER_THETA_E);
  emitter.energy = light_tree_word_float(word + CYCLES_WEBGPU_TREE_EMITTER_ENERGY);
  emitter.light.id = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_LIGHT));
  emitter.object_id = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_OBJECT));
  emitter.visibility_flag = std::bit_cast<int>(
      cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_VISIBILITY));
  emitter.bit_trail = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_BIT_TRAIL));
  return emitter;
}

ccl_device_inline void setup_light_tree(LightGlobals &globals, const uint descriptor)
{
  globals.data.integrator.use_direct_light = true;
  globals.tree_lamp_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_LAMPS);
  globals.tree_node_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_NODES);
  globals.tree_emitter_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_EMITTERS);
}

}  // namespace ccl

#include "kernel/light/tree.h"

namespace ccl {

ccl_device_inline bool webgpu_light_tree_select(KernelGlobals kg,
                                     const float rand,
                                     const float3 P,
                                     const float3 N,
                                     const int runtime_flags,
                                     LightSample *light)
{
  if (!light_tree_sample<false>(kg, rand, P, N, 0.0f, OBJECT_NONE, runtime_flags, light)) {
    return false;
  }
  light->prim = ~light_fetch(kg, light_tree_emitters_tag{}, light->emitter_id).light.id;
  return true;
}

ccl_device_inline float webgpu_light_tree_pdf(KernelGlobals kg,
                                    const float3 P,
                                    const float3 N,
                                    const uint path_flag,
                                    const uint emitter)
{
  return light_tree_pdf<false>(kg, P, N, 0.0f, path_flag, OBJECT_NONE, emitter, OBJECT_NONE);
}

}  // namespace ccl

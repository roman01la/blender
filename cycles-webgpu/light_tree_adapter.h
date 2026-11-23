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

ccl_device_inline float4 light_tree_word_float4(const uint word)
{
  return make_float4(light_tree_word_float(word),
                     light_tree_word_float(word + 1),
                     light_tree_word_float(word + 2),
                     light_tree_word_float(word + 3));
}

ccl_device_inline uint light_tree_object_word(KernelGlobals kg, const uint object)
{
  return kg->tree_object_offset + object * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
}

ccl_device_inline uint light_tree_triangle_index(KernelGlobals kg, const uint prim)
{
  return cycles_light_word(kg->tree_triangle_indices_offset + prim);
}

ccl_device_inline uint light_tree_shader_word(KernelGlobals kg, const uint shader)
{
  return kg->tree_shader_offset + shader * CYCLES_WEBGPU_TREE_SHADER_WORDS;
}

ccl_device_inline const KernelObject &light_fetch(KernelGlobals kg, objects_tag, const uint index)
{
  const uint word = light_tree_object_word(kg, index);
  const uint transform = word + CYCLES_WEBGPU_TREE_OBJECT_TRANSFORM;
  const uint inverse = word + CYCLES_WEBGPU_TREE_OBJECT_INVERSE_TRANSFORM;
  KernelObject &object = kg->object;
  object.tfm = {light_tree_word_float4(transform),
                light_tree_word_float4(transform + 4),
                light_tree_word_float4(transform + 8)};
  object.itfm = {light_tree_word_float4(inverse),
                 light_tree_word_float4(inverse + 4),
                 light_tree_word_float4(inverse + 8)};
  object.lightgroup = std::bit_cast<int>(
      cycles_light_word(word + CYCLES_WEBGPU_TREE_OBJECT_LIGHTGROUP));
  return object;
}

ccl_device_inline uint light_fetch(KernelGlobals kg, object_flag_tag, const uint object)
{
  return cycles_light_word(light_tree_object_word(kg, object) + CYCLES_WEBGPU_TREE_OBJECT_FLAGS);
}

ccl_device_inline uint light_fetch(KernelGlobals kg, object_prim_offset_tag, const uint object)
{
  return cycles_light_word(light_tree_object_word(kg, object) + CYCLES_WEBGPU_TREE_OBJECT_PRIM_OFFSET);
}

ccl_device_inline uint light_fetch(KernelGlobals kg, object_lookup_offset_tag, const uint object)
{
  return cycles_light_word(light_tree_object_word(kg, object) +
                           CYCLES_WEBGPU_TREE_OBJECT_LOOKUP_OFFSET);
}

ccl_device_inline uint light_fetch(KernelGlobals kg, light_to_tree_tag, const uint object)
{
  return cycles_light_word(light_tree_object_word(kg, object) + CYCLES_WEBGPU_TREE_OBJECT_TO_TREE);
}

ccl_device_inline uint light_fetch(KernelGlobals kg, triangle_to_tree_tag, const uint index)
{
  return cycles_light_word(kg->tree_triangle_to_tree_offset + index);
}

ccl_device_inline packed_uint3 light_fetch(KernelGlobals kg, tri_vindex_tag, const uint prim)
{
  const uint vertex = 3 * light_tree_triangle_index(kg, prim);
  return make_uint3(vertex, vertex + 1, vertex + 2);
}

ccl_device_inline packed_float3 light_fetch(KernelGlobals kg, tri_verts_tag, const uint index)
{
  return light_tree_word_vector(kg->tree_triangle_offset +
                                (index / 3) * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS +
                                3 * (index % 3));
}

ccl_device_inline uint light_fetch(KernelGlobals kg, tri_shader_tag, const uint prim)
{
  const uint index = light_tree_triangle_index(kg, prim);
  return cycles_light_word(kg->tree_triangle_offset + index * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS +
                           CYCLES_WEBGPU_TREE_TRIANGLE_SHADER);
}

ccl_device_inline KernelShader light_fetch(KernelGlobals kg, shaders_tag, const uint index)
{
  KernelShader shader{};
  shader.flags = std::bit_cast<int>(
      cycles_light_word(light_tree_shader_word(kg, index) + CYCLES_WEBGPU_TREE_SHADER_FLAGS));
  return shader;
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
  else if (lamp.type == LIGHT_SUN) {
    lamp.sun.angle = light_tree_word_float(word + CYCLES_WEBGPU_LAMP_SUN_ANGLE);
    lamp.sun.one_minus_cosangle = light_tree_word_float(word + CYCLES_WEBGPU_LAMP_SUN_ONE_MINUS_COS);
    lamp.sun.pdf = light_tree_word_float(word + CYCLES_WEBGPU_LAMP_SUN_PDF);
    lamp.sun.eval_fac = light_tree_word_float(word + CYCLES_WEBGPU_LAMP_SUN_EVAL);
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
  emitter.object_id = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_OBJECT));
  const int id = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_LIGHT));
  const uint detail = cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_DETAIL);
  if (id < 0) {
    emitter.light.id = id;
  }
  else if (emitter.object_id == OBJECT_NONE) {
    emitter.mesh = {id, std::bit_cast<int>(detail)};
  }
  else {
    emitter.triangle = {id, EmissionSampling(detail)};
  }
  emitter.visibility_flag = std::bit_cast<int>(
      cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_VISIBILITY));
  emitter.bit_trail = std::bit_cast<int>(cycles_light_word(word + CYCLES_WEBGPU_TREE_EMITTER_BIT_TRAIL));
  return emitter;
}

ccl_device_inline void setup_light_tree(LightGlobals &globals, const uint descriptor)
{
  globals.data.kernel_features = 0;
  globals.data.integrator.use_direct_light = true;
  auto &integrator = globals.data.integrator;
  integrator.use_light_tree = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_NODE_COUNT) != 0;
  if (!integrator.use_light_tree) {
    globals.distribution_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_DISTRIBUTION_OFFSET);
    integrator.num_distribution = int(cycles_light_word(descriptor + CYCLES_WEBGPU_DISTRIBUTION_COUNT));
    integrator.distribution_pdf_lights = light_tree_word_float(
        descriptor + CYCLES_WEBGPU_DISTRIBUTION_LAMP_PDF);
    integrator.distribution_pdf_triangles = light_tree_word_float(
        descriptor + CYCLES_WEBGPU_DISTRIBUTION_TRIANGLE_PDF);
  }
  globals.tree_lamp_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_LAMPS);
  globals.tree_node_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_NODES);
  globals.tree_emitter_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_EMITTERS);
  globals.tree_object_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_OBJECTS);
  globals.tree_triangle_indices_offset = cycles_light_word(descriptor +
                                                            CYCLES_WEBGPU_TREE_TRIANGLE_INDICES);
  globals.tree_triangle_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_TRIANGLES);
  globals.tree_shader_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_SHADERS);
  globals.tree_triangle_to_tree_offset = cycles_light_word(descriptor +
                                                            CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE);
}

}  // namespace ccl

#include "kernel/light/tree.h"

namespace ccl {

ccl_device_inline float webgpu_light_tree_pdf(KernelGlobals kg,
                                    const float3 P,
                                    const float3 N,
                                    const uint path_flag,
                                    const uint emitter)
{
  return light_tree_pdf<false>(kg, P, N, 0.0f, path_flag, OBJECT_NONE, emitter, OBJECT_NONE);
}

}  // namespace ccl

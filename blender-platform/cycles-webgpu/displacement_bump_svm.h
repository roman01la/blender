/* SPDX-License-Identifier: Apache-2.0 */

/* Stock Both displacement evaluates residual bump in the undisplaced frame.
 * Keep the full differential for Set Bump and the compact one for derivative
 * nodes. Leaving the frame restores P/dP while retaining the new normal. */
struct WebGPUSvmDisplacementBump {
  uint offset = SVM_STACK_INVALID;
  float3 position;
  float3 normal;
  differential3 full;
  differential3 compact;
};

template<typename Host, typename FetchSurface>
ccl_device_inline bool webgpu_svm_enter_displacement_bump(const Host &host,
                                                          const FetchSurface surface,
                                                          const WebGPUSvmBumpInputs &bump,
                                                          const uint packed,
                                                          const bool smooth,
                                                          WebGPUSvmDisplacementBump &state)
{
  const uint offset = packed & 0xffu;
  const uint position_slot = (packed >> 8u) & 0xffu;
  const uint normal_slot = (packed >> 16u) & 0xffu;
  if (state.offset != SVM_STACK_INVALID || offset + SVM_BUMP_EVAL_STATE_SIZE > SVM_STACK_SIZE ||
      position_slot == 0u)
  {
    return false;
  }
  WebGPUSvmAttributeSample position;
  WebGPUSvmTriangleCorners points;
  if (!host.attribute_sample(position_slot - 1u, position) ||
      position.mode != WEBGPU_SVM_ATTRIBUTE_INTERPOLATE ||
      !host.triangle_corners(position_slot - 1u, points))
  {
    return false;
  }

  /* svm_node_enter_bump_eval(): the original positions are vertex attributes.
   * Use their barycentric derivatives, then transform the full dual to world. */
  const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
  const float3 dpdu = points.corners[1] - points.corners[0];
  const float3 dpdv = points.corners[2] - points.corners[0];
  state.position = transform_point(&tfm, make_float3(position.value));
  state.full.dx = transform_direction(&tfm, dpdu * bump.u.dx + dpdv * bump.v.dx);
  state.full.dy = transform_direction(&tfm, dpdu * bump.u.dy + dpdv * bump.v.dy);
  const float3 Ng = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
  state.compact = differential_from_compact(Ng, differential_make_compact(state.full));

  /* primitive_normal_set_undisplaced(): true displacement prevents static
   * transform application, so the flat case uses the original face winding. */
  state.normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
  if (!smooth || normal_slot != 0u) {
    WebGPUSvmAttributeSample original;
    if (smooth && !host.attribute_sample(normal_slot - 1u, original)) {
      return false;
    }
    if (!smooth || original.mode != WEBGPU_SVM_ATTRIBUTE_MISSING) {
      const float3 normal = safe_normalize(smooth ? make_float3(original.value) : cross(dpdu, dpdv));
      const Transform inverse = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
      state.normal = safe_normalize(transform_direction_transposed(&inverse, normal));
      if (surface(CYCLES_WEBGPU_SVM_INTERSECTION).w == 2u) {
        state.normal = -state.normal;
      }
    }
  }
  state.offset = offset;
  return true;
}

/* SPDX-License-Identifier: Apache-2.0 */

/* The ordered tail keeps the base reflectance of each opaque dielectric branch.
 * Anisotropic tangents and Coat use the main material's specular, layers and
 * transmission records. The consumers expand the layers at the incident angle. */
template<typename Fetch, typename Host>
ccl_device_inline bool webgpu_svm_capture_principled_branch(
    const Fetch fetch,
    uint &offset,
    const uint end,
    const SVMNodeClosureBsdf &closure,
    const WebGPUSvmStackUsage &written,
    const WebGPUSvmNormalSlots &normals,
    const float *stack,
    const float3 shading_N,
    const WebGPUSvmContextWords path,
    const Host &host,
    const uint branch_index,
    WebGPUSvmMaterial &material,
    uint &lobe_candidates)
{
  SVMNodePrincipledBsdfData data;
  if (!webgpu_svm_read(fetch, offset, end, data)) {
    return false;
  }
  const uint emission_color = data.emission_color.x.bits | data.emission_color.y.bits |
                              data.emission_color.z.bits;
  const bool no_emission = (data.emission_strength.bits & 0x7fffffffu) == 0 ||
                           (emission_color & 0x7fffffffu) == 0;
  if ((stack_valid(closure.mix_weight_offset) &&
       !webgpu_svm_stack_has(written, closure.mix_weight_offset)) ||
      !no_emission || data.alpha.bits != __float_as_uint(1.0f) ||
      ((data.transmission_weight.bits | data.subsurface_weight.bits) & 0x7fffffffu) != 0 ||
      !webgpu_svm_input_valid(written, data.sheen_weight))
  {
    return false;
  }
  const bool mapped = normals.find_bump(data.normal_offset) != 0u;
  if (mapped && !webgpu_svm_stack_has3(written, data.normal_offset)) {
    return false;
  }
  const float mix = stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
  WebGPUSvmCoat coat{};
  if (!webgpu_svm_read_white_coat(data, written, stack, normals.base, coat)) {
    return false;
  }
  const float sheen_weight = stack_load(stack, data.sheen_weight);
  if (!webgpu_svm_finite(mix) || mix < 0.0f || mix > 1.0f ||
      !webgpu_svm_finite(sheen_weight))
  {
    return false;
  }
  const bool sheen_candidate = webgpu_svm_input_on_stack(data.sheen_weight) ||
                               sheen_weight > CLOSURE_WEIGHT_CUTOFF;
  const uint white = __float_as_uint(1.0f);
  WebGPUSvmMaterial branch{};
  if (!webgpu_svm_principled<true>(data,
                                  written,
                                  stack,
                                  mapped ? data.normal_offset : normals.base,
                                  mix,
                                  shading_N,
                                  branch))
  {
    return false;
  }
  const bool reflective = (path.w & CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS) == 0u ||
                          (path.x & PATH_RAY_VISIBILITY_DIFFUSE) == 0u;
  const float eta = reflective && branch.ggx ? branch.eta : 1.0f;
  const float mixed_coat_weight = reflective && coat.weight > CLOSURE_WEIGHT_CUTOFF ?
                                      coat.weight * mix : 0.0f;
  const bool anisotropy_candidate = branch.ggx && stack_valid(data.tangent_offset) &&
      (webgpu_svm_input_on_stack(data.anisotropic) || branch.anisotropy_aspect > 0.0f);
  const bool metal_candidate = webgpu_svm_input_on_stack(data.metallic) ||
                               branch.metallic > CLOSURE_WEIGHT_CUTOFF;
  if (metal_candidate && anisotropy_candidate) {
    return false;
  }
  const float3 local_N = mapped ? stack_load_float3(stack, data.normal_offset) : zero_float3();
  float4 layers = make_float4(branch.surface_tangent, mixed_coat_weight);
  float base_mix = mix;
  if (sheen_candidate) {
    /* Sheen shares the tangent/Coat record. A small nonzero Coat weight still
     * affects the stock Sheen normal even when Coat has no allocated lobe. */
    if (coat.possible || coat.weight > 0.0f || anisotropy_candidate ||
        data.sheen_tint.x.bits != white || data.sheen_tint.y.bits != white ||
        data.sheen_tint.z.bits != white || !webgpu_svm_input_valid(written, data.sheen_roughness))
    {
      return false;
    }
    const float roughness = stack_load(stack, data.sheen_roughness);
    if (!webgpu_svm_finite(roughness)) {
      return false;
    }
    if (sheen_weight > CLOSURE_WEIGHT_CUTOFF) {
      const float3 N = safe_normalize(safe_normalize_fallback(local_N, shading_N));
      layers = host.setup_paired_sheen(N, sheen_weight * mix, saturatef(roughness));
      base_mix = closure_layering_weight(make_spectrum(layers.w), make_spectrum(mix)).x;
    }
  }
  const bool first = branch_index == 0u;
  if (first) {
    material.ordered_principled_tangent_0 = layers;
  }
  else {
    material.ordered_principled_tangent_1 = layers;
  }
  /* Select the record values to keep the four shader stack indices static. */
  const float4 previous_coat = material.ordered_principled_coat;
  material.ordered_principled_coat = make_float4(first ? coat.ior : previous_coat.x,
                                                first ? coat.alpha : previous_coat.y,
                                                first ? previous_coat.z : coat.ior,
                                                first ? previous_coat.w : coat.alpha);
  const uint model = branch.multiscatter ? uint(CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX) :
                                          uint(CYCLES_WEBGPU_MATERIAL_GGX);
  /* The ordered flags and models are separate from ordinary Glossy closure IDs. */
  const uint flags = (mapped ? CYCLES_WEBGPU_ORDERED_PRINCIPLED_MAPPED_NORMAL : 0u) |
      (!reflective && metal_candidate ? CYCLES_WEBGPU_ORDERED_PRINCIPLED_SKIP_METAL_REFLECTION : 0u);
  const float metal_or_aspect = metal_candidate ? branch.metallic : -branch.anisotropy_aspect;
  host.write_ordered_lobe(
      2u * branch_index, model, branch.color, branch.specular_tint, branch.alpha);
  host.write_ordered_lobe(2u * branch_index + 1u,
                          flags,
                          make_float3(metal_or_aspect, eta, branch.f0),
                          local_N,
                          base_mix);
  material.ordered_principled_normal |= mapped;
  material.ordered_principled_layers |= coat.possible || anisotropy_candidate || sheen_candidate ||
                                       metal_candidate;
  material.ordered_principled_sheen |= sheen_candidate;
  const bool specular_candidate = branch.ggx &&
      (branch.metallic < 1.0f || webgpu_svm_input_on_stack(data.metallic)) &&
      (branch.eta != 1.0f || webgpu_svm_input_on_stack(data.specular_ior_level));
  material.ordered_principled_specular_candidates += uint(specular_candidate);
  material.ordered_principled_metal_candidates += uint(metal_candidate);
  material.ggx |= branch.ggx || coat.possible;
  lobe_candidates += 1u + uint(specular_candidate) + uint(metal_candidate) + uint(coat.possible) +
                     uint(sheen_candidate);
  return true;
}

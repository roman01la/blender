/* SPDX-License-Identifier: Apache-2.0 */

#if defined(CYCLES_WEBGPU_ORDERED_CLOSURES) && !defined(CYCLES_WEBGPU_EXCLUDE_STANDALONE)
namespace ccl {
namespace {

struct OrderedPrincipledCoat {
  float3 N = zero_float3();
  float weight = 0.0f;
  float sample_weight = 0.0f;
  float alpha = 0.0f;
  float ior = 1.0f;
  float energy_scale = 1.0f;
};

struct OrderedPrincipledBranch {
  ReflectiveClosures closures;
  GgxFloat4 bump;
  float3 specular_N;
  OrderedPrincipledCoat coat;
  GgxFloat4 sheen;
};

ccl_device OrderedPrincipledBranch read_ordered_principled_branch(
    const GgxFloat4 material,
    const unsigned int index,
    const GgxFloat4 tangent_layer,
    const GgxFloat4 layers,
    const GgxFloat4 geometric,
    const GgxFloat4 specular_normal,
    const GgxFloat4 wi,
    const float blur_alpha)
{
  const unsigned int address = cycles_ordered_tail_index(material.x) + 4u * index;
  const GgxFloat4 base = cycles_material_lobe(address);
  const GgxFloat4 tint = cycles_material_lobe(address + 1u);
  GgxFloat4 weights = cycles_material_lobe(address + 2u);
  GgxFloat4 bump = cycles_material_lobe(address + 3u);
  const uint flags = uint(weights.w);
  const bool mapped = (flags & 1u) != 0u;
  const bool skip_metal = (flags & 2u) != 0u;
  const float aspect = min(weights.x, 0.0f);
  const float metallic = max(weights.x, 0.0f);
  weights.x = skip_metal ? 0.0f : metallic;
  weights.w = bump.w;
  bump.w = mapped ? CYCLES_WEBGPU_NORMAL_CLOSURE + material.z : 0.0f;
  const float3 specular_N = rgb(cycles_bump_specular_normal(bump, geometric, wi, specular_normal));

  /* The producer already applied the caustics filter. Follow the white-tint
   * branch of stock bsdf_coat_setup(), with no independent Coat Normal. */
  OrderedPrincipledCoat coat;
  const float signed_ior = index == 0u ? layers.x : layers.z;
  /* A zero IOR with nonzero extra weight records a precomputed Sheen LTC.
   * Its attenuation is already included in the base branch mix. */
  const GgxFloat4 sheen = signed_ior == 0.0f ? tangent_layer : GgxFloat4{};
  if (signed_ior != 0.0f && tangent_layer.w > 0.0f) {
    const float alpha = index == 0u ? layers.y : layers.w;
    /* Default-input linking can supply the geometry normal explicitly. The
     * bytecode's missing-input default instead uses the branch normal. */
    const float3 coat_N = signed_ior < 0.0f ? rgb(specular_normal) : specular_N;
    MicrofacetBsdf bsdf{};
    float mix = weights.w;
    if (setup_white_coat(bsdf, tangent_layer.w, fabsf(signed_ior), alpha, coat_N, rgb(wi),
                         blur_alpha, mix))
    {
      coat = {bsdf.N, bsdf.weight.x, bsdf.sample_weight, bsdf.alpha_x, bsdf.ior, bsdf.energy_scale};
    }
    weights.w = mix;
  }
  if (skip_metal && metallic > CLOSURE_WEIGHT_CUTOFF) {
    /* Stock avoids allocating the metallic lobe on diffuse caustic paths
     * and still attenuates the lower layers. */
    weights.w *= 1.0f - metallic;
  }

  ReflectiveClosures closures;
  setup_closures(closures, base, tint, weights, GgxFloat4{0.0f, 0.0f, aspect, 0.0f}, tangent_layer,
                 specular_N, rgb(wi));
  blur_reflective_ggx(closures, blur_alpha);
  return {closures, bump, specular_N, coat, sheen};
}

ccl_device float3 ordered_principled_sheen_normal(const OrderedPrincipledBranch &branch,
                                                 const GgxFloat4 normal)
{
  return safe_normalize_fallback(rgb(branch.bump.w != 0.0f ? branch.bump : normal),
                                 rgb(normal));
}

ccl_device SheenBsdf ordered_principled_sheen(const OrderedPrincipledBranch &branch,
                                             const GgxFloat4 normal,
                                             const float3 wi)
{
  SheenBsdf sheen{.weight = make_spectrum(branch.sheen.w),
                 .type = CLOSURE_BSDF_SHEEN_ID,
                 .sample_weight = branch.sheen.w,
                 .N = ordered_principled_sheen_normal(branch, normal),
                 .roughness = branch.sheen.z,
                 .transformA = branch.sheen.x,
                 .transformB = branch.sheen.y};
  make_orthonormals_safe_tangent(sheen.N, wi, &sheen.T, &sheen.B);
  return sheen;
}

ccl_device MicrofacetBsdf ordered_principled_coat(const OrderedPrincipledBranch &branch)
{
  const auto &coat = branch.coat;
  return {.weight = make_spectrum(coat.weight),
          .type = CLOSURE_BSDF_MICROFACET_GGX_ID,
          .sample_weight = coat.sample_weight,
          .N = coat.N,
          .alpha_x = coat.alpha,
          .alpha_y = coat.alpha,
          .ior = coat.ior,
          .energy_scale = coat.energy_scale,
          .fresnel_type = MicrofacetFresnel::DIELECTRIC};
}

template<typename Evaluate>
[[gnu::always_inline]] auto with_ordered_principled_metal(
    const OrderedPrincipledBranch &branch, const Evaluate evaluate)
{
  const ReflectiveClosures &closures = branch.closures;
  FresnelF82Tint fresnel{.thin_film = {},
                        .f0 = closures.metal_f0,
                        .b = closures.metal_tint_b};
  const MicrofacetBsdf bsdf{.weight = closures.metal_weight,
                            .type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                            .sample_weight = closures.metal_sample_weight,
                            .N = branch.specular_N,
                            .alpha_x = closures.alpha_x,
                            .alpha_y = closures.alpha_y,
                            .ior = 1.0f,
                            .energy_scale = closures.metal_energy_scale,
                            .fresnel_type = MicrofacetFresnel::F82_TINT,
                            .fresnel = &fresnel,
                            .T = closures.tangent};
  return evaluate(bsdf);
}

template<typename Evaluate>
[[gnu::always_inline]] auto with_ordered_principled_specular(
    const OrderedPrincipledBranch &branch, const Evaluate evaluate)
{
  const ReflectiveClosures &closures = branch.closures;
  FresnelGeneralizedSchlick fresnel{.thin_film = {},
                                    .tint = {one_spectrum(), zero_spectrum()},
                                    .f0 = closures.specular_f0,
                                    .f90 = one_spectrum(),
                                    .exponent = -closures.eta};
  MicrofacetBsdf bsdf{.weight = closures.specular_weight,
                      .type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                      .sample_weight = closures.specular_sample_weight,
                      .N = branch.specular_N,
                      .alpha_x = closures.alpha_x,
                      .alpha_y = closures.alpha_y,
                      .ior = closures.eta,
                      .energy_scale = closures.specular_energy_scale,
                      .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                      .fresnel = &fresnel,
                      .T = closures.tangent};
  return evaluate(bsdf);
}

template<typename Evaluate>
ccl_device void for_ordered_principled_glossy(const OrderedPrincipledBranch &branch,
                                              const Evaluate evaluate)
{
  if (branch.coat.weight > 0.0f) {
    evaluate(ordered_principled_coat(branch));
  }
  if (branch.closures.has_metal) {
    with_ordered_principled_metal(branch, evaluate);
  }
  if (branch.closures.has_specular) {
    with_ordered_principled_specular(branch, evaluate);
  }
}

enum class OrderedPrincipledLobe { Layer, Metal, Specular, Diffuse, Count };

struct OrderedPrincipledEval {
  Spectrum value = zero_spectrum();
  Spectrum diffuse = zero_spectrum();
  float pdf = 0.0f;
  float roughness_pdf = 0.0f;
  float mass = 0.0f;
};

ccl_device void evaluate_ordered_principled_branch(
    OrderedPrincipledEval &result,
    const OrderedPrincipledBranch &branch,
    const GgxFloat4 normal,
    const float3 wi,
    const float3 wo,
    const OrderedPrincipledLobe skip = OrderedPrincipledLobe::Count)
{
  const ReflectiveClosures &closures = branch.closures;
  const bool valid_glossy = valid_specular_eval(rgb(normal), branch.specular_N, wo);
  if (branch.sheen.w > 0.0f && skip != OrderedPrincipledLobe::Layer) {
    const auto sheen = ordered_principled_sheen(branch, normal, wi);
    const float shadowing = cycles_bump_shadowing(
        normal, GgxFloat4{sheen.N.x, sheen.N.y, sheen.N.z, branch.bump.w}, vector(wo), true);
    if (shadowing > 0.0f) {
      float pdf = 0.0f;
      const auto *closure = reinterpret_cast<const ShaderClosure *>(&sheen);
      const Spectrum eval = bsdf_sheen_eval(closure, wi, wo, &pdf) * (shadowing * sheen.weight);
      result.value += eval;
      result.diffuse += eval;
      const float weighted_pdf = pdf * sheen.sample_weight;
      result.pdf += weighted_pdf;
      result.roughness_pdf += weighted_pdf;
    }
    result.mass += sheen.sample_weight;
  }
  if (branch.coat.weight > 0.0f && skip != OrderedPrincipledLobe::Layer) {
    if (valid_specular_eval(rgb(normal), branch.coat.N, wo)) {
      const auto coat = ordered_principled_coat(branch);
      float pdf = 0.0f;
      const auto *closure = reinterpret_cast<const ShaderClosure *>(&coat);
      result.value += coat.weight * bsdf_microfacet_ggx_eval(
                                        &energy_globals, closure, wi, wo, &pdf);
      const float weighted_pdf = pdf * coat.sample_weight;
      result.pdf += weighted_pdf;
      result.roughness_pdf += weighted_pdf * (coat.alpha_x * coat.alpha_y);
    }
    result.mass += branch.coat.sample_weight;
  }
  if (closures.has_metal && skip != OrderedPrincipledLobe::Metal) {
    float pdf = 0.0f;
    if (valid_glossy) {
      accumulate_lobe<Metal>(closures, branch.specular_N, wi, wo, result.value, pdf);
    }
    result.mass += closures.metal_sample_weight;
    result.pdf += pdf;
    result.roughness_pdf += pdf * (closures.alpha_x * closures.alpha_y);
  }
  if (closures.has_specular && skip != OrderedPrincipledLobe::Specular) {
    float pdf = 0.0f;
    if (valid_glossy) {
      accumulate_lobe<Specular>(closures, branch.specular_N, wi, wo, result.value, pdf);
    }
    result.mass += closures.specular_sample_weight;
    result.pdf += pdf;
    result.roughness_pdf += pdf * (closures.alpha_x * closures.alpha_y);
  }
  if (closures.has_diffuse && skip != OrderedPrincipledLobe::Diffuse) {
    float pdf = 0.0f;
    result.diffuse += accumulate_diffuse(closures, normal, branch.bump, wi, wo, result.value, pdf);
    result.mass += closures.diffuse_sample_weight;
    result.pdf += pdf;
    result.roughness_pdf += pdf;
  }
}

ccl_device GgxFloat4 eval_ordered_principled(
    const GgxFloat4 material,
    const GgxFloat4 first_tangent_layer,
    const GgxFloat4 second_tangent_layer,
    const GgxFloat4 layers,
    const GgxFloat4 normal,
    const GgxFloat4 geometric,
    const GgxFloat4 specular_normal,
    const GgxFloat4 wi,
    const GgxFloat4 wo,
    const float blur_alpha,
    GgxFloat4 &diffuse_ratio)
{
  OrderedPrincipledEval result;
  for (unsigned int index = 0; index < 2u; ++index) {
    const auto branch = read_ordered_principled_branch(
        material, index, index == 0u ? first_tangent_layer : second_tangent_layer, layers,
        geometric, specular_normal, wi, blur_alpha);
    evaluate_ordered_principled_branch(result, branch, normal, rgb(wi), rgb(wo));
  }
  diffuse_ratio = vector(safe_divide(result.diffuse, result.value));
  return {result.value.x, result.value.y, result.value.z, safe_divide(result.pdf, result.mass)};
}

ccl_device unsigned int sample_ordered_principled(
    const GgxFloat4 material,
    const GgxFloat4 first_tangent_layer,
    const GgxFloat4 second_tangent_layer,
    const GgxFloat4 layers,
    const GgxFloat4 normal,
    const GgxFloat4 geometric,
    const GgxFloat4 specular_normal,
    const GgxFloat4 wi,
    const GgxFloat4 random,
    const float blur_alpha,
    GgxFloat4 &weight,
    GgxFloat4 &out,
    GgxFloat4 &diffuse_ratio)
{
  const auto first = read_ordered_principled_branch(
      material, 0, first_tangent_layer, layers, geometric, specular_normal, wi, blur_alpha);
  const auto second = read_ordered_principled_branch(
      material, 1, second_tangent_layer, layers, geometric, specular_normal, wi, blur_alpha);
  constexpr uint lobes_per_branch = uint(OrderedPrincipledLobe::Count);
  const float masses[2u * lobes_per_branch] = {first.coat.sample_weight + first.sheen.w,
                                              first.closures.metal_sample_weight,
                                              first.closures.specular_sample_weight,
                                              first.closures.diffuse_sample_weight,
                                              second.coat.sample_weight + second.sheen.w,
                                              second.closures.metal_sample_weight,
                                              second.closures.specular_sample_weight,
                                              second.closures.diffuse_sample_weight};
  float total = 0.0f;
  for (const float mass : masses) {
    total += mass;
  }
  if (!(total > 0.0f)) {
    return LABEL_NONE;
  }
  const float selector = random.z * total;
  float prefix = 0.0f;
  uint selected = 2u * lobes_per_branch;
  float z = random.z;
  /* Match stock closure order within each branch and across the Mix. */
  for (uint index = 0; index < 2u * lobes_per_branch; ++index) {
    const float mass = masses[index];
    const float end = prefix + mass;
    if (mass > 0.0f && selected == 2u * lobes_per_branch) {
      selected = index;
    }
    if (selector < end) {
      selected = index;
      z = (selector - prefix) / mass;
      break;
    }
    prefix = end;
  }
  const bool select_first = selected < lobes_per_branch;
  const uint branch_base = select_first ? 0u : lobes_per_branch;
  const auto lobe = OrderedPrincipledLobe(selected - branch_base);
  const OrderedPrincipledBranch picked = select_first ? first : second;
  const bool sample_sheen = lobe == OrderedPrincipledLobe::Layer && picked.sheen.w > 0.0f;
  const bool sample_lambert = lobe == OrderedPrincipledLobe::Diffuse;
  const bool diffuse = sample_lambert || sample_sheen;
  float3 wo = zero_float3();
  Spectrum value = zero_spectrum();
  float pdf = 0.0f;
  float roughness_squared = 1.0f;
  int label;
  if (sample_sheen) {
    const auto bsdf = ordered_principled_sheen(picked, normal, rgb(wi));
    label = bsdf_sheen_sample(reinterpret_cast<const ShaderClosure *>(&bsdf),
                              rgb(geometric), rgb(wi), make_float2(rgb(random)),
                              &value, &wo, &pdf);
    value *= bsdf.weight * cycles_bump_shadowing(
        normal, GgxFloat4{bsdf.N.x, bsdf.N.y, bsdf.N.z, picked.bump.w}, vector(wo), false);
  }
  else if (sample_lambert) {
    const DiffuseBsdf bsdf{.N = rgb(picked.bump.w != 0.0f ? picked.bump : normal)};
    label = bsdf_diffuse_sample(reinterpret_cast<const ShaderClosure *>(&bsdf),
                                rgb(geometric), rgb(wi), make_float2(rgb(random)),
                                &value, &wo, &pdf);
    value *= picked.closures.diffuse_weight * cycles_bump_shadowing(normal, picked.bump, vector(wo),
                                                                  false);
  }
  else {
    const auto sample_glossy = [&](const MicrofacetBsdf &bsdf) __attribute__((always_inline)) {
      float eta;
      float2 alpha;
      const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
      const int sampled = bsdf_microfacet_ggx_sample(
          &energy_globals, closure, rgb(geometric), rgb(wi), make_float3(random.x, random.y, z),
          &value, &wo, &pdf, &alpha, &eta);
      value *= bsdf.weight;
      roughness_squared = bsdf.alpha_x * bsdf.alpha_y;
      return sampled;
    };
    if (lobe == OrderedPrincipledLobe::Layer) {
      label = sample_glossy(ordered_principled_coat(picked));
    }
    else if (lobe == OrderedPrincipledLobe::Metal) {
      label = with_ordered_principled_metal(picked, sample_glossy);
    }
    else {
      label = with_ordered_principled_specular(picked, sample_glossy);
    }
  }
  if (!(pdf > 0.0f)) {
    return LABEL_NONE;
  }
  const float selected_pdf = pdf * masses[selected];
  OrderedPrincipledEval result{
      value, diffuse ? value : zero_spectrum(), selected_pdf,
      selected_pdf * roughness_squared, masses[selected]};
  evaluate_ordered_principled_branch(result, first, normal, rgb(wi), wo,
                                     select_first ? lobe : OrderedPrincipledLobe::Count);
  evaluate_ordered_principled_branch(result, second, normal, rgb(wi), wo,
                                     select_first ? OrderedPrincipledLobe::Count : lobe);
  const float mixed_pdf = safe_divide(result.pdf, result.mass);
  if (!(mixed_pdf > 0.0f) || is_zero(result.value)) {
    return LABEL_NONE;
  }
  diffuse_ratio = vector(safe_divide(result.diffuse, result.value));
  const Spectrum throughput = result.value / mixed_pdf;
  weight = {throughput.x, throughput.y, throughput.z, mixed_pdf};
  out = {wo.x, wo.y, wo.z, safe_divide(result.roughness_pdf, result.pdf)};
  return label;
}

}  // namespace
}  // namespace ccl
#endif

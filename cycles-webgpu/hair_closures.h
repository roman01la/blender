/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Material records hold the raw SVM parameters and the triangle differential
 * aligned to the front-facing side. The stock setup, sample and eval functions
 * run again at each stage so they can retain their native closure layouts. */
template<typename Evaluate>
ccl_device auto with_hair_closure(const GgxFloat4 material,
                                 const GgxFloat4 tangent,
                                 const GgxFloat4 parameters,
                                 const GgxFloat4 lobe,
                                 const GgxFloat4 hair,
                                 const float3 N,
                                 const float3 Ng,
                                 const float3 wi,
                                 const float blur,
                                 const Evaluate evaluate)
{
  ShaderData sd{};
  sd.N = N;
  sd.Ng = Ng;
  sd.wi = wi;
  sd.dPdu = rgb(tangent);
  sd.type = PRIMITIVE_TRIANGLE;
  sd.lcg_state = uint(hair.z) | (uint(hair.w) << 16u);

  const Spectrum weight = rgb(material);
  if (material.w == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG) {
    HuangHairExtra extra{};
    extra.R = lobe.x;
    extra.TT = lobe.y;
    extra.TRT = lobe.z;
    extra.pixel_coverage = 1.0f;
    HuangHairBSDF bsdf{};
    bsdf.weight = weight;
    bsdf.sigma = rgb(parameters);
    bsdf.eta = lobe.w;
    bsdf.tilt = tangent.w;
    bsdf.roughness = parameters.w;
    bsdf.aspect_ratio = hair.x;
    bsdf.extra = &extra;
    sd.num_closure = 1;
    const int flags = bsdf_hair_huang_setup(&sd, &bsdf, 0);
    auto *sc = reinterpret_cast<ShaderClosure *>(&bsdf);
    if (flags) {
      bsdf_hair_huang_blur(sc, blur);
    }
    return evaluate(sd, bsdf, flags);
  }
  if (material.w == CYCLES_WEBGPU_MATERIAL_HAIR_CHIANG) {
    ChiangHairBSDF bsdf{};
    bsdf.weight = weight;
    bsdf.sigma = rgb(parameters);
    bsdf.eta = lobe.w;
    bsdf.alpha = tangent.w;
    bsdf.v = parameters.w;
    bsdf.s = lobe.x;
    bsdf.m0_roughness = lobe.y;
    const int flags = bsdf_hair_chiang_setup(&sd, &bsdf);
    bsdf_hair_chiang_blur(reinterpret_cast<ShaderClosure *>(&bsdf), blur);
    return evaluate(sd, bsdf, flags);
  }

  HairBsdf bsdf{};
  bsdf.weight = weight;
  bsdf.N = N;
  bsdf.T = rgb(tangent);
  bsdf.offset = tangent.w;
  bsdf.roughness1 = parameters.w;
  bsdf.roughness2 = lobe.x;
  const int flags = material.w == CYCLES_WEBGPU_MATERIAL_HAIR_REFLECTION ?
                        bsdf_hair_reflection_setup(&bsdf) : bsdf_hair_transmission_setup(&bsdf);
  return evaluate(sd, bsdf, flags);
}

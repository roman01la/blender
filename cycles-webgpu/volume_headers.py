# SPDX-License-Identifier: Apache-2.0

"""Expose stock homogeneous volume helpers independently of volume traversal."""

from pathlib import Path
import re


def prepare_volume_headers(work: Path, cycles: Path, scattering: bool = False) -> None:
    def section(path: str, start: str, end: str) -> str:
        source = (cycles / "kernel" / path).read_text()
        first = source.index(start)
        return source[first:source.index(end, first)]

    # Give the Vulkan backend typed storage for the phase parameters that the
    # generic ShaderClosure normally overlays on padding and the surface normal.
    types = section("types.h", "struct ccl_align(16) ShaderClosure {", "/* Shader Data")
    volume_storage = """struct ccl_align(16) ShaderClosure {
  SHADER_CLOSURE_VOLUME_BASE;
  float param[3];
};

"""
    original_types = (cycles / "kernel" / "types.h").read_text()
    overlay = work / "kernel" / "types.h"
    overlay.parent.mkdir(exist_ok=True)
    overlay.write_text(original_types.replace(types, volume_storage))

    closure = section("svm/closure.h", "ccl_device_inline void svm_alloc_closure_volume_scatter(",
                      "ccl_device void svm_node_closure_set_weight(")
    # Principled Volume is the last volume node. Exclude the surface-only handlers that follow.
    end = closure.index("\n}\n", closure.index("ccl_device_noinline void svm_node_principled_volume("))
    closure = closure[:end + len("\n}\n")].replace(
        "volume->alpha = stack_load(stack, param_extra);",
        "static_cast<volatile DraineVolume *>(volume)->alpha = stack_load(stack, param_extra);")
    # The packed volume stack contributes phase closures only. Copy their
    # fields at constant typed offsets for SPIR-V private addressing.
    phase = """ccl_device_inline void volume_shader_copy_phases(
    ccl_private ShaderVolumePhases *phases, const ccl_private ShaderData *sd)
{
  phases->num_closure = sd->num_closure;
#pragma clang loop unroll(full)
  for (int i = 0; i < MAX_CLOSURE; ++i) {
    ClosureType type = CLOSURE_NONE_ID;
    float r = 0.0f, g = 0.0f, b = 0.0f, sample_weight = 0.0f;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
    if (i < sd->num_closure) {
      const volatile auto &from = sd->closure[i];
      r = from.weight.x;
      g = from.weight.y;
      b = from.weight.z;
      type = from.type;
      sample_weight = from.sample_weight;
      p0 = type == CLOSURE_VOLUME_RAYLEIGH_ID ? 0.0f : from.param[0];
      p1 = (type == CLOSURE_VOLUME_FOURNIER_FORAND_ID ||
            type == CLOSURE_VOLUME_DRAINE_ID) ? from.param[1] : 0.0f;
      p2 = type == CLOSURE_VOLUME_FOURNIER_FORAND_ID ? from.param[2] : 0.0f;
    }
    volatile auto &to = phases->closure[i];
    to.weight.x = r;
    to.weight.y = g;
    to.weight.z = b;
    to.type = type;
    to.sample_weight = sample_weight;
    to.param[0] = p0;
    to.param[1] = p1;
    to.param[2] = p2;
  }
}

"""
    choose_phase = section("integrator/volume_shader.h",
                           "const ccl_device_inline ccl_private ShaderVolumeClosure *volume_shader_phase_pick(",
                           "ccl_device_inline float _volume_shader_phase_eval_mis(")
    # SPIR-V logical addressing cannot select between private pointers. Use the
    # sampled index, retaining Cycles' original reservoir selection and MIS.
    phase += choose_phase.replace(
        "const ccl_device_inline ccl_private ShaderVolumeClosure *", "ccl_device_inline int ").replace(
        "  return &phases->closure[sampled];", "  return sampled;")
    phase += section("integrator/volume_shader.h", "ccl_device_inline float _volume_shader_phase_eval_mis(",
                     "ccl_device float volume_shader_phase_eval(").replace(
        "const ccl_private ShaderVolumeClosure *skip_svc,", "const int skip_phase,").replace(
        "if (svc == skip_svc)", "if (i == skip_phase)")
    phase += section("integrator/volume_shader.h", "ccl_device int volume_shader_phase_sample(",
                     "#  if defined(__PATH_GUIDING__)").replace(
        "const ccl_private ShaderVolumeClosure *svc,", "const ccl_private ShaderVolumeClosure *svc,\n"
        "                                          const int sampled,").replace(
        "sd, phases, *wo, svc, phase_eval", "sd, phases, *wo, sampled, phase_eval")
    eval_helpers = section("film/light_passes.h", "ccl_device_inline void bsdf_eval_init(ccl_private BsdfEval *eval, Spectrum value)",
                     "ccl_device_inline void bsdf_eval_accum(")
    eval_helpers += section("film/light_passes.h", "ccl_device_inline void bsdf_eval_accum(ccl_private BsdfEval *eval, Spectrum value)",
                     "ccl_device_inline bool bsdf_eval_is_zero(")
    eval_helpers += section("film/light_passes.h", "ccl_device_inline bool bsdf_eval_is_zero(",
                            "ccl_device_inline Spectrum bsdf_eval_pass_diffuse_weight(")
    segment = section("integrator/shade_volume.h", "struct VolumeShaderCoefficients {", "struct EquiangularCoefficients {")
    segment += section("integrator/shade_volume.h", "ccl_device Spectrum volume_emission_integrate(", "/* Volume Integration */")
    blackbody = section("svm/math_util.h", "ccl_device float3 svm_math_blackbody_color_rec709(",
                        "ccl_device_inline float3 svm_math_gamma_color(")
    # The stock coefficients and polynomial are unchanged. Use dynamic native
    # vector indexing for the seven table rows instead of selecting pointers.
    blackbody = re.sub(r"  ccl_constant float \*[rgb] = blackbody_table_[rgb]\[i\];\n", "", blackbody)
    blackbody = re.sub(r"([rgb])\[(\d)\]",
                       r"volume_blackbody_coefficient<\2>(blackbody_table_\1, i)", blackbody)
    blackbody = """template<int column, size_t columns>
ccl_device_inline float volume_blackbody_coefficient(const float (&table)[7][columns], const int i)
{
  using Rows = float __attribute__((ext_vector_type(4)));
  const Rows first{table[0][column], table[1][column], table[2][column], table[3][column]};
  const Rows last{table[4][column], table[5][column], table[6][column], 0.0f};
  return i < 4 ? first[i & 3] : last[i & 3];
}

""" + blackbody
    (work / "volume_helpers.h").write_text(
        "CCL_NAMESPACE_BEGIN\n\n" + blackbody + closure + eval_helpers + phase + segment + "CCL_NAMESPACE_END\n")
    if not scattering:
        return

    # Cycles' CPU RGB accessor uses pointer arithmetic on float3 fields.
    # A native vector gives SPIR-V dynamic component reads without pointer casts.
    closure_path = Path("kernel/closure/volume.h")
    closure_volume = (cycles / closure_path).read_text()
    channel_get = "  return GET_SPECTRUM_CHANNEL(value, channel);"
    if closure_volume.count(channel_get) != 1:
        raise ValueError("Expected the stock volume color channel accessor")
    (work / closure_path).parent.mkdir(exist_ok=True)
    (work / closure_path).write_text(closure_volume.replace(
        channel_get,
        "  using RGB = float __attribute__((ext_vector_type(3)));\n"
        "  return RGB{value.x, value.y, value.z}[channel];"))

    # Keep the stock distance, equiangular, and MIS integrators. The adapter
    # supplies homogeneous SVM coefficients and uses the unguided sampling
    # probability when the stock volume contribution buffers are unavailable.
    integrator = "integrator/shade_volume.h"
    method = section("integrator/volume_stack.h", "enum VolumeSampleMethod {",
                     "ccl_device VolumeSampleMethod volume_stack_sample_method")
    result = section(integrator, "struct VolumeIntegrateResult {", "/* We use both volume")
    result = re.sub(r"  ShaderVolumePhases (direct|indirect)_phases;\n", "", result)
    equiangular = section(integrator, "struct EquiangularCoefficients {",
                          "/* Evaluate extinction coefficient")
    equiangular += section(integrator, "ccl_device float volume_equiangular_sample(",
                           "/* Compute ray segment directly visible")
    state = section(integrator, "struct VolumeIntegrateState {",
                    "/* Accumulate transmittance for equiangular")
    mis = section(integrator, "ccl_device_inline void volume_direct_scatter_mis(", r"/** \} */")
    init = section(integrator, "ccl_device_inline void volume_integrate_state_init(",
                   "/* Compute guided volume scatter probability")
    probability = section(integrator, "ccl_device_inline Spectrum\nvolume_scatter_probability_homogeneous",
                          "/* Homogeneous volume distance sampling")
    unguided = "  const Spectrum attenuation = 1.0f - volume_color_transmittance(coeff.sigma_t, ray_length);"
    probability = probability[:probability.index(unguided)] + unguided + "\n  return attenuation;\n}\n\n"
    probability = probability.replace("KernelGlobals kg,\n                                       ", "")
    probability = re.sub(r"                                       ccl_global float \*ccl_restrict render_buffer,\n", "",
                         probability)
    probability = probability.replace("const ccl_private VolumeShaderCoefficients &coeff,\n"
                                      "                                       ccl_private VolumeIntegrateState &vstate)",
                                      "const ccl_private VolumeShaderCoefficients &coeff)")
    homogeneous = section(integrator, "ccl_device_forceinline void volume_integrate_homogeneous(",
                          "/* heterogeneous volume distance sampling")
    before = """  sd->P = ray->P + ray->D * ray->tmin;
  VolumeShaderCoefficients coeff ccl_optional_struct_init;
  if (!volume_shader_sample(kg, state, sd, &coeff)) {
    return;
  }

"""
    if homogeneous.count(before) != 1:
        raise ValueError("Expected the stock homogeneous volume shader lookup")
    homogeneous = homogeneous.replace(before, "")
    homogeneous = homogeneous.replace("ccl_private ShaderData *ccl_restrict sd,",
                                      "ccl_private VolumeShaderCoefficients &coeff,\n"
                                      "                                                         const uint runtime_flag,")
    homogeneous = homogeneous.replace("                                                         ccl_global float *ccl_restrict\n"
                                      "                                                             render_buffer,\n", "")
    homogeneous = homogeneous.replace("    guiding_record_volume_emission(kg, state, emission);\n", "")
    homogeneous = homogeneous.replace("sd->runtime_flag", "runtime_flag")
    homogeneous = homogeneous.replace("kg, state, render_buffer, ray_length, coeff, vstate",
                                      "state, ray_length, coeff")
    homogeneous = re.sub(r" +volume_shader_copy_phases\(&result\.(?:direct|indirect)_phases, sd\);\n",
                         "", homogeneous)
    light = section("light/sample.h", "ccl_device_inline void shadow_ray_setup(",
                    "/* Create shadow ray towards light sample. */")
    light += section("light/sample.h", "ccl_device_inline void light_sample_to_volume_shadow_ray(",
                     "/* Multiple importance sampling weights. */")
    light += section("light/sample.h", "ccl_device_inline float light_sample_mis_weight_nee(",
                     "/* Next event estimation sampling.")
    light += section("light/sample.h", "/* Early path termination of shadow rays. */",
                     "/* This function should be used to compute a modified ray start position")
    clamp = section("film/light_passes.h", "ccl_device_forceinline void film_clamp_light(",
                    "/* --------------------------------------------------------------------\n * Pass accumulation utilities.")
    (work / "volume_scatter_helpers.h").write_text(
        "CCL_NAMESPACE_BEGIN\n\n" + method + result + equiangular + state + mis + init +
        probability + homogeneous + light + clamp + "CCL_NAMESPACE_END\n")


def prepare_absorption_headers(work: Path, cycles: Path) -> None:
    path = Path("kernel/geom/object.h")
    source = (cycles / path).read_text()
    start = source.index("ccl_device_inline Transform object_fetch_transform(")
    end = source.index("\n}\n", start)
    body = source[start:end]
    inverse = "return kernel_data_fetch(objects, object).itfm;"
    if body.count(inverse) != 1:
        raise ValueError("Expected the stock inverse transform fetch")
    # Read the packed inverse transform directly. Two traversal phases would
    # otherwise retain a mutable KernelObject private allocation.
    source = source[:start] + body.replace(inverse, "return absorption_inverse_transform(kg, object);") + source[end:]
    output = work / path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)

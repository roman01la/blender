/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_shader.hh"

#include "BKE_global.hh"
#include "GPU_state.hh"
#include "MEM_guardedalloc.h"
#include "gpu_context_private.hh"

#include "gpu_shader_dependency_private.hh"
#include "gpu_shader_interface.hh"
#include "webgpu_context.hh"
#include "webgpu_texture.hh"

#include <emscripten.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>

namespace blender::gpu {

using namespace shader;

EM_JS(char *,
      translate_to_wgsl,
      (const uint32_t *words,
       size_t byte_size,
       uint32_t depth_lo,
       uint32_t depth_hi,
       uint32_t packed_normals,
       int *status,
       uint32_t *binding_masks),
      {
        const translator = Module.shaderTranslator;
        const memory64 = typeof words != 'number';
        const nullPointer = memory64 ? 0n : 0;
        if (!translator)
          return nullPointer;

        const size = Number(byte_size);
        const input = translator.naga_alloc(size);
        new Uint8Array(translator.memory.buffer, input, size)
            .set(HEAPU8.subarray(Number(words), Number(words) + size));
        const result = translator.naga_translate(input, size, depth_lo, depth_hi, packed_normals);
        if (result == 0 && binding_masks) {
          HEAPU32.set(
              new Uint32Array(translator.memory.buffer, translator.naga_binding_masks_ptr(), 18),
              Number(binding_masks) / 4);
        }
        const length = translator.naga_result_size();
        const output = _malloc(memory64 ? BigInt(length + 1) : length + 1);
        if (!output)
          return nullPointer;
        const address = Number(output);
        HEAPU8.set(new Uint8Array(translator.memory.buffer, translator.naga_result_ptr(), length),
                   address);
        HEAPU8[address + length] = 0;
        HEAP32[Number(status) / 4] = result;
        return memory64 ? BigInt(output) : output;
      });

static std::string combined_glsl(const StringRefNull stage_define,
                                 Span<StringRefNull> sources,
                                 bool skip_preprocessor,
                                 bool uses_uint_textures)
{
  const char *uint_extensions = uses_uint_textures ?
                                    "#extension GL_EXT_samplerless_texture_functions : require\n" :
                                    "";
  const GeneratedSourceList extensions{{"gpu_shader_glsl_extension.glsl",
                                        {},
                                        "#version 450\n"
                                        "#extension GL_GOOGLE_cpp_style_line_directive : require\n"
                                        "#define GPU_ARB_clip_control\n"
                                        "#define GPU_ARB_derivative_control\n"
                                        "#define gl_VertexID gl_VertexIndex\n"
                                        "#define gl_InstanceID gl_InstanceIndex\n"
                                        "#define gpu_InstanceIndex gl_InstanceIndex\n"
                                        "#define gpu_BaseInstance (0)\n"
                                        "#define GPU_PROVOKING_VERTEX_LAST\n" +
                                            std::string(uint_extensions) +
                                            std::string(stage_define)}};
  std::string code;
  for (const StringRefNull part :
       gpu_shader_dependency_get_resolved_source("gpu_shader_compat_glsl.glsl", extensions))
  {
    code += part;
  }
  /* WGSL has no isnan/isinf built-ins. Inspect the bits so the test also works
   * for non-finite values that originated in GPU buffers. */
  code += R"(
bvec4 blender_isnan(vec4 value)
{
  return greaterThan(floatBitsToUint(value) & uvec4(0x7fffffffu), uvec4(0x7f800000u));
}
bool blender_isnan(float value) { return blender_isnan(vec4(value)).x; }
bvec2 blender_isnan(vec2 value) { return blender_isnan(vec4(value, 0, 0)).xy; }
bvec3 blender_isnan(vec3 value) { return blender_isnan(vec4(value, 0)).xyz; }

bvec4 blender_isinf(vec4 value)
{
  return equal(floatBitsToUint(value) & uvec4(0x7fffffffu), uvec4(0x7f800000u));
}
bool blender_isinf(float value) { return blender_isinf(vec4(value)).x; }
bvec2 blender_isinf(vec2 value) { return blender_isinf(vec4(value, 0, 0)).xy; }
bvec3 blender_isinf(vec3 value) { return blender_isinf(vec4(value, 0)).xyz; }

#define isnan blender_isnan
#define isinf blender_isinf
)";
  if (uses_uint_textures) {
    code += R"(
#define usampler2D utexture2D
#define usampler2DArray utexture2DArray
#define usampler3D utexture3D
)";
  }
  /* Blender places a string placeholder for the version at the head of the
   * source span. */
  for (const StringRefNull part : sources.drop_front(1)) {
    code += part;
  }
  if (!skip_preprocessor) {
    code = Shader::run_preprocessor(code, G.debug & G_DEBUG_GPU_SHADER_NO_DCE);
  }
  return code;
}

static wgpu::ShaderModule shader_module_from_spirv(const char *name,
                                                   glslang_stage_t stage,
                                                   const Span<uint32_t> spirv,
                                                   const uint64_t depth_bindings,
                                                   const uint32_t packed_normals,
                                                   std::string &error,
                                                   uint32_t *binding_masks = nullptr)
{
  int status = -1;
  const std::unique_ptr<char, decltype(&std::free)> wgsl(
      translate_to_wgsl(spirv.data(),
                        spirv.size() * sizeof(uint32_t),
                        uint32_t(depth_bindings),
                        uint32_t(depth_bindings >> 32),
                        packed_normals,
                        &status,
                        binding_masks),
      std::free);
  if (!wgsl || status != 0) {
    error = wgsl ? wgsl.get() : "could not access the browser shader translator";
    return nullptr;
  }

  std::string suffix = std::to_string(stage);
  if (packed_normals != 0) {
    suffix += "_packed_" + std::to_string(packed_normals);
  }
  Shader::dump_source_to_disk(name, std::string(name) + "_" + suffix, ".wgsl", wgsl.get());
  wgpu::ShaderSourceWGSL source;
  source.code = wgsl.get();
  wgpu::ShaderModuleDescriptor desc;
  desc.label = name;
  desc.nextInChain = &source;
  return webgpu_active_device().CreateShaderModule(&desc);
}

/* The overloads of GLSL built-ins are implicit roots of Blender's shader preprocessor.
 * Define their implementations after its dead-code elimination so their helpers survive. */
static constexpr const char *unsigned_sampler_definitions = R"(
uvec4 blender_webgpu_integer_2d(utexture2D image, vec2 uv, float lod)
{
  ivec2 size = textureSize(image, 0);
  ivec2 texel = ivec2(clamp(floor(uv * vec2(size)), vec2(0), vec2(size - 1)));
  return texelFetch(image, texel, 0);
}

uvec4 blender_webgpu_integer_3d(utexture3D image, vec3 uv, float lod)
{
  ivec3 size = textureSize(image, 0);
  ivec3 texel = ivec3(clamp(floor(uv * vec3(size)), vec3(0), vec3(size - 1)));
  return texelFetch(image, texel, 0);
}

uvec4 blender_webgpu_integer_2d_array(utexture2DArray image, vec3 uv, float lod)
{
  ivec3 size = textureSize(image, 0);
  vec3 coordinate = vec3(uv.xy * vec2(size.xy), uv.z + 0.5);
  ivec3 texel = ivec3(clamp(floor(coordinate), vec3(0), vec3(size - 1)));
  return texelFetch(image, texel, 0);
}

uvec4 texture(utexture2D image, vec2 uv)
{
  return blender_webgpu_integer_2d(image, uv, 0.0);
}

uvec4 texture(utexture3D image, vec3 uv)
{
  return blender_webgpu_integer_3d(image, uv, 0.0);
}

uvec4 texture(utexture2DArray image, vec3 uv)
{
  return blender_webgpu_integer_2d_array(image, uv, 0.0);
}

uvec4 textureLod(utexture2D image, vec2 uv, float lod)
{
  return blender_webgpu_integer_2d(image, uv, lod);
}

uvec4 textureLod(utexture3D image, vec3 uv, float lod)
{
  return blender_webgpu_integer_3d(image, uv, lod);
}

uvec4 textureLod(utexture2DArray image, vec3 uv, float lod)
{
  return blender_webgpu_integer_2d_array(image, uv, lod);
}
)";

static wgpu::ShaderModule compile_stage(const char *name,
                                        glslang_stage_t stage,
                                        std::string glsl,
                                        const uint64_t depth_bindings,
                                        std::string &error,
                                        uint32_t *binding_masks,
                                        std::vector<uint32_t> *constant_ids,
                                        std::vector<uint32_t> *retained_spirv = nullptr)
{
  static const bool initialized = glslang_initialize_process() != 0;
  if (!initialized) {
    error = "could not initialize glslang";
    return nullptr;
  }

  constexpr auto messages = glslang_messages_t(GLSLANG_MSG_SPV_RULES_BIT |
                                               GLSLANG_MSG_VULKAN_RULES_BIT);
  glslang_input_t input{};
  input.language = GLSLANG_SOURCE_GLSL;
  input.stage = stage;
  input.client = GLSLANG_CLIENT_VULKAN;
  input.client_version = GLSLANG_TARGET_VULKAN_1_1;
  input.target_language = GLSLANG_TARGET_SPV;
  input.target_language_version = GLSLANG_TARGET_SPV_1_3;
  const size_t sampler_extension = glsl.find("GL_EXT_samplerless_texture_functions");
  if (sampler_extension != std::string::npos) {
    glsl.insert(glsl.find('\n', sampler_extension) + 1, unsigned_sampler_definitions);
  }
  input.code = glsl.c_str();
  input.default_version = 450;
  input.messages = messages;
  input.resource = glslang_default_resource();

  const std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)> shader(
      glslang_shader_create(&input), glslang_shader_delete);
  if (!glslang_shader_preprocess(shader.get(), &input) ||
      !glslang_shader_parse(shader.get(), &input))
  {
    error = glslang_shader_get_info_log(shader.get());
    return nullptr;
  }

  const std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)> program(
      glslang_program_create(), glslang_program_delete);
  glslang_program_add_shader(program.get(), shader.get());
  if (!glslang_program_link(program.get(), messages)) {
    error = glslang_program_get_info_log(program.get());
    return nullptr;
  }

  glslang_spv_options_t options{};
  options.disable_optimizer = true;
  glslang_program_SPIRV_generate_with_options(program.get(), stage, &options);
  const size_t word_count = glslang_program_SPIRV_get_size(program.get());
  if (word_count == 0) {
    const char *messages = glslang_program_SPIRV_get_messages(program.get());
    error = messages ? messages : "glslang did not emit SPIR-V";
    return nullptr;
  }
  const Span<uint32_t> spirv(glslang_program_SPIRV_get_ptr(program.get()), word_count);
  if (constant_ids) {
    /* OpDecorate has operands (target, decoration, value) for SpecId. Capture IDs
     * even if their use was removed in this stage. */
    constexpr uint32_t op_decorate = 71, spec_id = 1;
    for (size_t i = 5; i < word_count; i += spirv[i] >> 16) {
      if (spirv[i] == (4u << 16 | op_decorate) && spirv[i + 2] == spec_id) {
        constant_ids->push_back(spirv[i + 3]);
      }
    }
  }
  if (retained_spirv) {
    retained_spirv->assign(spirv.begin(), spirv.end());
  }
  return shader_module_from_spirv(name, stage, spirv, depth_bindings, 0, error, binding_masks);
}

static bool is_uint_sampler(const ImageType type)
{
  return ELEM(type, ImageType::Uint2D, ImageType::Uint2DArray, ImageType::Uint3D);
}

static const char *texel_buffer_type(const ImageType type)
{
  switch (type) {
    case ImageType::FloatBuffer:
      return "samplerBuffer";
    case ImageType::IntBuffer:
      return "isamplerBuffer";
    case ImageType::UintBuffer:
      return "usamplerBuffer";
    default:
      return nullptr;
  }
}

static const char *sampled_dimension(ImageType type)
{
  switch (type) {
    case ImageType::Float1D:
      return "1D";
    case ImageType::Float1DArray:
      return "1DArray";
    case ImageType::Uint2D:
    case ImageType::Depth2D:
    case ImageType::Float2D:
      return "2D";
    case ImageType::Depth2DArray:
    case ImageType::Uint2DArray:
    case ImageType::Float2DArray:
      return "2DArray";
    case ImageType::Uint3D:
    case ImageType::Float3D:
      return "3D";
    case ImageType::FloatCube:
      return "Cube";
    case ImageType::FloatCubeArray:
      return "CubeArray";
    default:
      return nullptr;
  }
}

using Resource = ShaderCreateInfo::Resource;

/* Keep image slots separate from uniform buffers in group 3. WebGPU guarantees four groups. */
static constexpr int image_binding_start = 64;
static constexpr int subpass_binding_start = image_binding_start + GPU_MAX_IMAGE;

struct StorageTextureType {
  wgpu::TextureViewDimension dimension = wgpu::TextureViewDimension::Undefined;
  const char *glsl_type = nullptr;
};

static StorageTextureType storage_texture_type(const ImageType type)
{
  using D = wgpu::TextureViewDimension;
  switch (type) {
    case ImageType::Float2D:
      return {D::e2D, "image2D"};
    case ImageType::Uint2D:
      return {D::e2D, "uimage2D"};
    case ImageType::Int2D:
      return {D::e2D, "iimage2D"};
    case ImageType::Float2DArray:
      return {D::e2DArray, "image2DArray"};
    case ImageType::Uint2DArray:
      return {D::e2DArray, "uimage2DArray"};
    case ImageType::Int2DArray:
      return {D::e2DArray, "iimage2DArray"};
    case ImageType::Float3D:
      return {D::e3D, "image3D"};
    case ImageType::Uint3D:
      return {D::e3D, "uimage3D"};
    case ImageType::Int3D:
      return {D::e3D, "iimage3D"};
    default:
      return {};
  }
}

static wgpu::StorageTextureAccess storage_texture_access(const Qualifier qualifier)
{
  const bool read = flag_is_set(qualifier, Qualifier::read);
  const bool write = flag_is_set(qualifier, Qualifier::write);
  using Access = wgpu::StorageTextureAccess;
  if (read && write) {
    return Access::ReadWrite;
  }
  if (read) {
    return Access::ReadOnly;
  }
  return write ? Access::WriteOnly : Access::Undefined;
}

struct AtomicTextureType {
  ImageType type;
  const char *image;
  const char *sampler;
  const char *component;
  const char *vector;
  int dimensions;
};

static constexpr AtomicTextureType atomic_texture_types[] = {
    {ImageType::AtomicUint2D, "uimage2DAtomic", "usampler2DAtomic", "uint", "uvec4", 2},
    {ImageType::AtomicUint2DArray, "uimage2DArrayAtomic", "usampler2DArrayAtomic", "uint", "uvec4", 3},
    {ImageType::AtomicUint3D, "uimage3DAtomic", "usampler3DAtomic", "uint", "uvec4", 3},
    {ImageType::AtomicInt2D, "iimage2DAtomic", "isampler2DAtomic", "int", "ivec4", 2},
    {ImageType::AtomicInt2DArray, "iimage2DArrayAtomic", "isampler2DArrayAtomic", "int", "ivec4", 3},
    {ImageType::AtomicInt3D, "iimage3DAtomic", "isampler3DAtomic", "int", "ivec4", 3},
};

static const AtomicTextureType &atomic_texture_type(ImageType type)
{
  for (const AtomicTextureType &candidate : atomic_texture_types) {
    if (candidate.type == type) {
      return candidate;
    }
  }
  webgpu_draw_unsupported("this atomic image type");
}

void WebGPUShader::init(const ShaderCreateInfo &info, const bool is_codegen_only)
{
  BuiltinBits point_builtins = BuiltinBits::NONE;
  for (const auto &builtin : info.builtins_) {
    if (builtin.conditions.evaluate(info.compilation_constants_)) {
      point_builtins |= builtin.bit & (BuiltinBits::POINT_SIZE | BuiltinBits::POINT_COORD);
      barycentric_coords_ |= bool(builtin.bit & BuiltinBits::BARYCENTRIC_COORD);
    }
  }
  point_sprites_ = point_builtins != BuiltinBits::NONE;
  point_coord_ = bool(point_builtins & BuiltinBits::POINT_COORD);
  if (!is_codegen_only &&
      std::any_of(info.fragment_outputs_.begin(),
                  info.fragment_outputs_.end(),
                  [](const auto &out) { return out.blend != DualBlend::NONE; }) &&
      !webgpu_active_device().HasFeature(wgpu::FeatureName::DualSourceBlending))
  {
    error_ = "the WebGPU adapter does not support dual-source color blending";
    return;
  }
  if (point_sprites_ || barycentric_coords_) {
    for (const auto &attribute : info.vertex_inputs_) {
      if (point_sprites_) {
        point_vertex_id_location_ = std::max(point_vertex_id_location_,
                                             uint32_t(attribute.index) + 1);
      }
      pull_inputs_.push_back({uint32_t(attribute.index), attribute.type, attribute.name.c_str()});
    }
  }
  if (!info.geometry_source_.is_empty()) {
    error_ = "geometry shaders require WebGPU lowering";
    return;
  }
  for (const auto &input : info.subpass_inputs_) {
    if (!input.conditions.evaluate(info.compilation_constants_)) {
      continue;
    }
    if (input.index < 0 || input.index >= GPU_FB_MAX_COLOR_ATTACHMENT ||
        !ELEM(input.img_type,
              ImageType::Float2D,
              ImageType::Float2DArray,
              ImageType::Uint2D,
              ImageType::Uint2DArray,
              ImageType::Int2D,
              ImageType::Int2DArray))
    {
      error_ = "this WebGPU subpass input is unsupported";
      return;
    }
    uses_uint_textures_ = true;
  }
  for (const Resource &resource : info.resources_get_all_()) {
    if (!resource.conditions.evaluate(info.compilation_constants_)) {
      continue;
    }
    if (resource.bind_type == Resource::BindType::UNIFORM_BUFFER &&
        resource.slot >= image_binding_start)
    {
      error_ = "uniform buffer slot overlaps the WebGPU storage texture range";
      return;
    }
    if (resource.bind_type == Resource::BindType::IMAGE) {
      if (is_atomic_type(resource.image.type)) {
        if (resource.slot < 0 || resource.slot >= GPU_MAX_IMAGE ||
            !ELEM(resource.image.format, TextureFormat::SINT_32, TextureFormat::UINT_32))
        {
          error_ = "this atomic texture slot or format is unsupported";
          return;
        }
        uses_atomic_images_ = true;
        continue;
      }
      const auto format = webgpu_storage_texture_format(resource.image.format);
      const auto access = storage_texture_access(resource.image.qualifiers);
      if (resource.slot < 0 || resource.slot >= GPU_MAX_IMAGE || !format.glsl_layout ||
          !storage_texture_type(resource.image.type).glsl_type ||
          access == wgpu::StorageTextureAccess::Undefined ||
          (access == wgpu::StorageTextureAccess::ReadWrite && !format.read_write))
      {
        error_ = "this storage texture slot, type, format, or access mode is unsupported";
        return;
      }
      if (!is_codegen_only && access == wgpu::StorageTextureAccess::ReadWrite &&
          !ELEM(format.format,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureFormat::R32Uint,
                wgpu::TextureFormat::R32Sint) &&
          !webgpu_active_device().HasFeature(wgpu::FeatureName::TextureFormatsTier2))
      {
        error_ = "this read-write texture format requires WebGPU texture-formats-tier2";
        return;
      }
      if (!is_codegen_only && format.requires_tier1 &&
          !webgpu_active_device().HasFeature(wgpu::FeatureName::TextureFormatsTier1))
      {
        error_ = "this storage texture format requires WebGPU texture-formats-tier1";
        return;
      }
      if (access != wgpu::StorageTextureAccess::WriteOnly &&
          !webgpu_supports_storage_texture_reads())
      {
        error_ = "this browser does not support WGSL storage texture reads";
        return;
      }
    }
    if (resource.bind_type == Resource::BindType::SAMPLER && is_atomic_type(resource.sampler.type)) {
      if (resource.slot < 0 || resource.slot >= 64) {
        error_ = "atomic texture binding is outside Blender's sampler slots";
        return;
      }
      atomic_sampler_slots_ |= uint64_t(1) << resource.slot;
      continue;
    }
    if (resource.bind_type == Resource::BindType::ACCELERATION_STRUCTURE ||
        (resource.bind_type == Resource::BindType::SAMPLER &&
         !sampled_dimension(resource.sampler.type) && !texel_buffer_type(resource.sampler.type)))
    {
      error_ = "shader uses a resource type that has no WebGPU binding yet";
      return;
    }
    if (resource.bind_type == Resource::BindType::SAMPLER) {
      uses_uint_textures_ |= is_uint_sampler(resource.sampler.type);
    }
    if (resource.bind_type == Resource::BindType::SAMPLER &&
        ELEM(resource.sampler.type, ImageType::Depth2D, ImageType::Depth2DArray))
    {
      if (resource.slot < 0 || resource.slot >= 64) {
        error_ = "depth texture binding is outside Blender's sampler slots";
        return;
      }
      depth_bindings_ |= uint64_t(1) << resource.slot;
    }
  }
}

static void print_texel_buffers(std::ostream &out, const ShaderCreateInfo &info)
{
  std::set<int> slots;
  for (const Resource &resource : info.resources_get_all_()) {
    if (resource.bind_type == Resource::BindType::SAMPLER &&
        texel_buffer_type(resource.sampler.type) &&
        resource.conditions.evaluate(info.compilation_constants_))
    {
      slots.insert(resource.slot);
    }
  }
  if (slots.empty()) {
    return;
  }
  /* Blender's source optimizer knows texelFetch as a built-in. Preserve these
   * overloads and their dependencies until glslang resolves their custom types. */
  out << R"(
#pragma blender dead_code_elimination off
#define samplerBuffer BlenderFloatTexels
#define isamplerBuffer BlenderIntTexels
#define usamplerBuffer BlenderUintTexels
struct BlenderFloatTexels { uint slot; };
struct BlenderIntTexels { uint slot; };
struct BlenderUintTexels { uint slot; };
)";
  for (const int slot : slots) {
    out << "layout(set=2,binding=" << slot * 2 << ",std430) readonly buffer BlenderWords" << slot
        << " { uint blender_words_" << slot << "[]; };\n";
    out << "layout(set=2,binding=" << slot * 2 + 1 << ",std140) uniform BlenderTexelLayout" << slot
        << " { uvec4 blender_texel_layout_" << slot << "; };\n";
  }
  out << "uvec4 blender_texel_layout(uint slot) { switch (slot) {\n";
  for (const int slot : slots) {
    out << "case " << slot << "u: return blender_texel_layout_" << slot << ";\n";
  }
  out << "default: return uvec4(0); } }\n";
  out << "uint blender_texel_word(uint slot, uint word) { switch (slot) {\n";
  for (const int slot : slots) {
    out << "case " << slot << "u: return blender_words_" << slot << "[word];\n";
  }
  out << R"(
default: return 0u; } }

uint blender_texel_width(uvec4 metadata, uint component)
{
  return metadata.z == 10u && component == 3u ? 2u : metadata.z;
}

uint blender_texel_bits(uint slot, uint texel, uint component, uvec4 metadata)
{
  if (metadata.z == 10u) {
    return bitfieldExtract(blender_texel_word(slot, texel), int(component * 10u),
                           int(blender_texel_width(metadata, component)));
  }
  uint byte_offset = (texel * metadata.y + component) * (metadata.z >> 3u);
  return bitfieldExtract(blender_texel_word(slot, byte_offset >> 2u),
                         int((byte_offset & 3u) << 3u), int(metadata.z));
}

int textureSize(samplerBuffer image) { return int(blender_texel_layout(image.slot).x); }
int textureSize(isamplerBuffer image) { return int(blender_texel_layout(image.slot).x); }
int textureSize(usamplerBuffer image) { return int(blender_texel_layout(image.slot).x); }

uvec4 texelFetch(usamplerBuffer image, int index)
{
  uvec4 metadata = blender_texel_layout(image.slot);
  uvec4 result = uvec4(0, 0, 0, 1);
  if (uint(index) < metadata.x) {
    for (uint i = 0; i < metadata.y; ++i) {
      result[i] = blender_texel_bits(image.slot, uint(index), i, metadata);
    }
  }
  return result;
}

ivec4 texelFetch(isamplerBuffer image, int index)
{
  uvec4 metadata = blender_texel_layout(image.slot);
  ivec4 result = ivec4(0, 0, 0, 1);
  if (uint(index) < metadata.x) {
    for (uint i = 0; i < metadata.y; ++i) {
      int word = int(blender_texel_bits(image.slot, uint(index), i, metadata));
      result[i] = bitfieldExtract(word, 0, int(blender_texel_width(metadata, i)));
    }
  }
  return result;
}

vec4 texelFetch(samplerBuffer image, int index)
{
  uvec4 metadata = blender_texel_layout(image.slot);
  vec4 result = vec4(0, 0, 0, 1);
  if (uint(index) < metadata.x) {
    for (uint i = 0; i < metadata.y; ++i) {
      uint word = blender_texel_bits(image.slot, uint(index), i, metadata);
      uint width = blender_texel_width(metadata, i);
      if ((metadata.w & 2u) != 0u) {
        result[i] = width == 16u ? unpackHalf2x16(word).x : uintBitsToFloat(word);
      }
      else if ((metadata.w & 1u) != 0u) {
        int signed_word = bitfieldExtract(int(word), 0, int(width));
        result[i] = max(-1.0, float(signed_word) / float((1u << (width - 1u)) - 1u));
      }
      else {
        result[i] = float(word) / float((1u << width) - 1u);
      }
    }
  }
  return result;
}
#pragma blender dead_code_elimination on
)";
}

struct AtomicTextureSlot {
  int binding;
  bool read;
  bool write;
};

static void print_atomic_texture_operations(std::ostream &out,
                                            const AtomicTextureType &type,
                                            const std::vector<AtomicTextureSlot> &slots,
                                            const bool sampler)
{
  if (slots.empty()) {
    return;
  }
  const char *handle = sampler ? type.sampler : type.image;
  const char *coordinate = type.dimensions == 2 ? "ivec2" : "ivec3";
  const char *axes = type.dimensions == 2 ? "xy" : "xyz";
  out << "\n#undef " << handle << "\n#define " << handle << " BlenderAtomic_" << handle
      << "\nstruct " << handle << " { int binding; };\n";
  for (const AtomicTextureSlot &slot : slots) {
    const int group = sampler ? 2 : 3;
    out << "layout(set=" << group << ",binding=" << slot.binding << ",std430) "
        << (slot.write ? "" : "readonly ") << "buffer BlenderAtomicBuffer_" << group << "_"
        << slot.binding << " { uvec4 dimensions; " << type.component << " pixels[]; } " << handle
        << "_" << slot.binding << ";\n";
  }

  out << coordinate << (sampler ? " textureSize(" : " imageSize(") << handle << " image"
      << (sampler ? ", int lod) { if (lod != 0) return " + std::string(coordinate) + "(0); " :
                    ") { ")
      << "switch (image.binding) {\n";
  for (const AtomicTextureSlot &slot : slots) {
    out << "case " << slot.binding << ": return " << coordinate << "(" << handle << "_"
        << slot.binding << ".dimensions." << axes << ");\n";
  }
  out << "default: return " << coordinate << "(0); } }\n";

  auto print_operation = [&](const char *result,
                              const char *name,
                              const char *arguments,
                              const char *operation,
                              const char *operands,
                              const bool read) {
    const bool store = std::string_view(result) == "void";
    const bool fetch = std::string_view(result) == type.vector;
    out << result << " " << name << "(" << handle << " image, " << coordinate << " coord"
        << arguments << ") {\n";
    if (sampler) {
      out << "if (lod != 0) return " << result << "(0);\n";
    }
    out << "switch (image.binding) {\n";
    for (const AtomicTextureSlot &slot : slots) {
      if (read ? !slot.read : !slot.write) {
        continue;
      }
      out << "case " << slot.binding << ": {\n"
          << "uvec4 dimensions = " << handle << "_" << slot.binding << ".dimensions;\n"
          << "if (any(lessThan(coord, " << coordinate << "(0))) || any(greaterThanEqual(coord, "
          << coordinate << "(dimensions." << axes << ")))) break;\n"
          << "uint index = uint(coord.x) + dimensions.x * (uint(coord.y)"
          << (type.dimensions == 2 ? "" : " + dimensions.y * uint(coord.z)") << ");\n";
      if (!store) {
        out << "return ";
      }
      if (fetch) {
        out << type.vector << "(";
      }
      const bool atomic_load = read && slot.write;
      if (atomic_load) {
        out << "atomicAdd(";
      }
      else if (!read) {
        out << operation << "(";
      }
      out << handle << "_" << slot.binding << ".pixels[index]";
      if (atomic_load) {
        out << ", " << type.component << "(0))";
      }
      else if (!read) {
        out << operands << ")";
      }
      if (fetch) {
        out << ", 0, 0, 1)";
      }
      out << ";" << (store ? " return;" : "") << " }\n";
    }
    out << "default: break; }";
    if (!store) {
      out << " return " << result << "(0);";
    }
    out << " }\n";
  };

  print_operation(type.vector,
                  sampler ? "texelFetch" : "imageLoad",
                  sampler ? ", int lod" : "",
                  "",
                  "",
                  true);
  if (sampler) {
    return;
  }
  const std::string store_arguments = ", " + std::string(type.vector) + " value";
  print_operation("void", "imageStore", store_arguments.c_str(), "atomicExchange", ", value.x", false);
  const std::string value_argument = ", " + std::string(type.component) + " value";
  for (const char *suffix : {"Add", "Min", "Max", "And", "Or", "Xor", "Exchange"}) {
    print_operation(type.component,
                    ("imageAtomic" + std::string(suffix)).c_str(),
                    value_argument.c_str(),
                    ("atomic" + std::string(suffix)).c_str(),
                    ", value",
                    false);
  }
}

static void print_atomic_textures(std::ostream &out, const ShaderCreateInfo &info)
{
  const auto resources = info.resources_get_all_();
  bool has_atomic = false;
  for (const AtomicTextureType &type : atomic_texture_types) {
    std::vector<AtomicTextureSlot> samplers;
    std::vector<AtomicTextureSlot> images;
    for (const Resource &resource : resources) {
      if (!resource.conditions.evaluate(info.compilation_constants_)) {
        continue;
      }
      if (resource.bind_type == Resource::BindType::SAMPLER && resource.sampler.type == type.type) {
        samplers.push_back({resource.slot * 2, true, false});
      }
      if (resource.bind_type == Resource::BindType::IMAGE && resource.image.type == type.type) {
        images.push_back({image_binding_start + resource.slot,
                          flag_is_set(resource.image.qualifiers, Qualifier::read),
                          flag_is_set(resource.image.qualifiers, Qualifier::write)});
      }
    }
    if (samplers.empty() && images.empty()) {
      continue;
    }
    if (!has_atomic) {
      out << "\n#pragma blender dead_code_elimination off\n";
      has_atomic = true;
    }
    print_atomic_texture_operations(out, type, samplers, true);
    print_atomic_texture_operations(out, type, images, false);
  }
  if (has_atomic) {
    out << "\n#pragma blender dead_code_elimination on\n";
  }
}

const ShaderCreateInfo &WebGPUShader::patch_create_info(const ShaderCreateInfo &info)
{
  return info;
}

static void print_constants(std::ostream &out, const ShaderCreateInfo &info)
{
  std::ostringstream initializers;
  for (size_t id = 0; id < info.specialization_constants_.size(); ++id) {
    const SpecializationConstant &constant = info.specialization_constants_[id];
    const std::string storage = "blender_specialization_" + std::to_string(id);
    out << constant.type << " " << constant.name << ";\n";
    initializers << constant.name << " = ";
    out << "layout(constant_id=" << id << ") const ";
    if (constant.type == Type::float_t) {
      out << "uint " << storage << "=" << constant.value.u << "u;\n";
      initializers << "uintBitsToFloat(" << storage << ");\n";
    }
    else {
      out << constant.type << " " << storage << "=";
      switch (constant.type) {
        case Type::bool_t:
          out << (constant.value.u ? "true" : "false");
          break;
        case Type::uint_t:
          out << constant.value.u << "u";
          break;
        default:
          out << constant.value.i;
          break;
      }
      out << ";\n";
      initializers << storage << ";\n";
    }
  }
  if (!info.specialization_constants_.is_empty()) {
    /* Keep normal GLSL scope rules for shader parameters that share a constant's name.
     * The assignments also avoid OpSpecConstantOp for expressions using these values. */
    out << "void blender_init_specializations() {\n" << initializers.str() << "}\n";
  }
  for (const CompilationConstant &constant : info.compilation_constants_) {
    out << "const " << constant.type << " " << constant.name << "=";
    switch (constant.type) {
      case Type::bool_t:
        out << (constant.value.u ? "true" : "false");
        break;
      case Type::uint_t:
        out << constant.value.u << "u";
        break;
      default:
        out << constant.value.i;
        break;
    }
    out << ";\n";
  }
}

static void print_resources(std::ostream &out, Span<Resource> resources, const char *frequency)
{
  StringRefNull current;
  for (const Resource &resource : resources) {
    if (resource.info_name != current) {
      current = resource.info_name;
      out << "\n#define CREATE_INFO_RES_" << frequency << "_" << current << " \\\n";
    }
    const int slot = resource.slot;
    switch (resource.bind_type) {
      case Resource::BindType::UNIFORM_BUFFER:
        out << "layout(set=3,binding=" << slot << ",std140) uniform _"
            << resource.uniformbuf.name.str_no_array() << " { " << resource.uniformbuf.type_name
            << " " << resource.uniformbuf.name << "; };";
        break;
      case Resource::BindType::STORAGE_BUFFER:
        out << "layout(set=1,binding=" << slot << ",std430) ";
        if (!flag_is_set(resource.storagebuf.qualifiers, Qualifier::write)) {
          out << "readonly ";
        }
        out << "buffer _" << resource.storagebuf.name.str_no_array() << " { "
            << resource.storagebuf.type_name << " " << resource.storagebuf.name << "; };";
        break;
      case Resource::BindType::IMAGE: {
        if (is_atomic_type(resource.image.type)) {
          const char *type = atomic_texture_type(resource.image.type).image;
          out << "const " << type << " " << resource.image.name << " = " << type << "("
              << image_binding_start + slot << ");";
          break;
        }
        const auto format = webgpu_storage_texture_format(resource.image.format);
        const auto access = storage_texture_access(resource.image.qualifiers);
        out << "layout(" << format.glsl_layout << ",set=3,binding=" << image_binding_start + slot
            << ") uniform ";
        if (access == wgpu::StorageTextureAccess::ReadOnly) {
          out << "readonly ";
        }
        if (access == wgpu::StorageTextureAccess::WriteOnly) {
          out << "writeonly ";
        }
        out << storage_texture_type(resource.image.type).glsl_type << " " << resource.image.name
            << ";";
        break;
      }
      case Resource::BindType::SAMPLER: {
        const auto name = resource.sampler.name;
        const auto dimension = sampled_dimension(resource.sampler.type);
        if (is_atomic_type(resource.sampler.type)) {
          const char *type = atomic_texture_type(resource.sampler.type).sampler;
          out << "const " << type << " " << name << " = " << type << "(" << slot * 2 << ");";
        }
        else if (const char *type = texel_buffer_type(resource.sampler.type)) {
          out << "const " << type << " " << name << " = " << type << "(" << slot << "u);";
        }
        else if (is_uint_sampler(resource.sampler.type)) {
          out << "layout(set=2,binding=" << slot * 2 << ") uniform utexture" << dimension << " "
              << name << ";";
        }
        else if (dimension) {
          out << "layout(set=2,binding=" << slot * 2 << ") uniform sampler" << dimension << " "
              << name << ";";
        }
        break;
      }
      default:
        break;
    }
    out << " \\\n";
  }
  out << "\n";
}

std::string WebGPUShader::resources_declare(const ShaderCreateInfo &info) const
{
  std::ostringstream out;
  print_constants(out, info);
  print_texel_buffers(out, info);
  if (uses_atomic_images_ || atomic_sampler_slots_) {
    print_atomic_textures(out, info);
  }
  print_resources(out, info.pass_resources_, "PASS");
  print_resources(out, info.batch_resources_, "BATCH");
  print_resources(out, info.geometry_resources_, "GEOMETRY");

  StringRefNull active;
  for (const auto &shared : info.shared_variables_) {
    if (shared.info_name != active) {
      active = shared.info_name;
      out << "\n#define CREATE_INFO_RES_SHARED_VARS_" << active << " \\\n";
    }
    out << "shared " << shared.type << " " << shared.name << "; \\\n";
  }
  out << "\n";

  /* WebGPU does not have Vulkan push constants. Binding zero is reserved for
   * the per-draw uniform block in std140 layout. */
  if (point_sprites_) {
    out << "struct BlenderPointState { vec2 pixel_to_ndc; float fixed_size; "
           "int instance; uint indexed; };\n";
  }
  if (!info.push_constants_.is_empty() || point_sprites_ || barycentric_coords_) {
    out << "layout(set=0,binding=0,std140) uniform BlenderDrawConstants {\n";
    for (const auto &constant : info.push_constants_) {
      out << constant.type << " " << constant.name;
      if (constant.array_size > 0) {
        out << "[" << constant.array_size << "]";
      }
      out << ";\n";
    }
    if (point_sprites_) {
      out << "BlenderPointState blender_point_state;\n";
    }
    if (point_sprites_ || barycentric_coords_) {
      out << "uvec4 blender_vertex_stream_offsets[4];\n";
    }
    out << "};\n";
  }
  return out.str();
}

static const char *interpolation(Interpolation interp)
{
  switch (interp) {
    case Interpolation::FLAT:
      return "flat";
    case Interpolation::NO_PERSPECTIVE:
      return "noperspective";
    default:
      return "smooth";
  }
}

static int location_count(Type type)
{
  if (type == Type::float4x4_t) {
    return 4;
  }
  return type == Type::float3x3_t ? 3 : 1;
}

static int print_interfaces(std::ostream &out,
                            std::ostream &transfer,
                            const ShaderCreateInfo &info,
                            const bool is_vertex)
{
  int location = 0;
  for (const StageInterfaceInfo *iface : info.vertex_out_interfaces_) {
    const bool named = !iface->instance_name.is_empty();
    if (named) {
      /* Keep the source-facing struct private. Naga imports each stage location as
       * an independent WGSL input or output instead of binding a whole block. */
      out << "struct " << iface->name << " {\n";
      for (const auto &member : iface->inouts) {
        out << member.type << " " << member.name << ";\n";
      }
      out << "} " << iface->instance_name << ";\n";
    }
    for (const auto &member : iface->inouts) {
      out << "layout(location=" << location << ") " << interpolation(member.interp) << " "
          << (is_vertex ? "out" : "in") << " " << member.type << " ";
      if (named) {
        const std::string varying = "blender_varying_" + std::to_string(location);
        const std::string field = std::string(iface->instance_name) + "." +
                                  std::string(member.name.str_no_array());
        out << varying << member.name.str_only_array();
        transfer << (is_vertex ? varying : field) << " = " << (is_vertex ? field : varying)
                 << ";\n";
      }
      else {
        out << member.name;
      }
      out << ";\n";
      location += location_count(member.type);
    }
  }
  return location;
}

std::string WebGPUShader::vertex_interface_declare(const ShaderCreateInfo &info) const
{
  std::ostringstream out;
  std::ostringstream transfer;
  const bool can_pull = point_sprites_ || barycentric_coords_;
  for (const auto &attribute : info.vertex_inputs_) {
    if (can_pull) {
      out << "#ifdef BLENDER_VERTEX_PULL\n"
          << attribute.type << " " << attribute.name << ";\n#else\n";
    }
    out << "layout(location=" << attribute.index << ") in " << attribute.type << " "
        << attribute.name << ";\n";
    if (can_pull) {
      out << "#endif\n";
    }
  }
  const int location = print_interfaces(out, transfer, info, true);
  if (point_coord_) {
    out << "layout(location=" << location << ") noperspective out vec2 blender_point_coord;\n";
  }
  if (barycentric_coords_) {
    /* Use the same corner order and interpolation as Blender's OpenGL geometry shader. */
    out << "layout(location=" << location << ") smooth out vec3 gpu_BaryCoord;\n";
  }
  if (can_pull) {
    out << "#ifdef BLENDER_VERTEX_PULL\n"
           "#undef gl_VertexID\n"
           "#undef gl_InstanceID\n"
           "#undef gpu_InstanceIndex\n"
           "layout(set=0,binding=1,std140) uniform BlenderVertexPullCommand {\n"
           "  uvec4 blender_vertex_pull_cmd;\n"
           "};\n"
           "uint blender_pulled_vertex;\n"
           "uint blender_load_vertex(uint ordinal);\n"
           "void blender_load_vertex_attributes(uint vertex);\n"
           "#define gl_VertexID int(blender_pulled_vertex)\n"
           "#define gl_InstanceID int(blender_vertex_pull_cmd.z + uint(gl_InstanceIndex))\n"
           "#define gpu_InstanceIndex gl_InstanceID\n"
           "#endif\n";
  }
  if (point_sprites_) {
    /* The four strip vertices share the attributes of a single point instance. */
    out << "float blender_point_size;\n"
           "#define gl_PointSize blender_point_size\n"
           "#ifndef BLENDER_VERTEX_PULL\n"
           "#undef gl_VertexID\n"
           "#undef gl_InstanceID\n"
           "#undef gpu_InstanceIndex\n"
        << "layout(location=" << point_vertex_id_location_ << ") in uint blender_point_index;\n"
        << "#define gl_VertexID (blender_point_state.indexed != 0u ? "
           "int(blender_point_index) : gl_InstanceIndex)\n"
           "#define gl_InstanceID blender_point_state.instance\n"
           "#define gpu_InstanceIndex blender_point_state.instance\n"
           "#endif\n";
  }
  out << "layout(constant_id=1023) const bool blender_texture_target = false;\n"
         "void blender_vertex_main();\n"
         "void main() {\n";
  if (!info.specialization_constants_.is_empty()) {
    out << "blender_init_specializations();\n";
  }
  if (point_sprites_) {
    out << "blender_point_size = blender_point_state.fixed_size;\n";
  }
  if (can_pull) {
    out << "#ifdef BLENDER_VERTEX_PULL\n"
           "blender_pulled_vertex = blender_load_vertex(uint(gl_VertexIndex)"
        << (point_sprites_ ? " / 6u" : "") << ");\n"
        << "blender_load_vertex_attributes(blender_pulled_vertex);\n"
           "#endif\n";
  }
  if (barycentric_coords_) {
    out << "uint bary_corner = uint(gl_VertexIndex) % 3u;\n"
           "gpu_BaryCoord = vec3(bary_corner == 0u, bary_corner == 1u, bary_corner == 2u);\n";
  }
  out << "blender_vertex_main();\n";
  out << transfer.str();
  if (point_sprites_) {
    out << "#ifdef BLENDER_VERTEX_PULL\n"
           "const uint corners[6] = uint[6](0u, 1u, 2u, 2u, 1u, 3u);\n"
           "uint corner_index = corners[uint(gl_VertexIndex) % 6u];\n"
           "vec2 corner = vec2(corner_index & 1u, corner_index >> 1u);\n"
           "#else\n"
           "vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);\n"
           "#endif\n";
    if (point_coord_) {
      out << "blender_point_coord = corner;\n";
    }
    out << "if (blender_point_size > 0.0 && gl_Position.w > 0.0 && "
           "all(lessThanEqual(abs(gl_Position.xyz), vec3(gl_Position.w)))) {\n"
           "  gl_Position.xy += (corner - 0.5) * vec2(1.0, -1.0) * "
           "    blender_point_state.pixel_to_ndc * (blender_point_size * "
           "gl_Position.w);\n"
           "} else { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }\n";
  }
  out << "if (blender_texture_target) { gl_Position.y = -gl_Position.y; }\n"
         "gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;\n"
         "}\n"
         "#define main blender_vertex_main\n";
  return out.str();
}

std::string WebGPUShader::fragment_interface_declare(const ShaderCreateInfo &info) const
{
  std::ostringstream out;
  std::ostringstream transfer;
  std::ostringstream transfer_out;
  const int location = print_interfaces(out, transfer, info, false);
  if (point_coord_) {
    out << "layout(location=" << location
        << ") noperspective in vec2 blender_point_coord;\n"
           "#define gl_PointCoord blender_point_coord\n";
  }
  if (barycentric_coords_) {
    out << "layout(location=" << location << ") smooth in vec3 gpu_BaryCoord;\n";
  }
  for (const auto &input : info.subpass_inputs_) {
    if (!input.conditions.evaluate(info.compilation_constants_)) {
      continue;
    }
    const std::string texture = "blender_subpass_" + std::to_string(input.index);
    const char *prefix = ELEM(input.img_type, ImageType::Uint2D, ImageType::Uint2DArray) ? "u" :
                         ELEM(input.img_type, ImageType::Int2D, ImageType::Int2DArray)   ? "i" :
                                                                                         "";
    out << "layout(set=3,binding=" << subpass_binding_start + input.index << ") uniform "
        << prefix << "texture2D " << texture << ";\n"
        << input.type << " " << input.name << ";\n";
    const auto swizzle = std::string_view("xyzw").substr(0, to_component_count(input.type));
    transfer << input.name << " = texelFetch(" << texture << ", ivec2(gl_FragCoord.xy), 0)."
             << swizzle << ";\n";
  }
  for (const auto &color : info.fragment_outputs_) {
    /* EEVEE uses a read-write RGBA32Float backing texture for its logical RG16 UNORM
     * normal target. Keep the shader's two-channel output and expand it at the boundary. */
    const bool gbuffer_normal = color.type == Type::float2_t &&
                                std::string_view(color.name.c_str()).ends_with("_gbuf_normal");
    out << "layout(location=" << color.index;
    if (color.blend != DualBlend::NONE) {
      out << ",index=" << (color.blend == DualBlend::SRC_1 ? 1 : 0);
    }
    if (gbuffer_normal) {
      const std::string physical = "blender_gbuffer_normal";
      out << ") out vec4 " << physical << ";\n"
          << color.type << " " << color.name << ";\n";
      transfer_out << physical << " = vec4(roundEven(clamp(" << color.name
                   << ", 0.0, 1.0) * 65535.0) / 65535.0, 0.0, 1.0);\n";
    }
    else {
      out << ") out " << color.type << " " << color.name << ";\n";
    }
  }
  if (info.fragment_source_ == "eevee_surf_depth.bsl.hh") {
    /* Fixed-function depth writes can round differently from the rasterized fragment position
     * used by a later EEVEE material pass. Store the fragment depth so that Equal matches. */
    transfer_out << "gl_FragDepth = gl_FragCoord.z;\n";
  }
  if (!info.specialization_constants_.is_empty() || !transfer.view().empty() ||
      !transfer_out.view().empty())
  {
    out << "void blender_fragment_main();\n"
           "void main() {\n";
    if (!info.specialization_constants_.is_empty()) {
      out << "blender_init_specializations();\n";
    }
    out << transfer.str()
        << "blender_fragment_main();\n"
        << transfer_out.str()
        << "}\n"
           "#define main blender_fragment_main\n";
  }
  return out.str();
}

std::string WebGPUShader::compute_layout_declare(const ShaderCreateInfo &info) const
{
  std::ostringstream out;
  out << "layout(local_size_x=" << info.compute_layout_.local_size_x
      << ",local_size_y=" << info.compute_layout_.local_size_y
      << ",local_size_z=" << info.compute_layout_.local_size_z << ") in;\n";
  if (!info.specialization_constants_.is_empty()) {
    out << "void blender_init_specializations();\n"
           "void blender_compute_main();\n"
           "void main() {\n"
           "blender_init_specializations();\n"
           "blender_compute_main();\n"
           "}\n"
           "#define main blender_compute_main\n";
  }
  return out.str();
}

std::string WebGPUShader::geometry_layout_declare(const ShaderCreateInfo &) const
{
  return {};
}
std::string WebGPUShader::geometry_interface_declare(const ShaderCreateInfo &) const
{
  return {};
}
void WebGPUShader::geometry_shader_from_glsl(const ShaderCreateInfo &, MutableSpan<StringRefNull>)
{
}

void WebGPUShader::vertex_shader_from_glsl(const ShaderCreateInfo &info,
                                           MutableSpan<StringRefNull> sources)
{
  if (error_.empty()) {
    const bool can_use_packed_normals = std::any_of(
        info.vertex_inputs_.begin(), info.vertex_inputs_.end(), [](const auto &attribute) {
          return ELEM(
              attribute.type, Type::float_t, Type::float2_t, Type::float3_t, Type::float4_t);
        });
    std::string source = combined_glsl(
        "#define GPU_VERTEX_SHADER\n", sources, true, uses_uint_textures_);
    if (point_sprites_ || barycentric_coords_) {
      pull_vertex_source_ = source;
    }
    if (!skip_preprocessor) {
      source = Shader::run_preprocessor(source, G.debug & G_DEBUG_GPU_SHADER_NO_DCE);
    }
    vertex_ = compile_stage(name,
                            GLSLANG_STAGE_VERTEX,
                            std::move(source),
                            depth_bindings_,
                            error_,
                            vertex_bindings_.data(),
                            info.specialization_constants_.is_empty() ? nullptr :
                                                                        &vertex_constant_ids_,
                            can_use_packed_normals ? &vertex_spirv_ : nullptr);
  }
}

void WebGPUShader::fragment_shader_from_glsl(const ShaderCreateInfo &info,
                                             MutableSpan<StringRefNull> sources)
{
  if (error_.empty()) {
    const char *stage_define = info.fragment_source_ == "eevee_surf_volume.bsl.hh" ?
                                   "#define GPU_FRAGMENT_SHADER\n#define GPU_VOLUME_FROXEL_SHADER\n" :
                                   "#define GPU_FRAGMENT_SHADER\n";
    fragment_ = compile_stage(
        name,
        GLSLANG_STAGE_FRAGMENT,
        combined_glsl(stage_define, sources, skip_preprocessor, uses_uint_textures_),
        depth_bindings_,
        error_,
        fragment_bindings_.data(),
        info.specialization_constants_.is_empty() ? nullptr : &fragment_constant_ids_);
  }
}

void WebGPUShader::compute_shader_from_glsl(const ShaderCreateInfo &info,
                                            MutableSpan<StringRefNull> sources)
{
  if (error_.empty()) {
    compute_ = compile_stage(
        name,
        GLSLANG_STAGE_COMPUTE,
        combined_glsl(
            "#define GPU_COMPUTE_SHADER\n", sources, skip_preprocessor, uses_uint_textures_),
        depth_bindings_,
        error_,
        compute_bindings_.data(),
        info.specialization_constants_.is_empty() ? nullptr : &compute_constant_ids_);
  }
}

class DrawShaderInterface final : public ShaderInterface {
 public:
  DrawShaderInterface(const ShaderCreateInfo &info, Span<Resource> resources)
  {
    attr_len_ = info.vertex_inputs_.size();
    uniform_len_ = info.push_constants_.size();
    constant_len_ = info.specialization_constants_.size();
    std::fill(std::begin(builtins_), std::end(builtins_), -1);
    std::fill(std::begin(builtin_blocks_), std::end(builtin_blocks_), -1);
    std::fill(std::begin(attr_types_), std::end(attr_types_), 0);

    size_t names_size = 0;
    for (const auto &attr : info.vertex_inputs_) {
      names_size += attr.name.size() + 1;
    }
    for (const auto &uniform : info.push_constants_) {
      names_size += uniform.name.size() + 1;
    }
    for (const auto &constant : info.specialization_constants_) {
      names_size += constant.name.size() + 1;
    }
    for (const auto &resource : resources) {
      if (resource.bind_type == Resource::BindType::UNIFORM_BUFFER) {
        ubo_len_++;
        names_size += resource.uniformbuf.name.size() + 1;
      }
      if (resource.bind_type == Resource::BindType::SAMPLER) {
        uniform_len_++;
        names_size += resource.sampler.name.size() + 1;
      }
      if (resource.bind_type == Resource::BindType::IMAGE) {
        uniform_len_++;
        names_size += resource.image.name.size() + 1;
      }
      if (resource.bind_type == Resource::BindType::STORAGE_BUFFER) {
        ssbo_len_++;
        names_size += resource.storagebuf.name.size() + 1;
      }
    }
    if (names_size == 0) {
      return;
    }
    inputs_ = MEM_new_array<ShaderInput>(
        attr_len_ + ubo_len_ + uniform_len_ + ssbo_len_ + constant_len_, "WebGPU shader inputs");
    name_buffer_ = MEM_new_array_uninitialized<char>(names_size, "WebGPU input names");

    uint32_t offset = 0;
    size_t input_index = 0;
    for (const auto &attr : info.vertex_inputs_) {
      ShaderInput &input = inputs_[input_index++];
      input.location = input.binding = attr.index;
      copy_input_name(&input, attr.name, name_buffer_, offset);
      if (attr.index >= 0 && attr.index < GPU_VERT_ATTR_MAX_LEN) {
        attr_types_[attr.index] = uint8_t(attr.type);
        enabled_attr_mask_ |= 1u << attr.index;
      }
    }
    for (const auto &resource : resources) {
      if (resource.bind_type == Resource::BindType::UNIFORM_BUFFER) {
        ShaderInput &input = inputs_[input_index++];
        input.location = input.binding = resource.slot;
        copy_input_name(&input, resource.uniformbuf.name, name_buffer_, offset);
      }
    }
    for (const auto &uniform : info.push_constants_) {
      ShaderInput &input = inputs_[input_index];
      input.location = int32_t(input_index++ - attr_len_ - ubo_len_);
      input.binding = -1;
      copy_input_name(&input, uniform.name, name_buffer_, offset);
    }
    for (const auto &resource : resources) {
      if (resource.bind_type == Resource::BindType::SAMPLER) {
        ShaderInput &input = inputs_[input_index++];
        input.location = -1;
        input.binding = resource.slot;
        copy_input_name(&input, resource.sampler.name, name_buffer_, offset);
      }
      else if (resource.bind_type == Resource::BindType::IMAGE) {
        ShaderInput &input = inputs_[input_index++];
        input.location = -1;
        input.binding = resource.slot;
        copy_input_name(&input, resource.image.name, name_buffer_, offset);
        enabled_ima_mask_ |= 1u << resource.slot;
      }
    }
    set_image_formats_from_info(info);
    for (const auto &resource : resources) {
      if (resource.bind_type == Resource::BindType::STORAGE_BUFFER) {
        ShaderInput &input = inputs_[input_index++];
        input.location = input.binding = resource.slot;
        copy_input_name(&input, resource.storagebuf.name, name_buffer_, offset);
      }
    }
    /* The shared batch-expansion path uses this mask to bind its vertex and
     * index streams and to set the associated fetch uniforms. */
    for (const auto &resource : info.geometry_resources_) {
      if (resource.bind_type == Resource::BindType::STORAGE_BUFFER &&
          resource.conditions.evaluate(info.compilation_constants_))
      {
        ssbo_attr_mask_ |= 1u << resource.slot;
      }
    }
    for (int id = 0; id < constant_len_; id++) {
      ShaderInput &input = inputs_[input_index++];
      input.location = id;
      input.binding = -1;
      copy_input_name(&input, info.specialization_constants_[id].name, name_buffer_, offset);
    }
    sort_inputs();

    for (int builtin = 0; builtin < GPU_NUM_UNIFORMS; builtin++) {
      const char *name = builtin_uniform_name(GPUUniformBuiltin(builtin));
      if (const ShaderInput *input = name ? uniform_get(name) : nullptr) {
        builtins_[builtin] = input->location;
      }
    }
    for (int builtin = 0; builtin < GPU_NUM_UNIFORM_BLOCKS; builtin++) {
      const char *name = builtin_uniform_block_name(GPUUniformBlockBuiltin(builtin));
      if (const ShaderInput *input = name ? ubo_get(name) : nullptr) {
        builtin_blocks_[builtin] = input->binding;
      }
    }
  }
};

static size_t align_to(size_t offset, const size_t alignment)
{
  return (offset + alignment - 1) & ~(alignment - 1);
}

void WebGPUShader::init_draw_interface(const ShaderCreateInfo &info)
{
  auto resources = info.resources_get_all_();
  resources.remove_if([&info](const Resource &resource) {
    return !resource.conditions.evaluate(info.compilation_constants_);
  });
  interface = new DrawShaderInterface(info, resources);
  for (const int id : info.specialization_constants_.index_range()) {
    constant_keys_.push_back(std::to_string(id));
  }
  auto binding_visibility = [&](const uint32_t group,
                                const uint32_t binding) -> wgpu::ShaderStage {
    if (binding >= 128) {
      return wgpu::ShaderStage::None;
    }
    const uint32_t address = group * 128 + binding;
    const uint32_t word = address / 32;
    const uint32_t bit = 1u << (address % 32);
    return (vertex_bindings_[word] & bit ? wgpu::ShaderStage::Vertex : wgpu::ShaderStage::None) |
           (fragment_bindings_[word] & bit ? wgpu::ShaderStage::Fragment :
                                             wgpu::ShaderStage::None) |
           (compute_bindings_[word] & bit ? wgpu::ShaderStage::Compute : wgpu::ShaderStage::None);
  };
  auto add_binding = [&](const uint32_t group, wgpu::BindGroupLayoutEntry entry) {
    entry.visibility = binding_visibility(group, entry.binding);
    if (entry.visibility != wgpu::ShaderStage::None) {
      groups_[group].bindings.push_back(entry);
    }
  };

  for (const Resource &resource : resources) {
    wgpu::BindGroupLayoutEntry entry{};
    switch (resource.bind_type) {
      case Resource::BindType::UNIFORM_BUFFER:
        entry.binding = resource.slot;
        entry.buffer.type = wgpu::BufferBindingType::Uniform;
        add_binding(3, entry);
        break;
      case Resource::BindType::STORAGE_BUFFER: {
        const bool writable = flag_is_set(resource.storagebuf.qualifiers, Qualifier::write);
        if (writable && (binding_visibility(1, resource.slot) & wgpu::ShaderStage::Vertex)) {
          unsupported_draw_resources_ = true;
          break;
        }
        entry.binding = resource.slot;
        entry.buffer.type = writable ? wgpu::BufferBindingType::Storage :
                                       wgpu::BufferBindingType::ReadOnlyStorage;
        add_binding(1, entry);
        break;
      }
      case Resource::BindType::IMAGE:
        entry.binding = image_binding_start + resource.slot;
        if (is_atomic_type(resource.image.type)) {
          const bool writable = flag_is_set(resource.image.qualifiers, Qualifier::write);
          if (writable && (binding_visibility(3, entry.binding) & wgpu::ShaderStage::Vertex)) {
            unsupported_draw_resources_ = true;
            break;
          }
          entry.buffer.type = writable ? wgpu::BufferBindingType::Storage :
                                         wgpu::BufferBindingType::ReadOnlyStorage;
          entry.buffer.minBindingSize = 8 * sizeof(uint32_t);
          add_binding(3, entry);
          break;
        }
        entry.storageTexture.access = storage_texture_access(resource.image.qualifiers);
        entry.storageTexture.format = webgpu_storage_texture_format(resource.image.format).format;
        entry.storageTexture.viewDimension = storage_texture_type(resource.image.type).dimension;
        if (entry.storageTexture.access != wgpu::StorageTextureAccess::ReadOnly &&
            (binding_visibility(3, entry.binding) & wgpu::ShaderStage::Vertex))
        {
          unsupported_draw_resources_ = true;
          break;
        }
        add_binding(3, entry);
        break;
      case Resource::BindType::SAMPLER: {
        if (is_atomic_type(resource.sampler.type)) {
          entry.binding = resource.slot * 2;
          entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
          entry.buffer.minBindingSize = 8 * sizeof(uint32_t);
          add_binding(2, entry);
          break;
        }
        if (texel_buffer_type(resource.sampler.type)) {
          entry.binding = resource.slot * 2;
          entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
          add_binding(2, entry);
          entry.binding++;
          entry.buffer.type = wgpu::BufferBindingType::Uniform;
          entry.buffer.minBindingSize = 16;
          add_binding(2, entry);
          break;
        }
        using D = wgpu::TextureViewDimension;
        switch (resource.sampler.type) {
          case ImageType::Float1D:
          case ImageType::Uint2D:
          case ImageType::Depth2D:
          case ImageType::Float2D:
            entry.texture.viewDimension = D::e2D;
            break;
          case ImageType::Float1DArray:
          case ImageType::Depth2DArray:
          case ImageType::Uint2DArray:
          case ImageType::Float2DArray:
            entry.texture.viewDimension = D::e2DArray;
            break;
          case ImageType::Uint3D:
          case ImageType::Float3D:
            entry.texture.viewDimension = D::e3D;
            break;
          case ImageType::FloatCube:
            entry.texture.viewDimension = D::Cube;
            break;
          case ImageType::FloatCubeArray:
            entry.texture.viewDimension = D::CubeArray;
            break;
          default:
            unsupported_draw_resources_ = true;
            continue;
        }
        entry.binding = resource.slot * 2;
        entry.texture.sampleType = ELEM(resource.sampler.type,
                                        ImageType::Depth2D,
                                        ImageType::Depth2DArray) ?
                                       wgpu::TextureSampleType::Depth :
                                       wgpu::TextureSampleType::Float;
        if (is_uint_sampler(resource.sampler.type)) {
          entry.texture.sampleType = wgpu::TextureSampleType::Uint;
          add_binding(2, entry);
          break;
        }
        add_binding(2, entry);
        wgpu::BindGroupLayoutEntry sampler{};
        sampler.binding = entry.binding + 1;
        sampler.sampler.type = entry.texture.sampleType == wgpu::TextureSampleType::Depth ?
                                   wgpu::SamplerBindingType::NonFiltering :
                                   wgpu::SamplerBindingType::Filtering;
        add_binding(2, sampler);
        break;
      }
      default:
        unsupported_draw_resources_ = true;
        break;
    }
  }

  for (const auto &input : info.subpass_inputs_) {
    if (!input.conditions.evaluate(info.compilation_constants_)) {
      continue;
    }
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = subpass_binding_start + input.index;
    entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    using SampleType = wgpu::TextureSampleType;
    entry.texture.sampleType = ELEM(input.img_type, ImageType::Uint2D, ImageType::Uint2DArray) ?
                                   SampleType::Uint :
                               ELEM(input.img_type, ImageType::Int2D, ImageType::Int2DArray) ?
                                   SampleType::Sint :
                                   SampleType::UnfilterableFloat;
    add_binding(3, entry);
  }

  size_t size = 0;
  for (const auto &constant : info.push_constants_) {
    const uint32_t columns = constant.type == Type::float4x4_t ? 4 :
                             constant.type == Type::float3x3_t ? 3 :
                                                                 1;
    const uint32_t rows = to_component_count(constant.type) / columns;
    const uint32_t count = std::max(1, constant.array_size);
    const uint32_t element_size = columns > 1 ? 16 * columns : rows * sizeof(uint32_t);
    const uint32_t alignment = constant.array_size || columns > 1 || rows > 2 ? 16 : element_size;
    const uint32_t stride = constant.array_size ? align_to(element_size, 16) : element_size;
    size = align_to(size, alignment);
    uniforms_.push_back({size, count, stride, rows, columns});
    size += stride * count;
  }
  if (point_sprites_) {
    point_state_offset_ = align_to(size, 16);
    size = point_state_offset_ + 32;
  }
  if (point_sprites_ || barycentric_coords_) {
    pull_stream_offsets_ = align_to(size, 16);
    size = pull_stream_offsets_ + 16 * sizeof(uint32_t);
  }
  uniform_data_.resize(align_to(size, 16));
  if (!uniform_data_.empty()) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.buffer.type = wgpu::BufferBindingType::Uniform;
    entry.buffer.hasDynamicOffset = true;
    entry.buffer.minBindingSize = uniform_data_.size();
    add_binding(0, entry);
  }
}

void WebGPUShader::set_point_draw_state(const int instance, const bool indexed)
{
  int viewport[4];
  GPU_viewport_size_get_i(viewport);
  struct PointState {
    float pixel_to_ndc[2];
    float fixed_size;
    int32_t instance;
    uint32_t indexed;
  };
  static_assert(sizeof(PointState) == 20);
  const PointState point_state{{2.0f / std::max(1, viewport[2]), 2.0f / std::max(1, viewport[3])},
                               std::abs(Context::get()->state_manager->mutable_state.point_size),
                               instance,
                               uint32_t(indexed)};
  std::memcpy(uniform_data_.data() + point_state_offset_, &point_state, sizeof(point_state));
}

bool WebGPUShader::finalize(const ShaderCreateInfo *info)
{
  if (!error_.empty()) {
    std::cerr << "WebGPU shader '" << name << "': " << error_ << '\n';
    return false;
  }
  if (!(compute_ || (vertex_ && fragment_))) {
    return false;
  }
  init_draw_interface(*info);
  return true;
}

void WebGPUShader::warm_cache(int) {}
void WebGPUShader::bind(const shader::SpecializationConstants *state)
{
  bound_pipelines_ = &default_pipelines_;
  if (!state || state->is_empty() ||
      std::equal(state->values.begin(),
                 state->values.end(),
                 constants->values.begin(),
                 constants->values.end()))
  {
    return;
  }

  std::vector<uint32_t> key;
  key.reserve(state->values.size());
  for (const SpecializationConstant::Value &value : state->values) {
    key.push_back(value.u);
  }
  auto [variant, inserted] = specialized_pipelines_.try_emplace(std::move(key));
  bound_pipelines_ = &variant->second;
  if (!inserted) {
    return;
  }
  auto add_constants = [&](Span<uint32_t> ids, std::vector<wgpu::ConstantEntry> &entries) {
    for (const uint32_t id : ids) {
      if (id < constant_keys_.size()) {
        auto &entry = entries.emplace_back();
        entry.key = constant_keys_[id].c_str();
        /* Floats are passed as their exact bits to the u32 override. */
        entry.value = state->types[id] == Type::int_t ? double(state->values[id].i) :
                                                        double(state->values[id].u);
      }
    }
  };
  add_constants(vertex_constant_ids_, bound_pipelines_->vertex_constants);
  add_constants(fragment_constant_ids_, bound_pipelines_->fragment_constants);
  add_constants(compute_constant_ids_, bound_pipelines_->compute_constants);
}
void WebGPUShader::unbind() {}

void WebGPUShader::update_draw_uniform(int location,
                                       const int comp_len,
                                       const int array_size,
                                       const void *data)
{
  if (location < 0) {
    return;
  }
  if (size_t(location) >= uniforms_.size() || comp_len < 1 || array_size < 0) {
    webgpu_draw_unsupported("this uniform update");
  }

  const Uniform &uniform = uniforms_[location];
  const uint32_t elements = std::max(1, array_size);
  if (elements > uniform.array_len || uint32_t(comp_len) > uniform.rows * uniform.columns ||
      (uniform.columns > 1 && uint32_t(comp_len) != uniform.rows * uniform.columns))
  {
    webgpu_draw_unsupported("a uniform update outside its declared std140 range");
  }
  auto *destination = uniform_data_.data() + uniform.offset;
  auto *source = static_cast<const std::byte *>(data);
  const size_t column_bytes = uniform.columns > 1 ? uniform.rows * 4 : comp_len * 4;
  for (uint32_t element = 0; element < elements; element++) {
    for (uint32_t column = 0; column < uniform.columns; column++) {
      std::memcpy(
          destination + element * uniform.array_stride + column * 16, source, column_bytes);
      source += column_bytes;
    }
  }
}

void WebGPUShader::uniform_float(int location, int comp_len, int array_size, const float *data)
{
  update_draw_uniform(location, comp_len, array_size, data);
}

void WebGPUShader::uniform_int(int location, int comp_len, int array_size, const int *data)
{
  update_draw_uniform(location, comp_len, array_size, data);
}

static wgpu::BlendState blend_state(const GPUBlend mode)
{
  using F = wgpu::BlendFactor;
  using O = wgpu::BlendOperation;
  switch (mode) {
    case GPU_BLEND_NONE:
      return {};
    case GPU_BLEND_ALPHA:
    case GPU_BLEND_ALPHA_PREMULT:
      return {{O::Add, mode == GPU_BLEND_ALPHA ? F::SrcAlpha : F::One, F::OneMinusSrcAlpha},
              {O::Add, F::One, F::OneMinusSrcAlpha}};
    case GPU_BLEND_ADDITIVE:
      return {{O::Add, F::SrcAlpha, F::One}, {O::Add, F::Zero, F::One}};
    case GPU_BLEND_MIN:
      return {{O::Min, F::One, F::One}, {O::Min, F::One, F::One}};
    case GPU_BLEND_MAX:
      return {{O::Max, F::One, F::One}, {O::Max, F::One, F::One}};
    case GPU_BLEND_SUBTRACT:
      return {{O::ReverseSubtract, F::One, F::One}, {O::ReverseSubtract, F::One, F::One}};
    case GPU_BLEND_ADDITIVE_PREMULT:
      return {{O::Add, F::One, F::One}, {O::Add, F::One, F::One}};
    case GPU_BLEND_MULTIPLY:
      return {{O::Add, F::Dst, F::Zero}, {O::Add, F::DstAlpha, F::Zero}};
    case GPU_BLEND_INVERT:
      return {{O::Add, F::OneMinusDst, F::Zero}, {O::Add, F::Zero, F::One}};
    case GPU_BLEND_OIT:
      return {{O::Add, F::One, F::One}, {O::Add, F::Zero, F::OneMinusSrcAlpha}};
    case GPU_BLEND_BACKGROUND:
      return {{O::Add, F::OneMinusDstAlpha, F::SrcAlpha}, {O::Add, F::Zero, F::SrcAlpha}};
    case GPU_BLEND_CUSTOM:
      return {{O::Add, F::One, F::Src1}, {O::Add, F::One, F::Src1Alpha}};
    case GPU_BLEND_ALPHA_UNDER_PREMUL:
      return {{O::Add, F::OneMinusDstAlpha, F::One}, {O::Add, F::OneMinusDstAlpha, F::One}};
    case GPU_BLEND_OVERLAY_MASK_FROM_ALPHA:
      return {{O::Add, F::Zero, F::OneMinusSrcAlpha}, {O::Add, F::Zero, F::OneMinusSrcAlpha}};
    case GPU_BLEND_TRANSPARENCY:
      return {{O::Add, F::One, F::SrcAlpha}, {O::Add, F::Zero, F::SrcAlpha}};
    default:
      webgpu_draw_unsupported("this color blending mode");
  }
}

const wgpu::ShaderModule &WebGPUShader::vertex_module(const uint32_t packed_normals)
{
  if (packed_normals == 0) {
    return vertex_;
  }
  if (const auto found = packed_vertex_modules_.find(packed_normals);
      found != packed_vertex_modules_.end())
  {
    return found->second;
  }
  std::string error;
  auto module = shader_module_from_spirv(
      name, GLSLANG_STAGE_VERTEX, vertex_spirv_, depth_bindings_, packed_normals, error);
  if (!module) {
    std::cerr << "WebGPU shader '" << name << "': " << error << '\n';
    webgpu_draw_unsupported("rendering with this packed vertex input");
  }
  return packed_vertex_modules_.emplace(packed_normals, std::move(module)).first->second;
}

enum class PullComponent { Float, Unsigned, Signed, Unorm, Snorm };

struct PullAttributeFormat {
  wgpu::VertexFormat format;
  uint32_t bits;
  uint32_t lanes;
  PullComponent component;
};

static PullAttributeFormat pull_attribute_format(const WebGPUVertexAttribute &input)
{
  using W = wgpu::VertexFormat;
  using T = PullComponent;
  if (input.packed_snorm) {
    return {input.format, 10, 4, T::Snorm};
  }
  static constexpr PullAttributeFormat formats[] = {
      {W::Float32, 32, 1, T::Float},         {W::Float32x2, 32, 2, T::Float},
      {W::Float32x3, 32, 3, T::Float},       {W::Float32x4, 32, 4, T::Float},
      {W::Float16x2, 16, 2, T::Float},       {W::Float16x4, 16, 4, T::Float},
      {W::Uint32, 32, 1, T::Unsigned},       {W::Uint32x2, 32, 2, T::Unsigned},
      {W::Uint32x3, 32, 3, T::Unsigned},     {W::Uint32x4, 32, 4, T::Unsigned},
      {W::Sint32, 32, 1, T::Signed},         {W::Sint32x2, 32, 2, T::Signed},
      {W::Sint32x3, 32, 3, T::Signed},       {W::Sint32x4, 32, 4, T::Signed},
      {W::Uint16, 16, 1, T::Unsigned},       {W::Uint16x2, 16, 2, T::Unsigned},
      {W::Uint16x4, 16, 4, T::Unsigned},     {W::Sint16, 16, 1, T::Signed},
      {W::Sint16x2, 16, 2, T::Signed},       {W::Sint16x4, 16, 4, T::Signed},
      {W::Unorm16, 16, 1, T::Unorm},         {W::Unorm16x2, 16, 2, T::Unorm},
      {W::Unorm16x4, 16, 4, T::Unorm},       {W::Snorm16, 16, 1, T::Snorm},
      {W::Snorm16x2, 16, 2, T::Snorm},       {W::Snorm16x4, 16, 4, T::Snorm},
      {W::Uint8, 8, 1, T::Unsigned},         {W::Uint8x2, 8, 2, T::Unsigned},
      {W::Uint8x4, 8, 4, T::Unsigned},       {W::Sint8, 8, 1, T::Signed},
      {W::Sint8x2, 8, 2, T::Signed},         {W::Sint8x4, 8, 4, T::Signed},
      {W::Unorm8, 8, 1, T::Unorm},           {W::Unorm8x2, 8, 2, T::Unorm},
      {W::Unorm8x4, 8, 4, T::Unorm},         {W::Snorm8, 8, 1, T::Snorm},
      {W::Snorm8x2, 8, 2, T::Snorm},         {W::Snorm8x4, 8, 4, T::Snorm},
      {W::Unorm10_10_10_2, 10, 4, T::Unorm},
  };
  for (const auto &format : formats) {
    if (format.format == input.format) {
      return format;
    }
  }
  webgpu_draw_unsupported("pulling this vertex format");
}

static const char *pull_component_type(const PullComponent component)
{
  switch (component) {
    case PullComponent::Unsigned:
      return "uint";
    case PullComponent::Signed:
      return "int";
    default:
      return "float";
  }
}

static void pull_input_value(std::ostream &out,
                             const WebGPURenderPipelineKey &key,
                             uint32_t location,
                             uint32_t lanes,
                             const char *type)
{
  for (size_t slot = 0; slot < key.vertices.size(); slot++) {
    const auto &layout = key.vertices[slot];
    if (layout.stride == 0) {
      continue;
    }
    for (const auto &input : layout.attributes) {
      if (input.location != location) {
        continue;
      }
      const auto format = pull_attribute_format(input);
      out << type << "(";
      for (uint32_t lane = 0; lane < lanes; lane++) {
        if (lane != 0) {
          out << ", ";
        }
        if (lane >= format.lanes) {
          out << pull_component_type(format.component) << "(" << (lane == 3 ? 1 : 0) << ")";
          continue;
        }
        const bool packed = format.bits == 10;
        const uint32_t byte_offset = input.offset + (packed ? 0 : lane * format.bits / 8);
        const uint32_t bits = packed && lane == 3 ? 2 : format.bits;
        out << "blender_point_";
        switch (format.component) {
          case PullComponent::Float:
            out << (bits == 16 ? "half" : "float");
            break;
          case PullComponent::Signed:
            out << "sint";
            break;
          case PullComponent::Snorm:
            out << "snorm";
            break;
          case PullComponent::Unorm:
            out << "unorm";
            break;
          case PullComponent::Unsigned:
            out << "uint";
            break;
        }
        out << "(blender_vertex_stream_" << slot << "[(base_" << slot << " + " << byte_offset
            << "u) >> 2u], "
            << (packed ? std::to_string(lane * 10) + "u" :
                         "((base_" + std::to_string(slot) + " + " + std::to_string(byte_offset) +
                             "u) & 3u) * 8u")
            << ", " << bits << "u)";
      }
      out << ")";
      return;
    }
  }
  out << type << "(";
  for (uint32_t lane = 0; lane < lanes; lane++) {
    out << (lane == 0 ? "" : ", ") << (lane == 3 ? 1 : 0);
  }
  out << ")";
}

const wgpu::ShaderModule &WebGPUShader::vertex_pull_module(const WebGPURenderPipelineKey &key)
{
  PullModuleKey modules_key{key.pull_index_format, key.vertices};
  if (const auto it = vertex_pull_modules_.find(modules_key); it != vertex_pull_modules_.end()) {
    return it->second;
  }

  constexpr std::string_view vertex_define = "#define GPU_VERTEX_SHADER\n";
  std::string source = pull_vertex_source_;
  const size_t vertex_define_position = source.find(vertex_define);
  if (vertex_define_position == std::string::npos) {
    webgpu_draw_unsupported("finding the vertex pull shader declaration");
  }
  source.insert(vertex_define_position + vertex_define.size(), "#define BLENDER_VERTEX_PULL\n");

  std::ostringstream pull;
  if (key.pull_index_format != wgpu::IndexFormat::Undefined) {
    pull << "layout(set=0,binding=2,std430) readonly buffer BlenderPulledIndices "
            "{ uint blender_pulled_indices[]; };\n";
  }
  pull << "uint blender_load_vertex(uint ordinal) {\n"
          "uint at = blender_vertex_pull_cmd.x + ordinal;\n";
  if (key.pull_index_format == wgpu::IndexFormat::Uint16) {
    pull << "return ((blender_pulled_indices[at >> 1u] >> ((at & 1u) * 16u)) & 65535u) + "
            "blender_vertex_pull_cmd.y;\n";
  }
  else if (key.pull_index_format == wgpu::IndexFormat::Uint32) {
    pull << "return blender_pulled_indices[at] + blender_vertex_pull_cmd.y;\n";
  }
  else {
    pull << "return at;\n";
  }
  pull << "}\n";
  for (size_t slot = 0; slot < key.vertices.size(); slot++) {
    if (key.vertices[slot].stride != 0) {
      pull << "layout(set=0,binding=" << 3 + slot << ",std430) readonly buffer BlenderVertexStream"
           << slot << " { uint blender_vertex_stream_" << slot << "[]; };\n";
    }
  }
  pull << R"(
float blender_point_float(uint word, uint shift, uint bits) { return uintBitsToFloat(word); }
float blender_point_half(uint word, uint shift, uint bits) { return unpackHalf2x16(word)[shift >> 4u]; }
uint blender_point_uint(uint word, uint shift, uint bits)
{
  return bits == 32u ? word : bitfieldExtract(word, int(shift), int(bits));
}
int blender_point_sint(uint word, uint shift, uint bits)
{
  return bits == 32u ? int(word) : bitfieldExtract(int(word), int(shift), int(bits));
}
float blender_point_unorm(uint word, uint shift, uint bits)
{
  return float(blender_point_uint(word, shift, bits)) / float((1u << bits) - 1u);
}
float blender_point_snorm(uint word, uint shift, uint bits)
{
  return max(-1.0, float(blender_point_sint(word, shift, bits)) / float((1u << (bits - 1u)) - 1u));
}
void blender_load_vertex_attributes(uint vertex) {
)";
  for (size_t slot = 0; slot < key.vertices.size(); slot++) {
    if (key.vertices[slot].stride != 0) {
      pull << "uint base_" << slot << " = blender_vertex_stream_offsets[" << slot / 4 << "]["
           << slot % 4 << "] + vertex * " << key.vertices[slot].stride << "u;\n";
    }
  }
  for (const auto &input : pull_inputs_) {
    pull << input.name << " = ";
    if (ELEM(input.type, Type::float3x3_t, Type::float4x4_t)) {
      const uint32_t length = input.type == Type::float3x3_t ? 3 : 4;
      pull << input.type << "(";
      for (uint32_t column = 0; column < length; column++) {
        if (column != 0) {
          pull << ", ";
        }
        pull_input_value(
            pull, key, input.location + column, length, length == 3 ? "vec3" : "vec4");
      }
      pull << ")";
    }
    else {
      std::ostringstream type;
      type << input.type;
      pull_input_value(
          pull, key, input.location, to_component_count(input.type), type.str().c_str());
    }
    pull << ";\n";
  }
  pull << "}\n";
  source += pull.str();

  if (!skip_preprocessor) {
    source = Shader::run_preprocessor(source, G.debug & G_DEBUG_GPU_SHADER_NO_DCE);
  }
  std::string error;
  auto module = compile_stage(
      name, GLSLANG_STAGE_VERTEX, std::move(source), depth_bindings_, error, nullptr, nullptr);
  if (!module) {
    std::cerr << "WebGPU vertex pull shader '" << name << "': " << error << '\n';
    webgpu_draw_unsupported("rendering this vertex pull shader");
  }
  return vertex_pull_modules_.emplace(std::move(modules_key), std::move(module)).first->second;
}

wgpu::PipelineLayout WebGPUShader::vertex_pull_layout(const WebGPURenderPipelineKey &key) const
{
  const auto &device = webgpu_active_device();
  std::vector<wgpu::BindGroupLayoutEntry> bindings = groups_[0].bindings;
  /* Fixed-function vertex fetch can leave binding zero unused. The pulled variant uses it for
   * its stream offsets regardless of the original shader's reflected bindings. */
  if (bindings.empty()) {
    wgpu::BindGroupLayoutEntry draw{};
    draw.buffer.type = wgpu::BufferBindingType::Uniform;
    draw.buffer.hasDynamicOffset = true;
    draw.buffer.minBindingSize = uniform_data_.size();
    bindings.push_back(draw);
  }
  bindings.front().visibility |= wgpu::ShaderStage::Vertex;
  wgpu::BindGroupLayoutEntry command{};
  command.binding = 1;
  command.visibility = wgpu::ShaderStage::Vertex;
  command.buffer.type = wgpu::BufferBindingType::Uniform;
  command.buffer.hasDynamicOffset = true;
  command.buffer.minBindingSize = 16;
  bindings.push_back(command);
  auto add_stream = [&](uint32_t binding) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = binding;
    entry.visibility = wgpu::ShaderStage::Vertex;
    entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bindings.push_back(entry);
  };
  if (key.pull_index_format != wgpu::IndexFormat::Undefined) {
    add_stream(2);
  }
  for (size_t slot = 0; slot < key.vertices.size(); slot++) {
    if (key.vertices[slot].stride != 0) {
      add_stream(3 + slot);
    }
  }
  wgpu::BindGroupLayoutDescriptor desc{};
  desc.entryCount = bindings.size();
  desc.entries = bindings.data();
  std::array<wgpu::BindGroupLayout, 4> layouts{device.CreateBindGroupLayout(&desc),
                                               groups_[1].layout,
                                               groups_[2].layout,
                                               groups_[3].layout};
  size_t layout_count = layouts.size();
  while (!layouts[layout_count - 1]) {
    layout_count--;
  }
  wgpu::PipelineLayoutDescriptor pipeline{};
  pipeline.bindGroupLayoutCount = layout_count;
  pipeline.bindGroupLayouts = layouts.data();
  return device.CreatePipelineLayout(&pipeline);
}

uint64_t WebGPUShader::sampler_override_mask(const std::array<uint32_t, 18> &metadata)
{
  return uint64_t(metadata[16]) | (uint64_t(metadata[17]) << 32);
}

WebGPUShader::SamplerOverrides WebGPUShader::sampler_overrides(const uint64_t mask)
{
  SamplerOverrides overrides;
  if (mask != 0) {
    uint64_t active = mask & webgpu_emulated_sampler_mask();
    while (active != 0) {
      const uint32_t slot = std::countr_zero(active);
      overrides.emplace_back(slot, webgpu_emulated_sampler_parameters(slot));
      active &= active - 1;
    }
  }
  return overrides;
}

void WebGPUShader::append_sampler_overrides(std::vector<wgpu::ConstantEntry> &constants,
                                           const SamplerOverrides &overrides,
                                           const uint64_t mask)
{
  /* ID 1023 is the vertex orientation override. Sampler states follow it in WGSL. */
  static const auto keys = [] {
    std::array<std::string, 64> names;
    for (size_t slot = 0; slot < names.size(); slot++) {
      names[slot] = std::to_string(1024 + slot);
    }
    return names;
  }();
  for (const auto &[slot, flags] : overrides) {
    if (mask & (uint64_t(1) << slot)) {
      wgpu::ConstantEntry entry{};
      entry.key = keys[slot].c_str();
      entry.value = flags;
      constants.push_back(entry);
    }
  }
}

const wgpu::RenderPipeline &WebGPUShader::render_pipeline(const WebGPURenderPipelineKey &key)
{
  const uint64_t vertex_sampler_mask = sampler_override_mask(vertex_bindings_);
  const uint64_t fragment_sampler_mask = sampler_override_mask(fragment_bindings_);
  const auto samplers = sampler_overrides(vertex_sampler_mask | fragment_sampler_mask);
  auto &pipelines = samplers.empty() ? bound_pipelines_->pipelines.render_pipelines :
                                      bound_pipelines_->sampler_pipelines[samplers].render_pipelines;
  if (const auto found = pipelines.find(key); found != pipelines.end()) {
    return found->second;
  }
  if (!vertex_ || !fragment_ || unsupported_draw_resources_) {
    webgpu_draw_unsupported("rendering with this shader's resource bindings or output layout");
  }

  const wgpu::Device &device = webgpu_active_device();
  ensure_pipeline_layout(device);
  const auto layout = key.vertex_pulling ? vertex_pull_layout(key) : pipeline_layout_;

  const size_t vertex_streams = key.vertex_pulling ? 0 : key.vertices.size();
  std::vector<std::vector<wgpu::VertexAttribute>> attributes(vertex_streams);
  std::vector<wgpu::VertexBufferLayout> layouts(vertex_streams);
  uint32_t packed_normals = 0;
  for (size_t i = 0; i < vertex_streams; i++) {
    for (const auto &input : key.vertices[i].attributes) {
      wgpu::VertexAttribute attribute{};
      attribute.format = input.format;
      attribute.offset = input.offset;
      attribute.shaderLocation = input.location;
      attributes[i].push_back(attribute);
      if (input.packed_snorm) {
        packed_normals |= 1u << input.location;
      }
    }
    layouts[i].arrayStride = key.vertices[i].stride;
    layouts[i].stepMode = key.point_sprites ? wgpu::VertexStepMode::Instance :
                                              wgpu::VertexStepMode::Vertex;
    layouts[i].attributeCount = attributes[i].size();
    layouts[i].attributes = attributes[i].data();
  }

  const wgpu::BlendState blend = blend_state(key.blend);
  std::array<wgpu::ColorTargetState, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  size_t color_count = 0;
  for (size_t i = 0; i < colors.size(); i++) {
    if (key.targets.colors[i] == wgpu::TextureFormat::Undefined) {
      continue;
    }
    auto &color = colors[i];
    color.format = key.targets.colors[i];
    if (fragment_output_bits & (1u << i)) {
      color.writeMask = key.write_mask;
      color.blend = key.blend == GPU_BLEND_NONE ? nullptr : &blend;
    }
    else {
      color.writeMask = wgpu::ColorWriteMask::None;
    }
    color_count = i + 1;
  }
  wgpu::FragmentState fragment{};
  fragment.module = fragment_;
  fragment.entryPoint = "main";
  fragment.targetCount = color_count;
  fragment.targets = colors.data();
  auto fragment_constants = bound_pipelines_->fragment_constants;
  append_sampler_overrides(fragment_constants, samplers, fragment_sampler_mask);
  fragment.constantCount = fragment_constants.size();
  fragment.constants = fragment_constants.data();

  using Compare = wgpu::CompareFunction;
  using Op = wgpu::StencilOperation;
  wgpu::DepthStencilState depth{};
  depth.format = key.targets.depth;
  depth.depthWriteEnabled = key.depth_write;
  switch (key.depth_test) {
    case GPU_DEPTH_LESS:
      depth.depthCompare = Compare::Less;
      break;
    case GPU_DEPTH_LESS_EQUAL:
      depth.depthCompare = Compare::LessEqual;
      break;
    case GPU_DEPTH_EQUAL:
      depth.depthCompare = Compare::Equal;
      break;
    case GPU_DEPTH_GREATER:
      depth.depthCompare = Compare::Greater;
      break;
    case GPU_DEPTH_GREATER_EQUAL:
      depth.depthCompare = Compare::GreaterEqual;
      break;
    default:
      depth.depthCompare = Compare::Always;
      break;
  }
  depth.stencilFront.compare = key.stencil_test == GPU_STENCIL_EQUAL  ? Compare::Equal :
                               key.stencil_test == GPU_STENCIL_NEQUAL ? Compare::NotEqual :
                                                                        Compare::Always;
  depth.stencilBack = depth.stencilFront;
  depth.stencilReadMask = key.stencil_compare_mask;
  depth.stencilWriteMask = key.stencil_write_mask;
  if (key.stencil_op == GPU_STENCIL_OP_REPLACE) {
    depth.stencilFront.passOp = depth.stencilBack.passOp = Op::Replace;
  }
  else if (key.stencil_op == GPU_STENCIL_OP_COUNT_DEPTH_PASS) {
    depth.stencilFront.passOp = Op::DecrementWrap;
    depth.stencilBack.passOp = Op::IncrementWrap;
  }
  else if (key.stencil_op == GPU_STENCIL_OP_COUNT_DEPTH_FAIL) {
    depth.stencilFront.depthFailOp = Op::IncrementWrap;
    depth.stencilBack.depthFailOp = Op::DecrementWrap;
  }

  wgpu::ConstantEntry orientation{};
  orientation.key = "1023";
  orientation.value = key.targets.texture_coordinates ? 1 : 0;
  auto vertex_constants = bound_pipelines_->vertex_constants;
  vertex_constants.push_back(orientation);
  append_sampler_overrides(vertex_constants, samplers, vertex_sampler_mask);
  wgpu::RenderPipelineDescriptor pipeline{};
  pipeline.label = name;
  pipeline.layout = layout;
  pipeline.vertex.module = key.vertex_pulling ? vertex_pull_module(key) :
                                                vertex_module(packed_normals);
  pipeline.vertex.entryPoint = "main";
  pipeline.vertex.bufferCount = layouts.size();
  pipeline.vertex.buffers = layouts.data();
  pipeline.vertex.constantCount = vertex_constants.size();
  pipeline.vertex.constants = vertex_constants.data();
  pipeline.fragment = &fragment;
  if (depth.format != wgpu::TextureFormat::Undefined) {
    pipeline.depthStencil = &depth;
  }
  pipeline.primitive.topology = key.topology;
  pipeline.primitive.stripIndexFormat = key.strip_index_format;
  pipeline.primitive.cullMode = key.cull_mode;
  pipeline.primitive.frontFace = key.front_face;
  return pipelines.emplace(key, device.CreateRenderPipeline(&pipeline)).first->second;
}

void WebGPUShader::ensure_pipeline_layout(const wgpu::Device &device)
{
  if (pipeline_layout_) {
    return;
  }
  size_t count = groups_.size();
  while (count > 0 && groups_[count - 1].bindings.empty()) {
    count--;
  }
  std::array<wgpu::BindGroupLayout, 4> layouts;
  for (size_t i = 0; i < count; i++) {
    auto &group = groups_[i];
    wgpu::BindGroupLayoutDescriptor desc{};
    desc.entryCount = group.bindings.size();
    desc.entries = group.bindings.data();
    group.layout = layouts[i] = device.CreateBindGroupLayout(&desc);
  }
  wgpu::PipelineLayoutDescriptor desc{};
  desc.bindGroupLayoutCount = count;
  desc.bindGroupLayouts = layouts.data();
  pipeline_layout_ = device.CreatePipelineLayout(&desc);
}

void WebGPUShader::draw_pulled_vertices(const wgpu::RenderPassEncoder &pass,
                                        const wgpu::RenderPipeline &pipeline,
                                        const WebGPUVertexPullCommands &commands,
                                        const wgpu::Buffer &indices,
                                        const std::span<const wgpu::Buffer> streams,
                                        const std::span<const uint32_t> stream_offsets,
                                        const uint32_t count,
                                        const uint32_t direct_vertex_count,
                                        const uint32_t direct_instance_count)
{
  if (stream_offsets.size() > 16) {
    webgpu_draw_unsupported("this number of vertex pull streams");
  }
  if (point_sprites_) {
    set_point_draw_state(0);
  }
  if (!stream_offsets.empty()) {
    std::memcpy(uniform_data_.data() + pull_stream_offsets_,
                stream_offsets.data(),
                stream_offsets.size_bytes());
  }
  const auto uniform = webgpu_upload_draw_uniforms(uniform_data_);

  std::vector<wgpu::BindGroupEntry> entries(2);
  entries[0].buffer = uniform.buffer;
  entries[0].size = uniform_data_.size();
  entries[1].binding = 1;
  entries[1].buffer = commands.shader_arguments;
  entries[1].size = 16;
  if (indices) {
    auto &entry = entries.emplace_back();
    entry.binding = 2;
    entry.buffer = indices;
  }
  for (size_t slot = 0; slot < streams.size(); slot++) {
    if (streams[slot]) {
      auto &entry = entries.emplace_back();
      entry.binding = 3 + slot;
      entry.buffer = streams[slot];
    }
  }
  wgpu::BindGroupDescriptor desc{};
  desc.layout = pipeline.GetBindGroupLayout(0);
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  const auto group = webgpu_active_device().CreateBindGroup(&desc);
  for (uint32_t i = 1; i < groups_.size(); i++) {
    if (groups_[i].layout) {
      bind_resource_group(pass, i);
    }
  }
  for (uint32_t command = 0; command < count; command++) {
    const uint32_t offsets[2]{uniform.offset,
                              commands.shader_offset + command * commands.shader_stride};
    webgpu_render_set_bind_group(0, group, offsets);
    if (commands.draw_arguments) {
      pass.DrawIndirect(commands.draw_arguments, uint64_t(command) * 16);
    }
    else {
      pass.Draw(direct_vertex_count, direct_instance_count);
    }
  }
}

void WebGPUShader::bind_draw_resources(const wgpu::RenderPassEncoder &pass)
{
  bind_resources(pass);
}

void WebGPUShader::bind_compute_pipeline(const wgpu::ComputePassEncoder &pass)
{
  if (!compute_ || unsupported_draw_resources_) {
    webgpu_draw_unsupported("dispatching this compute shader's resource bindings");
  }
  const uint64_t sampler_mask = sampler_override_mask(compute_bindings_);
  const auto samplers = sampler_overrides(sampler_mask);
  auto &pipeline = samplers.empty() ? bound_pipelines_->pipelines.compute_pipeline :
                                     bound_pipelines_->sampler_pipelines[samplers].compute_pipeline;
  if (!pipeline) {
    const auto &device = webgpu_active_device();
    ensure_pipeline_layout(device);
    wgpu::ComputePipelineDescriptor desc{};
    desc.label = name;
    desc.layout = pipeline_layout_;
    desc.compute.module = compute_;
    desc.compute.entryPoint = "main";
    auto constants = bound_pipelines_->compute_constants;
    append_sampler_overrides(constants, samplers, sampler_mask);
    desc.compute.constantCount = constants.size();
    desc.compute.constants = constants.data();
    pipeline = device.CreateComputePipeline(&desc);
  }
  pass.SetPipeline(pipeline);
  bind_resources(pass);
}

template<typename Pass> void WebGPUShader::bind_resources(const Pass &pass)
{
  bind_uniforms(pass);
  for (uint32_t i = 1; i < groups_.size(); i++) {
    if (groups_[i].layout) {
      bind_resource_group(pass, i);
    }
  }
}

template<typename Pass>
void WebGPUShader::bind_resource_group(const Pass &pass, const uint32_t index)
{
  auto &group = groups_[index];
  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(group.bindings.size());
  for (const auto &layout : group.bindings) {
    wgpu::BindGroupEntry &entry = entries.emplace_back();
    entry.binding = layout.binding;
    if (index == 1) {
      entry.buffer = webgpu_bound_storage(layout.binding);
      if (!entry.buffer) {
        std::fprintf(
            stderr, "WebGPU shader '%s' requires storage buffer %u.\n", name, layout.binding);
        webgpu_draw_unsupported("using a shader with a missing storage buffer binding");
      }
    }
    else if (index == 3) {
      if (layout.binding >= subpass_binding_start) {
        entry.textureView = webgpu_subpass_input(layout.binding - subpass_binding_start);
        if (!entry.textureView) {
          webgpu_draw_unsupported("using a shader with a missing framebuffer input attachment");
        }
      }
      else if (layout.storageTexture.access != wgpu::StorageTextureAccess::BindingNotUsed) {
        entry.textureView = webgpu_bound_image(layout.binding - image_binding_start);
        if (!entry.textureView) {
          webgpu_draw_unsupported("using a shader with a missing storage texture binding");
        }
      }
      else if (layout.binding >= image_binding_start) {
        entry.buffer = webgpu_bound_atomic_texture(layout.binding - image_binding_start, true);
        if (!entry.buffer) {
          webgpu_draw_unsupported("using a shader with a missing atomic image binding");
        }
      }
      else {
        entry.buffer = webgpu_bound_uniform(layout.binding);
        if (!entry.buffer) {
          webgpu_draw_unsupported("using a shader with a missing uniform buffer binding");
        }
      }
    }
    else if (layout.buffer.type != wgpu::BufferBindingType::BindingNotUsed) {
      const auto slot = layout.binding / 2;
      if (atomic_sampler_slots_ & (uint64_t(1) << slot)) {
        entry.buffer = webgpu_bound_atomic_texture(slot, false);
      }
      else {
        const auto texels = webgpu_bound_texel_buffer(slot);
        entry.buffer = layout.binding % 2 == 0 ? texels.data : texels.metadata;
      }
      if (!entry.buffer) {
        webgpu_draw_unsupported("using a shader with a missing texture buffer binding");
      }
    }
    else {
      const auto texture = webgpu_bound_texture(layout.binding / 2);
      if (!texture.view) {
        webgpu_draw_unsupported("using a shader with a missing sampled texture binding");
      }
      if (layout.binding % 2 == 0) {
        entry.textureView = texture.view;
      }
      else {
        entry.sampler = texture.sampler;
      }
    }
  }
  const bool same_resources = std::equal(
      entries.begin(),
      entries.end(),
      group.last_entries.begin(),
      group.last_entries.end(),
      [](const wgpu::BindGroupEntry &a, const wgpu::BindGroupEntry &b) {
        return a.buffer.Get() == b.buffer.Get() && a.textureView.Get() == b.textureView.Get() &&
               a.sampler.Get() == b.sampler.Get();
      });
  if (!group.last_group || !same_resources) {
    wgpu::BindGroupDescriptor desc{};
    desc.layout = group.layout;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    group.last_group = webgpu_active_device().CreateBindGroup(&desc);
    group.last_entries = std::move(entries);
  }
  if constexpr (std::is_same_v<Pass, wgpu::RenderPassEncoder>) {
    webgpu_render_set_bind_group(index, group.last_group);
  }
  else {
    pass.SetBindGroup(index, group.last_group);
  }
}

template<typename Pass> void WebGPUShader::bind_uniforms(const Pass &pass)
{
  if (groups_[0].bindings.empty()) {
    if (groups_[0].layout) {
      bind_resource_group(pass, 0);
    }
    return;
  }
  const WebGPUUniformSlice slice = webgpu_upload_draw_uniforms(uniform_data_);
  auto [binding, inserted] = draw_uniform_bindings_.try_emplace(slice.buffer.Get());
  if (inserted) {
    wgpu::BindGroupEntry entry{};
    entry.buffer = slice.buffer;
    entry.size = uniform_data_.size();
    wgpu::BindGroupDescriptor desc{};
    desc.layout = groups_[0].layout;
    desc.entryCount = 1;
    desc.entries = &entry;
    binding->second = webgpu_active_device().CreateBindGroup(&desc);
  }
  if constexpr (std::is_same_v<Pass, wgpu::RenderPassEncoder>) {
    webgpu_render_set_bind_group(0, binding->second, {&slice.offset, 1});
  }
  else {
    pass.SetBindGroup(0, binding->second, 1, &slice.offset);
  }
}

}  // namespace blender::gpu

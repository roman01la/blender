# SPDX-License-Identifier: Apache-2.0

"""Build the Cycles WebGPU experiments and native references."""

import argparse
import re
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

from wgsl_loop_state import coalesce_loop_state


SOURCE = Path(__file__).resolve().parent
CYCLES = SOURCE.parent.parent / "blender" / "intern" / "cycles"
SVM_SAMPLER_BRIDGE = "_Z26cycles_svm_hardware_samplejjjjff"


def run(*args, stdout=None):
    subprocess.run(args, check=True, stdout=stdout)


def retarget(source: Path, wrapper: Path, output: Path) -> None:
    shader_ir = source.read_text()
    wrapper_ir = wrapper.read_text()
    # Use the Emscripten C++ headers to compile the source, then link its vector
    # ABI to the Vulkan entry point. No Wasm code is executed by the shader.
    for item in ("datalayout", "triple"):
        directive = rf'^target {item} = ".*"$'
        expected = re.search(directive, wrapper_ir, re.M)
        if expected is None:
            raise ValueError(f"Shader wrapper does not declare its target {item}")
        shader_ir, count = re.subn(directive, expected.group(0), shader_ir, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one target {item} in Cycles IR")
    shader_ir = re.sub(r' "target-(?:cpu|features)"="[^"]*"', "", shader_ir)
    shader_ir = re.sub(r"^define hidden ", "define hidden spir_func ", shader_ir, flags=re.M)
    # Match the calling convention of storage accessors and the shared film setting.
    shader_ir = re.sub(
        r"\b(call|declare) (?=(?:noundef )?i32 @_Z"
        r"(?:(?:17cycles_scene_word|19cycles_surface_word|17cycles_light_word|"
        r"25cycles_camera_sample_word|15cycles_svm_word|18cycles_volume_word)j|23cycles_film_transparentv)\b)",
        r"\1 spir_func ",
        shader_ir,
    )
    # Direct lighting also calls the C++ traversal kernel defined in a separate module.
    shader_ir = re.sub(r"\b(call|declare) (?=void @_Z17cycles_bvh2_trace)", r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(r"\b(call|declare) (?=void @_Z\d+cycles_volume_light_sample)",
                       r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(r"\b(call|declare) (?=<4 x float> @_Z\d+cycles_volume_shadow_transmittance)",
                       r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(r"\b(call|declare) (?=<4 x i32> @_Z18cycles_svm_contextjj)",
                       r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(
        rf"\b(call|declare) (?=(?:noundef )?<4 x float> @{SVM_SAMPLER_BRIDGE}\b)",
        r"\1 spir_func ",
        shader_ir,
    )
    shader_ir = re.sub(r"\b(call|declare) (?=(?:noundef )?(?:"
                       r"void @_Z29cycles_svm_write_ordered_lobe|<4 x float> @_Z20cycles_material_lobej\b))",
                       r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(r"\b(call|declare) (?=(?:noundef )?(?:float|void|<4 x float>) "
                       r"@_Z\d+cycles_query_(?:ao|bevel|raycast))",
                       r"\1 spir_func ", shader_ir)
    # Surface closures and their lookup accessor link across the BSDF and dispatch modules.
    shader_ir = re.sub(
        r"\b(call|declare) (?=(?:noundef )?(?:float @_Z17cycles_ggx_lookupj\b|"
        r"<4 x float> @_Z(?:15cycles_ggx_eval|\d+cycles_standalone_eval)|"
        r"i32 @_Z(?:17cycles_ggx_sample|\d+cycles_standalone_sample)|"
        r"float @_Z\d+cycles_bump_shadowing|<4 x float> @_Z\d+cycles_bump_specular_normal))",
        r"\1 spir_func ",
        shader_ir,
    )
    output.write_text(shader_ir)


def normalize_spirv(code: bytes, hardware_svm_textures: bool = False) -> bytes:
    if len(code) < 20 or len(code) % 4:
        raise ValueError("Invalid SPIR-V binary length")
    words = struct.unpack(f"<{len(code) // 4}I", code)
    if words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V magic")
    instructions = []
    int_types = {}
    vector_types = {}
    has_sampler_bridge = False
    cursor = 5
    while cursor < len(words):
        length, opcode = words[cursor] >> 16, words[cursor] & 0xFFFF
        if length == 0 or cursor + length > len(words):
            raise ValueError("Invalid SPIR-V instruction length")
        operands = words[cursor + 1 : cursor + length]
        cursor += length
        # Retain just the outlined sampler stub's name so WGSL generation can
        # locate its definition. Other variants let spirv-opt strip their debug data.
        if hardware_svm_textures and opcode in {2, 3, 4, 5, 6, 7, 8, 317, 330}:
            if opcode != 5:
                continue
            name = struct.pack(f"<{len(operands) - 1}I", *operands[1:]).split(b"\0", 1)[0]
            if name != SVM_SAMPLER_BRIDGE.encode():
                continue
            has_sampler_bridge = True
        if opcode == 21:  # OpTypeInt
            int_types[operands[0]] = operands[1:]
        elif opcode == 23:  # OpTypeVector
            vector_types[operands[0]] = operands[1:]
        instructions.append((opcode, operands))

    if hardware_svm_textures and not has_sampler_bridge:
        raise ValueError("Missing hardware SVM sampler bridge in SPIR-V")

    header = list(words[:5])
    added_types = []
    ints = {definition: type_id for type_id, definition in int_types.items()}
    vectors = {definition: type_id for type_id, definition in vector_types.items()}

    def new_id():
        identifier = header[3]
        header[3] += 1
        return identifier

    def ensure_type(types, definition, opcode):
        if definition not in types:
            types[definition] = new_id()
            added_types.append((opcode, (types[definition], *definition)))
        return types[definition]

    private_pointer_types = {operands[0] for opcode, operands in instructions
                             if opcode == 32 and operands[1] == 7}  # Function storage
    private_pointers = {operands[1] for opcode, operands in instructions
                        if opcode in {59, 65, 66} and operands[0] in private_pointer_types}
    integer_value_types = {}
    in_function = False
    for opcode, operands in instructions:
        if opcode == 54:  # OpFunction
            in_function = True
        elif opcode == 56:  # OpFunctionEnd
            in_function = False
        # Integer operands come from typed instructions or global constants
        # (including specialization constants and undef). A type ID in the
        # first operand of a function instruction identifies its result type.
        if (in_function or opcode == 1 or 41 <= opcode <= 52) and len(operands) >= 2:
            type_id, value_id = operands[:2]
            component, _ = vector_types.get(type_id, (type_id, 0))
            if component in int_types:
                integer_value_types[value_id] = type_id

    def signed_integer_type(type_id):
        component, length = vector_types.get(type_id, (type_id, 0))
        width, signed = int_types[component]
        if signed:
            return type_id
        signed_type = ensure_type(ints, (width, 1), 21)
        if length:
            signed_type = ensure_type(vectors, (signed_type, length), 23)
        return signed_type

    # SVM's packed byte fields may survive optimization as unsigned scalar
    # SSA values. WGSL has no u8. Preserve truncation with an explicit mask.
    byte_types = {type_id for type_id, definition in int_types.items() if definition == (8, 0)}
    byte_mask = None
    if byte_types:
        uint_type = ints[(32, 0)]

    def truncate_byte(result, value):
        nonlocal byte_mask
        if byte_mask is None:
            byte_mask = new_id()
            added_types.append((43, (uint_type, byte_mask, 255)))  # OpConstant
        return 199, (uint_type, result, value, byte_mask)  # OpBitwiseAnd

    normalized = []
    first_function = None
    for opcode, operands in instructions:
        if (opcode == 62 and len(operands) > 2 and operands[0] in private_pointers and
                operands[2] in {1, 3}):  # OpStore Volatile[|Aligned]
            # This qualifier keeps LLVM's local struct accesses scalar. Naga's
            # SPIR-V importer does not read the extra memory operands.
            operands = operands[:2]
        if (opcode == 61 and len(operands) > 3 and operands[2] in private_pointers and
                operands[3] in {1, 3}):  # OpLoad Volatile[|Aligned]
            operands = operands[:3]
        if byte_types:
            if (opcode == 21 and operands[0] in byte_types) or (opcode == 17 and operands == (39,)):
                continue  # OpTypeInt and OpCapability Int8
            if opcode == 28 and operands[1] in byte_types:  # OpTypeArray
                # Private closure storage retains C++ padding as byte arrays.
                # The shader addresses typed members and never reads the padding.
                operands = (operands[0], uint_type, operands[2])
            if opcode == 113:  # OpUConvert
                result_type, result, value = operands
                if result_type in byte_types and integer_value_types[value] == uint_type:
                    opcode, operands = truncate_byte(result, value)
                elif result_type == uint_type and integer_value_types[value] in byte_types:
                    opcode = 83  # OpCopyObject
            elif opcode in {128, 196} and operands[0] in byte_types:  # OpIAdd, OpShiftLeftLogical
                # Preserve byte truncation for SVM stack offsets and feature flags.
                _, result, *arguments = operands
                widened = new_id()
                normalized.append((opcode, (uint_type, widened, *arguments)))
                opcode, operands = truncate_byte(result, widened)
            elif opcode in {1, 43, 46, 83, 169, 194, 197, 198, 199, 245} and operands[0] in byte_types:
                # Undef, constant, null, copy, select, logical shift, bitwise ops, and phi.
                operands = (uint_type, *operands[1:])
        # SPIR-V permits unsigned integer types in signed instructions. Naga
        # uses the type's signedness to emit WGSL, so make it explicit.
        if opcode == 110:  # OpConvertFToS
            result_type, result, value = operands
            signed_type = signed_integer_type(result_type)
            if signed_type != result_type:
                signed_result = new_id()
                normalized.append((opcode, (signed_type, signed_result, value)))
                opcode, operands = 124, (result_type, result, signed_result)  # OpBitcast
        elif opcode == 111:  # OpConvertSToF
            result_type, result, value = operands
            value_type = integer_value_types[value]
            signed_type = signed_integer_type(value_type)
            if signed_type != value_type:
                signed_value = new_id()
                normalized.append((124, (signed_type, signed_value, value)))  # OpBitcast
                operands = (result_type, result, signed_value)
        elif opcode in {135, 138, 139, 195}:  # OpSDiv, OpSRem, OpSMod, OpShiftRightArithmetic
            result_type, result, left, right = operands
            signed_type = signed_integer_type(result_type)
            def signed_operand(value):
                if integer_value_types[value] == signed_type:
                    return value
                identifier = new_id()
                normalized.append((124, (signed_type, identifier, value)))  # OpBitcast
                return identifier

            left = signed_operand(left)
            if opcode != 195:
                right = signed_operand(right)
            if result_type != signed_type:
                signed_result = new_id()
                normalized.append((opcode, (signed_type, signed_result, left, right)))
                opcode, operands = 124, (result_type, result, signed_result)
            else:
                operands = (signed_type, result, left, right)
        if opcode == 54 and first_function is None:  # OpFunction
            first_function = len(normalized)

        # Naga also rejects the optional Aligned operand that LLVM adds to
        # OpLoad/OpStore. Keep all other memory-access flags.
        required = {61: 3, 62: 2}.get(opcode)  # OpLoad and OpStore
        if required is not None and operands[required : required + 1] == (2,):
            if len(operands) != required + 2:
                raise ValueError("Invalid SPIR-V Aligned memory operand")
            operands = operands[:required]
        normalized.append((opcode, operands))

    if added_types:
        normalized[first_function:first_function] = added_types
    output = header
    for opcode, operands in normalized:
        output.extend([(len(operands) + 1) << 16 | opcode, *operands])
    return struct.pack(f"<{len(output)}I", *output)


def build_lowering_plugin(llvm: Path, work: Path) -> Path:
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    plugin = work / ("bvh2-lowering" + extension)
    flags = subprocess.check_output(
        [llvm / "llvm-config", "--cxxflags", "--ldflags", "--libs", "core", "analysis", "passes"],
        text=True,
    )
    run(
        llvm / "clang++",
        "-O2", "-shared", "-fPIC",
        SOURCE / "bvh2_lowering.cc",
        "-o", plugin,
        *shlex.split(flags),
    )
    return plugin


def lower_spirv(llvm: Path, work: Path, optimized: Path, plugin: Path,
                structured_cfg: bool = False, canonicalize_loops: bool = False) -> Path:
    lowered = work / "lowered.bc"
    control_flow = ("structurizecfg" if structured_cfg else
                    "reg2mem,cycles-webgpu-dispatch-cfg,mem2reg")
    if canonicalize_loops:
        # Give nested loops dedicated exits before structurizing them. Shared
        # exits can swallow back edges in Noise octaves or volume shadow traversal.
        control_flow = f"loop-simplify,lcssa,{control_flow}"
    passes = (
        "function(cycles-webgpu-hoist-resources,cycles-webgpu-legalize-values,lower-switch,"
        f"{control_flow},cycles-webgpu-legalize-values,adce),verify"
    )
    run(llvm / "opt", f"-load-pass-plugin={plugin}", f"-passes={passes}", optimized, "-o", lowered)
    return lowered


def build(options) -> None:
    options.output.mkdir(parents=True, exist_ok=True)
    extended_queries = options.extended_queries or options.advanced_queries
    headers = options.emscripten_sysroot / "include"
    cxx_flags = ("-std=c++20", "-DNDEBUG", "-O2", "-fno-math-errno", "-I", CYCLES)
    if options.volume_reference:
        cxx_flags += ("-fno-strict-aliasing", "-D__MAX_CLOSURE__=2")
    svm = (options.svm or options.svm_compact or options.sampled_emission or options.svm_query or
           options.shadow_materials)
    if svm:
        # The stock procedural texture hash uses wrapping signed arithmetic.
        cxx_flags += ("-fwrapv",)
    light = options.direct_light or options.wavefront_light
    volume_shadow = options.volume_scatter_shadow or options.volume_extinction_shadow
    absorption_shadow = options.absorption_shadow or volume_shadow
    absorption = options.absorption or absorption_shadow or options.volume_interval
    volume = options.volume_reference or options.volume_scatter
    intersect_shadow = options.intersect_shadow or options.transparent_shadow or absorption_shadow
    surface = options.surface or options.surface_compact or options.surface_query
    compact = (options.bvh2_compact or options.surface_compact or options.svm_compact or
               options.sampled_emission or options.svm_query or options.surface_query or
               intersect_shadow or options.subsurface_compact or absorption or options.volume_scatter)
    kernel = ("bvh2" if options.bvh2 or options.bvh2_compact or surface or light or
              intersect_shadow else "camera")
    dispatch = kernel
    adaptive = next((stage for stage, enabled in (
        ("check", options.adaptive_check),
        ("filter-x", options.adaptive_filter_x),
        ("filter-y", options.adaptive_filter_y),
    ) if enabled), None)
    if light:
        dispatch = "direct_light"
    elif surface:
        dispatch = "surface"
    elif options.path_update:
        dispatch = "path_update"
    elif options.shade_background:
        kernel, dispatch = "direct_light", "shade_background"
    elif options.denoise_surface:
        kernel, dispatch = "camera", "denoise_surface"
    elif options.film_surface:
        kernel, dispatch = "camera", "film_surface"
    elif options.film_ao:
        kernel, dispatch = "bvh2", "film_ao"
    elif options.sampled_emission:
        kernel, dispatch = "svm", "sampled_emission"
    elif options.svm or options.svm_compact or options.svm_query:
        kernel = dispatch = "svm"
    elif absorption:
        kernel, dispatch = "bvh2", "absorption"
    elif volume:
        kernel = "volume"
        dispatch = "volume_scatter" if options.volume_scatter else kernel
    elif options.subsurface or options.subsurface_compact:
        kernel, dispatch = "bvh2", "subsurface"
    elif adaptive:
        kernel = dispatch = "adaptive"
    elif options.adaptive_film:
        kernel = dispatch = "adaptive_film"
        cxx_flags += ("-I", CYCLES.parent / "atomic")

    adaptive_camera = ("slots" if options.adaptive_camera_slots else
                       "rays" if options.adaptive_camera_rays else None)
    shader_name = f"{dispatch.replace('_', '-')}{'-compact' if compact else ''}.wgsl"
    if options.film_ao_prepare:
        shader_name = "film-ao-prepare.wgsl"
    if options.svm_core:
        shader_name = ("svm-compact-textures.wgsl" if options.svm_textures else
                       "svm-compact-core.wgsl")
    if options.validated_svm_words:
        shader_name = shader_name.removesuffix(".wgsl") + "-validated.wgsl"
    if options.hardware_svm_textures:
        shader_name = shader_name.removesuffix(".wgsl") + "-hardware.wgsl"
    if options.sampled_emission:
        shader_name = "sampled-emission.wgsl"
    if options.svm_query:
        if options.advanced_queries:
            shader_name = "svm-query-advanced.wgsl"
        elif options.ao_only_queries:
            shader_name = "svm-query-ao-textures.wgsl"
        else:
            shader_name = "svm-query-textures.wgsl" if extended_queries else "svm-query.wgsl"
    if options.surface_query:
        shader_name = "surface-query.wgsl"
    if adaptive:
        shader_name = f"adaptive-{adaptive}.wgsl"
    elif adaptive_camera:
        shader_name = ("adaptive-camera-rays-retire.wgsl" if adaptive_camera == "rays" else
                       "adaptive-camera-slots.wgsl")
    elif options.stock_camera:
        shader_name = "stock-camera.wgsl"
    elif options.primary_camera:
        shader_name = "primary-camera.wgsl"
    elif options.wavefront_light:
        shader_name = "wavefront-light.wgsl"
    if options.defer_shadow:
        shader_name = ("wavefront-light-deferred-multi-point.wgsl" if options.point_distribution
                       else "wavefront-light-deferred.wgsl")
    if intersect_shadow:
        shader_name = ("intersect-shadow-transparent.wgsl" if options.transparent_shadow else
                       "intersect-shadow.wgsl")
    if absorption:
        shader_name = "absorption-shadow.wgsl" if options.absorption_shadow else "absorption-compact.wgsl"
    if options.volume_interval:
        shader_name = "volume-interval.wgsl"
    if options.volume_scatter:
        shader_name = "volume-scatter.wgsl"
    if volume_shadow:
        shader_name = ("volume-scatter-shadow.wgsl" if options.volume_scatter_shadow else
                       "volume-extinction-shadow.wgsl")
    if options.volume_grid:
        shader_name = shader_name.removesuffix(".wgsl") + "-grid.wgsl"
    if options.volume_guides:
        shader_name = shader_name.removesuffix(".wgsl") + "-guides.wgsl"
    if options.shadow_materials:
        shader_name = (shader_name.removesuffix(".wgsl") + "-materials.wgsl" if volume_shadow else
                       "intersect-shadow-transparent-materials.wgsl")
    if options.reflection_only:
        shader_name = shader_name.removesuffix(".wgsl") + "-reflection.wgsl"

    if options.light_tree:
        shader_name = shader_name.removesuffix(".wgsl") + "-tree.wgsl"

    if options.denoise_background:
        shader_name = shader_name.removesuffix(".wgsl") + "-guides.wgsl"
    if options.standalone_closures:
        shader_name = shader_name.removesuffix(".wgsl") + "-closures.wgsl"
    if options.hair_closures:
        shader_name = shader_name.removesuffix(".wgsl") + "-hair.wgsl"
    if options.volume_forward:
        shader_name = shader_name.removesuffix(".wgsl") + "-volume.wgsl"

    shader_flags = []
    if options.validated_svm_words:
        shader_flags.append("-DCYCLES_WEBGPU_VALIDATED_SVM_WORDS=1")
    ordered_closures = (options.svm_compact or options.svm_query or options.wavefront_light or
                        options.film_surface or
                        (options.standalone_closures and (options.path_update or options.denoise_surface)))
    if ordered_closures:
        shader_flags.append("-DCYCLES_WEBGPU_ORDERED_CLOSURES=1")
    if options.volume_grid:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_GRID=1")
    if options.volume_guides:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_GUIDES=1")
    if options.volume_forward:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_FORWARD=1")
    if (options.path_update or options.denoise_surface) and not options.standalone_closures:
        shader_flags.append("-DCYCLES_WEBGPU_EXCLUDE_STANDALONE=1")
    if (options.wavefront_light or options.shade_background) and not options.hair_closures:
        shader_flags.append("-DCYCLES_WEBGPU_EXCLUDE_HAIR=1")
    if options.svm_core or options.svm_query or options.shadow_materials:
        shader_flags.append("-DCYCLES_WEBGPU_SVM_CORE=1")
    if extended_queries or options.shadow_materials or options.svm_textures:
        shader_flags.append("-DCYCLES_WEBGPU_SVM_TEXTURES=1")
    if extended_queries or options.svm_textures:
        shader_flags.append("-DCYCLES_WEBGPU_SVM_PROJECTED_IMAGES=1")
    if options.hardware_svm_textures:
        shader_flags.append("-DCYCLES_WEBGPU_HARDWARE_SVM_TEXTURES=1")
    if options.advanced_queries:
        shader_flags.append("-DCYCLES_WEBGPU_SVM_ADVANCED_QUERIES=1")
    if options.sampled_emission:
        shader_flags.extend(("-DCYCLES_WEBGPU_SAMPLED_EMISSION=1", "-DCYCLES_WEBGPU_SURFACE=1"))
    if options.svm_query:
        shader_flags.extend(("-DCYCLES_WEBGPU_SCENE_QUERIES=1", "-DCYCLES_WEBGPU_SURFACE=1"))
    if options.ao_only_queries:
        shader_flags.append("-DCYCLES_WEBGPU_AO_ONLY_QUERIES=1")
    if options.surface_query:
        shader_flags.append("-DCYCLES_WEBGPU_QUERY_HITS=1")
    if options.denoise_background:
        shader_flags.append("-DCYCLES_WEBGPU_DENOISE_BACKGROUND=1")
    if options.film_ao:
        shader_flags.extend(("-DCYCLES_WEBGPU_FILM_AO=1",
                             "-DCYCLES_WEBGPU_INTERSECT_SHADOW=1",
                             "-DCYCLES_WEBGPU_TRANSPARENT_SHADOW=1",
                             "-DCYCLES_PRIMARY_RAY_QUEUE=1",
                             "-DCYCLES_WEBGPU_COMPACT_PATHS=1"))
    if options.film_ao_prepare:
        shader_flags.append("-DCYCLES_WEBGPU_FILM_AO_PREPARE=1")
    if options.film_surface:
        shader_flags.extend(("-DCYCLES_WEBGPU_FILM_SURFACE=1",
                             "-DCYCLES_PRIMARY_RAY_QUEUE=1",
                             "-DCYCLES_WEBGPU_EXCLUDE_HAIR=1",
                             "-DCYCLES_WEBGPU_COMPACT_PATHS=1"))
    if options.denoise_surface:
        shader_flags.extend(("-DCYCLES_WEBGPU_DENOISE_SURFACE=1",
                             "-DCYCLES_WEBGPU_COMPACT_PATHS=1"))
    if options.light_tree or options.volume_scatter:
        shader_flags.append("-DCYCLES_WEBGPU_LIGHT_TREE=1")
    if options.point_distribution:
        shader_flags.append("-DCYCLES_WEBGPU_POINT_DISTRIBUTION=1")
    if options.defer_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_DEFER_SHADOW=1")
    if intersect_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_INTERSECT_SHADOW=1")
    if absorption:
        shader_flags.append("-DCYCLES_WEBGPU_ABSORPTION=1")
    if absorption_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_ABSORPTION_SHADOW=1")
    if options.volume_interval:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_INTERVAL=1")
    if volume_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW=1")
    if options.volume_scatter_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_VOLUME_SCATTER_SHADOW=1")
    if options.volume_scatter:
        shader_flags.extend(("-DCYCLES_WEBGPU_VOLUME_SCATTER=1",
                             "-DCYCLES_WEBGPU_WAVEFRONT_LIGHT=1", "-DCYCLES_WEBGPU_DEFER_SHADOW=1",
                             "-DCYCLES_WEBGPU_EXCLUDE_HAIR=1"))
    if options.transparent_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_TRANSPARENT_SHADOW=1")
    if options.shadow_materials:
        shader_flags.extend(("-DCYCLES_WEBGPU_SHADOW_SVM=1", "-DCYCLES_WEBGPU_SURFACE=1"))
        if volume_shadow:
            shader_flags.append("-DCYCLES_WEBGPU_VOLUME_SHADOW_MATERIALS=1")
            shader_flags.append("-DCYCLES_WEBGPU_COMPACT_MATERIAL_SHADOWS=1")
    if options.bvh2_compact:
        shader_flags.append("-DCYCLES_WEBGPU_OFFSET_CLOSEST=1")
    if options.subsurface or options.subsurface_compact:
        shader_flags.append("-DCYCLES_WEBGPU_SUBSURFACE=1")
    if adaptive:
        shader_flags.append(f"-DCYCLES_WEBGPU_ADAPTIVE_{adaptive.replace('-', '_').upper()}=1")
    if options.reflection_only:
        shader_flags.append("-DCYCLES_WEBGPU_REFLECTION_ONLY=1")
    if (options.primary_camera or options.stock_camera or adaptive_camera or options.path_update or
            options.denoise_surface):
        shader_flags.append("-DCYCLES_PRIMARY_RAY_QUEUE=1")
    if options.primary_camera or options.stock_camera or options.adaptive_camera_rays:
        shader_flags.append("-DCYCLES_WEBGPU_CAMERA_SAMPLE=1")
        cxx_flags += ("-I", CYCLES.parent / "atomic")
    if options.stock_camera or adaptive_camera:
        shader_flags.append("-DCYCLES_WEBGPU_STOCK_CAMERA=1")
    if adaptive_camera:
        shader_flags.append(f"-DCYCLES_WEBGPU_ADAPTIVE_CAMERA_{adaptive_camera.upper()}=1")
    if options.path_update:
        shader_flags.append("-DCYCLES_WEBGPU_PATH_UPDATE=1")
    if surface:
        shader_flags.append("-DCYCLES_WEBGPU_SURFACE=1")
    if compact or options.wavefront_light or options.path_update or options.adaptive_film:
        shader_flags.append("-DCYCLES_WEBGPU_COMPACT_PATHS=1")
    if options.wavefront_light or options.shade_background:
        shader_flags.append("-DCYCLES_WEBGPU_WAVEFRONT_LIGHT=1")
    if options.shade_background:
        shader_flags.extend(("-DCYCLES_WEBGPU_SHADE_BACKGROUND=1",
                             "-DCYCLES_WEBGPU_COMPACT_PATHS=1",
                             "-DCYCLES_PRIMARY_RAY_QUEUE=1"))
    structured_dispatch = (kernel != "camera" or options.primary_camera or
                           options.stock_camera or adaptive_camera or options.path_update or
                           options.denoise_surface or options.film_surface)
    with tempfile.TemporaryDirectory(prefix="cycles-webgpu-", dir=options.output) as temporary:
        work = Path(temporary)
        wrapper_ir = work / "dispatch.ll"
        build_cxx_flags = cxx_flags
        if (extended_queries or options.shadow_materials or options.svm_textures or
                options.sampled_emission or (options.svm_compact and not options.svm_core)):
            # Share the stock Perlin arithmetic across SVM call sites.
            noise_path = Path("kernel/svm/noise.h")
            if options.sampled_emission:
                stock_closure = (CYCLES / "kernel/svm/closure.h").read_text()
                closure_skip = re.search(
                    r"ccl_device_inline int svm_node_closure_bsdf_skip\(.*?^}", stock_closure,
                    re.M | re.S)
                if closure_skip is None:
                    raise ValueError("Cannot find stock SVM closure bytecode skip")
                header = work / "kernel/svm/webgpu_sampled_closure_skip.h"
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text("CCL_NAMESPACE_BEGIN\n" + closure_skip[0] +
                                  "\nCCL_NAMESPACE_END\n")
            stock_noise = (CYCLES / noise_path).read_text()
            outlined, count = re.subn(r"ccl_device_noinline_cpu float perlin_[1-4]d\(",
                                      r"__attribute__((noinline)) \g<0>", stock_noise)
            if count < 4:
                raise ValueError("Cannot find the four stock Perlin noise dimensions")
            header = work / noise_path
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text(outlined)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.light_tree or options.volume_scatter:
            from light_tree_headers import prepare_light_tree_headers

            overlay = prepare_light_tree_headers(work, CYCLES)
            build_cxx_flags = ("-I", overlay, *cxx_flags)
        if absorption:
            from volume_headers import prepare_absorption_headers

            prepare_absorption_headers(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.volume_reference:
            from volume_headers import prepare_volume_headers

            prepare_volume_headers(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.volume_scatter or volume_shadow:
            from volume_headers import prepare_volume_headers

            volume_overlay = work / "volume"
            volume_overlay.mkdir()
            prepare_volume_headers(volume_overlay, CYCLES, scattering=options.volume_scatter)
            if options.volume_grid:
                from volume_grid_headers import prepare_volume_grid_headers

                prepare_volume_grid_headers(volume_overlay, CYCLES)
        if options.subsurface or options.subsurface_compact:
            from subsurface_headers import prepare_subsurface_headers

            prepare_subsurface_headers(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if (options.path_update or options.wavefront_light or options.denoise_surface or
                options.film_surface):
            from bump_headers import prepare_bump_header

            prepare_bump_header(work, CYCLES)
        if options.denoise_surface:
            from denoising_headers import prepare_denoising_header

            overlay = prepare_denoising_header(work, CYCLES)
            build_cxx_flags = ("-I", overlay, *build_cxx_flags)
        if options.volume_scatter_shadow and options.volume_guides:
            from denoising_headers import prepare_volume_denoising_header

            prepare_volume_denoising_header(work, CYCLES)
        if options.path_update:
            from transparent_headers import prepare_transparent_ray_header

            prepare_transparent_ray_header(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.path_update or options.film_surface:
            # A purely reflective Fresnel closure leaves cos_HO unused. Initialize
            # it so LLVM cannot fold the unsampled refraction branch to a WGSL NaN.
            microfacet_path = Path("kernel/closure/bsdf_microfacet.h")
            source = (CYCLES / microfacet_path).read_text()
            declaration = "  float cos_HO;"
            if source.count(declaration) != 1:
                raise ValueError("Expected one stock microfacet refraction cosine")
            source = source.replace(declaration, "  float cos_HO = 0.0f;")
            if options.path_update:
                # Packed surface closures have zero film thickness. LLVM's
                # SPIR-V backend cannot lower the conductor iridescence
                # temporaries that survive in the generic sampler.
                thin_film_guard = "if (fresnel->thin_film.thickness > THINFILM_THICKNESS_CUTOFF)"
                if source.count(thin_film_guard) != 6:
                    raise ValueError("Expected six stock Fresnel thin-film guards")
                source = source.replace(thin_film_guard, "if constexpr (false)")
            header = work / microfacet_path
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text(source)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.standalone_closures or options.wavefront_light:
            from hair_headers import prepare_hair_headers

            prepare_hair_headers(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.svm_query:
            from scene_query_headers import prepare_scene_query_headers

            overlay = prepare_scene_query_headers(work, CYCLES, outline_noise=extended_queries)
            build_cxx_flags = ("-I", overlay, *build_cxx_flags)
        run(
            options.llvm_bin / "clang",
            "--no-default-config",
            "-x", "hlsl",
            "-target", "spirv-unknown-vulkan-compute",
            "-Xclang", "-hlsl-entry", "-Xclang", "main",
            *shader_flags,
            "-O2", "-S", "-emit-llvm",
            "-I", SOURCE,
            "-I", work,
            SOURCE / ("subsurface_compact_dispatch.hlsl" if options.subsurface_compact
                      else f"{dispatch}_dispatch.hlsl"),
            "-o", wrapper_ir,
        )
        shader_modules = [wrapper_ir]
        kernels = ([dispatch] if options.defer_shadow else
                   [kernel, dispatch] if light else [kernel])
        if options.shade_background or options.film_ao:
            kernels.append("camera")
        if options.volume_scatter:
            kernels.append("direct_light")
        if volume_shadow:
            kernels.append("volume")
        if options.sampled_emission or options.svm_query:
            kernels.append("bvh2")
        if options.shadow_materials:
            kernels.append("svm")
        if options.wavefront_light or options.path_update or options.denoise_surface or options.film_surface:
            kernels.append("ggx")
        # Keep scanline induction variables at 32 bits for the WebGPU backend.
        hair_bsdf = options.standalone_closures or options.hair_closures
        narrow_indvars = (adaptive or hair_bsdf or options.svm_query or options.transparent_shadow or
                          options.subsurface_compact or volume or absorption)
        narrow_cxx_flags = ("-mllvm", "-indvars-widen-indvars=false") if narrow_indvars else ()
        for name in kernels:
            cpp_ir = work / f"{name}-kernel.ll"
            shader_ir = work / f"{name}-shader.ll"
            module_flags = build_cxx_flags
            if (options.volume_scatter or volume_shadow) and name == "volume":
                module_flags = ("-I", volume_overlay, "-fno-strict-aliasing",
                                "-D__MAX_CLOSURE__=8", *module_flags)
            run(
                options.llvm_bin / "clang++",
                "--no-default-config",
                "-target", "wasm64-unknown-emscripten",
                "-msimd128",
                "-isystem", headers / "c++" / "v1",
                "-isystem", headers,
                "-D_POSIX_C_SOURCE=200809L",
                "-fno-vectorize", "-fno-slp-vectorize",
                *module_flags,
                *narrow_cxx_flags,
                *shader_flags,
                "-S", "-emit-llvm",
                SOURCE / f"{name}_kernels.cc",
                "-o", cpp_ir,
            )
            retarget(cpp_ir, wrapper_ir, shader_ir)
            shader_modules.append(shader_ir)
        run(options.llvm_bin / "llvm-link", *shader_modules, "-o", work / "linked.bc")
        optimized = work / "optimized.bc"
        plugin = build_lowering_plugin(options.llvm_bin, work) if structured_dispatch else None
        if adaptive:
            film_accessors = "main,_Z25cycles_adaptive_film_readj,_Z26cycles_adaptive_film_writejj"
            run(
                options.llvm_bin / "opt",
                f"-load-pass-plugin={plugin}",
                f"-internalize-public-api-list={film_accessors}",
                "-indvars-widen-indvars=false",
                "-passes=internalize,default<O2>,cycles-webgpu-lower-film,default<O2>,globaldce",
                work / "linked.bc",
                "-o", optimized,
            )
        else:
            # Keep storage handles and the private closure payloads within the entry point.
            inline_all = (svm or light or options.shade_background or options.path_update or
                          options.denoise_surface or options.film_surface or options.film_ao or options.transparent_shadow or
                          options.subsurface_compact or volume or absorption)
            opt_plugin = (f"-load-pass-plugin={plugin}",) if inline_all else ()
            inline_helpers = "cycles-webgpu-inline-helpers,always-inline," if inline_all else ""
            # SPIR-V cannot index a vector load widened from the scalar SVM stack.
            scalar_stack = ("-vectorize-slp=false",) if (
                svm or options.transparent_shadow or options.subsurface_compact or
                volume or absorption) else ()
            narrow_indices = ("-indvars-widen-indvars=false",) if narrow_indvars else ()
            run(
                options.llvm_bin / "opt",
                *opt_plugin,
                *scalar_stack,
                *narrow_indices,
                "-internalize-public-api-list=main",
                f"-passes=internalize,{inline_helpers}default<O2>,globaldce",
                work / "linked.bc",
                "-o", optimized,
            )
        shader = optimized
        if structured_dispatch:
            # Keep the original branches and loops in traversal and shading.
            # Dispatching each block through a switch keeps excessive
            # per-lane state live on Metal.
            shader = lower_spirv(options.llvm_bin, work, optimized, plugin,
                                 structured_cfg=(options.shade_background or options.adaptive_film or
                                                 options.bvh2_compact or intersect_shadow or
                                                 options.surface_compact or options.surface_query or svm or
                                                 options.path_update or options.wavefront_light or
                                                 options.denoise_surface or options.film_surface or options.film_ao or options.subsurface or
                                                 options.subsurface_compact or
                                                 volume or absorption),
                                 canonicalize_loops=svm or volume_shadow)
        run(
            options.llvm_bin / "llc",
            "-O0" if structured_dispatch else "-O2", "-filetype=obj",
            # Disable vector_select_to_faceforward_lowering (rule 1 in LLVM 22.1.8).
            "-spirvprelegalizercombiner-disable-rule=1",
            shader,
            "-o", work / "kernel.spv",
        )
        # LLVM retains names of resource bindings as unused 8-bit constants.
        # Optimize those away before validation.
        strip_debug = () if options.hardware_svm_textures else ("--strip-debug",)
        run(
            options.spirv_bin / "spirv-opt",
            "--target-env=vulkan1.2",
            *strip_debug, "-O", "--trim-capabilities", "--skip-validation",
            work / "kernel.spv",
            "-o", work / "optimized.spv",
        )
        normalized = work / "normalized.spv"
        normalized.write_bytes(normalize_spirv((work / "optimized.spv").read_bytes(),
                                              options.hardware_svm_textures))
        if options.svm_query:
            from scene_query_headers import narrow_scene_query_indices

            query = work / "query-indexing.spv"
            query.write_bytes(narrow_scene_query_indices(normalized.read_bytes()))
            run(options.spirv_bin / "spirv-opt", "--target-env=vulkan1.2",
                "--eliminate-dead-code-aggressive", "--eliminate-dead-const", "--trim-capabilities",
                query, "-o", normalized)
        run(options.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", normalized)
        run("node", SOURCE / "translate.mjs", options.translator, normalized, work / shader_name)
        if options.hardware_svm_textures:
            shader_path = work / shader_name
            sampler = (SOURCE / "svm-hardware-textures.wgsl").read_text()
            shader, count = re.subn(
                rf"fn {SVM_SAMPLER_BRIDGE}\([^\n]*\) -> vec4<f32> \{{[^{{}}]*\}}",
                lambda _: sampler,
                shader_path.read_text(),
            )
            if count != 1:
                raise ValueError("Expected one outlined SVM atlas sampler in generated WGSL")
            shader_path.write_text("// Cycles WebGPU hardware texture atlas\n" + shader)
        if options.validated_svm_words:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU validated SVM node bounds\n")
        if extended_queries:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU projected query images\n")
        if options.advanced_queries:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU advanced procedural queries\n")
        if svm and not (options.sampled_emission or options.shadow_materials):
            shader_path = work / shader_name
            shader_path.write_text("// Cycles WebGPU AOV output\n" + shader_path.read_text())
        if options.sampled_emission:
            # LLVM's SPIR-V backend emits writable storage even for HLSL StructuredBuffer.
            # Match the explicit WebGPU bind group layout: only direct radiance is writable.
            shader_path = work / shader_name
            source = shader_path.read_text()
            for binding in (0, 1, 2, 4, 5, 6, 7):
                source, count = re.subn(
                    rf"(@group\(0\)\s+@binding\({binding}\)\s+var<storage, )read_write(?=>)",
                    r"\g<1>read", source,
                )
                if count != 1:
                    raise ValueError(f"Expected one sampled emission declaration at binding {binding}")
            shader_path.write_text(source)
        if (options.wavefront_light or options.path_update or options.shade_background or
                intersect_shadow or options.denoise_surface or options.svm_query):
            shader_path = work / shader_name
            source = shader_path.read_text()
            coalesced = coalesce_loop_state(source)
            if options.light_tree:
                textured = ("// Cycles Light Tree textured emitters\n"
                            if options.wavefront_light else "")
                coalesced = ("// Cycles Light Tree finite point\n"
                             "// Cycles Light Tree triangle emitters\n" + textured + coalesced)
            if intersect_shadow:
                coalesced = "// Cycles WebGPU shadow targets\n" + coalesced
            if coalesced != source:
                shader_path.write_text(coalesced)
        if svm and not (options.sampled_emission or options.shadow_materials):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU reflective caustics visibility\n")
                shader.write("// Cycles WebGPU glass caustics visibility\n")
                shader.write("// Cycles WebGPU compact Voronoi Distance\n")
                shader.write("// Cycles WebGPU weighted diffuse Principled mixtures\n")
        if options.svm_compact or options.svm_query or options.film_surface:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Glossy displacement Bump\n")
                shader.write("// Cycles WebGPU Diffuse and Glossy displacement Bump\n")
        if svm and not (options.sampled_emission or options.shadow_materials):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU residual displacement bump\n")
        if (options.svm_compact or options.svm_query or options.shadow_materials or
                options.wavefront_light or options.standalone_closures or options.film_surface):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Glossy and Transparent\n")
        if options.wavefront_light and options.light_tree:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU lamp sampling flags\n")
        if ((options.wavefront_light or options.shade_background) and options.light_tree or
                options.volume_scatter):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU analytic Light Falloff\n")
                shader.write("// Cycles WebGPU ray-length lamp emission\n")
                shader.write("// Cycles WebGPU mapped world SVM\n")
        if (options.svm_compact or options.svm_query or options.sampled_emission or
                options.shadow_materials):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Transparent emission\n")
        if (options.svm_compact or options.svm_query or options.shadow_materials or
                options.wavefront_light or options.path_update or options.film_surface or
                options.denoise_surface):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU closure compositions\n")
        if (options.svm_compact or options.svm_query or options.wavefront_light or
                options.path_update or options.film_surface or options.denoise_surface):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU single Principled Coat\n")
        if options.svm_compact or options.svm_query or options.sampled_emission:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU coated Principled emission\n")
        if (svm and not options.shadow_materials or options.wavefront_light or
                options.standalone_closures):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU standalone surface closures v2\n")
        if (svm and not options.shadow_materials or options.wavefront_light or
                options.standalone_closures or options.film_surface):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU mixed Diffuse closures v4\n")
                shader.write("// Cycles WebGPU independent mixed Bump normals\n")
        if ordered_closures:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU ordered closures\n")
                shader.write("// Cycles WebGPU integer ordered tail indices\n")
                shader.write("// Cycles WebGPU paired Principled layers with metallic\n")
        if ordered_closures or options.shadow_materials:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU ordered Transparent and Translucent closures\n")
        if svm or surface or options.hair_closures or options.standalone_closures:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU hair BSDFs\n")
        if svm or options.subsurface_compact or options.path_update or options.denoise_surface:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Christensen-Burley subsurface\n")
        if (options.svm_compact or options.svm_query or options.subsurface_compact):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU subsurface bump normals\n")
        if options.path_update:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU homogeneous absorption bounds\n")
        if options.surface_compact or options.surface_query or options.denoise_surface:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU volume guide surface hits\n")
        if svm or options.path_update or options.denoise_surface or options.shade_background:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU constant transparent surface\n")
        if svm or options.path_update or options.transparent_shadow:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU varying transparent surface\n")
        if svm or options.path_update or options.wavefront_light or options.denoise_surface:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU mixed Transparent surface closures v2\n")
        if (svm or options.path_update or options.denoise_surface or options.shade_background or
                options.wavefront_light or options.transparent_shadow):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU ray portals\n")
        if options.volume_scatter or (
                options.light_tree and (options.wavefront_light or options.shade_background)):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU stock mixed-light distribution\n")
        if options.stock_camera or options.adaptive_camera_rays:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU camera lens\n")
        if (svm and not options.shadow_materials or options.path_update or
                options.stock_camera or options.adaptive_camera_rays):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU ray differentials\n")
        if options.wavefront_light or options.shade_background or options.volume_scatter:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU procedural sky world\n")
        if options.path_update or options.shade_background:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Emission pass deltas\n")
        if options.wavefront_light or options.path_update or options.shade_background:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU per-lobe film lighting\n")
        if options.film_surface or options.film_ao:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU filtered film layout 6\n")
                if options.film_surface:
                    shader.write("// Cycles WebGPU volume geometry passes\n")
        if options.shadow_materials:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU cubic transparent shadows\n")
                shader.write("// Cycles WebGPU extended shadow SVM\n")
                if not volume_shadow:
                    shader.write("// Cycles WebGPU varying Ambient Occlusion shadows\n")
                else:
                    shader.write("// Cycles WebGPU compact material shadows\n")
        if (options.volume_interval or options.volume_scatter or options.volume_extinction_shadow or
                (options.volume_scatter_shadow and options.shadow_materials) or
                (options.wavefront_light and options.volume_forward) or
                (svm and not options.sampled_emission and not options.shadow_materials)):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU mixed surface volume boundaries\n")
        if ((svm and not options.sampled_emission and not options.shadow_materials) or
                options.wavefront_light or options.path_update or options.denoise_surface or
                options.subsurface_compact or options.film_surface):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU material layout 7\n")
        if options.volume_scatter or volume_shadow:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU literal additive volume\n")
        if options.volume_interval or options.volume_scatter or volume_shadow:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU extended volume stack\n")
        if options.volume_scatter:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU varying volume shadow dispatch\n")
        if (options.volume_scatter or options.volume_forward or options.surface_compact or
                options.surface_query):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU volume ray differentials\n")
        if svm:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU Principled Alpha\n")
        if volume_shadow:
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU transparent volume shadows\n")
        if (adaptive == "check" or options.wavefront_light or options.shade_background or
                options.path_update or options.denoise_surface or absorption or
                options.volume_interval or options.volume_scatter or volume_shadow):
            with (work / shader_name).open("a") as shader:
                shader.write("\n// Cycles WebGPU render buffer ABI 2\n")
        if (compact or options.surface or options.path_update or options.wavefront_light or
                adaptive or adaptive_camera or options.adaptive_film or options.shade_background or
                options.denoise_surface or options.film_surface or options.film_ao):
            shutil.move(work / shader_name, options.output / shader_name)
            return
        if options.direct_light:
            reference = "direct-light-reference"
            reference_source = "direct_light_reference.cc"
            reference_flags = ("-DCYCLES_WEBGPU_SURFACE=1",)
        elif options.svm:
            reference = "svm-reference"
            reference_source = "svm_reference.cc"
            reference_flags = ()
        elif options.volume_reference:
            reference = "volume-reference"
            reference_source = "volume_reference.cc"
            reference_flags = ("-I", work)
        elif options.subsurface:
            reference = "subsurface-reference"
            reference_source = "subsurface_reference.cc"
            reference_flags = ("-I", work)
        else:
            reference = "bvh2-reference" if options.bvh2 else "reference"
            reference_source = "bvh2_reference.cc" if options.bvh2 else "reference.cc"
            reference_flags = ()
        run(
            options.llvm_bin / "clang++",
            *reference_flags,
            *cxx_flags,
            *shader_flags,
            *(SOURCE / f"{name}_kernels.cc" for name in kernels),
            SOURCE / reference_source,
            "-o", work / reference,
        )
        result_reference = reference
        if kernel == "camera":
            result_reference = "primary-reference.json" if options.primary_camera else "reference.json"
            reference_args = ("--primary",) if options.primary_camera else ()
            with (work / result_reference).open("w") as output:
                run(work / reference, *reference_args, stdout=output)
            if options.primary_camera:
                sampled_reference = "sampled-primary-reference.json"
                with (work / sampled_reference).open("w") as output:
                    run(work / reference, "--sampled-primary", stdout=output)
                shutil.move(work / sampled_reference, options.output / sampled_reference)
        if options.svm or options.subsurface or options.volume_reference:
            reference_json = reference + ".json"
            with (work / reference_json).open("w") as output:
                run(work / reference, stdout=output)
            shutil.move(work / reference_json, options.output / reference_json)
        for result in (shader_name, result_reference):
            shutil.move(work / result, options.output / result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--spirv-bin", required=True, type=Path)
    parser.add_argument("--emscripten-sysroot", required=True, type=Path)
    parser.add_argument("--translator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reflection-only", action="store_true",
                        help="build a smaller wavefront light or path-update shader for opaque scenes")
    parser.add_argument("--defer-shadow", action="store_true",
                        help="emit shadow rays for a separate wavefront intersection stage")
    parser.add_argument("--svm-core", action="store_true",
                        help="build the smaller SVM shader for materials without extended nodes")
    parser.add_argument("--light-tree", action="store_true",
                        help="use stock Cycles Light Tree and world importance sampling")
    parser.add_argument("--point-distribution", action="store_true",
                        help="sample point lights using the stock Cycles light distribution")
    parser.add_argument("--denoise-background", action="store_true",
                        help="write denoising albedo for background and forward light hits")
    parser.add_argument("--standalone-closures", action="store_true",
                        help="include standalone surface closures in path update or denoising")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--denoise-surface", action="store_true",
                       help="compile Cycles' denoising guides for compact surfaces")
    modes.add_argument("--intersect-shadow", action="store_true",
                       help="build opaque shadow intersection for compact queued paths")
    modes.add_argument("--transparent-shadow", action="store_true",
                       help="build compact shadow traversal with constant RGB transparency")
    modes.add_argument("--bvh2", action="store_true", help="compile static triangle BVH2 traversal")
    modes.add_argument("--stock-camera", action="store_true",
                       help="build the camera kernel for stock Cycles work tiles")
    modes.add_argument("--primary-camera", action="store_true",
                       help="compile the camera kernel with the BVH2 ray buffer layout")
    modes.add_argument("--surface", action="store_true",
                       help="compile flat triangle surface setup")
    modes.add_argument("--bvh2-compact", action="store_true",
                       help="compile BVH2 traversal over compacted paths")
    modes.add_argument("--surface-compact", action="store_true",
                       help="compile triangle surface setup over compacted paths")
    modes.add_argument("--surface-query", action="store_true",
                       help="compile compact surface setup with the SVM query intersection tail")
    modes.add_argument("--direct-light", action="store_true",
                       help="compile diffuse point lighting with opaque triangle shadows")
    modes.add_argument("--svm", action="store_true", help="compile the diffuse SVM subset")
    modes.add_argument("--svm-compact", action="store_true",
                       help="compile the SVM stage over compacted paths")
    modes.add_argument("--svm-query", action="store_true",
                       help="compile SVM with Cycles BVH ray queries over compacted paths")
    modes.add_argument("--sampled-emission", action="store_true",
                       help="shade sampled triangle emission with SVM")
    modes.add_argument("--wavefront-light", action="store_true",
                       help="compile direct lighting with per-path SVM materials")
    modes.add_argument("--film-ao", action="store_true",
                       help="compile the Ambient Occlusion render pass shadow rays")
    modes.add_argument("--film-ao-prepare", action="store_true",
                       help="prepare Ambient Occlusion rays for varying transparent shadows")
    modes.add_argument("--film-surface", action="store_true",
                       help="compile filtered Normal and closure color passes")
    modes.add_argument("--shade-background", action="store_true",
                       help="compile environment and forward area emission for miss paths")
    modes.add_argument("--path-update", action="store_true",
                       help="compile diffuse continuation and radiance accumulation")
    modes.add_argument("--absorption", action="store_true",
                       help="track homogeneous volume absorption along compacted camera paths")
    modes.add_argument("--absorption-shadow", action="store_true",
                       help="integrate homogeneous volume absorption along shadow rays")
    modes.add_argument("--volume-reference", action="store_true",
                       help="verify stock homogeneous volume SVM and phase functions")
    modes.add_argument("--volume-interval", action="store_true",
                       help="track the homogeneous scattering volume boundary")
    modes.add_argument("--volume-scatter", action="store_true",
                       help="compile stock homogeneous volume scattering and Light Tree sampling")
    modes.add_argument("--volume-scatter-shadow", action="store_true",
                       help="integrate attenuation and accumulate direct volume lighting")
    modes.add_argument("--volume-extinction-shadow", action="store_true",
                       help="integrate homogeneous volume extinction along surface shadow rays")
    modes.add_argument("--subsurface", action="store_true",
                       help="compile the stock Random Walk subsurface kernel for static triangles")
    modes.add_argument("--subsurface-compact", action="store_true",
                       help="compile Random Walk and Christensen–Burley over queued subsurface paths")
    modes.add_argument("--adaptive-camera-slots", action="store_true",
                       help="compile adaptive camera slot candidate lookup")
    modes.add_argument("--adaptive-camera-rays", action="store_true",
                       help="compile camera rays for admitted adaptive samples")
    modes.add_argument("--adaptive-check", action="store_true",
                       help="compile Cycles' adaptive convergence check")
    modes.add_argument("--adaptive-filter-x", action="store_true",
                       help="compile Cycles' horizontal convergence filter")
    modes.add_argument("--adaptive-filter-y", action="store_true",
                       help="compile Cycles' vertical convergence filter")
    modes.add_argument("--adaptive-film", action="store_true",
                       help="compile Cycles' adaptive film pass writers")
    parser.add_argument("--svm-textures", action="store_true",
                        help="include common textures and color nodes in the compact core SVM")
    parser.add_argument("--validated-svm-words", action="store_true",
                        help="use prevalidated typed SVM payloads in the compact texture shader")
    parser.add_argument("--hardware-svm-textures", action="store_true",
                        help="use hardware bicubic image filtering when the compact SVM atlas is available")
    parser.add_argument("--extended-queries", action="store_true",
                        help="include selected texture and color nodes with SVM ray queries")
    parser.add_argument("--advanced-queries", action="store_true",
                        help="include Gabor and advanced Voronoi with SVM ray queries")
    parser.add_argument("--ao-only-queries", action="store_true",
                        help="build the textured SVM query pipeline for Ambient Occlusion")
    parser.add_argument("--hair-closures", action="store_true",
                        help="include stock Hair BSDFs in the deferred direct light shader")
    parser.add_argument("--volume-forward", action="store_true",
                        help="use stock volume Light Tree MIS for forward emission")
    parser.add_argument("--volume-guides", action="store_true",
                        help="write Cycles' volume albedo and normal guides")
    parser.add_argument("--volume-grid", action="store_true",
                        help="add NanoVDB grid attributes and stock heterogeneous transport")
    parser.add_argument("--shadow-materials", action="store_true",
                        help="evaluate SVM tint at transparent shadow hits")
    options = parser.parse_args()
    options.film_ao |= options.film_ao_prepare
    if options.shadow_materials and not (options.transparent_shadow or
                                        (options.volume_scatter_shadow and options.volume_guides)):
        parser.error("--shadow-materials requires transparent or guided volume shadows")
    if options.extended_queries and not options.svm_query:
        parser.error("--extended-queries requires --svm-query")
    if options.advanced_queries and not options.svm_query:
        parser.error("--advanced-queries requires --svm-query")
    if options.ao_only_queries and not (options.extended_queries and not options.advanced_queries):
        parser.error("--ao-only-queries requires --extended-queries without --advanced-queries")
    if options.hair_closures and not (options.wavefront_light and options.defer_shadow):
        parser.error("--hair-closures requires --wavefront-light --defer-shadow")
    if options.volume_forward and not (options.light_tree and
            (options.shade_background or (options.wavefront_light and options.defer_shadow))):
        parser.error("--volume-forward requires Light Tree background or deferred lighting")
    if options.volume_guides and not (options.volume_scatter or options.volume_scatter_shadow):
        parser.error("--volume-guides requires --volume-scatter or --volume-scatter-shadow")
    if options.volume_grid and not (options.volume_scatter or options.volume_scatter_shadow or
                                    options.volume_extinction_shadow):
        parser.error("--volume-grid requires a volume transport mode")
    if options.svm_core and not options.svm_compact:
        parser.error("--svm-core requires --svm-compact")
    if options.svm_textures and not options.svm_core:
        parser.error("--svm-textures requires --svm-core")
    if options.validated_svm_words and not (options.svm_compact and options.svm_textures):
        parser.error("--validated-svm-words requires --svm-compact --svm-core --svm-textures")
    if options.hardware_svm_textures and not (options.svm_compact and options.svm_textures):
        parser.error("--hardware-svm-textures requires --svm-compact --svm-core --svm-textures")
    if options.standalone_closures and not (options.path_update or options.denoise_surface):
        parser.error("--standalone-closures requires --path-update or --denoise-surface")
    if options.reflection_only and not (options.wavefront_light or options.path_update):
        parser.error("--reflection-only requires --wavefront-light or --path-update")
    if options.defer_shadow and not options.wavefront_light:
        parser.error("--defer-shadow requires --wavefront-light")
    if options.point_distribution and not options.defer_shadow:
        parser.error("--point-distribution requires --wavefront-light --defer-shadow")
    if options.light_tree and (options.point_distribution or not (
            options.shade_background or options.defer_shadow)):
        parser.error("--light-tree requires --shade-background or deferred wavefront lighting")
    if options.denoise_background and not options.shade_background:
        parser.error("--denoise-background requires --shade-background")
    build(options)

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
        r"25cycles_camera_sample_word|15cycles_svm_word)j|23cycles_film_transparentv)\b)",
        r"\1 spir_func ",
        shader_ir,
    )
    # Direct lighting also calls the C++ traversal kernel defined in a separate module.
    shader_ir = re.sub(r"\b(call|declare) (?=void @_Z17cycles_bvh2_trace)", r"\1 spir_func ", shader_ir)
    shader_ir = re.sub(r"\b(call|declare) (?=<4 x i32> @_Z18cycles_svm_contextjj)",
                       r"\1 spir_func ", shader_ir)
    # GGX closures and their lookup accessor link across the BSDF and dispatch modules.
    shader_ir = re.sub(
        r"\b(call|declare) (?=(?:noundef )?(?:float @_Z17cycles_ggx_lookupj\b|"
        r"<4 x float> @_Z15cycles_ggx_eval|i32 @_Z17cycles_ggx_sample))",
        r"\1 spir_func ",
        shader_ir,
    )
    output.write_text(shader_ir)


def normalize_spirv(code: bytes) -> bytes:
    if len(code) < 20 or len(code) % 4:
        raise ValueError("Invalid SPIR-V binary length")
    words = struct.unpack(f"<{len(code) // 4}I", code)
    if words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V magic")
    instructions = []
    int_types = {}
    vector_types = {}
    cursor = 5
    while cursor < len(words):
        length, opcode = words[cursor] >> 16, words[cursor] & 0xFFFF
        if length == 0 or cursor + length > len(words):
            raise ValueError("Invalid SPIR-V instruction length")
        operands = words[cursor + 1 : cursor + length]
        if opcode == 21:  # OpTypeInt
            int_types[operands[0]] = operands[1:]
        elif opcode == 23:  # OpTypeVector
            vector_types[operands[0]] = operands[1:]
        instructions.append((opcode, operands))
        cursor += length

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
        if byte_types:
            if (opcode == 21 and operands[0] in byte_types) or (opcode == 17 and operands == (39,)):
                continue  # OpTypeInt and OpCapability Int8
            if opcode == 113:  # OpUConvert
                result_type, result, value = operands
                if result_type in byte_types and integer_value_types[value] == uint_type:
                    opcode, operands = truncate_byte(result, value)
                elif result_type == uint_type and integer_value_types[value] in byte_types:
                    opcode = 83  # OpCopyObject
            elif opcode == 128 and operands[0] in byte_types:  # OpIAdd
                # Keep byte overflow for packed SVM stack offset comparisons.
                _, result, *arguments = operands
                widened = new_id()
                normalized.append((opcode, (uint_type, widened, *arguments)))
                opcode, operands = truncate_byte(result, widened)
            elif opcode in {1, 43, 46, 83, 169, 197, 198, 199, 245} and operands[0] in byte_types:
                # Undef, constant, null, copy, select, bitwise or/xor/and, and phi.
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
        # Give the fractal loops dedicated exits before structurizing them.
        # A shared exit can turn the 3D Hybrid octave back edge into an exit.
        control_flow = f"loop-simplify,lcssa,{control_flow}"
    passes = (
        "function(cycles-webgpu-hoist-resources,cycles-webgpu-legalize-values,lower-switch,"
        f"{control_flow},cycles-webgpu-legalize-values,adce),verify"
    )
    run(llvm / "opt", f"-load-pass-plugin={plugin}", f"-passes={passes}", optimized, "-o", lowered)
    return lowered


def build(options) -> None:
    options.output.mkdir(parents=True, exist_ok=True)
    headers = options.emscripten_sysroot / "include"
    cxx_flags = ("-std=c++20", "-DNDEBUG", "-O2", "-fno-math-errno", "-I", CYCLES)
    svm = options.svm or options.svm_compact
    if svm:
        # The stock procedural texture hash uses wrapping signed arithmetic.
        cxx_flags += ("-fwrapv",)
    light = options.direct_light or options.wavefront_light
    surface = options.surface or options.surface_compact
    compact = (options.bvh2_compact or options.surface_compact or options.svm_compact or
               options.intersect_shadow or options.subsurface_compact)
    kernel = ("bvh2" if options.bvh2 or options.bvh2_compact or surface or light or
              options.intersect_shadow else "camera")
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
    elif options.svm or options.svm_compact:
        kernel = dispatch = "svm"
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
    if options.svm_core:
        shader_name = "svm-compact-core.wgsl"
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
    if options.intersect_shadow:
        shader_name = "intersect-shadow.wgsl"
    if options.reflection_only:
        shader_name = shader_name.removesuffix(".wgsl") + "-reflection.wgsl"

    if options.light_tree:
        shader_name = shader_name.removesuffix(".wgsl") + "-tree.wgsl"

    if options.denoise_background:
        shader_name = shader_name.removesuffix(".wgsl") + "-guides.wgsl"

    shader_flags = []
    if options.svm_core:
        shader_flags.append("-DCYCLES_WEBGPU_SVM_CORE=1")
    if options.denoise_background:
        shader_flags.append("-DCYCLES_WEBGPU_DENOISE_BACKGROUND=1")
    if options.denoise_surface:
        shader_flags.extend(("-DCYCLES_WEBGPU_DENOISE_SURFACE=1",
                             "-DCYCLES_WEBGPU_COMPACT_PATHS=1"))
    if options.light_tree:
        shader_flags.append("-DCYCLES_WEBGPU_LIGHT_TREE=1")
    if options.point_distribution:
        shader_flags.append("-DCYCLES_WEBGPU_POINT_DISTRIBUTION=1")
    if options.defer_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_DEFER_SHADOW=1")
    if options.intersect_shadow:
        shader_flags.append("-DCYCLES_WEBGPU_INTERSECT_SHADOW=1")
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
                           options.denoise_surface)
    with tempfile.TemporaryDirectory(prefix="cycles-webgpu-", dir=options.output) as temporary:
        work = Path(temporary)
        wrapper_ir = work / "dispatch.ll"
        build_cxx_flags = cxx_flags
        if options.light_tree:
            from light_tree_headers import prepare_light_tree_headers

            overlay = prepare_light_tree_headers(work, CYCLES)
            build_cxx_flags = ("-I", overlay, *cxx_flags)
        if options.subsurface or options.subsurface_compact:
            from subsurface_headers import prepare_subsurface_headers

            prepare_subsurface_headers(work, CYCLES)
            build_cxx_flags = ("-I", work, *build_cxx_flags)
        if options.denoise_surface:
            from denoising_headers import prepare_denoising_header

            overlay = prepare_denoising_header(work, CYCLES)
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
            SOURCE / ("subsurface_compact_dispatch.hlsl" if options.subsurface_compact
                      else f"{dispatch}_dispatch.hlsl"),
            "-o", wrapper_ir,
        )
        shader_modules = [wrapper_ir]
        kernels = ([dispatch] if options.defer_shadow else
                   [kernel, dispatch] if light else [kernel])
        if options.shade_background:
            kernels.append("camera")
        if options.wavefront_light or options.path_update or options.denoise_surface:
            kernels.append("ggx")
        # Keep scanline induction variables at 32 bits for the WebGPU backend.
        adaptive_cxx_flags = ("-mllvm", "-indvars-widen-indvars=false") if adaptive else ()
        for name in kernels:
            cpp_ir = work / f"{name}-kernel.ll"
            shader_ir = work / f"{name}-shader.ll"
            run(
                options.llvm_bin / "clang++",
                "--no-default-config",
                "-target", "wasm64-unknown-emscripten",
                "-msimd128",
                "-isystem", headers / "c++" / "v1",
                "-isystem", headers,
                "-D_POSIX_C_SOURCE=200809L",
                "-fno-vectorize", "-fno-slp-vectorize",
                *build_cxx_flags,
                *adaptive_cxx_flags,
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
            # Private C++ aggregates cannot cross function calls in WGSL.
            opt_plugin = (f"-load-pass-plugin={plugin}",) if svm else ()
            inline_helpers = "cycles-webgpu-inline-helpers,always-inline," if svm else ""
            run(
                options.llvm_bin / "opt",
                *opt_plugin,
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
                                                 options.bvh2_compact or options.intersect_shadow or
                                                 options.surface_compact or svm or
                                                 options.path_update or options.wavefront_light or
                                                 options.denoise_surface or options.subsurface or
                                                 options.subsurface_compact),
                                 canonicalize_loops=svm)
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
        run(
            options.spirv_bin / "spirv-opt",
            "--target-env=vulkan1.2",
            "--strip-debug", "-O", "--trim-capabilities", "--skip-validation",
            work / "kernel.spv",
            "-o", work / "optimized.spv",
        )
        normalized = work / "normalized.spv"
        normalized.write_bytes(normalize_spirv((work / "optimized.spv").read_bytes()))
        run(options.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", normalized)
        run("node", SOURCE / "translate.mjs", options.translator, normalized, work / shader_name)
        if (options.wavefront_light or options.path_update or options.shade_background or
                options.intersect_shadow or options.denoise_surface):
            shader_path = work / shader_name
            source = shader_path.read_text()
            coalesced = coalesce_loop_state(source)
            if options.light_tree:
                coalesced = "// Cycles Light Tree finite point\n" + coalesced
            if coalesced != source:
                shader_path.write_text(coalesced)
        if (compact or options.surface or options.path_update or options.wavefront_light or
                adaptive or adaptive_camera or options.adaptive_film or options.shade_background or
                options.denoise_surface):
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
        if options.svm or options.subsurface:
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--denoise-surface", action="store_true",
                       help="compile Cycles' denoising guides for compact surfaces")
    modes.add_argument("--intersect-shadow", action="store_true",
                       help="build opaque shadow intersection for compact queued paths")
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
    modes.add_argument("--direct-light", action="store_true",
                       help="compile diffuse point lighting with opaque triangle shadows")
    modes.add_argument("--svm", action="store_true", help="compile the diffuse SVM subset")
    modes.add_argument("--svm-compact", action="store_true",
                       help="compile the SVM stage over compacted paths")
    modes.add_argument("--wavefront-light", action="store_true",
                       help="compile direct lighting with per-path SVM materials")
    modes.add_argument("--shade-background", action="store_true",
                       help="compile environment and forward area emission for miss paths")
    modes.add_argument("--path-update", action="store_true",
                       help="compile diffuse continuation and radiance accumulation")
    modes.add_argument("--subsurface", action="store_true",
                       help="compile the stock Random Walk subsurface kernel for static triangles")
    modes.add_argument("--subsurface-compact", action="store_true",
                       help="compile Random Walk over queued subsurface paths")
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
    options = parser.parse_args()
    if options.svm_core and not options.svm_compact:
        parser.error("--svm-core requires --svm-compact")
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

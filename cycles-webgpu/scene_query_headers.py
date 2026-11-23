# SPDX-License-Identifier: Apache-2.0
"""Use Cycles' shader ray queries with the static WebGPU BVH adapter."""

import struct


def prepare_scene_query_headers(work, cycles, *, outline_noise=False):
    """Keep the stock geometric kernels and supply WebGPU traversal and SVM adapters."""
    overlay = work / "scene-query"
    svm = overlay / "kernel" / "svm"
    svm.mkdir(parents=True)
    kernels = (
        ("ao", "template<uint64_t node_feature_mask"),
        ("raycast", "ccl_device_inline void svm_raycast_attr_eval_and_store("),
        ("bevel", "template<uint64_t node_feature_mask"),
    )
    parts = ['#include "kernel/integrator/path_state.h"', '#include "kernel/sample/mapping.h"']
    for name, end in kernels:
        source = (cycles / "kernel" / "svm" / f"{name}.h").read_text()
        start_at = source.index("#ifdef __SHADER_RAYTRACE__\n") + len("#ifdef __SHADER_RAYTRACE__\n")
        end_at = source.index(end, start_at)
        parts.append("CCL_NAMESPACE_BEGIN\n" + source[start_at:end_at] + "CCL_NAMESPACE_END")
    (svm / "webgpu_scene_queries.h").write_text("\n\n".join(parts) + "\n")
    if outline_noise:
        stock = (cycles / "kernel" / "svm" / "noisetex.h").read_text()
        (svm / "noisetex.h").write_text(outline_noise_selection(stock))
    return overlay


def outline_noise_selection(stock):
    """Share each stock Noise fractal across the value and color outputs."""
    selection = "ccl_device float noise_select("
    insert_at = "ccl_device void noise_texture_1d("
    if stock.count(selection) != 1 or stock.count(insert_at) != 1:
        raise ValueError("Cannot locate the stock Cycles Noise texture functions")
    stock = stock.replace(selection, "ccl_device float stock_noise_select(")
    settings = ("const float detail, const float roughness, const float lacunarity, "
                "const float offset, const float gain, const int type")
    arguments = "detail, roughness, lacunarity, offset, gain, type"
    wrappers = []
    for dimension in range(1, 5):
        axes = "xyzw"[:dimension]
        scalars = ", ".join(f"const float {axis}" for axis in axes)
        coordinates = "x" if dimension == 1 else f"make_float{dimension}({', '.join(axes)})"
        coordinate_type = "float" if dimension == 1 else f"float{dimension}"
        components = "p" if dimension == 1 else ", ".join(f"p.{axis}" for axis in axes)
        # A scalar signature avoids SPIR-V aggregate pointers and bool parameter decorations.
        wrappers.append(f"""__attribute__((noinline)) ccl_device float webgpu_scalar_noise_{dimension}(
    {scalars}, {settings}, const uint normalize)
{{
  return stock_noise_select({coordinates}, {arguments}, normalize != 0);
}}

ccl_device float noise_select(const {coordinate_type} p, {settings}, const bool normalize)
{{
  return webgpu_scalar_noise_{dimension}({components}, {arguments}, normalize);
}}
""")
    return stock.replace(insert_at, "\n".join(wrappers) + "\n" + insert_at)


def narrow_scene_query_indices(code):
    """Drop pointer-width conversions from stock BVH private array indices.

    LLVM expands byte offsets on the LocalIntersection records into unsigned
    64-bit member indices even with 32-bit induction variables. The indices
    are bounded by the local hit count or are constant struct field numbers.
    WGSL uses 32-bit indexing for both.
    """
    words = struct.unpack(f"<{len(code) // 4}I", code)
    header = list(words[:5])
    instructions = []
    narrow = {}
    uint_type = None
    uint_constants = {}
    wide_types = set()
    cursor = 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        operands = words[cursor + 1:cursor + count]
        instructions.append((opcode, operands))
        if opcode == 21 and operands[1:] == (32, 0):  # OpTypeInt
            uint_type = operands[0]
        elif opcode == 21 and operands[1:] == (64, 0):
            wide_types.add(operands[0])
        elif opcode == 43 and operands[0] == uint_type:  # OpConstant
            uint_constants[operands[2]] = operands[1]
        elif opcode in (113, 114) and operands[0] in wide_types:  # OpUConvert, OpSConvert
            narrow[operands[1]] = operands[2]
        cursor += count
    if not wide_types:
        return code
    if uint_type is None:
        raise ValueError("Scene query shaders require 32-bit unsigned indexing")

    new_constants = []
    for opcode, operands in instructions:
        if opcode != 43 or operands[0] not in wide_types:
            continue
        _, value_id, low, high = operands
        if high != 0 or low >= 0x80000000:
            continue
        if low not in uint_constants:
            uint_constants[low] = header[3]
            header[3] += 1
            new_constants.append((43, (uint_type, uint_constants[low], low)))
        narrow[value_id] = uint_constants[low]

    first_function = next(i for i, (opcode, _) in enumerate(instructions) if opcode == 54)
    instructions[first_function:first_function] = new_constants
    for opcode, operands in instructions:
        if opcode in (65, 66):  # OpAccessChain, OpInBoundsAccessChain
            operands = operands[:3] + tuple(narrow.get(i, i) for i in operands[3:])
        header.extend(((len(operands) + 1) << 16 | opcode, *operands))
    return struct.pack(f"<{len(header)}I", *header)

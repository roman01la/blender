# SPDX-License-Identifier: Apache-2.0
"""Normalize the fused experimental closure shader's camera binding and compute ABI."""

import re
import subprocess
from pathlib import Path


def check_linked_symbols(llvm: Path, bitcode: Path) -> None:
    # Old Clang HLSL materializes the named cbuffer array separately. The SPIR-V
    # stage below reconnects it. Every shader function must be fully linked.
    names = subprocess.check_output(
        [llvm / "llvm-nm", "--undefined-only", "--just-symbol-name", bitcode], text=True
    ).splitlines()
    if unresolved := sorted(set(names) - {"camera_rows"}):
        raise ValueError("Fused shader has unresolved symbols: " + ", ".join(unresolved))


def normalize_camera(spirv: Path, binary: Path, work: Path) -> None:
    asm = work / "local-closures-camera.spvasm"
    subprocess.run([spirv / "spirv-dis", binary, "-o", asm], check=True)
    text = asm.read_text()

    # This pinned HLSL backend loses cbuffer metadata when the associated
    # array is read in an outlined helper. Keep the original camera layout.
    array = "%_arr_v4uint_uint_15"
    matches = re.findall(
        rf"^\s*(%\w+) = OpVariable %_ptr_Uniform_{re.escape(array[1:])} Uniform$",
        text,
        flags=re.M,
    )
    if not matches and re.search(r"\bOpDecorate %\w+ Binding 5\b", text):
        return
    if len(matches) != 1 or re.search(r"\bOpDecorate %\w+ Binding 5\b", text):
        raise ValueError("Expected one unbound 15-row camera uniform and an unused slot 5")
    variable = matches[0]

    # Validate the exact register and element layout before reparenting.
    for expression in (
        rf"OpDecorate {array} ArrayStride 16", rf"{array} = OpTypeArray %v4uint %uint_15",
        r"%v4uint = OpTypeVector %uint 4", r"%uint = OpTypeInt 32 0",
        r"%uint_15 = OpConstant %uint 15",
    ):
        if not re.search(expression + r"\s*$", text, re.M):
            raise ValueError("Unexpected camera uniform layout")

    decl = (
        f"%cycles_camera_block = OpTypeStruct {array}\n"
        "%cycles_camera_pointer = OpTypePointer Uniform %cycles_camera_block\n"
        f"{variable} = OpVariable %cycles_camera_pointer Uniform"
    )
    text = re.sub(
        rf"^ *{variable} = OpVariable %_ptr_Uniform_{array[1:]} Uniform$",
        decl, text, count=1, flags=re.M,
    )
    ann = text.index("               OpDecorate ")
    text = text[:ann] + f"""               OpDecorate %cycles_camera_block Block
               OpMemberDecorate %cycles_camera_block 0 Offset 0
               OpDecorate {variable} DescriptorSet 0
               OpDecorate {variable} Binding 5
""" + text[ann:]

    # The array lengths and every used row are fixed and below 15. The
    # 64-bit constants generated for C++ GEPs have no place in WebGPU.
    def access(match: re.Match) -> str:
        row = int(match[2])
        if not (0 <= row < 15 and re.search(rf"^\s*%uint_{row} = OpConstant %uint {row}$", text, re.M)):
            raise ValueError(f"Unexpected static camera row {row}")
        return f"{match[1]} %uint_0 %uint_{row}"

    text, count = re.subn(
        rf"(\bOp(?:InBounds)?AccessChain %\w+ {variable}) %ulong_(\d+)\b",
        access, text,
    )
    remaining_access = re.findall(
        rf"\bOp(?:InBounds)?AccessChain %\w+ {variable} ([^\n]*)", text,
    )
    if count == 0 or len(remaining_access) != count or re.search(
        rf"\bOp(?:Load|Store) .*{variable}(?:\W|$)", text
    ):
        raise ValueError("Unexpected access to the camera uniform")

    asm.write_text(text)
    assembled = work / "local-closures-camera.spv"
    subprocess.run([spirv / "spirv-as", "--target-env", "vulkan1.2", asm, "-o", assembled],
                   check=True)
    subprocess.run([
        spirv / "spirv-opt", "--target-env=vulkan1.2", "--eliminate-dead-code-aggressive",
        "--eliminate-dead-const", "--trim-capabilities", assembled, "-o", binary,
    ], check=True)


def finalize_dispatch(shader: Path, width: int, read_only: tuple[int, ...], marker: str) -> None:
    source = shader.read_text()
    # Keep the host bind groups read-only where HLSL has StructuredBuffer.
    for binding in read_only:
        source, count = re.subn(
            rf"(@group\(0\) @binding\({binding}\)\s+var<storage, )read_write(?=>)",
            r"\g<1>read", source,
        )
        if count != 1:
            raise ValueError(f"Expected one read-only binding at slot {binding}")
    before = "fn main(@builtin(global_invocation_id) param: vec3<u32>) {"
    after = f"""fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {{
    let param = vec3<u32>(id.x + id.y * groups.x * {width}u, 0u, 0u);"""
    if source.count(before) != 1 or f"@workgroup_size({width}, 1, 1)" not in source:
        raise ValueError("Unexpected fused entry point")
    source = source.replace(before, after)
    shader.write_text(marker + "\n// Cycles WebGPU linear workgroup grid\n" + source)


def finalize_wgsl(shader: Path, *, deferred: bool = False) -> None:
    if deferred:
        finalize_dispatch(shader, 576, (0, 1, 3, 6, 8, 10),
                          "// Cycles WebGPU thread-local Light Tree closures")
    else:
        finalize_dispatch(shader, 576, (0, 1, 3, 6, 7, 8, 10),
                          "// Cycles WebGPU experimental thread-local closures")


def finalize_shadow_wgsl(shader: Path) -> None:
    finalize_dispatch(shader, 64, (0, 1, 2, 3), "// Cycles WebGPU thread-local closure shadows")

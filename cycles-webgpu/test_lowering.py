# SPDX-License-Identifier: Apache-2.0

"""Verify that SPIR-V lowering retains the types of private scalar accesses."""

import argparse
from pathlib import Path
import tempfile

from build import build_lowering_plugin, lower_spirv, run


SCALAR_ACCESSES = """target triple = "spirv-unknown-vulkan-compute"

@result = internal addrspace(10) global i32 zeroinitializer

define void @main() "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" {
entry:
  %float_slot = alloca float, align 16
  %word_slot = alloca i32, align 8
  store volatile i32 1065353216, ptr %float_slot, align 16
  store volatile float -0.0, ptr %word_slot, align 8
  %word = load volatile i32, ptr %float_slot, align 16
  %float = load volatile float, ptr %word_slot, align 8
  %bits = bitcast float %float to i32
  %combined = xor i32 %word, %bits
  store volatile i32 %combined, ptr addrspace(10) @result, align 4
  ret void
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--spirv-bin", required=True, type=Path)
    options = parser.parse_args()
    llvm = options.llvm_bin

    with tempfile.TemporaryDirectory(prefix="cycles-webgpu-lowering-") as directory:
        work = Path(directory)
        source = work / "scalar-accesses.ll"
        source.write_text(SCALAR_ACCESSES)
        plugin = build_lowering_plugin(llvm, work)
        lowered = lower_spirv(llvm, work, source, plugin, structured_cfg=True)
        lowered_source = work / "lowered.ll"
        run(llvm / "llvm-dis", lowered, "-o", lowered_source)
        text = lowered_source.read_text()
        for access in (
            "store volatile float 1.000000e+00, ptr %float_slot, align 16",
            "store volatile i32 -2147483648, ptr %word_slot, align 8",
            "load volatile float, ptr %float_slot, align 16",
            "load volatile i32, ptr %word_slot, align 8",
        ):
            if access not in text:
                raise AssertionError(f"Lowering did not preserve the typed access: {access}")

        shader = work / "scalar-accesses.spv"
        run(
            llvm / "llc",
            "-O0", "-filetype=obj", "-spirvprelegalizercombiner-disable-rule=1",
            lowered, "-o", shader,
        )
        run(options.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", shader)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-2.0-or-later

import re
import subprocess
import sys
from pathlib import Path


def main():
    emcc, ispc, *arguments = sys.argv[1:]
    output_index = arguments.index("-o") + 1
    output = Path(arguments[output_index])
    llvm_path = output.with_suffix(".ll")
    arguments[output_index] = str(llvm_path)

    # The generic ISPC targets preserve OIDN's eight-channel CPU layout.
    # ISPC's direct WebAssembly target supports only four lanes.
    subprocess.run([ispc, "--emit-llvm-text", *arguments], check=True)
    llvm = llvm_path.read_text(encoding="utf-8")
    intrinsic = re.search(r"@llvm\.(?:aarch64|arm|x86)\.[\w.]+", llvm)
    if intrinsic:
        raise RuntimeError(f"ISPC emitted a host-specific intrinsic: {intrinsic.group()}")
    llvm = re.sub(r'^target datalayout = ".*"\n', "", llvm, flags=re.MULTILINE)
    llvm = re.sub(r'^target triple = ".*"', 'target triple = "wasm64-unknown-emscripten"',
                  llvm, flags=re.MULTILINE)
    llvm = re.sub(r' "target-(?:cpu|features)"="[^"]*"', "", llvm)
    llvm_path.write_text(llvm, encoding="utf-8")

    subprocess.run(
        [emcc, "-m64", "-pthread", "-msimd128", "-fPIC", "-O3", "-c", str(llvm_path),
         "-o", str(output)],
        check=True,
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0

"""Adapt the stock Principled Hair integration counters to WGSL."""

import re
from pathlib import Path


def prepare_hair_headers(work: Path, cycles: Path) -> None:
    relative = Path("kernel/closure/bsdf_principled_hair_huang.h")
    source = (cycles / relative).read_text()
    # Simpson integration uses bounded intervals. WGSL uses 32-bit indices.
    source = source.replace("size_t", "uint")
    # Total internal reflection leaves the refraction output unwritten. Keep
    # it finite so LLVM does not replace unused paths with literal WGSL NaNs.
    source = re.sub(r"float (cos_theta_t[123]);", r"float \1 = 0.0f;", source)
    header = work / relative
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(source)

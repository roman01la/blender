# SPDX-License-Identifier: Apache-2.0

"""Use Cycles' subsurface helpers with the static-triangle WebGPU BVH."""

from pathlib import Path


def prepare_subsurface_headers(work: Path, cycles: Path) -> None:
    integrator = work / "kernel" / "integrator"
    integrator.mkdir(parents=True)
    upstream = cycles / "kernel" / "integrator"
    source = (upstream / "subsurface_random_walk.h").read_text()
    # The adapter instantiates the same local traversal template for static triangles.
    (integrator / "subsurface_random_walk.h").write_text(
        source.replace('#include "kernel/bvh/bvh.h"\n', ""))

    source = (upstream / "subsurface.h").read_text()
    start = source.index("ccl_device_inline bool subsurface_entry_bounce(")
    end = source.index("ccl_device int subsurface_bounce(", start)
    (integrator / "subsurface_entry.h").write_text(
        "CCL_NAMESPACE_BEGIN\n\n" + source[start:end] + "CCL_NAMESPACE_END\n")

    closure = work / "kernel" / "closure"
    closure.mkdir()
    volume = (cycles / "kernel" / "closure" / "volume.h").read_text()
    # A dynamic native-vector extract avoids selecting pointers to RGB components
    # on LLVM's Vulkan target.
    (closure / "volume.h").write_text(volume.replace(
        "  return GET_SPECTRUM_CHANNEL(value, channel);",
        "  using Channels = float __attribute__((ext_vector_type(4)));\n"
        "  return Channels{value.x, value.y, value.z, 0.0f}[channel];"))

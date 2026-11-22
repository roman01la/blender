# SPDX-License-Identifier: Apache-2.0
"""Expose the stock ray advance and closure picker to the small WebGPU adapter."""


def prepare_transparent_ray_header(work, cycles):
    utility = (cycles / "kernel" / "bvh" / "util.h").read_text()
    start = utility.index("ccl_device_forceinline float intersection_t_offset(")
    end = utility.index("\n}\n", start) + len("\n}\n")
    surface = (cycles / "kernel" / "integrator" / "surface_shader.h").read_text()
    start_pick = surface.index("const ccl_device_inline ccl_private ShaderClosure *"
                               "surface_shader_bsdf_bssrdf_pick(")
    end_pick = surface.index("\n}\n", start_pick) + len("\n}\n")
    (work / "transparent_ray_util.h").write_text(
        "CCL_NAMESPACE_BEGIN\n" + utility[start:end] + surface[start_pick:end_pick] +
        "CCL_NAMESPACE_END\n")

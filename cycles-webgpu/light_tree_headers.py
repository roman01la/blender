# SPDX-License-Identifier: Apache-2.0
"""Adapt the pinned Cycles light tree headers to shader storage-backed records."""

import re


def _replace(source, original, replacement=""):
    if source.count(original) != 1:
        raise ValueError(f"Unexpected stock light kernel source near {original!r}")
    return source.replace(original, replacement, 1)


def _remove_between(source, start, end, replacement=""):
    start_at = source.index(start)
    end_at = source.index(end, start_at)
    return source[:start_at] + replacement + source[end_at:]


def _light_tree_header(source):
    # The synchronized scene admits points, areas, the world map and triangles.
    # Keep stock mesh-instance traversal, bounds and PDF evaluation.
    for name in ("spot", "sun"):
        source = _replace(source, f'#include "kernel/light/{name}.h"\n')
    source = _replace(source, """      case LIGHT_SPOT:
        dir = klight->spot.dir;
        break;
""")
    source = _replace(source, """      case LIGHT_SUN:
        dir = centroid;
        break;
""")
    source = _replace(source, """      case LIGHT_SPOT:
        is_visible = spot_light_tree_parameters<in_volume_segment>(
            klight, centroid, P_c, bcone, cos_theta_u, distance, point_to_centroid, energy);
        break;
""")
    source = _replace(source, """      case LIGHT_SUN:
        is_visible = sun_light_tree_parameters<in_volume_segment>(
            centroid, bcone.theta_e, t, cos_theta_u, distance, point_to_centroid, theta_d);
        break;
""")
    source = _remove_between(
        source,
        "  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) {",
        "  return 0;")
    source = _remove_between(
        source, "/* If the function is called in volume", "CCL_NAMESPACE_END")

    # Stock devices can take an address in a contiguous array. In the WebGPU
    # shaders, extend the lifetime of the record decoded from storage instead.
    pattern = (
        r"const ccl_global (KernelLight|KernelLightTreeNode|KernelLightTreeEmitter) \*(\w+) = "
        r"&kernel_data_fetch\((.*?)\);"
    )
    source, count = re.subn(
        pattern,
        lambda match: (
            f"const {match[1]} &{match[2]}_record = kernel_data_fetch({match[3]});\n"
            f"  const {match[1]} *{match[2]} = &{match[2]}_record;"
        ),
        source,
        flags=re.DOTALL,
    )
    if count != 15:
        raise ValueError(f"Unexpected number of stock Light Tree pointer loads: {count}")
    return source


def prepare_light_tree_headers(work, cycles):
    """Shadow stock headers only in Light Tree shader compilations."""
    overlay = work / "stock-light-tree"
    light = overlay / "kernel" / "light"
    light.mkdir(parents=True)
    source = cycles / "kernel" / "light"
    (light / "tree.h").write_text(_light_tree_header((source / "tree.h").read_text()))
    background = (source / "background.h").read_text()
    # The synchronized host scene admits only map sampling. Omit portal and
    # procedural-sun branches without changing stock map sampling or its PDF.
    background = _remove_between(
        background,
        "ccl_device_inline bool background_portal_data_fetch_and_check_side",
        "template<bool in_volume_segment>")
    (light / "background.h").write_text(background)
    # The Wasm C++ ABI would otherwise pass vectors by aggregate pointers
    # across functions after retargeting the code to Vulkan.
    (light / "common.h").write_text(_replace(
        (source / "common.h").read_text(),
        "ccl_device float3 disk_light_sample(",
        "ccl_device_forceinline float3 disk_light_sample("))
    mapping = overlay / "kernel" / "sample"
    mapping.mkdir()
    source_mapping = (cycles / "kernel" / "sample" / "mapping.h").read_text()
    (mapping / "mapping.h").write_text(_replace(
        source_mapping,
        "ccl_device float2 sample_uniform_disk(",
        "ccl_device_forceinline float2 sample_uniform_disk("))
    return overlay

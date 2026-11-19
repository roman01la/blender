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
    # Static analytic lamps need no triangle geometry, local mesh-light transforms,
    # or link-specific roots. Keep the importance, selection and PDF code intact.
    for name in ("spot", "sun", "triangle"):
        source = _replace(source, f'#include "kernel/light/{name}.h"\n')
    source = _remove_between(
        source, "ccl_device_inline bool is_light", "ccl_device_inline bool is_leaf")
    source = _remove_between(
        source,
        "template<bool in_volume_segment>\nccl_device void light_tree_to_local_space",
        "/* This is the general function")
    source = _remove_between(
        source, "  if (is_light(kemitter)) {", "  return true;\n}", """  const KernelLight lamp = kernel_data_fetch(lights, ~kemitter->light.id);
  centroid = lamp.co;
  switch (lamp.type) {
    case LIGHT_POINT:
      dir = safe_normalize(P - centroid);
      break;
    case LIGHT_AREA:
      dir = lamp.area.dir;
      break;
    case LIGHT_BACKGROUND:
      centroid = make_float3(0.0f, 0.0f, 1.0f);
      dir = make_float3(0.0f, 0.0f, -1.0f);
      break;
    default:
      return false;
  }
""")
    source = _remove_between(source, "  if (is_mesh(kemitter)) {", "  KernelBoundingCone bcone;")
    source = _remove_between(
        source, "  if (is_triangle(kemitter)) {", "  is_visible |= has_transmission;", """  const KernelLight lamp = kernel_data_fetch(lights, ~kemitter->light.id);
  switch (lamp.type) {
    case LIGHT_POINT:
      is_visible = point_light_tree_parameters<in_volume_segment>(
          &lamp, centroid, P_c, cos_theta_u, distance, point_to_centroid);
      bcone.theta_o = 0.0f;
      break;
    case LIGHT_AREA:
      is_visible = area_light_tree_parameters<in_volume_segment>(
          &lamp, centroid, P_c, N_or_D, bcone.axis, cos_theta_u, distance, point_to_centroid);
      break;
    case LIGHT_BACKGROUND:
      is_visible = background_light_tree_parameters<in_volume_segment>(
          centroid, t, cos_theta_u, distance, point_to_centroid, theta_d);
      break;
    default:
      return;
  }

""")
    source = _remove_between(
        source, "  if (in_volume_segment) {\n    /* Vector that forms", "  light_tree_importance",
        """  if (in_volume_segment) {
    point_to_centroid = -compute_v(centroid, P, N_or_D, bcone.axis, t);
  }

""")
    source = _remove_between(
        source, "  const ccl_global KernelLightTreeEmitter *kemitter = &kernel_data_fetch(light_tree_emitters,\n                                                                         selected_index);",
        "  return selected_index;")
    source = _remove_between(
        source,
        "  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) {",
        "  return 0;")
    source = _remove_between(
        source, "      if (node_index < 0) {", "    }\n\n    /* Inner node. */", "      break;\n")
    source = _remove_between(
        source, "  int subtree_root_index;", "  float pdf = 1.0f;",
        "  uint bit_trail = kemitter->bit_trail;\n\n")
    source = _replace(source, "if (emitter == target_emitter)", "if (emitter == index_emitter)")
    source = _remove_between(
        source, "      if (subtree_root_index != -1) {", "      return pdf;")
    source = _remove_between(
        source, "/* If the function is called in volume", "CCL_NAMESPACE_END")

    # Stock devices can take an address in a contiguous array. In the WebGPU
    # shaders, extend the lifetime of the record decoded from storage instead.
    pattern = (
        r"const ccl_global (KernelLightTreeNode|KernelLightTreeEmitter) \*(\w+) = "
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
    if count != 7:
        raise ValueError(f"Unexpected number of stock Light Tree pointer loads: {count}")
    return source


def prepare_light_tree_headers(work, cycles):
    """Shadow stock headers only in analytic Light Tree shader compilations."""
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

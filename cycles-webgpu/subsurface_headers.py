# SPDX-License-Identifier: Apache-2.0

"""Use Cycles' subsurface helpers with the static-triangle WebGPU BVH."""

from pathlib import Path


def prepare_subsurface_headers(work: Path, cycles: Path) -> None:
    integrator = work / "kernel" / "integrator"
    integrator.mkdir(parents=True)
    upstream = cycles / "kernel" / "integrator"
    # Make the private hit record's alignment padding a legal WGSL word array.
    types = (cycles / "kernel" / "types.h").read_text()
    types = types.replace("  Intersection hits[LOCAL_MAX_HITS];\n  float3 Ng[LOCAL_MAX_HITS];",
                          "  Intersection hits[LOCAL_MAX_HITS];\n"
                          "  uint _webgpu_pad[3];\n  float3 Ng[LOCAL_MAX_HITS];")
    (work / "kernel" / "types.h").write_text(types)
    source = (upstream / "subsurface_random_walk.h").read_text()
    # The adapter instantiates the same local traversal template for static triangles.
    (integrator / "subsurface_random_walk.h").write_text(
        source.replace('#include "kernel/bvh/bvh.h"\n', ""))

    source = (upstream / "subsurface.h").read_text()
    # Use field copies for private BVH hits. Dynamic aggregate copies otherwise
    # become opaque-pointer memcpy calls that LLVM's SPIR-V backend cannot lower.
    bvh = work / "kernel" / "bvh"
    bvh.mkdir()
    utilities = (cycles / "kernel" / "bvh" / "util.h").read_text()
    copy_helpers = """ccl_device_inline void webgpu_copy(ccl_private Intersection &dst,
                                          const volatile ccl_private Intersection &src)
{
  dst.t = src.t;
  dst.u = src.u;
  dst.v = src.v;
  dst.prim = src.prim;
  dst.object = src.object;
  dst.type = src.type;
}

ccl_device_inline void webgpu_copy(ccl_private float3 &dst,
                                          const volatile ccl_private float3 &src)
{
  dst.x = src.x;
  dst.y = src.y;
  dst.z = src.z;
}

"""
    utilities = utilities.replace("ccl_device_inline void sort_intersections_and_normals(",
                                  copy_helpers +
                                  "ccl_device_inline void sort_intersections_and_normals(")
    utilities = utilities.replace(
        """        Intersection tmp_hit = hits[j];
        float3 tmp_Ng = Ng[j];
        hits[j] = hits[j + 1];
        Ng[j] = Ng[j + 1];
        hits[j + 1] = tmp_hit;
        Ng[j + 1] = tmp_Ng;""",
        """        Intersection tmp_hit;
        float3 tmp_Ng;
        webgpu_copy(tmp_hit, hits[j]);
        webgpu_copy(tmp_Ng, Ng[j]);
        webgpu_copy(hits[j], hits[j + 1]);
        webgpu_copy(Ng[j], Ng[j + 1]);
        webgpu_copy(hits[j + 1], tmp_hit);
        webgpu_copy(Ng[j + 1], tmp_Ng);""")
    (bvh / "util.h").write_text(utilities)

    disk = (upstream / "subsurface_disk.h").read_text()
    disk = disk.replace('#include "kernel/bvh/bvh.h"\n', "")
    disk = disk.replace('#include "kernel/integrator/guiding.h"\n', "")
    # The same object transform helper with only its static branch enabled.
    disk = disk.replace("      Transform itfm;\n"
                        "      object_fetch_transform_motion_test(kg, object, time, &itfm);",
                        "      const Transform itfm = "
                        "object_fetch_transform(kg, object, OBJECT_INVERSE_TRANSFORM);")
    disk = disk.replace("ss_isect.hits[0] = ss_isect.hits[hit];",
                        "webgpu_copy(ss_isect.hits[0], ss_isect.hits[hit]);")
    disk = disk.replace("ss_isect.Ng[0] = ss_isect.Ng[hit];",
                        "webgpu_copy(ss_isect.Ng[0], ss_isect.Ng[hit]);")
    # Path guiding is disabled on the WebGPU device.
    start = disk.index("      guiding_record_bssrdf_bounce(")
    end = disk.index("      return true;", start)
    (integrator / "subsurface_disk.h").write_text(disk[:start] + disk[end:])

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

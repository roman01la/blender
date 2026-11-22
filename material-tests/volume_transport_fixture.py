# SPDX-License-Identifier: GPL-2.0-or-later

from hashlib import sha256
from math import pi
from pathlib import Path

import bpy
from mathutils import Vector


def material(name, shader, inputs):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    tree = result.node_tree
    tree.nodes.clear()
    node = tree.nodes.new(shader)
    for key, value in inputs.items():
        node.inputs[key].default_value = value
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(node.outputs[0], output.inputs["Surface" if shader == "ShaderNodeBsdfDiffuse"
                                                  else "Volume"])
    return result


def smoke(fixture, interpolation):
    path = Path(fixture.volume_grid)
    data = bpy.data.volumes.new("Transport probe grid")
    data.filepath = str(path)
    obj = bpy.data.objects.new(data.name, data)
    fixture.scene.collection.objects.link(obj)
    if not data.grids.load():
        raise RuntimeError(f"Could not load grid {path.name}")
    grid = next(grid for grid in data.grids if grid.name.startswith("density"))
    bpy.context.view_layer.update()
    bounds = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).bound_box
    lower = Vector(min(corner[axis] for corner in bounds) for axis in range(3))
    upper = Vector(max(corner[axis] for corner in bounds) for axis in range(3))
    scale = 2.8 / max(upper - lower)
    obj.scale = (scale,) * 3
    obj.rotation_euler.x = pi / 2
    obj.location = Vector((-0.23, 0.0, 0.1)) - obj.rotation_euler.to_matrix() @ (
        (lower + upper) * (0.5 * scale))
    shader = material("Transport probe smoke", "ShaderNodeVolumePrincipled", {
        "Color": (0.7, 0.81, 1, 1), "Density": 2.4, "Density Attribute": grid.name,
        "Anisotropy": 0.22,
    })
    shader.cycles.volume_interpolation = interpolation
    data.materials.append(shader)
    return {"grid": grid.name, "digest": sha256(path.read_bytes()).hexdigest()}


def configure(fixture, node, options):
    import _cycles

    scene = fixture.scene
    sphere = fixture.sphere
    name = options["media"]
    for obj in list(scene.objects):
        if obj.name.startswith("Transport probe "):
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.dither_intensity = 0
    scene.cycles.use_denoising = options["denoise"]
    if options["denoise"]:
        if not _cycles.with_openimagedenoise:
            raise RuntimeError("Open Image Denoise is required for this probe")
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        scene.cycles.denoising_use_gpu = fixture.device == "WEBGPU"
        scene.cycles.denoising_quality = "HIGH"
        scene.cycles.denoising_prefilter = "ACCURATE"
        for layer in scene.view_layers:
            layer.cycles.use_denoising = True
            layer.cycles.denoising_store_passes = False

    sphere.hide_render = name in {"surface", "grid"}
    sphere.location = (0.28, -0.08, 0.01)
    sphere.scale = (0.88, 0.88, 0.88)
    if name != "surface":
        node.inputs["Color"].default_value = (0.74, 0.44, 0.25, 1)
        node.inputs["Density"].default_value = 0.84
        node.inputs["Anisotropy"].default_value = 0.18

    camera = scene.camera
    inside = name == "overlap-inside"
    camera.location = (0.16, -0.21, 0.12) if inside else (3.4, -4.2, 2.1)
    target = Vector((0.05, 1.0, -0.25) if inside else (0.0, 0.0, -0.03))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 41 if inside else 55

    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, -1.28))
    floor = bpy.context.object
    floor.name = "Transport probe floor"
    floor.data.materials.append(material("Transport probe matte", "ShaderNodeBsdfDiffuse", {
        "Color": (0.35, 0.41, 0.46, 1),
    }))
    if name.startswith("overlap"):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.76,
                                             location=(-0.28, 0.07, 0.05))
        obj = bpy.context.object
        obj.name = "Transport probe intersecting mesh"
        if name == "overlap-absorption":
            shader = material("Transport probe absorption", "ShaderNodeVolumeAbsorption", {
                "Color": (0.25, 0.63, 0.81, 1), "Density": 0.72,
            })
        else:
            shader = material("Transport probe scattering", "ShaderNodeVolumeScatter", {
                "Color": (0.48, 0.68, 0.94, 1), "Density": 0.7, "Anisotropy": -0.24,
            })
        obj.data.materials.append(shader)

    if name.startswith("grid"):
        if not fixture.volume_grid:
            raise ValueError("The grid volume probe needs a smoke VDB file")
        return smoke(fixture, options.get("interpolation", "LINEAR"))
    return name

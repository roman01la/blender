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


def mixed_boundary(scatter, additive, transparent=False, principled=False):
    tree = scatter.id_data
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    if transparent:
        surface = tree.nodes.new("ShaderNodeBsdfTransparent")
        surface.inputs["Color"].default_value = (0.79, 0.91, 0.96, 1)
    elif principled:
        surface = tree.nodes.new("ShaderNodeBsdfPrincipled")
        surface.inputs["Transmission Weight"].default_value = 1
        surface.inputs["Roughness"].default_value = 0.12
        surface.inputs["IOR"].default_value = 1.33
    else:
        surface = tree.nodes.new("ShaderNodeBsdfGlass")
        surface.distribution = "GGX"
        surface.inputs["Roughness"].default_value = 0.12
        surface.inputs["IOR"].default_value = 1.33
    tree.links.new(surface.outputs[0], output.inputs["Surface"])

    absorption = tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (0.3, 0.63, 0.87, 1)
    absorption.inputs["Density"].default_value = 0.76
    if additive:
        scatter.inputs["Density"].default_value = 0.22
        add = tree.nodes.new("ShaderNodeAddShader")
        tree.links.new(absorption.outputs[0], add.inputs[0])
        tree.links.new(scatter.outputs[0], add.inputs[1])
        tree.links.new(add.outputs[0], output.inputs["Volume"])
    else:
        tree.links.new(absorption.outputs[0], output.inputs["Volume"])
    return tree, output, surface


def bump_surface(tree, shader):
    noise = tree.nodes.new("ShaderNodeTexNoise")
    position = tree.nodes.new("ShaderNodeNewGeometry")
    normal = tree.nodes.new("ShaderNodeBump")
    normal.inputs["Strength"].default_value = 0.18
    normal.inputs["Distance"].default_value = 0.10
    tree.links.new(position.outputs["Position"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], normal.inputs["Height"])
    tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])


def varying_boundary(scatter, bump):
    tree, output, transparent = mixed_boundary(scatter, additive=True, transparent=True)
    path = tree.nodes.new("ShaderNodeLightPath")
    tint = tree.nodes.new("ShaderNodeMixRGB")
    tint.inputs[1].default_value = (0.79, 0.91, 0.96, 1)
    tint.inputs[2].default_value = (0.54, 0.73, 0.83, 1)
    tree.links.new(path.outputs["Is Shadow Ray"], tint.inputs[0])
    tree.links.new(tint.outputs[0], transparent.inputs["Color"])
    if bump:
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        bump_surface(tree, diffuse)
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.57
        tree.links.new(diffuse.outputs[0], mix.inputs[1])
        tree.links.new(transparent.outputs[0], mix.inputs[2])
        tree.links.new(mix.outputs[0], output.inputs["Surface"])


def configure_depth_output(scene, active):
    layer = scene.view_layers[0]
    layer.use_pass_z = active
    scene.render.use_compositing = active
    if not active:
        return

    scene.render.compositor_device = "CPU"
    compositor = scene.compositing_node_group
    if compositor is None:
        compositor = bpy.data.node_groups.new("Volume Depth data pass", "CompositorNodeTree")
        scene.compositing_node_group = compositor
        compositor.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    compositor.nodes.clear()
    layers = compositor.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    scale = compositor.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 0.1
    compositor.links.new(layers.outputs["Depth"], scale.inputs[0])
    output = compositor.nodes.new("NodeGroupOutput")
    compositor.links.new(scale.outputs[0], output.inputs["Image"])


def configure(fixture, node, options):
    import _cycles

    scene = fixture.scene
    sphere = fixture.sphere
    name = options["media"]
    for obj in list(scene.objects):
        if obj.name.startswith("Transport probe "):
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.dither_intensity = 0
    scene.render.film_transparent = name == "transparent-volume-film"
    configure_depth_output(scene, options.get("depth", False))
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

    sphere.hide_render = name in {"surface", "grid", "separated-volume-bounds"}
    sphere.location = (0.28, -0.08, 0.01)
    sphere.scale = (0.88, 0.88, 0.88)
    if name != "surface":
        node.inputs["Color"].default_value = (0.74, 0.44, 0.25, 1)
        node.inputs["Density"].default_value = 0.84
        node.inputs["Anisotropy"].default_value = 0.18
    if name.startswith("mixed-boundary"):
        bump = name.endswith("-bump")
        tree, _, glass = mixed_boundary(
            node, additive=bump or name.endswith("-additive"), principled=bump)
        if bump:
            bump_surface(tree, glass)
    elif name in {"transparent-volume-boundary", "transparent-volume-film"}:
        mixed_boundary(node, additive=True, transparent=True)
    elif name in {"transparent-volume-varying", "transparent-volume-bump"}:
        varying_boundary(node, bump=name == "transparent-volume-bump")
        if options.get("depth"):
            # Keep the absorbing medium and transparent boundary. A single-sample
            # Depth pass should not depend on matching random volume events.
            node.inputs["Density"].default_value = 0.0

    camera = scene.camera
    inside = name.endswith("-inside")
    camera.location = (0.16, -0.21, 0.12) if inside else (3.4, -4.2, 2.1)
    target = Vector((0.05, 1.0, -0.25) if inside else (0.0, 0.0, -0.03))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 41 if inside else 55

    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, -1.28))
    floor = bpy.context.object
    floor.name = "Transport probe floor"
    matte = material("Transport probe matte", "ShaderNodeBsdfDiffuse", {
        "Color": (0.35, 0.41, 0.46, 1),
    })
    floor.data.materials.append(matte)
    if name in {"ao-scatter", "ao-scatter-bump"}:
        tree = matte.node_tree
        diffuse = next(node for node in tree.nodes if node.type == "BSDF_DIFFUSE")
        ao = tree.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 16
        ao.inputs["Distance"].default_value = 1.8
        ao.inputs["Color"].default_value = diffuse.inputs["Color"].default_value
        tree.links.new(ao.outputs["Color"], diffuse.inputs["Color"])
        if name == "ao-scatter-bump":
            bump_surface(tree, diffuse)
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

    if name == "separated-volume-bounds":
        camera.data.lens = 37
        for i in range(12):
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2, radius=0.26,
                location=((i % 4 - 1.5) * 0.78, (i // 4 - 1) * 0.78, -0.24))
            obj = bpy.context.object
            obj.name = f"Transport probe medium {i}"
            obj.data.materials.append(sphere.data.materials[0])

    if name.startswith("grid"):
        if not fixture.volume_grid:
            raise ValueError("The grid volume probe needs a smoke VDB file")
        return smoke(fixture, options.get("interpolation", "LINEAR"))
    return name

# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from mathutils import Vector


def configure(fixture, node, mode):
    scene = fixture.scene
    sphere = fixture.sphere
    camera = scene.camera
    sphere.location = (0, 0, 0)
    sphere.scale = (1, 1, 1)
    camera.location = (3.4, -4.2, 2.1)
    light = bpy.data.objects["Light"]
    light.data.size = 1.12 if mode.startswith("shadow") else 2.5
    scene.cycles.samples = 96
    scene.cycles.use_adaptive_sampling = False
    scene.view_layers[0].cycles.use_pass_volume_majorant = mode == "requested-majorant"
    camera.data.lens = 55
    for obj in list(scene.objects):
        if obj.name.startswith("Absorption receiver"):
            bpy.data.objects.remove(obj, do_unlink=True)
    is_clear = mode.endswith("-clear")
    node.inputs["Color"].default_value = (1, 1, 1, 1) if is_clear else (0.18, 0.55, 0.83, 1)
    node.inputs["Density"].default_value = 0.9

    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.4, 0.4, 0.4, 1)
    if mode.startswith("inside"):
        camera.location = (0.04, 0.09, 0.12)
        target = Vector((0, 0, -1))
    else:
        target = Vector((0, 0, 0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    if mode.startswith("shadow"):
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
        sphere.location = light.location * 0.48
        sphere.scale = (0.84, 0.65, 0.72)
        bpy.ops.mesh.primitive_plane_add(size=4)
        receiver = bpy.context.object
        receiver.name = "Absorption receiver"
        material = bpy.data.materials.get("Absorption receiver material")
        if not material:
            material = bpy.data.materials.new("Absorption receiver material")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
            diffuse.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1)
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        receiver.data.materials.append(material)

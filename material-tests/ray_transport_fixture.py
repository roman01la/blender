# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from mathutils import Vector


def plate(center, normal, size, material):
    bpy.ops.mesh.primitive_plane_add(size=size, location=center)
    obj = bpy.context.object
    obj.name = "Ray transport plate"
    obj.rotation_euler = Vector(normal).to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def configure(fixture, node, tree, output, mode):
    scene = fixture.scene
    camera = scene.camera
    camera.location = (3.4, -4.2, 2.1)
    fixture.sphere.hide_render = False
    for obj in list(scene.objects):
        if obj.name.startswith("Ray transport plate"):
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = mode == "wireframe-adaptive"
    scene.cycles.adaptive_min_samples = 16
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

    if mode.startswith("wireframe"):
        if mode == "wireframe-far":
            camera.location *= 1.4
            camera.data.lens = 55 * 1.4
        else:
            camera.data.lens = 55
        reflected = mode.startswith("wireframe-reflection")
        node.inputs["Size"].default_value = 0 if mode.endswith("-zero") else (
            0.25 if reflected else 1.5)
        emission = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(node.outputs["Fac"], emission.inputs["Color"])
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
        fixture.material.cycles.emission_sampling = "NONE"
        if reflected:
            fixture.sphere.hide_render = True
            outgoing = Vector((1, 0, 0))
            mirror_normal = (camera.location.normalized() + outgoing).normalized()
            mirror = bpy.data.materials.get("Ray transport mirror") or bpy.data.materials.new(
                "Ray transport mirror")
            mirror.use_nodes = True
            bsdf = mirror.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Metallic"].default_value = 0.8
            bsdf.inputs["Roughness"].default_value = 0.16
            plate((0, 0, 0), mirror_normal, 1.8, mirror)
            plate((2, 0, 0), outgoing, 3, fixture.material).visible_camera = False
        return

    camera.data.lens = 55
    if mode in {"portal-shadow", "portal-shadow-clear"}:
        receiver = bpy.data.materials.get("Ray transport receiver") or bpy.data.materials.new(
            "Ray transport receiver")
        receiver.use_nodes = True
        receiver_tree = receiver.node_tree
        receiver_tree.nodes.clear()
        diffuse = receiver_tree.nodes.new("ShaderNodeBsdfDiffuse")
        receiver_output = receiver_tree.nodes.new("ShaderNodeOutputMaterial")
        receiver_tree.links.new(diffuse.outputs[0], receiver_output.inputs["Surface"])
        fixture.sphere.data.materials[0] = receiver
        if mode == "portal-shadow":
            light_axis = bpy.data.objects["Light"].location.normalized()
            plate(light_axis * 2.35, light_axis, 5, fixture.material).visible_camera = False
        return

    fixture.sphere.hide_render = True
    camera_axis = camera.location.normalized()
    plate((0, 0, 0), camera_axis, 1.6, fixture.material)
    node.inputs["Color"].default_value = (0.68, 0.45, 0.28, 1)
    target = bpy.data.materials.get("Ray transport target") or bpy.data.materials.new(
        "Ray transport target")
    target.use_nodes = True
    target.cycles.emission_sampling = "NONE"
    target_tree = target.node_tree
    target_tree.nodes.clear()
    emission = target_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.75, 0.64, 0.52, 1)
    destination = target_tree.nodes.new("ShaderNodeOutputMaterial")
    target_tree.links.new(emission.outputs[0], destination.inputs["Surface"])
    if mode == "portal-stationary":
        plate(-camera_axis * 1.3, camera_axis, 1.8, target)
        return
    ahead = Vector((1, 0, 0))
    offset = tree.nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    offset.inputs[1].default_value = (8, 0, 0)
    tree.links.new(tree.nodes.new("ShaderNodeNewGeometry").outputs["Position"], offset.inputs[0])
    tree.links.new(offset.outputs[0], node.inputs["Position"])
    node.inputs["Direction"].default_value = (0, 1, 0) if mode == "portal-away" else ahead
    plate((11, 0, 0), ahead, 3, target)
    depth = target_tree.nodes.new("ShaderNodeLightPath").outputs["Portal Depth"]
    target_tree.links.new(depth, emission.inputs["Strength"])

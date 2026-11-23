# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from mathutils import Vector


def configure(scene, settings):
    lamp = scene.objects["Light"]
    previous = lamp.data
    kind = settings["kind"]
    light = lamp.data = bpy.data.lights.new("Probe analytic light", kind)
    bpy.data.lights.remove(previous)
    lamp.visible_camera = settings.get("camera_visible", False)
    lamp.location = (1.2, -1.8, 2.5)
    lamp.rotation_euler = (-lamp.location).to_track_quat("-Z", "Y").to_euler()
    light.energy = settings.get("energy", 1.7 if kind == "SUN" else 45.0)
    if color := settings.get("color"):
        light.color = color
    light.use_shadow = settings.get("cast_shadow", True)
    light.cycles.use_multiple_importance_sampling = settings.get("light_mis", True)
    if kind == "AREA":
        light.size = 1.3
    if kind == "POINT":
        light.shadow_soft_size = 0
    if settings.get("camera_visible"):
        camera = scene.camera.location
        lamp.location = camera * 0.55 + Vector((-0.55, -0.05, 0.45))
        lamp.rotation_euler = (camera - lamp.location).to_track_quat("-Z", "Y").to_euler()
        if kind == "POINT":
            light.energy = 12
            light.shadow_soft_size = 0.32
        else:
            light.energy = 0.8
            light.size = 0.62
    scene.cycles.use_light_tree = settings.get("light_tree", False)
    scene.cycles.light_sampling_threshold = settings.get("light_threshold", 0)
    fill = scene.objects.get("Falloff point fill")
    if scene.cycles.use_light_tree and fill is None:
        fill = bpy.data.objects.new("Falloff point fill", bpy.data.lights.new("Falloff point fill", "POINT"))
        scene.collection.objects.link(fill)
        fill.location = (-2, 1, 2)
        fill.data.energy = 9
    if fill is not None:
        fill.hide_render = not scene.cycles.use_light_tree
        fill.data.shadow_soft_size = settings.get("fill_radius", 0)
    blocker = scene.objects.get("Probe off-camera blocker")
    if "cast_shadow" in settings and blocker is None:
        bpy.ops.mesh.primitive_plane_add(size=3.8, location=lamp.location * 0.52)
        blocker = bpy.context.object
        blocker.name = "Probe off-camera blocker"
        blocker.rotation_euler = lamp.location.to_track_quat("Z", "Y").to_euler()
        blocker.visible_camera = blocker.visible_diffuse = blocker.visible_glossy = False
        blocker.visible_transmission = blocker.visible_volume_scatter = False
    if blocker is not None:
        blocker.hide_render = "cast_shadow" not in settings
    light.use_nodes = True
    tree = light.node_tree
    emission = next(node for node in tree.nodes if node.type == "EMISSION")
    if output := settings.get("falloff"):
        falloff = tree.nodes.new("ShaderNodeLightFalloff")
        falloff.inputs["Strength"].default_value = 1
        falloff.inputs["Smooth"].default_value = settings["smooth"]
        tree.links.new(falloff.outputs[output], emission.inputs["Strength"])
    if "ray_length" in settings:
        length = tree.nodes.new("ShaderNodeLightPath").outputs["Ray Length"]
        divisor = settings["ray_length"]
        if divisor is not None:
            divide = tree.nodes.new("ShaderNodeMath")
            divide.operation = "DIVIDE"
            divide.inputs[1].default_value = divisor
            tree.links.new(length, divide.inputs[0])
            length = divide.outputs["Value"]
        remap = tree.nodes.new("ShaderNodeMapRange")
        remap.interpolation_type = "SMOOTHERSTEP"
        remap.clamp = settings.get("clamp", True)
        tree.links.new(length, remap.inputs["Value"])
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(remap.outputs["Result"], mix.inputs[0])
        tree.links.new(emission.outputs[0], mix.inputs[1 if settings.get("inverted") else 2])
        output = next(node for node in tree.nodes if node.type == "OUTPUT_LIGHT")
        tree.links.new(mix.outputs[0], output.inputs["Surface"])
    if temperature := settings.get("temperature"):
        blackbody = tree.nodes.new("ShaderNodeBlackbody")
        blackbody.inputs["Temperature"].default_value = temperature
        tree.links.new(blackbody.outputs["Color"], emission.inputs["Color"])
    if "emission_strength" in settings:
        emission.inputs["Strength"].default_value = settings["emission_strength"]
    return tree

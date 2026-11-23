# SPDX-License-Identifier: GPL-2.0-or-later

from math import pi

import bpy

from mixed_glossy_fixture import configure_render_outputs


def configure(scene, settings):
    world = scene.world
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 1.25
    tree.links.new(background.outputs[0], output.inputs["Surface"])

    environment = tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images["Probe HDRI"]
    color = environment.outputs["Color"]
    if settings.get("mapped"):
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.13, -0.31, pi)
        mapping.inputs["Scale"].default_value = (0.8, 1.15, 1)
        tree.links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs[0], environment.inputs["Vector"])

    if settings.get("color"):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.mode = "HSV"
        tree.links.new(color, separate.inputs["Color"])
        threshold = tree.nodes.new("ShaderNodeValue")
        threshold.outputs[0].default_value = 0.3
        greater = tree.nodes.new("ShaderNodeMath")
        greater.operation = "GREATER_THAN"
        tree.links.new(separate.outputs[2], greater.inputs[0])
        tree.links.new(threshold.outputs[0], greater.inputs[1])
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        tree.links.new(greater.outputs[0], mix.inputs[0])
        tree.links.new(separate.outputs[2], mix.inputs[6])
        tree.links.new(threshold.outputs[0], mix.inputs[7])
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.mode = "HSV"
        tree.links.new(separate.outputs[0], combine.inputs[0])
        tree.links.new(separate.outputs[1], combine.inputs[1])
        tree.links.new(mix.outputs[2], combine.inputs[2])
        color = combine.outputs["Color"]
    tree.links.new(color, background.inputs["Color"])

    world.cycles.sampling_method = "MANUAL" if settings.get("sampled", True) else "NONE"
    world.cycles_visibility.camera = settings.get("camera", True)
    scene.cycles.use_light_tree = settings.get("light_tree", True)
    scene.objects["Fill"].hide_render = settings.get("single_area", False)
    configure_render_outputs(scene, settings)
    return tree

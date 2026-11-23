# SPDX-License-Identifier: GPL-2.0-or-later

from math import pi

import bpy


def configure(scene, settings):
    mode = settings["mode"]
    blocker = scene.objects.get("Film AO blocker")
    if blocker is not None:
        material = blocker.data.materials[0]
        bpy.data.objects.remove(blocker, do_unlink=True)
        bpy.data.materials.remove(material)
    scene.world.light_settings.distance = 3.6
    graphs = []
    if mode != "clear":
        bpy.ops.mesh.primitive_plane_add(size=4.8, location=(0, 0, 1.45))
        blocker = bpy.context.object
        blocker.name = "Film AO blocker"
        blocker.visible_camera = blocker.visible_diffuse = blocker.visible_glossy = False
        blocker.visible_transmission = blocker.visible_volume_scatter = False
        material = bpy.data.materials.new("Film AO shadow material")
        material.use_nodes = material.use_transparent_shadow = True
        blocker.data.materials.append(material)
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        shader = shadow_surface(tree, mode)
        if mode.startswith("backface"):
            material.cycles.emission_sampling = "NONE"
            blocker.rotation_euler.x = pi if mode == "backface-open" else 0
        tree.links.new(shader.outputs[0], output.inputs["Surface"])
        graphs.append(tree)

    output_mode = settings.get("output", "ao")
    layer = scene.view_layers[0]
    layer.use_pass_ambient_occlusion = output_mode != "combined-only"
    scene.render.use_compositing = output_mode == "ao"
    if not scene.render.use_compositing:
        return graphs
    compositor = scene.compositing_node_group
    if compositor is None:
        compositor = bpy.data.node_groups.new("Film AO data pass", "CompositorNodeTree")
        compositor.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = compositor
    compositor.nodes.clear()
    layers = compositor.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    composite = compositor.nodes.new("NodeGroupOutput")
    compositor.links.new(layers.outputs["Ambient Occlusion"], composite.inputs["Image"])
    graphs.append(compositor)
    return graphs


def shadow_surface(tree, mode):
    if mode == "opaque":
        return tree.nodes.new("ShaderNodeBsdfDiffuse")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    if mode == "tinted":
        transparent.inputs["Color"].default_value = (0.23, 0.58, 0.83, 1)
    elif mode.startswith("varying"):
        uv = tree.nodes.new("ShaderNodeTexCoord")
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 7
        tree.links.new(uv.outputs["UV"], noise.inputs["Vector"])
        tint = noise.outputs["Factor"]
        if mode == "varying-tinted":
            ramp = tree.nodes.new("ShaderNodeMixRGB")
            ramp.inputs[1].default_value = (0.15, 0.52, 0.8, 1)
            ramp.inputs[2].default_value = (0.77, 0.39, 0.3, 1)
            tree.links.new(tint, ramp.inputs[0])
            tint = ramp.outputs[0]
        tree.links.new(tint, transparent.inputs["Color"])
    elif mode.startswith("backface"):
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 0.25
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(geometry.outputs["Backfacing"], mix.inputs[0])
        tree.links.new(transparent.outputs[0], mix.inputs[1])
        tree.links.new(emission.outputs[0], mix.inputs[2])
        return mix
    return transparent

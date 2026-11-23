# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from mixed_principled_fixture import bump, varying_factor


def diffuse_lobe(tree, color):
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = color
    return diffuse


def glossy_lobe(tree, color, distribution, roughness):
    glossy = tree.nodes.new("ShaderNodeBsdfAnisotropic")
    glossy.distribution = distribution
    glossy.inputs["Color"].default_value = color
    glossy.inputs["Roughness"].default_value = roughness
    return glossy


def mix_lobes(tree, first, second, reverse, factor="Z"):
    mix = tree.nodes.new("ShaderNodeMixShader")
    if isinstance(factor, str):
        tree.links.new(varying_factor(tree, reverse, factor), mix.inputs[0])
    else:
        mix.inputs[0].default_value = 1.0 - factor if reverse else factor
    branches = (second, first) if reverse else (first, second)
    for shader, socket in zip(branches, mix.inputs[1:], strict=True):
        tree.links.new(shader.outputs[0], socket)
    return mix


def bump_lobes(tree, lobes, scale, invert):
    bump(tree, lobes[0], scale, invert)
    normal = lobes[0].inputs["Normal"].links[0].from_socket
    for lobe in lobes[1:]:
        tree.links.new(normal, lobe.inputs["Normal"])


def ordered_mix(tree, settings):
    diffuse = [diffuse_lobe(tree, color) for color in ((0.46, 0.10, 0.05, 1), (0.06, 0.45, 0.17, 1))]
    roughness = (0.16, 0.62)
    if "collapse" in settings:
        roughness = (roughness[settings["collapse"]],) * 2
    glossy = [
        glossy_lobe(tree, (0.47, 0.60, 0.79, 1), "GGX", roughness[0]),
        glossy_lobe(tree, (0.79, 0.43, 0.16, 1), settings["distribution"], roughness[1]),
    ]
    if settings.get("outside") == "anisotropic":
        glossy[1].inputs["Anisotropy"].default_value = 0.6
    if settings.get("outside") == "rough-diffuse":
        diffuse[1].inputs["Roughness"].default_value = 0.3
    mode = settings["bump"]
    first_pair = (diffuse[0], glossy[0])
    second_pair = (glossy[1], diffuse[1])
    if mode == "shared":
        bump_lobes(tree, first_pair + second_pair, 5, False)
    if mode == "both":
        for index, (lobe, scale) in enumerate(zip(first_pair + second_pair, (5, 7, 9, 11), strict=True)):
            bump(tree, lobe, scale, index % 2 == 1)
    if settings.get("explicit_normal"):
        normal = tree.nodes.new("ShaderNodeNewGeometry").outputs["Normal"]
        for lobe in first_pair + second_pair:
            tree.links.new(normal, lobe.inputs["Normal"])
    reverse = settings.get("reverse", False)
    first = mix_lobes(tree, *first_pair, reverse, "X")
    second = mix_lobes(tree, *second_pair, reverse, 0.23)
    return mix_lobes(tree, first, second, reverse)


def mixed_pair(tree, settings):
    diffuse = diffuse_lobe(tree, (0.63, 0.15, 0.06, 1))
    glossy = glossy_lobe(tree, (0.12, 0.48, 0.74, 1), settings["distribution"], 0.33)
    mode = settings["bump"]
    if mode in {"both", "diffuse", "shared"}:
        bump(tree, diffuse, 5, False)
    if mode in {"both", "glossy"}:
        bump(tree, glossy, 9, True)
    if mode == "shared":
        shared_normal = diffuse.inputs["Normal"].links[0].from_socket
        tree.links.new(shared_normal, glossy.inputs["Normal"])
    return mix_lobes(tree, diffuse, glossy, settings.get("reverse", False))


def shader_displacement(tree, output, height=None):
    displacement = tree.nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Scale"].default_value = 0.22
    if height is None:
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 3
        tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
        height = noise.outputs["Fac"]
    tree.links.new(height, displacement.inputs["Height"])
    tree.links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])


def displaced_glossy(material, output, settings):
    tree = material.node_tree
    material.displacement_method = "BUMP"
    glossy = glossy_lobe(tree, (0.74, 0.54, 0.32, 1), settings["distribution"], 0.29)
    mode = settings["bump"]
    if mode in {"local", "independent", "shared-height"}:
        bump(tree, glossy, 7, True)
    if mode in {"displacement", "independent", "shared-height"}:
        # Blender converts the shared normal vector to a scalar height.
        height = glossy.inputs["Normal"].links[0].from_socket if mode == "shared-height" else None
        shader_displacement(tree, output, height)
    return glossy


def displaced_mixed_pair(material, output, settings):
    tree = material.node_tree
    material.displacement_method = "BUMP"
    diffuse = diffuse_lobe(tree, (0.63, 0.15, 0.06, 1))
    glossy = glossy_lobe(tree, (0.26, 0.52, 0.78, 1), settings["distribution"], 0.34641016)
    mode = settings["bump"]
    height = None
    if mode in {"local", "independent-height", "shared-height"}:
        bump_lobes(tree, (diffuse, glossy), 7, True)
        if mode == "shared-height":
            height = diffuse.inputs["Normal"].links[0].from_socket
    elif mode == "independent-normals":
        bump(tree, diffuse, 5, False)
        bump(tree, glossy, 9, True)
    if mode not in {"flat", "local"}:
        shader_displacement(tree, output, height)
    return mix_lobes(tree, diffuse, glossy, settings.get("reverse", False), settings.get("factor", 0.37))


def configure(scene, material, output, settings):
    tree = material.node_tree
    if settings.get("mixed_displacement_bump"):
        surface = displaced_mixed_pair(material, output, settings)
    elif settings.get("displacement_bump"):
        surface = displaced_glossy(material, output, settings)
    elif settings.get("ordered"):
        surface = ordered_mix(tree, settings)
    else:
        surface = mixed_pair(tree, settings)
    tree.links.new(surface.outputs[0], output.inputs["Surface"])
    return configure_render_outputs(scene, settings)


def configure_render_outputs(scene, settings):
    normal_pass = settings.get("normal_pass", False)
    glossy_color_pass = settings.get("glossy_color_pass", False)
    layer = scene.view_layers[0]
    layer.use_pass_normal = normal_pass
    layer.use_pass_glossy_color = glossy_color_pass
    scene.render.use_compositing = normal_pass or glossy_color_pass
    scene.cycles.use_denoising = settings.get("denoise", False)
    if scene.cycles.use_denoising:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        scene.cycles.denoising_use_gpu = scene.cycles.device == "GPU"
        scene.cycles.denoising_quality = "HIGH"
        scene.cycles.denoising_prefilter = "ACCURATE"
        layer.cycles.use_denoising = True
        layer.cycles.denoising_store_passes = False
    if not scene.render.use_compositing:
        return None

    compositor = scene.compositing_node_group
    if compositor is None:
        compositor = bpy.data.node_groups.new("Probe film pass", "CompositorNodeTree")
        scene.compositing_node_group = compositor
        compositor.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    compositor.nodes.clear()
    layers = compositor.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    composite = compositor.nodes.new("NodeGroupOutput")
    if glossy_color_pass:
        compositor.links.new(layers.outputs["Glossy Color"], composite.inputs["Image"])
        return compositor
    # Map the signed Normal data to the display range without clipping negative axes.
    encode = compositor.nodes.new("ShaderNodeVectorMath")
    encode.operation = "MULTIPLY_ADD"
    encode.inputs[1].default_value = encode.inputs[2].default_value = (0.5, 0.5, 0.5)
    compositor.links.new(layers.outputs["Normal"], encode.inputs[0])
    compositor.links.new(encode.outputs["Vector"], composite.inputs["Image"])
    return compositor

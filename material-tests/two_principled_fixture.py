# SPDX-License-Identifier: GPL-2.0-or-later

from mixed_glossy_fixture import configure_render_outputs, mix_lobes


def tangent_normal(fixture, tree, inverted):
    image = fixture.float_image
    image.colorspace_settings.name = "Non-Color"
    image.pixels = [
        value
        for y in range(4)
        for x in range(4)
        for value in (0.23 + 0.15 * x, 0.20 + 0.19 * y, 0.94, 1.0)
    ]
    image.pack()
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    normal = tree.nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.uv_map = "UVMap"
    normal.inputs["Strength"].default_value = 0.75
    color = texture.outputs["Color"]
    if inverted:
        reverse = tree.nodes.new("ShaderNodeVectorMath")
        reverse.operation = "MULTIPLY_ADD"
        reverse.inputs[1].default_value = (-1.0, 0.5, 1.0)
        reverse.inputs[2].default_value = (1.0, 0.4, 0.0)
        tree.links.new(color, reverse.inputs[0])
        color = reverse.outputs["Vector"]
    tree.links.new(color, normal.inputs["Color"])
    return normal.outputs["Normal"]


def configure(fixture, tree, output, settings):
    first = tree.nodes.new("ShaderNodeBsdfPrincipled")
    second = tree.nodes.new("ShaderNodeBsdfPrincipled")
    first.inputs["Base Color"].default_value = (0.68, 0.12, 0.043, 1)
    second.inputs["Base Color"].default_value = (0.11, 0.34, 0.65, 1)
    for branch, roughness, ior, specular in ((first, 0.2, 1.45, 0.5),
                                             (second, 0.55, 1.59, 0.62)):
        branch.distribution = settings.get("distribution", "MULTI_GGX")
        branch.inputs["Roughness"].default_value = roughness
        branch.inputs["IOR"].default_value = ior
        branch.inputs["Specular IOR Level"].default_value = specular

    if settings.get("strong_specular"):
        first.inputs["IOR"].default_value = 2.3
        second.inputs["IOR"].default_value = 2.1
    mode = settings.get("normal", "mapped")
    if mode != "flat":
        tree.links.new(tangent_normal(fixture, tree, False), first.inputs["Normal"])
    if mode == "independent":
        tree.links.new(tangent_normal(fixture, tree, True), second.inputs["Normal"])
    if settings.get("linked_lobes"):
        roughness = tree.nodes.new("ShaderNodeTexImage")
        roughness.image = fixture.image
        tree.links.new(roughness.outputs["Color"], first.inputs["Roughness"])
        facing = tree.nodes.new("ShaderNodeLayerWeight")
        facing.inputs["Blend"].default_value = 0.23
        specular = tree.nodes.new("ShaderNodeMath")
        specular.operation = "MULTIPLY"
        specular.inputs[1].default_value = 4.0
        tree.links.new(facing.outputs["Fresnel"], specular.inputs[0])
        tree.links.new(specular.outputs[0], second.inputs["Specular IOR Level"])
    if settings.get("zero_specular_first"):
        first.inputs["Specular IOR Level"].default_value = 0.0
    if settings.get("metallic_pair"):
        first.inputs["Metallic"].default_value = 0.55
        second.inputs["Metallic"].default_value = 0.9
        if settings.get("linked_metallic"):
            metal = tree.nodes.new("ShaderNodeTexImage")
            metal.image = fixture.image
            tree.links.new(metal.outputs["Color"], first.inputs["Metallic"])
    if settings.get("anisotropy") or settings.get("bump_second"):
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = fixture.image
        if settings.get("anisotropy"):
            for branch, bias, rotation in ((first, 0.54, 0.12), (second, 0.67, 0.31)):
                aspect = tree.nodes.new("ShaderNodeMath")
                aspect.operation = "MULTIPLY_ADD"
                aspect.inputs[1].default_value = 0.25
                aspect.inputs[2].default_value = bias
                tree.links.new(texture.outputs["Color"], aspect.inputs[0])
                tree.links.new(aspect.outputs[0], branch.inputs["Anisotropic"])
                branch.inputs["Anisotropic Rotation"].default_value = rotation
        if settings.get("bump_second"):
            bump = tree.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.25
            tree.links.new(texture.outputs["Color"], bump.inputs["Height"])
            tree.links.new(bump.outputs["Normal"], second.inputs["Normal"])
    for branch, weight in zip((first, second), settings.get("coat_weights", (0.0, 0.0))):
        branch.inputs["Coat Weight"].default_value = weight
        branch.inputs["Coat Roughness"].default_value = 0.2
    if "sheen_weights" in settings:
        for branch, weight in zip((first, second), settings["sheen_weights"], strict=True):
            branch.inputs["Sheen Weight"].default_value = weight
            branch.inputs["Sheen Roughness"].default_value = settings.get("sheen_roughness", 0.5)
    if settings.get("outside") == "coat-tint":
        second.inputs["Coat Tint"].default_value = (0.2, 0.8, 0.4, 1.0)

    mix = mix_lobes(tree, first, second, settings.get("reverse", False),
                   settings.get("factor", 0.8))
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    compositor = configure_render_outputs(fixture.scene, settings)
    if compositor is not None:
        fixture.scene.render.compositor_device = "CPU"
    return compositor

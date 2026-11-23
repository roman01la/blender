# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from glossy_transparency_fixture import external_rgba
from mixed_glossy_fixture import configure_render_outputs, diffuse_lobe
from mixed_principled_fixture import bump


def set_factor(tree, socket, value, reverse=False):
    if isinstance(value, (int, float)):
        socket.default_value = 1.0 - value if reverse else value
        return
    if reverse:
        inverse = tree.nodes.new("ShaderNodeMath")
        inverse.operation = "SUBTRACT"
        inverse.inputs[0].default_value = 1.0
        tree.links.new(value, inverse.inputs[1])
        value = inverse.outputs[0]
    tree.links.new(value, socket)


def mix_bsdfs(tree, first, second, factor, reverse=False):
    mix = tree.nodes.new("ShaderNodeMixShader")
    set_factor(tree, mix.inputs[0], factor, reverse)
    branches = (second, first) if reverse else (first, second)
    for shader, socket in zip(branches, mix.inputs[1:], strict=True):
        tree.links.new(shader.outputs[0], socket)
    return mix


def select_image(tree, choices, selectors, socket_type):
    selected = choices[0]
    for selector, choice in zip(selectors, choices[1:], strict=True):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = socket_type
        is_color = socket_type == "RGBA"
        inputs = mix.inputs[6:8] if is_color else mix.inputs[2:4]
        tree.links.new(selector, mix.inputs[0])
        tree.links.new(selected, inputs[0])
        tree.links.new(choice, inputs[1])
        selected = mix.outputs[2 if is_color else 0]
    return selected


def paper_inputs(fixture, tree, settings):
    if not settings.get("image"):
        return None, 0.63
    sources = [external_rgba(fixture, 16, "wireframe")]
    if settings.get("random_images"):
        sources = [fixture.image, sources[0], fixture.float_image]
    images = []
    for source in sources:
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = source
        images.append(texture)
    if len(images) == 1:
        return images[0].outputs["Color"], images[0].outputs["Alpha"]

    random = tree.nodes.new("ShaderNodeObjectInfo").outputs["Random"]
    selectors = []
    for operation, threshold in (("LESS_THAN", 0.33), ("GREATER_THAN", 0.66)):
        compare = tree.nodes.new("ShaderNodeMath")
        compare.operation = operation
        compare.inputs[1].default_value = threshold
        tree.links.new(random, compare.inputs[0])
        selectors.append(compare.outputs[0])
    return (select_image(tree, [image.outputs["Color"] for image in images], selectors, "RGBA"),
            select_image(tree, [image.outputs["Alpha"] for image in images], selectors, "FLOAT"))


def principled_translucent_surface(fixture, tree, settings):
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.distribution = settings.get("distribution", "GGX")
    principled.inputs["IOR"].default_value = 1.45
    principled.inputs["Roughness"].default_value = 0.5
    principled.inputs["Metallic"].default_value = settings.get("metallic", 0)
    translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
    color, alpha = paper_inputs(fixture, tree, settings)
    if color is not None:
        coordinate = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.25, 0.8, 1)
        tree.links.new(coordinate.outputs["UV"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], color.node.inputs["Vector"])
        tree.links.new(color, principled.inputs["Base Color"])
        tree.links.new(color, translucent.inputs["Color"])
    set_factor(tree, principled.inputs["Alpha"], settings.get("fixed_alpha", alpha))
    if settings.get("only_principled"):
        return principled
    reverse = settings.get("order") == "reverse"
    if "mix" in settings:
        return mix_bsdfs(tree, principled, translucent, settings["mix"], reverse)
    addition = tree.nodes.new("ShaderNodeAddShader")
    branches = (translucent, principled) if reverse else (principled, translucent)
    for shader, socket in zip(branches, addition.inputs, strict=True):
        tree.links.new(shader.outputs[0], socket)
    return addition


def paper_surface(fixture, tree, settings):
    if settings.get("shadow") == "opaque":
        return diffuse_lobe(tree, (0.55, 0.38, 0.23, 1.0))
    if settings.get("principled_translucent"):
        return principled_translucent_surface(fixture, tree, settings)
    diffuse = diffuse_lobe(tree, (0.55, 0.38, 0.23, 1.0))
    pair = settings.get("pair")
    translucent = None
    if pair != "transparent":
        translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
        translucent.inputs["Color"].default_value = (0.27, 0.55, 0.71, 1.0)
    color, alpha = paper_inputs(fixture, tree, settings)
    shades = [diffuse, translucent] if translucent else [diffuse]
    if color:
        for shade in shades:
            tree.links.new(color, shade.inputs["Color"])
    if settings.get("bump"):
        normal = tree.nodes.new("ShaderNodeBump")
        normal.inputs["Strength"].default_value = 0.36
        normal.inputs["Distance"].default_value = 0.11
        tree.links.new(color, normal.inputs["Height"])
        for shade in shades:
            tree.links.new(normal.outputs["Normal"], shade.inputs["Normal"])

    if pair == "translucent":
        return mix_bsdfs(tree, diffuse, translucent, 0.3)
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    if settings.get("shadow") == "mixed":
        transparent.inputs["Color"].default_value = (0.82, 0.55, 0.37, 1.0)
    reverse = settings.get("order") == "reverse"
    if pair == "transparent":
        return mix_bsdfs(tree, transparent, diffuse, alpha, reverse)
    if settings.get("order") == "middle":
        # Same closure weights as mixing 63% of the 70/30 paper pair.
        inner = mix_bsdfs(tree, transparent, translucent, 0.189 / 0.559)
        return mix_bsdfs(tree, diffuse, inner, 0.559)
    opaque = mix_bsdfs(tree, diffuse, translucent, 0.3, reverse)
    return mix_bsdfs(tree, transparent, opaque, alpha, reverse)


def off_camera_bump(scene):
    # Synchronize a Bump shader to enable ray differentials elsewhere in the scene.
    bpy.ops.mesh.primitive_plane_add(size=0.1, location=scene.camera.location * 4)
    hidden = bpy.context.object
    hidden.name = "Off-camera paper Bump"
    hidden.visible_camera = hidden.visible_shadow = False
    material = bpy.data.materials.new("Off-camera paper Bump")
    material.use_nodes = True
    hidden.data.materials.append(material)
    tree = material.node_tree
    tree.nodes.clear()
    diffuse = diffuse_lobe(tree, (0.8, 0.7, 0.6, 1.0))
    bump(tree, diffuse, 5.0, False)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
    return tree


def configure(fixture, tree, output, settings):
    scene = fixture.scene
    scene.cycles.transparent_max_bounces = 8
    scene.render.film_transparent = settings.get("alpha", False)
    if light := scene.objects.get("Paper transmission light"):
        data = light.data
        bpy.data.objects.remove(light, do_unlink=True)
        bpy.data.lights.remove(data)
    for name in ("Paper closure shadow probe", "Off-camera paper Bump"):
        if previous := scene.objects.get(name):
            material = previous.data.materials[0]
            bpy.data.objects.remove(previous, do_unlink=True)
            bpy.data.materials.remove(material)

    trees = [tree]
    if settings.get("global_bump"):
        trees.append(off_camera_bump(scene))
    if settings.get("shadow"):
        from transparent_fixture import plate

        light = scene.objects["Light"]
        direction = light.location.normalized()
        occluder = plate(direction * 2.35, direction)
        occluder.name = "Paper closure shadow probe"
        occluder.visible_camera = False
        material = bpy.data.materials.new("Probe paper shadow")
        material.use_nodes = True
        occluder.data.materials.append(material)
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        trees.append(tree)
    else:
        back = bpy.data.lights.new("Paper transmission light", "POINT")
        back.energy = 530
        light = bpy.data.objects.new("Paper transmission light", back)
        scene.collection.objects.link(light)
        light.location = (-2.5, 3.5, -0.5)
    surface = paper_surface(fixture, tree, settings)
    tree.links.new(surface.outputs[0], output.inputs["Surface"])
    if compositor := configure_render_outputs(scene, settings):
        trees.append(compositor)
    return trees

# SPDX-License-Identifier: GPL-2.0-or-later

from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import zlib

import bpy

from mixed_glossy_fixture import configure_render_outputs, glossy_lobe, mix_lobes


def png_chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload +
            struct.pack(">I", zlib.crc32(tag + payload)))


def external_rgba(fixture, depth, mode):
    name = f"Probe {mode} RGBA{depth}"
    if image := bpy.data.images.get(name):
        return image
    if not hasattr(fixture, "glossy_images"):
        fixture.glossy_images = TemporaryDirectory(prefix="blender-probe-glossy-", dir="/tmp")

    size = 8
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            # Each precision sample rounds to the same 8-bit value, 128.
            detail = 128 * 257 + 220 * ((x + y * 3) % size) // (size - 1) - 110
            if mode == "color":
                rgba = (detail, 0, 0, 65535)
            elif mode == "alpha":
                rgba = (65535, 65535, 65535, detail)
            else:
                alpha = (0, 17990, 40349, 65535)[(x + y) % 4]
                rgba = ((200 + 5 * x) * 257, (55 + 20 * y) * 257, 77 * 257, alpha)
            if depth == 16:
                rows.extend(struct.pack(">4H", *rgba))
            else:
                rows.extend((value + 128) // 257 for value in rgba)

    path = Path(fixture.glossy_images.name) / f"{mode}-{depth}.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, depth, 6, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )
    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = name
    image.colorspace_settings.name = "sRGB" if mode == "wireframe" else "Non-Color"
    image.alpha_mode = "STRAIGHT"
    return image


def test_texture(fixture, tree, settings, mode):
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = external_rgba(fixture, settings["depth"], mode)
    texture.interpolation = settings.get("interpolation", "Linear")
    return texture


def precision_surface(fixture, tree, settings):
    mode = settings["precision"]
    texture = test_texture(fixture, tree, settings, mode)
    if mode == "color":
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(texture.outputs["Color"], separate.inputs["Color"])
        value = separate.outputs["Red"]
    else:
        value = texture.outputs["Alpha"]
    detail = tree.nodes.new("ShaderNodeMath")
    detail.operation = "MULTIPLY_ADD"
    detail.inputs[1].default_value = 255.0
    detail.inputs[2].default_value = -127.5
    tree.links.new(value, detail.inputs[0])
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(detail.outputs[0], emission.inputs["Color"])
    return emission


def ordered_glossy_mix(fixture, tree, settings):
    sharp = glossy_lobe(tree, (0.2, 0.2, 0.2, 1), settings.get("sharp", "GGX"), 0.0)
    rough = glossy_lobe(tree, (0.2, 0.2, 0.2, 1), settings.get("rough", "BECKMANN"),
                        0.6131393313407898)
    fresnel = tree.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.25
    tint = tree.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs[6].default_value = (0.006273983, 0.006273983, 0.006273983, 1.0)
    tint.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
    tree.links.new(fresnel.outputs["Factor"], tint.inputs[0])
    for lobe in (sharp, rough):
        tree.links.new(tint.outputs[2], lobe.inputs["Color"])

    reverse = settings.get("reverse", False)
    # The control folds away the sharp Glossy branch in stock SVM.
    reflect = mix_lobes(tree, sharp, rough, reverse, 1.0 if settings.get("rough_only") else 0.147541)
    if settings.get("opaque") or settings.get("shadow") == "opaque":
        return reflect

    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = ((0.1234803, 0.11820004, 0.11820004, 1.0)
                                                  if settings.get("camera") else
                                                  (0.82, 0.55, 0.37, 1.0))
    if settings.get("camera"):
        factor = tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    elif settings.get("image_alpha"):
        image = test_texture(fixture, tree, settings, "wireframe")
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.28
        ramp.color_ramp.elements[1].position = 0.72
        tree.links.new(image.outputs["Alpha"], ramp.inputs["Factor"])
        factor = ramp.outputs["Color"]
        shadow_value = None
        if settings.get("shadow_ao"):
            ao = tree.nodes.new("ShaderNodeAmbientOcclusion")
            ao.inside = True
            ao.inputs["Distance"].default_value = 5.0
            shadow_value = ao.outputs["AO"]
        elif settings.get("shadow_voronoi"):
            voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
            voronoi.feature = "F1"
            voronoi.voronoi_dimensions = "3D"
            voronoi.inputs["Detail"].default_value = 0.0
            voronoi.normalize = settings.get("normalize", False)
            shadow_value = voronoi.outputs["Distance"]
        if shadow_value is not None:
            multiply = tree.nodes.new("ShaderNodeMath")
            multiply.operation = "MULTIPLY"
            tree.links.new(factor, multiply.inputs[0])
            tree.links.new(shadow_value, multiply.inputs[1])
            factor = multiply.outputs[0]
    else:
        return mix_lobes(tree, transparent, reflect, reverse, settings.get("factor", 0.57))
    if reverse:
        invert = tree.nodes.new("ShaderNodeMath")
        invert.operation = "SUBTRACT"
        invert.inputs[0].default_value = 1.0
        tree.links.new(factor, invert.inputs[1])
        factor = invert.outputs[0]
    surface = mix_lobes(tree, transparent, reflect, reverse, 0.0)
    tree.links.new(factor, surface.inputs[0])
    return surface


def glossy_mix(fixture, tree, settings):
    if settings.get("ordered_glossy_mix"):
        return ordered_glossy_mix(fixture, tree, settings)
    glossy = glossy_lobe(tree, (0.74, 0.54, 0.32, 1), settings["distribution"],
                         0.1414213478565216 if settings.get("classroom") else 0.29)
    if settings.get("shadow") == "opaque":
        return glossy
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    if settings.get("tinted"):
        transparent.inputs["Color"].default_value = (0.82, 0.55, 0.37, 1)

    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.43
    if "depth" in settings:
        texture = test_texture(fixture, tree, settings, "wireframe")
        tree.links.new(texture.outputs["Color"], glossy.inputs["Color"])
        factor = texture.outputs["Alpha"]
        if settings.get("reverse"):
            inverse = tree.nodes.new("ShaderNodeMath")
            inverse.operation = "SUBTRACT"
            inverse.inputs[0].default_value = 1.0
            tree.links.new(factor, inverse.inputs[1])
            factor = inverse.outputs[0]
        tree.links.new(factor, mix.inputs[0])
    closures = (glossy, transparent) if settings.get("reverse") else (transparent, glossy)
    for closure, socket in zip(closures, list(mix.inputs)[1:], strict=True):
        tree.links.new(closure.outputs[0], socket)
    return mix


def configure(fixture, tree, output, settings):
    scene = fixture.scene
    scene.cycles.transparent_max_bounces = 8
    scene.render.film_transparent = settings.get("alpha", False)
    if previous := bpy.data.objects.get("Glossy transparency shadow probe"):
        previous_material = previous.data.materials[0]
        bpy.data.objects.remove(previous, do_unlink=True)
        bpy.data.materials.remove(previous_material)
    trees = [tree]
    if settings.get("shadow"):
        from transparent_fixture import plate

        light = scene.objects["Light"]
        direction = light.location.normalized()
        occluder = plate(direction * 2.35, direction)
        occluder.name = "Glossy transparency shadow probe"
        occluder.visible_camera = False
        material = bpy.data.materials.new("Probe Glossy shadow")
        material.use_nodes = True
        occluder.data.materials.append(material)
        shadow_tree = material.node_tree
        shadow_tree.nodes.clear()
        shadow_surface = glossy_mix(fixture, shadow_tree, settings)
        shadow_output = shadow_tree.nodes.new("ShaderNodeOutputMaterial")
        shadow_tree.links.new(shadow_surface.outputs[0], shadow_output.inputs["Surface"])
        trees.append(shadow_tree)
    else:
        if settings.get("precision"):
            fixture.material.cycles.emission_sampling = "NONE"
        surface = precision_surface(fixture, tree, settings) if settings.get("precision") else (
            glossy_mix(fixture, tree, settings))
        tree.links.new(surface.outputs[0], output.inputs["Surface"])
    if compositor := configure_render_outputs(scene, settings):
        trees.append(compositor)
    return trees

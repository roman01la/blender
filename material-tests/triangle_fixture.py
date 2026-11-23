# SPDX-License-Identifier: GPL-2.0-or-later
"""Emissive triangle fixtures shared by the material and adaptive render probes."""

from math import pi

import bpy
from mathutils import Quaternion


def add_triangle(scene, name, position, size, color, strength, sampling):
    half = size / 2
    vertices = [(-half, -half, 0), (0, half, 0), (half, -half, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = position
    scene.collection.objects.link(obj)

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.cycles.emission_sampling = sampling
    tree = material.node_tree
    tree.nodes.clear()
    light = tree.nodes.new("ShaderNodeEmission")
    light.inputs["Color"].default_value = (*color, 1)
    light.inputs["Strength"].default_value = strength
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(light.outputs[0], output.inputs["Surface"])
    mesh.materials.append(material)
    return obj


def texture_emission(obj, pattern, flip_uv=False):
    material = obj.data.materials[0]
    tree = material.node_tree
    emission = tree.nodes.get("Emission")
    for node in list(tree.nodes):
        if node.type not in {"EMISSION", "OUTPUT_MATERIAL"}:
            tree.nodes.remove(node)
    output = tree.nodes.get("Material Output").inputs["Surface"]
    if not output.is_linked:
        tree.links.new(emission.outputs[0], output)
    if pattern is None:
        return

    coord = tree.nodes.new("ShaderNodeTexCoord")
    if pattern == "image":
        uv_layer = obj.data.uv_layers.active or obj.data.uv_layers.new(name="Emitter UV")
        # Use one side of the image so flipping UVs changes the emitter's
        # illumination as well as the texture at an individual sampled point.
        for corner, (u, v) in zip(uv_layer.data, ((0.04, 0.1), (0.18, 0.88), (0.43, 0.26)),
                                  strict=True):
            corner.uv = (1 - u if flip_uv else u, v)
        image_name = material.name + " emission image"
        image = bpy.data.images.get(image_name)
        if image is None:
            image = bpy.data.images.new(image_name, width=4, height=4, float_buffer=True)
            image.colorspace_settings.name = "Non-Color"
            r, g, b, _ = emission.inputs["Color"].default_value
            image.pixels[:] = [
                value for y in range(4) for x in range(4)
                for value in (r * (0.14 + 0.23 * x + 0.05 * y),
                              g * (0.2 + 0.17 * y + 0.06 * x),
                              b * (0.24 + 0.2 * (3 - x) + 0.04 * y), 1)
            ]
            image.pack()
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.extension = "EXTEND"
        tree.links.new(coord.outputs["UV"], texture.inputs["Vector"])
        tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    elif pattern == "noise":
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 4.5
        noise.inputs["Detail"].default_value = 1.5
        multiply = tree.nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = emission.inputs["Strength"].default_value
        tree.links.new(coord.outputs["Generated"], noise.inputs["Vector"])
        tree.links.new(noise.outputs["Fac"], multiply.inputs[0])
        tree.links.new(multiply.outputs[0], emission.inputs["Strength"])
    elif pattern == "sky":
        sky = tree.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "HOSEK_WILKIE"
        sky.sun_direction = (-0.946154, 0.061538, 0.317814)
        sky.turbidity = 2.9
        sky.ground_albedo = 0.3
        tree.links.new(coord.outputs["Reflection"], sky.inputs["Vector"])
        curves = tree.nodes.new("ShaderNodeRGBCurve")
        curves.mapping.curves[3].points.new(0.4, 0.34)
        curves.mapping.update()
        tree.links.new(sky.outputs["Color"], curves.inputs["Color"])
        hsv = tree.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = 0.73
        tree.links.new(curves.outputs["Color"], hsv.inputs["Color"])
        tree.links.new(hsv.outputs["Color"], emission.inputs["Color"])
    elif pattern == "pixel-wireframe":
        wireframe = tree.nodes.new("ShaderNodeWireframe")
        wireframe.use_pixel_size = True
        wireframe.inputs["Size"].default_value = 2
        tree.links.new(wireframe.outputs["Factor"], emission.inputs["Strength"])
    else:
        raise ValueError(f"Unknown emitter texture {pattern}")


def mix_transparent_emission(obj, mode):
    tree = obj.data.materials[0].node_tree
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    emission = tree.nodes.get("Emission")
    mix = tree.nodes.new("ShaderNodeMixShader")
    if mode == "backface":
        factor = tree.nodes.new("ShaderNodeNewGeometry").outputs["Backfacing"]
    else:
        coord = tree.nodes.new("ShaderNodeTexCoord")
        split = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(coord.outputs["Generated"], split.inputs[0])
        weight = tree.nodes.new("ShaderNodeMath")
        weight.operation = "MULTIPLY_ADD"
        weight.inputs[1].default_value = 0.6
        weight.inputs[2].default_value = 0.2
        tree.links.new(split.outputs["X"], weight.inputs[0])
        factor = weight.outputs[0]
    tree.links.new(factor, mix.inputs[0])
    tree.links.new(transparent.outputs[0], mix.inputs[1])
    tree.links.new(emission.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], tree.nodes.get("Material Output").inputs["Surface"])


def set_principled_emission(obj, coat_weight):
    tree = obj.data.materials[0].node_tree
    emission = tree.nodes.get("Emission")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0, 0, 0, 1)
    principled.inputs["Specular IOR Level"].default_value = 0
    principled.inputs["Coat Weight"].default_value = coat_weight
    principled.inputs["Coat Roughness"].default_value = 0.24
    principled.inputs["Coat IOR"].default_value = 2.2
    for name in ("Color", "Strength"):
        source = emission.inputs[name]
        target = principled.inputs[f"Emission {name}"]
        if source.is_linked:
            tree.links.new(source.links[0].from_socket, target)
        else:
            target.default_value = source.default_value
    tree.links.new(principled.outputs[0], tree.nodes.get("Material Output").inputs["Surface"])


def add_receiver_bump(receiver):
    tree = receiver.id_data
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.37
    tree.links.new(noise.outputs["Factor"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], receiver.inputs["Normal"])


def add_benchmark_emitters(scene, *, textured=False):
    """Keep the established camera, lights, world and materials in the 1440p test."""
    warm = add_triangle(scene, "Warm triangle emitter", (1.8, 1.5, 2.65), 1.45,
                        (1, 0.42, 0.18), 8, "FRONT_BACK")
    cool = add_triangle(scene, "Cool triangle emitter", (-2.1, 1.1, 2.2), 1.3,
                        (0.55, 0.85, 1), 5, "FRONT_BACK")
    cool.rotation_euler = (0.16, -0.25, 0.2)
    cool.scale = (0.85, 1.2, 1)
    if textured:
        texture_emission(warm, "image")
        texture_emission(cool, "noise")
    return warm, cool


class TriangleProbe:
    """Leave enough fill for the zero-emission control to use the same Light Tree path."""

    def __init__(self, scene):
        self.scene = scene
        self.light = next(obj for obj in scene.objects if obj.type == "LIGHT")
        point_light = bpy.data.lights.new("Probe point fill", "POINT")
        point_light.energy = 1
        point_light.shadow_soft_size = 0
        self.point = bpy.data.objects.new("Probe point fill", point_light)
        self.point.location = (-2, -1, 2)
        scene.collection.objects.link(self.point)
        self.background = scene.world.node_tree.nodes.get("Background")
        self.emitter = add_triangle(scene, "Probe triangle", (0, 0, 0), 2.6,
                                    (1, 0.4, 0.12), 8, "FRONT_BACK")
        self.instance = self.emitter.copy()
        self.instance.name = "Probe triangle instance"
        self.instance.location = (1.45, -2.5, 2.2)
        self.instance.rotation_euler = (0.1, -0.25, 0.3)
        self.instance.scale = (0.6, 0.8, 1)
        scene.collection.objects.link(self.instance)

    def configure(self, settings, receiver):
        scene = self.scene
        scene.cycles.samples = 128
        scene.render.dither_intensity = 0
        scene.cycles.use_light_tree = settings.get("light_tree", True)
        camera_hit = settings.get("camera_hit", False)
        scene.render.film_transparent = camera_hit
        self.light.data.energy = 4
        self.light.hide_render = settings.get("mesh_only", False)
        self.point.hide_render = not settings.get("point_fill", False)
        self.background.inputs["Strength"].default_value = 0
        sampling = settings["sampling"]
        glossy = settings.get("glossy", False)
        obj = self.emitter
        if camera_hit:
            obj.location = scene.camera.location * 0.46
            orientation = scene.camera.location.to_track_quat("-Z", "Y")
            if settings.get("reverse_facing"):
                orientation = orientation @ Quaternion((1, 0, 0), pi)
            obj.rotation_euler = orientation.to_euler()
            obj.scale = (0.77, 0.77, 0.77)
        elif glossy:
            obj.location = (1.65, -2.04, 1.02)
            obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
            obj.scale = (1.3, 1.3, 1)
        else:
            obj.location = (-2.45, -1.9, 2.5)
            reverse = sampling == "BACK" or settings.get("reverse_facing")
            obj.rotation_euler = (pi, 0, 0) if reverse else (0, 0, 0)
            obj.scale = (1, 1, 1)
        hidden = glossy or settings.get("camera_hidden", False)
        obj.visible_camera = not hidden
        sampled = not settings.get("camera_only", False)
        obj.visible_diffuse = obj.visible_glossy = obj.visible_transmission = sampled
        obj.visible_volume_scatter = obj.visible_shadow = sampled
        self.instance.visible_camera = not settings.get("camera_hidden", False)
        self.instance.hide_render = not settings.get("instanced", False)
        material = obj.data.materials[0]
        material.cycles.emission_sampling = sampling
        strength = material.node_tree.nodes.get("Emission").inputs["Strength"]
        strength.default_value = settings.get("strength", 8)
        texture = settings.get("texture")
        flip_uv = settings.get("flip_uv", False)
        texture_emission(obj, texture, flip_uv)
        coat = settings.get("principled_coat")
        if coat is not None:
            set_principled_emission(obj, coat)
        if mode := settings.get("transparent_emission"):
            mix_transparent_emission(obj, mode)
        if settings.get("receiver_bump"):
            add_receiver_bump(receiver)
        if glossy:
            receiver.inputs["Metallic"].default_value = 1
            receiver.inputs["Roughness"].default_value = 0.17
            receiver.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
        result = {
            "sampling": material.cycles.emission_sampling,
            "strength": strength.default_value,
            "lamps": [light.name for light in (self.light, self.point) if not light.hide_render],
            "emitters": [
                [item.name, list(item.location), [round(angle, 6) for angle in item.rotation_euler],
                 list(item.scale),
                 item.visible_camera, [list(polygon.vertices) for polygon in item.data.polygons]]
                for item in (obj, self.instance) if not item.hide_render
            ],
        }
        if texture:
            result["texture"] = {"pattern": texture, "flip_uv": flip_uv}
        if coat is not None:
            result["principled_coat"] = coat
            result["light_tree"] = scene.cycles.use_light_tree
        if mode:
            result["transparent_emission"] = mode
            result["light_tree"] = scene.cycles.use_light_tree
            result["emitter_graph"] = [(node.bl_idname, [(link.from_node.name, link.from_socket.identifier)
                                                       for socket in node.inputs for link in socket.links])
                                       for node in material.node_tree.nodes]
        return result

# SPDX-License-Identifier: GPL-2.0-or-later
"""Emissive triangle fixtures shared by the material and adaptive render probes."""

from math import pi

import bpy


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
    else:
        raise ValueError(f"Unknown emitter texture {pattern}")


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
        scene.cycles.use_light_tree = True
        self.light.data.energy = 4
        self.light.hide_render = settings.get("mesh_only", False)
        self.point.hide_render = not settings.get("point_fill", False)
        self.background.inputs["Strength"].default_value = 0
        sampling = settings["sampling"]
        glossy = settings.get("glossy", False)
        obj = self.emitter
        if glossy:
            obj.location = (1.65, -2.04, 1.02)
            obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
            obj.scale = (1.3, 1.3, 1)
        else:
            obj.location = (-2.45, -1.9, 2.5)
            obj.rotation_euler = (pi, 0, 0) if sampling == "BACK" else (0, 0, 0)
            obj.scale = (1, 1, 1)
        hidden = glossy or settings.get("camera_hidden", False)
        obj.visible_camera = not hidden
        self.instance.visible_camera = not settings.get("camera_hidden", False)
        self.instance.hide_render = not settings.get("instanced", False)
        material = obj.data.materials[0]
        material.cycles.emission_sampling = sampling
        strength = material.node_tree.nodes.get("Emission").inputs["Strength"]
        strength.default_value = settings.get("strength", 8)
        texture = settings.get("texture")
        flip_uv = settings.get("flip_uv", False)
        texture_emission(obj, texture, flip_uv)
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
        return result

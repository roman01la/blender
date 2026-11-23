# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from transparent_fixture import plate


def configure(fixture, shader, settings):
    scene = fixture.scene
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = settings["caustics"]
    scene.cycles.blur_glossy = 0.0
    sphere = fixture.sphere
    if previous := bpy.data.objects.get("Glass caustics receiver"):
        material = previous.data.materials[0]
        bpy.data.objects.remove(previous, do_unlink=True)
        bpy.data.materials.remove(material)

    diffuse_probe = settings["diffuse_probe"]
    sphere.visible_camera = not diffuse_probe
    sphere.visible_shadow = not diffuse_probe
    shader.inputs["Roughness"].default_value = 0.11
    shader.inputs["IOR"].default_value = 1.42
    if shader.bl_idname == "ShaderNodeBsdfPrincipled":
        shader.inputs["Base Color"].default_value = (0.24, 0.56, 0.72, 1)
        shader.inputs["Transmission Weight"].default_value = 0.3
    else:
        shader.distribution = "GGX"
        shader.inputs["Color"].default_value = (0.72, 0.84, 0.94, 1)
    if not diffuse_probe:
        return [shader.id_data]

    # Diffuse rays from the plate need to pass through the camera-hidden sphere.
    receiver = plate((0, 0, 0), scene.camera.location)
    receiver.name = "Glass caustics receiver"
    receiver.scale = (0.21,) * 3
    material = bpy.data.materials.new("Glass caustics receiver")
    material.use_nodes = True
    receiver.data.materials.append(material)
    tree = material.node_tree
    tree.nodes.clear()
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.64, 0.57, 0.44, 1)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
    return [shader.id_data, tree]

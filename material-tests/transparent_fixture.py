# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from mathutils import Vector


def plate(center, normal):
    bpy.ops.mesh.primitive_plane_add(size=5, location=center)
    obj = bpy.context.object
    obj.name = "Transparent proof plate"
    obj.rotation_euler = Vector(normal).to_track_quat('Z', 'Y').to_euler()
    return obj


def connect_tint(fixture, bsdf, texture):
    if not texture:
        return
    tree = bsdf.id_data
    if texture == "image":
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = fixture.image
        node.interpolation = "Linear"
    else:
        node = tree.nodes.new("ShaderNodeTexNoise")
        node.inputs["Scale"].default_value = 4.5
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(coordinates.outputs["UV"], node.inputs["Vector"])
    tree.links.new(node.outputs["Color"], bsdf.inputs["Color"])


def connect_query(shader):
    tree = shader.id_data
    ao = tree.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 8
    ao.inputs["Color"].default_value = (0.82, 0.6, 0.42, 1)
    ao.inputs["Distance"].default_value = 1.5
    tree.links.new(ao.outputs["Color"], shader.inputs["Color"])


def configure_mixed(fixture, material_node, mode):
    principled = mode.startswith("principled-")
    mode = mode.removeprefix("principled-")
    scene = fixture.scene
    samples = scene.cycles.samples
    shadow = mode in {"shadow", "opaque-shadow"}
    configure(fixture, material_node, "image-colored-shadow" if shadow else
              "alpha" if mode == "alpha" else
              "limited" if mode == "limited" else "tinted")
    scene.cycles.samples = samples
    if shadow:
        shadow_material = bpy.data.objects["Transparent proof plate"].data.materials[0]
        shadow_material.use_transparent_shadow = mode != "opaque-shadow"
        material_node = next(node for node in shadow_material.node_tree.nodes
                             if node.bl_idname == "ShaderNodeBsdfTransparent")
    tree = material_node.id_data
    diffuse = tree.nodes.new("ShaderNodeBsdfPrincipled" if principled else "ShaderNodeBsdfDiffuse")
    diffuse.inputs["Base Color" if principled else "Color"].default_value = (0.29, 0.47, 0.69, 1.0)
    if principled:
        diffuse.inputs["Roughness"].default_value = 0.32
        diffuse.inputs["Metallic"].default_value = 0.36
    mix = tree.nodes.new("ShaderNodeAddShader" if mode == "add" else "ShaderNodeMixShader")
    if mode == "add":
        inputs = (mix.inputs[0], mix.inputs[1])
    else:
        factor = mix.inputs[0]
        factor.default_value = 0.58
        if mode in {"linked", "alpha", "shadow", "opaque-shadow"}:
            noise = tree.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 3.0
            coordinates = tree.nodes.new("ShaderNodeTexCoord")
            tree.links.new(coordinates.outputs["UV"], noise.inputs["Vector"])
            tree.links.new(noise.outputs["Fac"], factor)
        inputs = (mix.inputs[1], mix.inputs[2])
    closures = (diffuse, material_node) if mode == "reverse" else (material_node, diffuse)
    for closure, socket in zip(closures, inputs, strict=True):
        tree.links.new(closure.outputs[0], socket)
    output = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(mix.outputs[0], output.inputs["Surface"])


def configure(fixture, material_node, mode):
    if mode.startswith("mixed-"):
        configure_mixed(fixture, material_node, mode.removeprefix("mixed-"))
        return
    query = mode.startswith("query-")
    mode = mode.removeprefix("query-")
    texture, _, variant = mode.partition("-")
    if texture in {"image", "noise"}:
        mode = variant
    else:
        texture = None
    scene = fixture.scene
    if obj := bpy.data.objects.get("Transparent proof plate"):
        material = obj.data.materials[0]
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)

    scene.cycles.samples = 32
    scene.cycles.transparent_max_bounces = 1 if mode == "limited" else 8
    scene.cycles.min_transparent_bounces = 3
    scene.render.film_transparent = mode == "alpha"
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    tint = (0.82, 0.48, 0.28, 1)
    if mode in {"unoccluded", "colored-shadow", "opaque-shadow"}:
        if mode == "unoccluded":
            return
        light = bpy.data.objects['Light']
        direction = light.location.normalized()
        if query:
            connect_query(material_node)
        shader = "ShaderNodeBsdfTransparent"
        obj = plate(direction * 2.35, direction)
        obj.visible_camera = False
    elif mode in {"clear", "tinted", "limited", "alpha"}:
        material_node.inputs["Color"].default_value = (1, 1, 1, 1) if mode == "clear" else tint
        connect_tint(fixture, material_node, texture)
        if mode == "alpha":
            return
        direction = scene.camera.location.normalized()
        shader = "ShaderNodeBsdfDiffuse" if query else "ShaderNodeEmission"
        obj = plate(-direction * 1.7, direction)
    else:
        raise ValueError(mode)

    material = bpy.data.materials.new("Transparent proof material")
    obj.data.materials.append(material)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    bsdf = tree.nodes.new(shader)
    bsdf.inputs["Color"].default_value = tint if shader == "ShaderNodeBsdfTransparent" else (
        0.73, 0.6, 0.38, 1.0)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs[0], output.inputs["Surface"])
    material.cycles.emission_sampling = "NONE"
    material.use_transparent_shadow = mode != "opaque-shadow"

    if shader == "ShaderNodeBsdfTransparent":
        connect_tint(fixture, bsdf, texture)
    elif query:
        connect_query(bsdf)

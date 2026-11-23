# SPDX-License-Identifier: GPL-2.0-or-later

from mixed_glossy_fixture import configure_render_outputs
from two_principled_fixture import tangent_normal


def configure(fixture, tree, output, settings):
    sphere = fixture.sphere
    for polygon in sphere.data.polygons:
        polygon.use_smooth = not settings.get("flat")
    transformed = settings.get("transformed")
    sphere.scale = (0.86, 1.24, 0.97) if transformed else (1, 1, 1)
    sphere.rotation_euler = (0.14, -0.22, 0.39) if transformed else (0, 0, 0)
    fixture.material.displacement_method = settings.get("method", "BOTH")

    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (2.9, 1.8, 1.0)
    tree.links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = fixture.image
    image.image.colorspace_settings.name = "Non-Color"
    tree.links.new(mapping.outputs["Vector"], image.inputs["Vector"])
    displacement = tree.nodes.new("ShaderNodeDisplacement")
    displacement.space = "OBJECT"
    displacement.inputs["Scale"].default_value = 0.16
    tree.links.new(image.outputs["Color"], displacement.inputs["Height"])
    tree.links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])

    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.64, 0.43, 0.24, 1)
    principled.inputs["Roughness"].default_value = 0.35
    if base := settings.get("map_base"):
        normal = tangent_normal(fixture, tree, False)
        normal.node.base = base
        normal.node.inputs["Strength"].default_value = 0.55
        tree.links.new(normal, principled.inputs["Normal"])
    tree.links.new(principled.outputs[0], output.inputs["Surface"])
    return configure_render_outputs(fixture.scene, settings)

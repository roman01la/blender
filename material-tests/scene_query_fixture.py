# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

ADVANCED_TEXTURES = {
    "gabor2d", "gabor3d", "gaborphase", "gaborintensity", "gaborimage",
    "voronoi1d", "voronoi2d", "voronoi3d", "voronoi4d", "voronoiedge", "voronoiradius",
    "voronoiimage", "combined",
}

NOISE_OPTIONS = {
    "noise1": ("1D", "MULTIFRACTAL"),
    "noise2": ("2D", "HYBRID_MULTIFRACTAL"),
    "noise3": ("3D", "RIDGED_MULTIFRACTAL"),
    "noise4": ("4D", "HETERO_TERRAIN"),
}


def configure(fixture, node, tree, output, mode, query_output):
    """Give the shader queries nontrivial hits while isolating their result from light visibility."""
    extended, mode = mode.startswith("extended-"), mode.removeprefix("extended-")
    texture = "noise"
    variant, _, base_mode = mode.partition("-")
    noise_options = NOISE_OPTIONS.get(variant) if extended else None
    if noise_options:
        variant, _, base_mode = base_mode.partition("-")
    if extended and variant in (ADVANCED_TEXTURES |
                                {"cubic", "smart", "tube", "sphere", "box", "environment", "mirror"}):
        texture, mode = variant, base_mode
    scene = fixture.scene
    scene.cycles.samples = 32
    if blocker := bpy.data.objects.get("Probe query blocker"):
        bpy.data.objects.remove(blocker, do_unlink=True)
    if mode in {"occluded", "other-hit"}:
        direction = scene.camera.location.normalized()
        center = 1.25 * direction if mode == "occluded" else (0, 0, 1.8)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
        blocker = bpy.context.active_object
        blocker.name = "Probe query blocker"
        blocker.visible_camera = False
        if mode == "occluded":
            blocker.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
            blocker.scale = (2.0, 2.0, 0.05)
        else:
            blocker.rotation_euler = (0.2, 0.1, -0.15)
            blocker.scale = (0.5, 0.4, 0.4)
    if node.bl_idname == "ShaderNodeRaycast":
        # Cycles supplies the shading point and normal for unlinked inputs.
        vectors = {"Position": (9 if mode == "miss" else 0, 0, 2.5),
                   "Direction": (0, 0, -1)}
        for name, vector in vectors.items():
            constant = tree.nodes.new("ShaderNodeCombineXYZ")
            for socket, value in zip(constant.inputs, vector, strict=True):
                socket.default_value = value
            tree.links.new(constant.outputs[0], node.inputs[name])
        node.inputs["Length"].default_value = 4
    elif node.bl_idname == "ShaderNodeAmbientOcclusion":
        node.inputs["Distance"].default_value = 1.5
    elif node.bl_idname == "ShaderNodeBevel":
        node.inputs["Radius"].default_value = 0 if mode == "zero-radius" else 0.55
    emission = tree.nodes.new("ShaderNodeEmission")
    color = node.outputs[query_output]
    if extended:
        if texture in ADVANCED_TEXTURES:
            sampled_color = make_advanced_texture(fixture, tree, texture)
        elif texture == "noise":
            sample = tree.nodes.new("ShaderNodeTexNoise")
            sample.inputs["Scale"].default_value = 2.58
            sampled_color = sample.outputs["Color"]
        else:
            environment = texture in {"environment", "mirror"}
            sample = tree.nodes.new("ShaderNodeTexEnvironment" if environment else "ShaderNodeTexImage")
            sample.image = fixture.image
            coordinates = tree.nodes.new("ShaderNodeTexCoord")
            if texture in {"cubic", "smart"}:
                sample.interpolation = texture.title()
                vector = coordinates.outputs["UV"]
            elif environment:
                sample.projection = "MIRROR_BALL" if texture == "mirror" else "EQUIRECTANGULAR"
                vector = coordinates.outputs["Normal"]
            else:
                sample.projection = texture.upper()
                vector = coordinates.outputs["Generated"]
            tree.links.new(vector, sample.inputs["Vector"])
            sampled_color = sample.outputs["Color"]
        if noise_options:
            noise = tree.nodes.new("ShaderNodeTexNoise")
            noise.noise_dimensions, noise.noise_type = noise_options
            noise.inputs["Distortion"].default_value = 0.4
            noise.inputs["Detail"].default_value = 2.3
            coordinates = tree.nodes.new("ShaderNodeTexCoord")
            if noise.noise_dimensions == "1D":
                separate = tree.nodes.new("ShaderNodeSeparateXYZ")
                tree.links.new(coordinates.outputs["Generated"], separate.inputs["Vector"])
                tree.links.new(separate.outputs["X"], noise.inputs["W"])
            else:
                tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
                if noise.noise_dimensions == "4D":
                    noise.inputs["W"].default_value = 0.87
            sampled_color = multiply_colors(tree, sampled_color, noise.outputs["Color"])
        color = multiply_colors(tree, color, sampled_color)
    tree.links.new(color, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    fixture.material.cycles.emission_sampling = "NONE"


def multiply_colors(tree, first, second):
    multiply = tree.nodes.new("ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs["Fac"].default_value = 1
    tree.links.new(first, multiply.inputs[1])
    tree.links.new(second, multiply.inputs[2])
    return multiply.outputs[0]


def make_advanced_texture(fixture, tree, texture):
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    if texture.startswith("gabor") or texture == "combined":
        gabor = tree.nodes.new("ShaderNodeTexGabor")
        gabor.gabor_type = "2D" if texture in {"gabor2d", "gaborphase", "gaborimage"} else "3D"
        gabor.inputs["Scale"].default_value = 3.4
        gabor.inputs["Frequency"].default_value = 1.7
        gabor.inputs["Anisotropy"].default_value = 0.55
        tree.links.new(coordinates.outputs["Generated"], gabor.inputs["Vector"])
        if texture not in {"combined", "gaborimage"}:
            output = {"gaborphase": "Phase", "gaborintensity": "Intensity"}.get(texture, "Value")
            return gabor.outputs[output]
    if texture.startswith("voronoi") or texture == "combined":
        voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
        modes = {"voronoi1d": ("1D", "F2", "EUCLIDEAN", "Distance"),
                 "voronoi2d": ("2D", "SMOOTH_F1", "MINKOWSKI", "Color"),
                 "voronoi3d": ("3D", "F1", "CHEBYCHEV", "Position"),
                 "voronoi4d": ("4D", "F2", "MANHATTAN", "Distance"),
                 "voronoiedge": ("3D", "DISTANCE_TO_EDGE", "EUCLIDEAN", "Distance"),
                 "voronoiradius": ("4D", "N_SPHERE_RADIUS", "EUCLIDEAN", "Radius")}
        dimensions, feature, metric, output = modes.get(texture, modes["voronoi4d"])
        voronoi.voronoi_dimensions = dimensions
        voronoi.feature = feature
        voronoi.distance = metric
        for name, value in (("W", 0.37), ("Scale", 2.2), ("Detail", 1.35),
                            ("Roughness", 0.53), ("Smoothness", 0.7), ("Exponent", 1.3)):
            if socket := voronoi.inputs.get(name):
                socket.default_value = value
        if vector := voronoi.inputs.get("Vector"):
            tree.links.new(coordinates.outputs["Generated"], vector)
        if dimensions == "1D":
            separate = tree.nodes.new("ShaderNodeSeparateXYZ")
            tree.links.new(coordinates.outputs["Generated"], separate.inputs["Vector"])
            tree.links.new(separate.outputs["X"], voronoi.inputs["W"])
        if texture not in {"combined", "voronoiimage"}:
            return voronoi.outputs[output]
    if texture == "combined":
        tree.links.new(gabor.outputs["Intensity"], voronoi.inputs["W"])
        first, second = gabor.outputs["Value"], voronoi.outputs[output]
    else:
        image = tree.nodes.new("ShaderNodeTexImage")
        image.image = fixture.image
        image.projection = "BOX"
        image.interpolation = "Cubic"
        tree.links.new(coordinates.outputs["Generated"], image.inputs["Vector"])
        first = image.outputs["Color"]
        second = gabor.outputs["Value"] if texture == "gaborimage" else voronoi.outputs[output]
    return multiply_colors(tree, first, second)

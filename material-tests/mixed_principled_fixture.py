# SPDX-License-Identifier: GPL-2.0-or-later

COLORS = ((0.71, 0.11, 0.05, 1), (0.04, 0.29, 0.66, 1))


def diffuse_principled(tree, color):
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Specular IOR Level"].default_value = 0
    return shader


def varying_factor(tree, reverse, axis="Z"):
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coordinates.outputs["Generated"], separate.inputs[0])
    factor = tree.nodes.new("ShaderNodeMath")
    factor.operation = "MULTIPLY_ADD"
    factor.use_clamp = True
    factor.inputs[1].default_value = -1.5 if reverse else 1.5
    factor.inputs[2].default_value = 1.25 if reverse else -0.25
    tree.links.new(separate.outputs[axis], factor.inputs[0])
    return factor.outputs[0]


def bump(tree, shader, scale, invert):
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    normal = tree.nodes.new("ShaderNodeBump")
    normal.invert = invert
    normal.inputs["Strength"].default_value = 0.65
    normal.inputs["Distance"].default_value = 0.13
    tree.links.new(noise.outputs["Fac"], normal.inputs["Height"])
    tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])


def nested_mix(tree, settings):
    colors = (*COLORS, (0.22, 0.66, 0.04, 1), (0.6, 0.19, 0.53, 1))
    # Lambert lobes with the same normal evaluate like one Diffuse node
    # fed by the weighted sum of their colors.
    reference = settings["mode"] == "nested-reference"
    reverse = settings.get("reverse", False)
    varying = settings.get("varying", True)
    if reference:
        leaves = colors
    else:
        shaders = [diffuse_principled(tree, color) for color in colors]
        if settings.get("explicit_normal"):
            normal = tree.nodes.new("ShaderNodeNewGeometry").outputs["Normal"]
            for shader in shaders:
                tree.links.new(normal, shader.inputs["Normal"])
        leaves = [shader.outputs[0] for shader in shaders]

    def mix(left, right, axis):
        node = tree.nodes.new("ShaderNodeMixRGB" if reference else "ShaderNodeMixShader")
        if varying:
            tree.links.new(varying_factor(tree, reverse, axis), node.inputs[0])
        else:
            node.inputs[0].default_value = 0.5
        branches = (right, left) if reverse else (left, right)
        for value, socket in zip(branches, node.inputs[1:], strict=True):
            if isinstance(value, tuple):
                socket.default_value = value
            else:
                tree.links.new(value, socket)
        return node.outputs[0]

    root = mix(mix(*leaves[:2], "X"), mix(*leaves[2:], "Y"), "Z")
    if reference:
        surface = tree.nodes.new("ShaderNodeBsdfDiffuse")
        tree.links.new(root, surface.inputs["Color"])
        return surface.outputs[0]
    return root


def configure(tree, output, settings):
    mode = settings["mode"]
    if mode in {"nested", "nested-reference"}:
        tree.links.new(nested_mix(tree, settings), output.inputs["Surface"])
        return
    if mode == "single-principled":
        surface = diffuse_principled(tree, COLORS[0])
    elif mode == "varying-diffuse-reference":
        colors = tree.nodes.new("ShaderNodeMixRGB")
        for socket, color in zip(colors.inputs[1:], COLORS, strict=True):
            socket.default_value = color
        tree.links.new(varying_factor(tree, False), colors.inputs[0])
        surface = tree.nodes.new("ShaderNodeBsdfDiffuse")
        tree.links.new(colors.outputs[0], surface.inputs["Color"])
    else:
        colors = COLORS
        if mode == "cutoff":
            # Keep distinct colors so SVM retains both closures. At 0.5, each
            # lobe's mean weight is 6.67e-6 times the requested color scale.
            scale = settings["scale"]
            colors = ((3e-5 * scale, 1e-5 * scale, 0, 1),
                      (0, 1e-5 * scale, 3e-5 * scale, 1))
        first = diffuse_principled(tree, colors[0])
        if settings.get("with_diffuse"):
            second = tree.nodes.new("ShaderNodeBsdfDiffuse")
            second.inputs["Color"].default_value = colors[1]
        else:
            second = diffuse_principled(tree, colors[1])
        if mode == "independent-bump":
            bump(tree, first, 5, False)
            bump(tree, second, 9, True)
        surface = tree.nodes.new("ShaderNodeMixShader")
        reverse = settings.get("reverse", False)
        if mode == "varying":
            tree.links.new(varying_factor(tree, reverse), surface.inputs[0])
        else:
            factor = settings["factor"]
            surface.inputs[0].default_value = 1 - factor if reverse else factor
        branches = (second, first) if reverse else (first, second)
        for shader, socket in zip(branches, surface.inputs[1:], strict=True):
            tree.links.new(shader.outputs[0], socket)
    tree.links.new(surface.outputs[0], output.inputs["Surface"])

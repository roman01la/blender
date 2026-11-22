# SPDX-License-Identifier: GPL-2.0-or-later
"""Build and render the same small material probes in desktop and web Blender."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import traceback
import zlib

import bpy
import gpu
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import VOLUME_TEXTURE_INPUTS, configure_node, json_key, value_json


MARKER = "MATERIAL_TEST "
BOX = 32
GRAPH_ONLY = {"NodeGroupInput", "NodeGroupOutput", "NodeReroute", "ShaderNodeGroup"}
DISPLACEMENT = {"ShaderNodeDisplacement", "ShaderNodeVectorDisplacement"}
LIGHTING_NODES = {"ShaderNodeLightAccumulation", "ShaderNodeLightEvaluation",
                  "ShaderNodeLightInfo", "ShaderNodeShadowRaycast"}
EEVEE_ONLY = LIGHTING_NODES | {"ShaderNodeEeveeSpecular", "ShaderNodeShaderToRGB"}
LINE_STYLE_NODES = {"ShaderNodeOutputLineStyle", "ShaderNodeUVAlongStroke"}
LAYER_CONTROLLERS = {
    "Subsurface": "Subsurface Weight", "Coat": "Coat Weight", "Sheen": "Sheen Weight",
    "Thin Film": "Thin Film Thickness", "Emission": "Emission Strength",
    "Transmission": "Transmission Weight",
}


def emit(kind, **data):
    print(MARKER + json.dumps({"event": kind, **data}, separators=(",", ":")), file=sys.__stdout__, flush=True)


def backend():
    return {"backend": gpu.platform.backend_type_get(), "renderer": gpu.platform.renderer_get(),
            "vendor": gpu.platform.vendor_get()}


def aim(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def available_socket(sockets, identifier=None):
    return next((socket for socket in sockets if socket.enabled and not socket.is_unavailable
                 and (identifier is None or socket.identifier == identifier)), None)


def graph_fingerprint(tree):
    descriptors = [{"id": item.bl_idname,
                    "inputs": [[sock.identifier, sock.type, value_json(getattr(sock, "default_value", None))]
                               for sock in item.inputs if sock.enabled and not sock.is_unavailable],
                    "outputs": [[sock.identifier, sock.type] for sock in item.outputs
                                if sock.enabled and not sock.is_unavailable]}
                   for item in tree.nodes]
    edges = sorted([link.from_node.name, link.from_socket.identifier,
                    link.to_node.name, link.to_socket.identifier] for link in tree.links)
    return hashlib.sha256(json_key([descriptors, edges]).encode()).hexdigest()


def configure_fixture_node(tree, node, settings):
    if settings.get("is_active_output") is False:
        # Blender keeps an output active when the tree has only one.
        if "target" in settings:
            configure_node(node, {"target": settings["target"]})
        active = tree.nodes.new(node.bl_idname)
        if hasattr(active, "target"):
            active.target = node.target
        for socket in node.inputs:
            destination = available_socket(active.inputs, socket.identifier)
            if destination is None:
                continue
            if hasattr(socket, "default_value"):
                destination.default_value = socket.default_value
            for link in socket.links:
                tree.links.new(link.from_socket, destination)
        active.is_active_output = True
    configure_node(node, settings)


class Fixture:
    def __init__(self, engine, device, lighting):
        self.engine = engine
        self.device = device
        self.scene = bpy.context.scene
        self.triangle_probe = None
        for obj in list(self.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
        self.sphere = bpy.context.object
        self.transformed_object = False
        for polygon in self.sphere.data.polygons:
            polygon.use_smooth = True
        mesh = self.sphere.data
        colors = mesh.color_attributes.new(name="Probe", type="FLOAT_COLOR", domain="POINT")
        for corner, point in zip(colors.data, mesh.vertices):
            corner.color = (0.5 + point.co.x * 0.2, 0.35, 0.45, 1.0)

        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        self.scene.collection.objects.link(camera)
        camera.location = (3.4, -4.2, 2.1)
        camera.data.lens = 55
        aim(camera, (0, 0, 0))
        self.scene.camera = camera
        point_only = lighting in {"point", "two-points"}
        lamp_type = "POINT" if point_only else "AREA"
        lamp = bpy.data.objects.new("Light", bpy.data.lights.new("Light", lamp_type))
        self.scene.collection.objects.link(lamp)
        lamp.location = (1, -2, 4)
        lamp.data.energy = 650
        if lamp_type == "POINT":
            lamp.data.shadow_soft_size = 0.0
        else:
            lamp.data.size = 2.5
            aim(lamp, (0, 0, 0))
        if lighting in {"two-points", "mixed-hdri"}:
            fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "POINT"))
            self.scene.collection.objects.link(fill)
            fill.location = (-3, -1, 2)
            fill.data.energy = 350
            fill.data.shadow_soft_size = 0 if point_only else 0.35
        world = bpy.data.worlds.new("Probe world")
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        background.inputs["Color"].default_value = ((0, 0, 0, 1) if point_only
                                                     else (0.2, 0.23, 0.26, 1))
        if lighting in {"area-hdri", "mixed-hdri"}:
            width, height = 32, 16
            image = bpy.data.images.new("Probe HDRI", width=width, height=height, float_buffer=True)
            image.pixels = [
                value
                for y in range(height)
                for x in range(width)
                for value in ((3.2, 1.7, 0.9, 1) if 8 <= x < 13 and 8 <= y < 12
                              else (0.14 + 0.006 * x, 0.18 + 0.005 * y, 0.25, 1))
            ]
            image.pack()
            environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            environment.image = image
            world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
            if engine == "cycles":
                world.cycles.sampling_method = "MANUAL" if lighting == "mixed-hdri" else "NONE"
                world.cycles.sample_map_resolution = 64
        self.scene.world = world

        self.material = None
        self.image = bpy.data.images.new("Probe texture", width=4, height=4, alpha=True)
        self.image.pixels = [v for i in range(16) for v in (0.25 if i % 2 else 0.75, 0.4, 0.2, 0.8)]
        self.image.pack()
        self.float_image = bpy.data.images.new("Probe float texture", width=4, height=4,
                                               alpha=True, float_buffer=True)
        self.float_image.pixels = self.image.pixels[:]
        self.float_image.pack()
        self.ies = bpy.data.texts.new("Probe.ies")
        # A rotationally symmetric, type C profile in LM-63 format.
        self.ies.write("IESNA:LM-63-1995\nTILT=NONE\n1 1000 1 3 1 1 2 0 0 0\n"
                       "1 1 10\n0 90 180\n0\n1000 450 0\n")
        self.ies_path = str(Path("/tmp") / f"material-probe-{os.getpid()}.ies")
        Path(self.ies_path).write_text(self.ies.as_string())
        self.group = None
        self.line_style = None

        self.scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"
        self.scene.render.resolution_x = self.scene.render.resolution_y = BOX
        self.scene.render.resolution_percentage = 100
        self.scene.render.use_compositing = self.scene.render.use_sequencer = False
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.view_settings.view_transform = "Standard"
        bpy.context.preferences.view.render_display_type = "NONE"
        if engine == "cycles":
            import _cycles
            _cycles.enable_print_stats()
            if device == "CPU":
                active = ["CPU"]
                self.scene.cycles.device = "CPU"
            else:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = device
                prefs.refresh_devices()
                active = []
                for candidate in prefs.devices:
                    candidate.use = candidate.type == device
                    if candidate.use:
                        active.append(candidate.name)
                if not active:
                    raise RuntimeError(f"No Cycles {device} device")
                self.scene.cycles.device = "GPU"
            self.scene.cycles.use_light_tree = lighting in {"area-world", "mixed-hdri"}
            self.scene.cycles.light_sampling_threshold = 0
            self.scene.cycles.sample_clamp_direct = 0
            self.scene.cycles.use_fast_gi = False
            self.scene.cycles.samples = 2
            self.scene.cycles.use_adaptive_sampling = False
            self.scene.cycles.use_denoising = False
            self.scene.cycles.max_bounces = 4
            emit("device", device=device, names=active)
        else:
            self.scene.eevee.taa_render_samples = 4

    def make_group(self):
        group = bpy.data.node_groups.new("Probe nested shader", "ShaderNodeTree")
        group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
        group.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
        source = group.nodes.new("NodeGroupInput")
        destination = group.nodes.new("NodeGroupOutput")
        group.links.new(source.outputs["Color"], destination.inputs["Color"])
        self.group = group
        return group

    def build_stroke(self, case):
        if self.engine != "eevee":
            raise NotImplementedError("Freestyle stroke shading uses Eevee")
        if not bpy.app.build_options.freestyle:
            raise NotImplementedError("Freestyle is disabled in this build")

        self.scene.render.use_freestyle = True
        self.scene.render.line_thickness = 2.0
        line_set = self.scene.view_layers[0].freestyle_settings.linesets.active
        previous = self.line_style
        style = self.line_style = bpy.data.linestyles.new("Probe stroke")
        line_set.linestyle = style
        if previous is not None:
            bpy.data.linestyles.remove(previous)
        style.thickness = 8.0
        style.texture_spacing = 0.15
        style.color = (0.8, 0.0, 0.1)
        style.use_nodes = True
        tree = style.node_tree
        nodes = {node.bl_idname: node for node in tree.nodes}
        texture = nodes["ShaderNodeTexImage"]
        texture.image = self.image
        output = nodes["ShaderNodeOutputLineStyle"]
        output.inputs["Color Fac"].default_value = 1.0
        node = nodes[case["node"]]
        configure_fixture_node(tree, node, case["settings"])
        if identifier := case.get("input"):
            socket = available_socket(node.inputs, identifier)
            if socket is None:
                raise ValueError(f"Inactive or missing input {identifier}")
            for link in list(socket.links):
                tree.links.remove(link)
            if case.get("linked"):
                tree.links.new(texture.outputs["Color"], socket)
            else:
                socket.default_value = case["value"]
        return graph_fingerprint(tree)

    def configure_volume_cube(self, camera_position):
        if self.sphere.data.name != "Probe volume cube":
            cube = bpy.data.meshes.new("Probe volume cube")
            vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                     (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            cube.from_pydata(vertices, [], faces)
            if self.material is not None:
                cube.materials.append(self.material)
            self.sphere.data = cube
        self.scene.eevee.taa_render_samples = 64
        self.scene.eevee.volumetric_tile_size = "2"
        self.scene.eevee.volumetric_samples = 128
        self.scene.camera.location = camera_position
        self.scene.camera.data.lens = 44
        aim(self.scene.camera, (0, 0, 0))
        world = self.scene.world.node_tree.nodes["Background"]
        world.inputs["Color"].default_value = (1, 1, 1, 1)

    def tiled_image(self):
        name = "Probe UDIM"
        if image := bpy.data.images.get(name):
            return image

        def png_chunk(tag, payload):
            return (struct.pack(">I", len(payload)) + tag + payload +
                    struct.pack(">I", zlib.crc32(tag + payload)))

        with TemporaryDirectory(prefix="blender-probe-udim-", dir="/tmp") as directory:
            directory = Path(directory)
            tiles = ((1001, 32, (210, 70, 35), 150), (1002, 64, (45, 190, 75), 230))
            for tile, size, color, alpha in tiles:
                rows = bytearray()
                for y in range(size):
                    rows.append(0)
                    for x in range(size):
                        shade = 1 if ((x // 8) ^ (y // 8)) & 1 else 2
                        rows.extend(channel // shade for channel in color)
                        rows.append(alpha)
                (directory / f"probe.{tile}.png").write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
                    + png_chunk(b"IDAT", zlib.compress(rows))
                    + png_chunk(b"IEND", b"")
                )
            image = bpy.data.images.load(str(directory / "probe.1001.png"), check_existing=False)
            image.name = name
            image.source = "TILED"
            image.filepath = str(directory / "probe.<UDIM>.png")
            if not any(tile.number == 1002 for tile in image.tiles):
                image.tiles.new(1002)
            image.pack()
            if len(image.packed_files) != len(tiles):
                raise RuntimeError("Could not pack the two probe image tiles")
        return image

    def build(self, case):
        if "volume_cube_camera" in case:
            self.configure_volume_cube(case["volume_cube_camera"])
        if bpy.app.build_options.freestyle:
            self.scene.render.use_freestyle = False
        previous = self.material
        tag = case["id"].rsplit("/", 1)[-1]
        self.material = bpy.data.materials.new("Probe_" + tag)
        self.material.use_nodes = True
        if case.get("variant") == "accurate-volume":
            self.material.volume_intersection_method = "ACCURATE"
        if previous is None:
            self.sphere.data.materials.append(self.material)
        else:
            self.sphere.data.materials[0] = self.material
            bpy.data.materials.remove(previous)
        transformed = case.get("variant") == "transformed-object"
        if transformed != self.transformed_object:
            self.sphere.location = (0.125, 0.375, 0.25) if transformed else (0, 0, 0)
            self.sphere.rotation_euler = (0.2, 0.3, 0.4) if transformed else (0, 0, 0)
            self.sphere.scale = (0.8, 1.1, 0.5) if transformed else (1, 1, 1)
            self.sphere.color = (0.12, 0.35, 0.6, 0.65) if transformed else (1, 1, 1, 1)
            self.sphere.pass_index = 2 if transformed else 0
            self.transformed_object = transformed
        if transformed:
            self.material.pass_index = 3
        tree = self.material.node_tree
        tree.nodes.clear()
        if self.group is not None:
            bpy.data.node_groups.remove(self.group)
            self.group = None
        layer = self.scene.view_layers[0]
        for aov in list(layer.aovs):
            layer.aovs.remove(aov)
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.55, 0.34, 0.2, 1)
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])

        node_id = case["node"]
        if node_id in LINE_STYLE_NODES:
            return self.build_stroke(case)
        if node_id == "ShaderNodeOutputMaterial":
            node = output
        elif node_id in GRAPH_ONLY and node_id != "NodeReroute":
            node = tree.nodes.new("ShaderNodeGroup")
            node.node_tree = self.make_group()
        elif node_id == "ShaderNodeBsdfDiffuse":
            node = diffuse
        else:
            node = tree.nodes.new(node_id)

        settings = case["settings"]
        if node_id == "NodeGroupOutput":
            group_output = next(item for item in node.node_tree.nodes if item.bl_idname == node_id)
            configure_fixture_node(node.node_tree, group_output, settings)
        else:
            configure_fixture_node(tree, node, settings)
        if case.get("variant") == "reference-object":
            reference = bpy.data.objects.get("Probe texture reference")
            if reference is None:
                reference = bpy.data.objects.new("Probe texture reference", None)
                self.scene.collection.objects.link(reference)
                reference.location = (-0.5, -0.25, -0.125)
                reference.rotation_euler = (0.1, 0.2, 0.3)
                reference.scale = (0.5, 1.25, 0.75)
            node.object = reference
        is_lighting = (node_id in LIGHTING_NODES or
                       (node_id == "ShaderNodeAttribute" and node.attribute_type == "LIGHT") or
                       (node_id == "ShaderNodeVectorTransform" and
                        "LIGHT" in {node.convert_from, node.convert_to}))
        if self.engine == "cycles" and (node_id in EEVEE_ONLY or is_lighting):
            raise NotImplementedError("Uses Eevee-specific material shading")
        if node_id == "ShaderNodeScript":
            raise NotImplementedError("Eevee does not evaluate OSL Script nodes" if self.engine == "eevee" else
                                      "OSL script fixture requires an OSL-capable native GPU")
        if node_id == "ShaderNodeTexIES":
            if node.mode == "INTERNAL":
                node.ies = self.ies
            else:
                node.filepath = self.ies_path
        if node_id == "ShaderNodeOutputMaterial" and hasattr(node, "target"):
            target = "CYCLES" if self.engine == "cycles" else "EEVEE"
            if node.target not in {"ALL", target}:
                raise NotImplementedError(f"Output targets {node.target}")
        if case.get("variant") == "udim":
            node.image = self.tiled_image()
            coords = tree.nodes.new("ShaderNodeTexCoord")
            scale = tree.nodes.new("ShaderNodeVectorMath")
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = (2.0, 1.0, 1.0)
            tree.links.new(coords.outputs["Generated"], scale.inputs[0])
            tree.links.new(scale.outputs["Vector"], node.inputs["Vector"])
        elif hasattr(node, "image"):
            node.image = self.float_image if case.get("variant") == "float-image" else self.image
        if node_id == "ShaderNodeAttribute" and not node.attribute_name:
            node.attribute_name = "Probe"
        if node_id == "ShaderNodeVertexColor" and not node.layer_name:
            node.layer_name = "Probe"

        def link_input(socket):
            if socket.type == "SHADER":
                source = tree.nodes.new("ShaderNodeBsdfDiffuse")
                source.inputs["Color"].default_value = (0.15, 0.66, 0.36, 1)
                tree.links.new(source.outputs[0], socket)
            else:
                coord = tree.nodes.new("ShaderNodeTexCoord")
                coordinates = coord.outputs["Generated"]
                if case.get("variant") in {"image-volume", "accurate-volume"}:
                    texture = tree.nodes.new("ShaderNodeTexImage")
                    texture.image = self.image
                    tree.links.new(coordinates, texture.inputs["Vector"])
                    tree.links.new(texture.outputs["Color"], socket)
                else:
                    tree.links.new(coordinates, socket)

        if case.get("variant") == "edit-curve":
            if hasattr(node, "color_ramp"):
                node.color_ramp.elements.new(0.42).color = (0.95, 0.2, 0.05, 0.65)
                link_input(node.inputs["Fac"])
            else:
                node.mapping.curves[0].points.new(0.42, 0.76)
                node.mapping.update()
                socket = next(sock for sock in node.inputs
                              if sock.identifier in {"Value", "Color", "Vector"})
                link_input(socket)

        if case.get("input"):
            sock = available_socket(node.inputs, case["input"])
            if sock is None:
                raise ValueError(f"Inactive or missing input {case['input']}")
            if node_id == "ShaderNodeBsdfPrincipled":
                for prefix, controller in LAYER_CONTROLLERS.items():
                    weight = node.inputs.get(controller)
                    if sock.name.startswith(prefix) and sock.name != controller and weight:
                        weight.default_value = 0.6
            if case.get("linked"):
                link_input(sock)
            else:
                original = value_json(sock.default_value)
                sock.default_value = case["value"]
                if value_json(sock.default_value) == original:
                    raise ValueError(f"Input {sock.identifier} stayed at its default")
        if node_id == "ShaderNodeBsdfPrincipled" and (
                any("subsurface" in name.lower() for name in case["settings"])):
            node.inputs["Subsurface Weight"].default_value = 0.6

        if node_id in GRAPH_ONLY:
            if node_id == "NodeReroute":
                link_input(node.inputs[0])
            else:
                link_input(node.inputs["Color"])
        if node_id in {"ShaderNodeMixShader", "ShaderNodeAddShader", "ShaderNodeShaderToRGB"}:
            for socket in node.inputs:
                if socket.type == "SHADER" and not socket.is_linked:
                    link_input(socket)
        if case.get("variant") == "lambert-closures":
            colors = ((0.625, 0.125, 0.25, 1), (0.125, 0.5, 0.75, 1))
            for socket, color in zip((item for item in node.inputs if item.type == "SHADER"), colors):
                socket.links[0].from_node.inputs["Color"].default_value = color
        if node_id == "ShaderNodeOutputAOV":
            aov = layer.aovs.add()
            aov.name = node.aov_name = "MaterialProbe"
            tree.links.new(tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"], node.inputs["Color"])
        if node_id not in {"ShaderNodeOutputMaterial", "ShaderNodeOutputAOV"}:
            wanted = None if node_id in GRAPH_ONLY else case["output"]
            primary = available_socket(node.outputs, wanted)
            if (primary is None) or (wanted is not None and primary.identifier != wanted):
                raise ValueError(f"Inactive or missing output {wanted}")
            if is_lighting:
                accumulation = (node if node_id == "ShaderNodeLightAccumulation" else
                                tree.nodes.new("ShaderNodeLightAccumulation"))
                if accumulation == node:
                    if not accumulation.inputs["Diffuse Light"].is_linked:
                        evaluation = tree.nodes.new("ShaderNodeLightEvaluation")
                        tree.links.new(evaluation.outputs["Factor"], accumulation.inputs["Diffuse Light"])
                else:
                    tree.links.new(primary, accumulation.inputs["Diffuse Light"])
                tree.links.new(accumulation.outputs["Shader"], output.inputs["Surface"])
            elif node_id in VOLUME_TEXTURE_INPUTS:
                tree.links.new(primary, output.inputs["Volume"])
                # Avoid obscuring the volume with an opaque boundary.
                tree.links.remove(output.inputs["Surface"].links[0])
            elif node_id in DISPLACEMENT:
                tree.links.new(primary, output.inputs["Displacement"])
            elif primary.type == "SHADER":
                if node != diffuse:
                    tree.links.new(primary, output.inputs["Surface"])
            else:
                tree.links.new(primary, diffuse.inputs["Color"])
        if any(not link.is_valid for link in tree.links):
            raise ValueError("Fixture has an invalid node link")
        if settings := case.get("triangle_emitter"):
            from triangle_fixture import TriangleProbe

            if self.triangle_probe is None:
                self.triangle_probe = TriangleProbe(self.scene)
            lighting = self.triangle_probe.configure(settings, node)
            return hashlib.sha256(json_key([graph_fingerprint(tree), lighting]).encode()).hexdigest()
        if query := case.get("query_fixture"):
            from scene_query_fixture import configure

            configure(self, node, tree, output, query, case["output"])
            return hashlib.sha256(json_key([graph_fingerprint(tree), query]).encode()).hexdigest()
        if transparency := case.get("transparent_fixture"):
            from transparent_fixture import configure

            configure(self, node, transparency)
            return hashlib.sha256(
                json_key([graph_fingerprint(tree), transparency]).encode()).hexdigest()
        if absorption := case.get("absorption_fixture"):
            from absorption_fixture import configure

            configure(self, node, absorption)
            return hashlib.sha256(
                json_key([graph_fingerprint(tree), absorption]).encode()).hexdigest()
        if volume := case.get("volume_transport"):
            from volume_transport_fixture import configure

            media = configure(self, node, volume)
            return hashlib.sha256(
                json_key([graph_fingerprint(tree), volume, media]).encode()).hexdigest()
        if transport := case.get("ray_transport_fixture"):
            from ray_transport_fixture import configure

            configure(self, node, tree, output, transport)
            return hashlib.sha256(
                json_key([graph_fingerprint(tree), transport]).encode()).hexdigest()
        return graph_fingerprint(tree)


def image_summary(path, include_pixels=False, size=(BOX, BOX)):
    # Loading the exported image verifies that a new, nonempty result was produced.
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = image.pixels[:]
        if (width, height) != size or len(pixels) != width * height * 4:
            raise ValueError("Render result has unexpected dimensions")
        center_indices = [4 * (y * width + x) for y in range(height // 4, 3 * height // 4)
                          for x in range(width // 4, 3 * width // 4)]
        center = [pixels[offset:offset + 3] for offset in center_indices]
        rgba = bytes(round(value * 255) for value in pixels)
        pixel_hash = hashlib.sha256(rgba).hexdigest()
        summary = {"size": [width, height], "pixel_sha256": pixel_hash, "mean_rgb": [
                    round(sum(pixel[channel] for pixel in center) / len(center), 6)
                    for channel in range(3)],
                "mean_alpha": round(sum(pixels[offset + 3] for offset in center_indices) /
                                    len(center_indices), 6),
                "magenta": sum(r > 0.94 and g < 0.03 and b > 0.94 for r, g, b in center) / len(center)}
        if include_pixels:
            summary["rgba_hex"] = rgba.hex()
        return summary
    finally:
        bpy.data.images.remove(image)


class Batch:
    def __init__(self, request):
        self.request = request
        self.fixture = Fixture(request["engine"], request["device"], request.get("lighting", "area-world"))
        self.fixture.volume_grid = request.get("volume_grid")
        if self.fixture.engine == "cycles":
            self.default_volume_bounces = self.fixture.scene.cycles.volume_bounces
        self.pending = iter(request["cases"])
        self.active = None
        self.outcome = None
        self.image = Path(request["image"])
        self.complete = 0

    def begin(self, case):
        self.active = case
        self.outcome = None
        self.image.unlink(missing_ok=True)
        emit("begin", id=case["id"])
        try:
            if self.fixture.engine == "cycles":
                self.fixture.scene.cycles.samples = case.get("cycles_samples", 2)
                self.fixture.scene.cycles.volume_bounces = case.get(
                    "cycles_volume_bounces", self.default_volume_bounces)
            size = case.get("resolution", BOX)
            self.fixture.scene.render.resolution_x = self.fixture.scene.render.resolution_y = size
            self.signature = self.fixture.build(case)
            emit("connected", id=case["id"], fingerprint=self.signature,
                 material=self.fixture.material.name)
        except NotImplementedError as error:
            self.end("outside_fixture", str(error))
        except (RuntimeError, ValueError, AttributeError, TypeError) as error:
            self.end("fixture_error", str(error))

    def end(self, status, reason=None, **extra):
        emit("result", id=self.active["id"], status=status, reason=reason, **extra)
        self.active = None
        self.complete += 1

    def finish_image(self):
        try:
            bpy.data.images["Render Result"].save_render(str(self.image), scene=self.fixture.scene)
            self.end("completed", fingerprint=self.signature,
                     gpu=backend() if self.fixture.engine == "eevee" else None,
                     image=image_summary(
                         self.image, include_pixels=bool(self.active.get("include_pixels") or
                                                          self.active.get("absorption_fixture")),
                         size=(self.fixture.scene.render.resolution_x,
                               self.fixture.scene.render.resolution_y)))
        except (KeyError, RuntimeError, ValueError) as error:
            self.end("render_error", str(error))

    def run_native(self):
        for case in self.pending:
            self.begin(case)
            if self.active is None:
                continue
            try:
                result = bpy.ops.render.render()
                if "FINISHED" not in result:
                    self.end("render_error", str(result))
                else:
                    self.finish_image()
            except RuntimeError as error:
                self.end("render_error", str(error))
        Path(self.fixture.ies_path).unlink(missing_ok=True)
        emit("done", completed=self.complete)

    def on_complete(self, *_):
        self.outcome = "completed"

    def on_cancel(self, *_):
        self.outcome = "cancelled"

    def tick(self):
        if bpy.app.is_job_running("RENDER"):
            return 0.05
        if self.active is not None:
            if self.outcome == "completed":
                self.finish_image()
            elif self.outcome == "cancelled":
                self.end("render_error", "Render cancelled")
            else:
                return 0.05
        case = next(self.pending, None)
        if case is None:
            Path(self.fixture.ies_path).unlink(missing_ok=True)
            bpy.app.handlers.render_complete.remove(self.on_complete)
            bpy.app.handlers.render_cancel.remove(self.on_cancel)
            emit("done", completed=self.complete)
            return None
        self.begin(case)
        if self.active is not None:
            try:
                window = bpy.context.window_manager.windows[0]
                with bpy.context.temp_override(window=window):
                    result = bpy.ops.render.render("INVOKE_DEFAULT")
                if "RUNNING_MODAL" not in result:
                    self.end("render_error", str(result))
            except RuntimeError as error:
                self.end("render_error", str(error))
        return 0.01


def run(request):
    from catalog import inventory

    try:
        if request.get("inventory"):
            Path(request["inventory"]).write_text(json.dumps(inventory(), ensure_ascii=False))
            emit("inventory")
            return
        batch = Batch(request)
        if sys.platform == "emscripten":
            bpy.app.handlers.render_complete.append(batch.on_complete)
            bpy.app.handlers.render_cancel.append(batch.on_cancel)
            bpy.app.timers.register(batch.tick)
        else:
            batch.run_native()
    except Exception as error:
        emit("fatal", reason=str(error))
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(json.loads(args.request.read_text()))

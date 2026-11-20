# SPDX-License-Identifier: GPL-2.0-or-later
"""Inventory Blender's registered material nodes using the live RNA definitions."""

import itertools
import json
import math


TREE_ONLY = {
    "NodeFrame": "layout",
    "NodeGroupInput": "group",
    "NodeGroupOutput": "group",
    "NodeReroute": "material",
    "ShaderNodeLightIterInternalInput": "internal",
    "ShaderNodeLightIterInternalOutput": "internal",
    "ShaderNodeOutputLineStyle": "line-style",
    "ShaderNodeUVAlongStroke": "line-style",
    "ShaderNodeOutputLight": "light",
    "ShaderNodeOutputWorld": "world",
}
VOLUME_TEXTURE_INPUTS = {
    "ShaderNodeVolumeAbsorption": "Color",
    "ShaderNodeVolumeCoefficients": "Scatter Coefficients",
    "ShaderNodeVolumePrincipled": "Color",
    "ShaderNodeVolumeScatter": "Color",
}
BASE_PROPERTIES = {
    "location", "location_absolute", "width", "height", "dimensions", "name",
    "label", "parent", "select", "show_options", "show_preview", "show_texture",
    "hide", "use_custom_color", "color", "warning_propagation", "mute",
    "show_expanded", "rna_type",
}


def value_json(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, set):
        return sorted(value)
    try:
        return [value_json(member) for member in value]
    except TypeError:
        return getattr(value, "name", None)


def describe_properties(owner, excluded=(), only=None):
    properties = {}
    for prop in owner.bl_rna.properties:
        name = prop.identifier
        if (name in excluded or name.startswith("bl_") or prop.is_readonly or
                (only is not None and name not in only)):
            continue
        default = getattr(owner, name, None)
        spec = {"name": prop.name, "type": prop.type, "default": value_json(default),
                "description": prop.description}
        if prop.type == "ENUM":
            # Some dynamic enums expose their declaration through enum_items_static.
            items = list(prop.enum_items) or list(prop.enum_items_static)
            spec["choices"] = [{"id": item.identifier, "name": item.name,
                                "description": item.description}
                               for item in items if item.identifier]
            spec["flag"] = prop.is_enum_flag
        if prop.type in {"FLOAT", "INT"}:
            for bound in ("hard_min", "hard_max", "soft_min", "soft_max"):
                spec[bound] = value_json(getattr(prop, bound))
        if prop.type == "POINTER":
            spec["target"] = prop.fixed_type.identifier
        properties[name] = spec
    return properties


def describe_socket(socket):
    return {"id": socket.identifier, "name": socket.name, "type": socket.type,
            "available": socket.enabled and not socket.is_unavailable,
            "default": describe_properties(socket, only={"default_value"}).get("default_value")}


def sockets(node):
    return {key: [describe_socket(sock) for sock in getattr(node, key)]
            for key in ("inputs", "outputs")}


def nested_settings(node):
    nested = {}
    for name in ("image_user", "texture_mapping", "color_mapping"):
        owner = getattr(node, name, None)
        if owner is not None:
            nested[name] = describe_properties(owner)
    if hasattr(node, "color_ramp"):
        ramp = node.color_ramp
        nested["color_ramp"] = describe_properties(ramp)
        nested["color_ramp.elements"] = {"type": "COLLECTION", "default": [
            {"position": item.position, "color": value_json(item.color)}
            for item in ramp.elements]}
    if hasattr(node, "mapping"):
        mapping = node.mapping
        nested["mapping"] = describe_properties(mapping)
        nested["mapping.curves"] = {"type": "COLLECTION", "default": [
            [value_json(point.location) for point in curve.points]
            for curve in mapping.curves]}
    return nested


def choices(spec):
    if spec["type"] == "BOOLEAN" and isinstance(spec["default"], bool):
        return [not spec["default"]]
    if spec["type"] == "ENUM":
        if spec["flag"]:
            return [v for v in ([choice["id"]] for choice in spec["choices"]) if v != spec["default"]]
        return [choice["id"] for choice in spec["choices"] if choice["id"] != spec["default"]]
    return []


def set_property(node, path, value):
    parts = path.split(".")
    owner = node
    for name in parts[:-1]:
        owner = getattr(owner, name)
    old = getattr(owner, parts[-1])
    setattr(owner, parts[-1], set(value) if isinstance(old, set) else value)
    actual = value_json(getattr(owner, parts[-1]))
    if isinstance(value, float) and isinstance(actual, float):
        if math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-8):
            return
    if actual != value:
        raise ValueError(f"{path} did not accept {value!r}")


def configure_node(node, settings):
    if node.bl_idname == "ShaderNodeMix" and "data_type" not in settings:
        if settings.keys() & {"blend_type", "clamp_result"}:
            node.data_type = "RGBA"
        elif "factor_mode" in settings:
            node.data_type = "VECTOR"
    for name, value in settings.items():
        set_property(node, name, value)


def flatten_settings(entry):
    settings = dict(entry["settings"])
    for path, specs in entry["nested"].items():
        if "default" not in specs:
            settings.update({f"{path}.{prop}": spec for prop, spec in specs.items()})
    return settings


def inventory():
    import bpy

    ids = sorted({name for name in dir(bpy.types) if name.startswith("ShaderNode")
                  and issubclass(getattr(bpy.types, name), bpy.types.ShaderNode)
                  and name != "ShaderNode"} | TREE_ONLY.keys())
    tree = bpy.data.node_groups.new("Material node inventory", "ShaderNodeTree")
    catalog = []
    try:
        for node_id in ids:
            scope = TREE_ONLY.get(node_id, "material")
            entry = {"id": node_id, "scope": scope}
            catalog.append(entry)
            try:
                node = tree.nodes.new(node_id)
            except RuntimeError as error:
                entry["unavailable"] = str(error)
                continue
            try:
                entry.update(name=node.bl_label, canonical_id=node.bl_idname,
                             description=node.bl_rna.description,
                             settings=describe_properties(node, BASE_PROPERTIES),
                             nested=nested_settings(node))
                entry["layouts"] = [{"settings": {}, **sockets(node)}]
                for name, spec in flatten_settings(entry).items():
                    for setting in choices(spec):
                        variant = tree.nodes.new(node_id)
                        try:
                            configure_node(variant, {name: setting})
                            entry["layouts"].append({"settings": {name: setting}, **sockets(variant)})
                        except (TypeError, ValueError, RuntimeError):
                            entry.setdefault("unavailable_choices", []).append({name: setting})
                        finally:
                            tree.nodes.remove(variant)
            finally:
                tree.nodes.remove(node)
    finally:
        bpy.data.node_groups.remove(tree)
    return {"blender": {"version": bpy.app.version_string,
                        "hash": bpy.app.build_hash.decode("ascii", errors="replace")},
            "nodes": catalog}


def json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def numeric_examples(socket):
    spec = socket["default"]
    if not spec:
        return []
    default = spec["default"]
    if isinstance(default, bool):
        return [not default]
    if spec["type"] == "ENUM":
        return choices(spec)
    if spec["type"] in {"FLOAT", "INT"} and isinstance(default, (int, float)):
        if spec["type"] == "INT":
            values = [0, 1, 4, spec["soft_min"], spec["soft_max"]]
        else:
            values = [0.0, 0.25, 0.75, 1.0, spec["soft_min"], spec["soft_max"]]
        return sorted({v for v in values if isinstance(v, (int, float)) and
                       abs(v) <= 1000 and spec["hard_min"] <= v <= spec["hard_max"] and
                       v != default})
    if isinstance(default, list) and all(isinstance(v, (int, float)) for v in default):
        colorful = [0.21, 0.54, 0.78] + ([0.73] if len(default) == 4 else [])
        return [v for v in ([0] * len(default), colorful) if len(v) == len(default) and v != default]
    if isinstance(default, str):
        return ["Probe"]
    return []


def cases(catalog, strategy):
    """Cases use stable socket IDs so a changed native layout cannot silently pass."""
    for entry in catalog["nodes"]:
        node_id = entry["id"]
        if ("unavailable" in entry or node_id != entry["canonical_id"] or
                entry["scope"] not in {"material", "group", "line-style"}):
            continue
        layouts = entry["layouts"]
        variants = layouts[:1] if strategy == "default" else layouts
        for layout in variants:
            settings = layout["settings"]
            outputs = [sock["id"] for sock in layout["outputs"] if sock["available"]]
            for output in outputs or [None]:
                base = {"node": node_id, "settings": settings, "output": output}
                yield base
                if strategy in {"default", "modes"}:
                    continue
                if not settings and (
                    node_id == "ShaderNodeObjectInfo" or
                    (node_id == "ShaderNodeTexCoord" and output in {"Normal", "Object", "Reflection"})
                ):
                    yield {**base, "variant": "transformed-object"}
                if not settings and node_id == "ShaderNodeTexCoord" and output == "Object":
                    yield {**base, "variant": "reference-object"}
                if not settings and node_id in {"ShaderNodeMixShader", "ShaderNodeAddShader"}:
                    closures = {**base, "variant": "lambert-closures"}
                    yield closures
                    if node_id == "ShaderNodeMixShader":
                        for factor in (0.0, 0.25, 1.0):
                            yield {**closures, "input": "Fac", "value": factor}
                        yield {**closures, "input": "Fac", "linked": True}
                if not settings and output == "Volume" and node_id in VOLUME_TEXTURE_INPUTS:
                    textured = {**base, "input": VOLUME_TEXTURE_INPUTS[node_id], "linked": True}
                    yield {**textured, "variant": "image-volume"}
                    if node_id == "ShaderNodeVolumeScatter":
                        yield {**textured, "variant": "accurate-volume"}
                for sock in layout["inputs"]:
                    if not sock["available"]:
                        continue
                    for value in numeric_examples(sock):
                        yield {**base, "input": sock["id"], "value": value}
                    if sock["type"] in {"VALUE", "INT", "BOOLEAN", "VECTOR", "RGBA", "SHADER"}:
                        yield {**base, "input": sock["id"], "linked": True}
        if strategy in {"options", "pairwise"}:
            for name, spec in flatten_settings(entry).items():
                if spec["type"] in {"FLOAT", "INT"}:
                    for value in numeric_examples({"default": spec}):
                        yield {"node": node_id, "settings": {name: value}, "output": None}
            if node_id in {"ShaderNodeTexImage", "ShaderNodeTexEnvironment"}:
                yield {"node": node_id, "settings": {}, "output": None, "variant": "float-image"}
        if strategy == "pairwise":
            for a, b in itertools.combinations(layouts[1:], 2):
                if a["settings"].keys() != b["settings"].keys():
                    yield {"node": node_id, "settings": a["settings"] | b["settings"], "output": None}
        if strategy in {"options", "pairwise"} and node_id in {"ShaderNodeRGBCurve", "ShaderNodeFloatCurve",
                                                 "ShaderNodeVectorCurve", "ShaderNodeValToRGB"}:
            yield {"node": node_id, "settings": {}, "output": None, "variant": "edit-curve"}

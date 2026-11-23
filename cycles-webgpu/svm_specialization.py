# SPDX-License-Identifier: Apache-2.0

"""Compile SVM subsets from the shared Cycles WebGPU evaluator."""

from pathlib import Path
import re


_CLOSURES = """
    END CLOSURE_BSDF CLOSURE_SET_WEIGHT CLOSURE_EMISSION CLOSURE_WEIGHT
    EMISSION_WEIGHT MIX_CLOSURE JUMP_IF_ZERO JUMP_IF_ONE
"""
_TEXTURED = """
    VALUE_F VALUE_V GEOMETRY TEX_COORD CONVERT ATTR ATTR_DERIVATIVE TEX_IMAGE
    TEX_SKY HSV MATH LIGHT_PATH OBJECT_INFO RGB_RAMP CURVES GAMMA BRIGHTCONTRAST
    MIX MIX_COLOR MIX_FLOAT MAPPING CLAMP INVERT
"""
_NODE_PROFILES = {
    "surface": _CLOSURES + _TEXTURED + """
        DISPLACEMENT SET_BUMP CLOSURE_SET_NORMAL TEX_NOISE VECTOR_MATH TEX_WAVE
        NORMAL_MAP NORMAL AOV_START AOV_COLOR AOV_VALUE
    """,
    "shadow": _CLOSURES + _TEXTURED + "SET_BUMP",
    "emission": _CLOSURES + _TEXTURED,
}


def prepare_specialized_svm(work: Path, cycles: Path, shader_source: Path,
                            profile: str) -> tuple[Path, str]:
    """Remove unused bodies from the shared SVM opcode dispatch."""
    source = shader_source.parent / "blender-platform/cycles-webgpu/svm_material_eval.h"
    text = source.read_text()
    anchor = "    host.visit(type, offset - 1);\n    switch (type) {\n"
    if text.count(anchor) != 1:
        raise ValueError("Cannot identify the Cycles WebGPU material opcode switch")
    begin, switch = text.split(anchor)
    if "\n      default:\n" not in switch:
        raise ValueError("Cannot find the top-level material opcode switch default")
    cases, end = switch.split("\n      default:\n", 1)
    allowed = {"NODE_" + node for node in _NODE_PROFILES[profile].split()}
    observed = set()

    # Cases at this indentation within the anchored switch are dispatch for
    # ShaderNodeType. Preserve a shared case body if any of its aliases is used.
    labels = r"(?:^      case NODE_\w+:(?: \{)?\n)+"
    if profile == "shadow":
        specialized = cases
        for match in reversed(list(re.finditer(labels, cases, flags=re.M))):
            nodes = set(re.findall(r"case (NODE_\w+):", match.group()))
            observed.update(nodes)
            if nodes & allowed:
                continue
            # Keep the handler's enclosing preprocessor guards in place. The
            # shadow traversal has a deep branch chain, so removing the entire
            # unused handler avoids adding extra switch destinations.
            closing = r"^      }\n" if match.group().endswith(" {\n") else r"^        break;\n"
            close = re.search(closing, cases[match.end():], flags=re.M)
            if close is None or re.search(labels, cases[match.end():match.end() + close.end()], re.M):
                raise ValueError(f"Cannot remove the SVM handler for {', '.join(sorted(nodes))}")
            specialized = (specialized[:match.start()] +
                           specialized[match.end() + close.end():])
    else:
        def prune(match: re.Match[str]) -> str:
            cases = match.group()
            nodes = set(re.findall(r"case (NODE_\w+):", cases))
            observed.update(nodes)
            return cases if nodes & allowed else cases + "        return false;\n"

        specialized = re.sub(labels, prune, cases, flags=re.M)
    if unknown := allowed - observed:
        raise ValueError(f"Unimplemented SVM nodes in {profile}: {', '.join(sorted(unknown))}")
    if specialized == cases:
        raise ValueError(f"No unused cases in the {profile} SVM profile")

    stock_registry = (cycles / "kernel/svm/node_types_template.h").read_text()
    node_ids = {}
    for derivative, name in re.findall(
        r"^SHADER_NODE_TYPE(_DERIVATIVE)?\((NODE_\w+)\)$", stock_registry, re.M
    ):
        node_ids[name] = len(node_ids)
        if derivative:
            node_ids[name + "_DERIVATIVE"] = len(node_ids)
    if "NODE_END" not in node_ids:
        raise ValueError("Cannot read the stock Cycles SVM node registry")
    manifest = ",".join(str(index) for index in sorted(node_ids[name] for name in allowed))
    header = work / "svm_scene_specialized.h"
    header.write_text(begin + anchor + specialized + "\n      default:\n" + end)
    return header, f"// Cycles WebGPU specialized SVM nodes: {manifest}\n"


_WGSL_IF = re.compile(r"( *)if (.+) \{")


def _wgsl_if_range(lines: list[str], start: int, indent: str) -> tuple[int, int]:
    branch_end = next(index for index in range(start + 1, len(lines))
                      if lines[index].strip() and not lines[index].startswith(indent + " "))
    end = branch_end
    if lines[branch_end] == indent + "} else {":
        end = next(index for index in range(branch_end + 1, len(lines))
                   if lines[index].strip() and not lines[index].startswith(indent + " "))
    if lines[end] != indent + "}":
        raise ValueError(f"Unexpected WGSL branch terminator at line {end + 1}")
    return branch_end, end


def _flatten_nested_wgsl_if(lines: list[str], start: int, peak: int, index: int) -> bool:
    outer = _WGSL_IF.fullmatch(lines[start])
    inner = _WGSL_IF.fullmatch(lines[start + 1])
    if not (outer and inner and inner[1] == outer[1] + "    "):
        return False
    inner_branch_end, inner_end = _wgsl_if_range(lines, start + 1, inner[1])
    if peak >= inner_end:
        return False
    outer_branch_end, outer_end = _wgsl_if_range(lines, start, outer[1])
    indent = outer[1]
    parent = f"cond_outer_{index}"
    child = f"cond_inner_{index}"
    # The inner conditional is the first statement of the outer branch. Short
    # circuiting preserves evaluation order, including reads in the condition.
    flattened = [f"{indent}let {parent} = {outer[2]};",
                 f"{indent}let {child} = {parent} && ({inner[2]});"]

    def append_branch(condition: str, begin: int, end: int, dedent: int = 0) -> None:
        if begin < end:
            flattened.append(f"{indent}if {condition} {{")
            flattened.extend(line[dedent:] for line in lines[begin:end])
            flattened.append(f"{indent}}}")

    append_branch(child, start + 2, inner_branch_end, dedent=4)
    append_branch(f"({parent} && !{child})", inner_branch_end + 1, inner_end, dedent=4)
    append_branch(parent, inner_end + 1, outer_branch_end)
    append_branch(f"!{parent}", outer_branch_end + 1, outer_end)
    lines[start:outer_end + 1] = flattened
    return True


def flatten_shadow_wgsl(source: str) -> str:
    """Reduce nesting in Naga's structured shadow WGSL for Chromium Tint."""
    lines = source.splitlines()
    index = 0
    while True:
        indent, peak = max((len(line) - len(line.lstrip()), offset)
                           for offset, line in enumerate(lines) if line.strip())
        # Tint counts both statement and compound scope recursion. Keep room
        # below its 127-level parser budget for the deepest evaluator branches.
        if indent <= 61 * 4:
            return "\n".join(lines) + "\n"
        for start in range(peak - 1, -1, -1):
            line = lines[start]
            depth = len(line) - len(line.lstrip())
            if depth >= indent or not line.endswith("{"):
                continue
            indent = depth
            if _flatten_nested_wgsl_if(lines, start, peak, index):
                index += 1
                break
        else:
            raise ValueError("Cannot reduce the specialized shadow WGSL nesting")

# SPDX-License-Identifier: GPL-2.0-or-later

"""Inventory the stock Cycles registries and check the WebGPU coverage ledger."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


WEB = Path(__file__).resolve().parent.parent
BLENDER = WEB.parent / "blender"
LEDGER = Path(__file__).with_name("source_coverage.json")
CYCLES = "intern/cycles/kernel/"
STATUSES = {"ported", "limited", "unsupported", "unverified", "native-only", "internal"}


def blank(match):
    return re.sub(r"[^\n]", " ", match[0])


def source(path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", blank, path.read_text(), flags=re.S)


def enum_body(text, enum):
    match = re.search(rf"\benum\s+{enum}\b(?:\s*:\s*\w+)?\s*\{{([^}}]*)\}};", text)
    if not match:
        raise ValueError(f"Cannot locate Blender's {enum} enum")
    return match[1], text.count("\n", 0, match.start(1)) + 1


def remove_directive(text, pattern):
    result, count = re.subn(pattern, blank, text, flags=re.M)
    if count != 1:
        raise ValueError(f"Expected one declaration matching {pattern}")
    return result


def enum_members(text, first_line, expand=None, previous_value=-1):
    value = previous_value
    for token in re.finditer(r"[^,]+", text):
        name = token[0].strip()
        if not name:
            continue
        line = first_line + text.count("\n", 0, token.start() + len(token[0]) - len(token[0].lstrip()))
        match = re.fullmatch(r"(\w+)(?:\s*=\s*(0x[0-9a-fA-F]+|\d+))?", name)
        if match:
            value = int(match[2], 0) if match[2] is not None else value + 1
            yield {"name": match[1], "value": value, "line": line}
        else:
            names = expand(name) if expand else ()
            if not names:
                raise ValueError(f"Unrecognized enum member at line {line}: {name}")
            for expanded in names:
                value += 1
                yield {"name": expanded, "value": value, "line": line}


def film_convert(name):
    match = re.fullmatch(r"DECLARE_FILM_CONVERT_KERNEL\((\w+)\)", name)
    if match:
        kernel = f"DEVICE_KERNEL_FILM_CONVERT_{match[1]}"
        return kernel, f"{kernel}_HALF_RGBA"
    return ()


def device_kernels(text):
    body, first_line = enum_body(text, "DeviceKernel")
    body = remove_directive(
        body, r"^#define DECLARE_FILM_CONVERT_KERNEL\(variant\)\s*\\\n"
              r"\s*DEVICE_KERNEL_FILM_CONVERT_##variant,\s*DEVICE_KERNEL_FILM_CONVERT_##variant##_HALF_RGBA\s*$")
    body = remove_directive(body, r"^#undef DECLARE_FILM_CONVERT_KERNEL$")
    return list(enum_members(body, first_line, film_convert))


def svm_nodes(text, template):
    body, first_line = enum_body(text, "ShaderNodeType")
    body = remove_directive(body, r"^#define SHADER_NODE_TYPE\(name\) name,$")
    body = remove_directive(body, r"^#define SHADER_NODE_TYPE_DERIVATIVE\(name\) name, name##_DERIVATIVE,$")
    body = remove_directive(body, r'^#include "node_types_template\.h"$')
    directives = {
        "#ifndef SHADER_NODE_TYPE",
        "#define SHADER_NODE_TYPE(name)",
        "#endif",
        "#ifndef SHADER_NODE_TYPE_DERIVATIVE",
        "#define SHADER_NODE_TYPE_DERIVATIVE(name) SHADER_NODE_TYPE(name)",
        "#undef SHADER_NODE_TYPE",
        "#undef SHADER_NODE_TYPE_DERIVATIVE",
    }
    rows = []
    for number, line in enumerate(template.splitlines(), 1):
        line = line.strip()
        if not line or re.sub(r"^#\s+", "#", line) in directives:
            continue
        match = re.fullmatch(r"SHADER_NODE_TYPE(_DERIVATIVE)?\((NODE_\w+)\)", line)
        if not match:
            raise ValueError(f"Unrecognized SVM entry at line {number}: {line}")
        rows.append({"name": match[2], "value": len(rows), "line": number})
        if match[1]:
            rows.append({"name": f"{match[2]}_DERIVATIVE", "value": len(rows), "line": number})
    for member in enum_members(body, first_line, previous_value=len(rows) - 1):
        rows.append({**member, "source": CYCLES + "svm/types.h"})
    return rows


def kernel_features(text):
    values = {}
    joined = re.sub(r"\\\n", " \x00", text)
    for match in re.finditer(r"^#[ \t]*define[ \t]+(KERNEL_FEATURE_\w+)[ \t]+([^\n]+)", joined, re.M):
        name, expression = match.groups()
        expression = expression.replace("\x00", " ").strip()
        shift = re.fullmatch(r"\(1ULL\s*<<\s*(\d+)ULL\)", expression)
        if shift:
            value = 1 << int(shift[1])
        else:
            value = 0
            for part in expression.strip("()").split("|"):
                feature = part.strip()
                if feature not in values:
                    raise ValueError(f"Unrecognized feature expression for {name}: {expression}")
                value |= values[feature]
        values[name] = value
        line = joined[:match.start()]
        yield {"name": name, "value": value, "line": line.count("\n") + line.count("\x00") + 1}


def inventory(blender):
    types = CYCLES + "types.h"
    svm = CYCLES + "svm/types.h"
    template = CYCLES + "svm/node_types_template.h"
    features = CYCLES + "features.h"
    sources = {name: source(blender / name) for name in (types, svm, template, features)}
    groups = {
        "DeviceKernel": (types, device_kernels(sources[types])),
        "ShaderNodeType": (template, svm_nodes(sources[svm], sources[template])),
        "ClosureType": (svm, enum_members(*enum_body(sources[svm], "ClosureType"))),
        "PassType": (types, enum_members(*enum_body(sources[types], "PassType"))),
        "KernelFeature": (features, kernel_features(sources[features])),
    }
    return {registry: [{"source": file, **member} for member in members]
            for registry, (file, members) in groups.items()}


def signature(members):
    table = "".join(f"{member['name']}={member['value']}\n" for member in members)
    return hashlib.sha256(table.encode()).hexdigest()


def coverage(blender, ledger, web=WEB):
    stock = inventory(blender)
    if ledger["schema"] != 1:
        raise ValueError("Unsupported Cycles coverage ledger schema")
    tracked = ledger["registries"]
    if set(stock) != set(tracked):
        raise ValueError("Coverage ledger registry names differ from the source inventory")
    result = {}
    errors = []
    evidence = {}
    for registry, members in stock.items():
        names = {member["name"] for member in members}
        if not names or len(names) != len(members):
            errors.append(f"{registry}: stock registry is empty or has duplicate names")
        baseline = tracked[registry]
        annotations = {}
        for group in baseline["groups"]:
            status, note = group["status"], group["note"]
            references = group.get("evidence", [])
            if status not in STATUSES or not note or (status != "unverified" and not references):
                raise ValueError(f"{registry}: invalid status or missing reason/evidence in {group}")
            for reference in references:
                if reference not in evidence:
                    path, separator, anchor = reference.partition("#")
                    root, _, relative = path.partition("/")
                    if root not in {"web", "blender"} or not separator or not anchor or not relative:
                        raise ValueError(f"Malformed evidence reference: {reference}")
                    file = {"web": web, "blender": blender}[root] / relative
                    evidence[reference] = file.is_file() and anchor in file.read_text()
                    if not evidence[reference]:
                        errors.append(f"Missing coverage evidence: {reference}")
            for name in group["members"]:
                if name in annotations:
                    errors.append(f"{registry}: duplicate ledger entry {name}")
                annotations[name] = {"status": status, "note": note, "evidence": references}
        errors.extend(f"{registry}: untracked {name}" for name in sorted(names - annotations.keys()))
        errors.extend(f"{registry}: stale entry {name}" for name in sorted(annotations.keys() - names))
        stock_signature = signature(members)
        if stock_signature != baseline["signature"]:
            errors.append(f"{registry}: stock names, ordering, or values changed ({stock_signature})")
        rows = [{**member, **annotations.get(member["name"], {"status": "untracked"})}
                for member in members]
        result[registry] = {"signature": stock_signature,
                            "counts": dict(sorted(Counter(row["status"] for row in rows).items())),
                            "members": rows}
    return {"schema": 1, "registries": result, "errors": errors}


def add_parser(commands):
    parser = commands.add_parser("source-coverage", help="audit the stock Cycles coverage ledger")
    parser.add_argument("--blender-source", type=Path, default=BLENDER)
    parser.add_argument("--output", type=Path, help="write ordered registry details as JSON")
    parser.add_argument("--check", action="store_true", help="fail on source or ledger drift")
    parser.add_argument("--require-audited", action="store_true",
                        help="also fail if any stock entries still need an implementation audit")


def run(args):
    try:
        report = coverage(args.blender_source, json.loads(LEDGER.read_text()))
    except (OSError, KeyError, ValueError) as error:
        print(f"Cannot audit Cycles coverage: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for registry, details in report["registries"].items():
        counts = ", ".join(f"{status} {count}" for status, count in details["counts"].items())
        print(f"{registry} ({len(details['members'])}): {counts}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    if args.require_audited:
        unverified = sum(details["counts"].get("unverified", 0)
                         for details in report["registries"].values())
        if unverified:
            print(f"{unverified} stock entries still need a WebGPU audit", file=sys.stderr)
            return 1
    return int((args.check or args.require_audited) and bool(report["errors"]))

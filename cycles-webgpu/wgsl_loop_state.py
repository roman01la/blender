# SPDX-License-Identifier: Apache-2.0

"""Simplify Naga's state copies around a lowered Cycles dispatch switch."""

from collections import Counter
import re


_PHI = r"phi_\d+_"
_TEMP = r"_e\d+"
_SCRATCH = r"local(?:_\d+)?"


class _UnrecognizedLoop(ValueError):
    """The generated shader does not match the supported form."""


def _require(condition):
    if not condition:
        raise _UnrecognizedLoop()


def _bindings(lines, pattern):
    bindings = {}
    for line in lines:
        match = re.fullmatch(pattern, line)
        _require(match is not None)
        lhs, rhs = match.groups()
        _require(lhs not in bindings)
        bindings[lhs] = rhs
    return bindings


def _coalesce(shader):
    lines = shader.splitlines()
    _require(lines.count("    loop {") == 1)
    loop = lines.index("    loop {")
    switch = next(i for i in range(loop, len(lines)) if lines[i].startswith("            switch "))
    end_switch = lines.index("            }", switch)
    resume = lines.index("            continue;", end_switch)
    carry = lines.index("        continuing {", resume)
    end_carry = lines.index("        }", carry)
    _require(lines[resume:carry] == ["            continue;", "        } else {",
                                      "            break;", "        }"])
    _require(lines[end_carry + 1] == "    }")

    # Values read at the head of the loop stay immutable through the switch.
    saved = _bindings(lines[loop + 1:switch - 1], fr"        let ({_TEMP}) = ({_PHI});")
    guard = re.fullmatch(fr"        if \(({_TEMP}) != 0u\) \{{", lines[switch - 1])
    _require(guard is not None and guard[1] in saved)
    _require(lines[switch] == f"            switch bitcast<i32>({guard[1]}) {{")
    incoming = {state: snapshot for snapshot, state in saved.items()}
    size = len(incoming)
    _require(size and len(saved) == size)

    # Recognize both halves of each outgoing -> scratch -> incoming copy.
    stores = lines[end_switch + 1:resume]
    copies = lines[carry + 1:end_carry]
    _require(len(stores) == len(copies) == 2 * size)
    loaded = _bindings(stores[:size], fr"            let ({_TEMP}) = ({_PHI});")
    saved_to_scratch = _bindings(stores[size:], fr"            ({_SCRATCH}) = ({_TEMP});")
    loaded_scratch = _bindings(copies[::2], fr"            let ({_TEMP}) = ({_SCRATCH});")
    restored = _bindings(copies[1::2], fr"            ({_PHI}) = ({_TEMP});")
    _require(set(restored) == set(incoming))
    _require(list(loaded_scratch) == list(restored.values()))
    _require(set(loaded_scratch.values()) == set(saved_to_scratch))
    outgoing = {loaded[saved_to_scratch[loaded_scratch[temporary]]]: state
                for state, temporary in restored.items()}
    _require(len(outgoing) == size and outgoing.keys().isdisjoint(incoming))

    # Storage can be coalesced only if each removed binding has the exact
    # expected type and no other reads or writes anywhere in the shader.
    before = "\n".join(lines[:loop]) + "\n"
    types = dict(re.findall(r"^    var (\w+): (.+);$", before, re.M))
    initialized = Counter(re.findall(fr"^    ({_PHI}) = ", before, re.M))
    mentions = Counter(re.findall(r"\b\w+\b", shader))
    for old, temporary in restored.items():
        scratch = loaded_scratch[temporary]
        new = loaded[saved_to_scratch[scratch]]
        _require(types[new] == types[scratch] == types[old])
        _require(mentions[scratch] == 3 and mentions[old] == 4 and initialized[old] == 1)
    _require(all(mentions[name] == 2 for name in loaded.keys() | loaded_scratch.keys()))

    # Each case has one straight-line update per outgoing state immediately
    # before its final break. Redirect writes to the incoming state. Any RHS
    # referring to an old value still reads its immutable loop-head snapshot,
    # so this keeps the parallel-copy semantics even when states exchange values.
    case_open = re.compile(r"                (?:case .+|default): \{")
    rewritten = []
    num_cases = 0
    pos = switch + 1
    while pos < end_switch:
        _require(case_open.fullmatch(lines[pos]) is not None)
        end_case = lines.index("                }", pos)
        _require(end_case < end_switch and lines[end_case - 1] == "                    break;")
        first_store = end_case - size - 1
        _require(first_store > pos)
        assignments = _bindings(lines[first_store:end_case - 1], fr"                    ({_PHI}) = (.+);")
        _require(assignments.keys() == outgoing.keys())
        rewritten.extend(lines[pos:first_store])
        for destination, value in assignments.items():
            old = outgoing[destination]
            if value != incoming[old]:
                rewritten.append(f"                    {old} = {value};")
        rewritten.extend(lines[end_case - 1:end_case + 1])
        num_cases += 1
        pos = end_case + 1
    # Each outgoing phi is declared once, written once per case and read
    # once after the switch. This excludes unrecognized RHS references.
    _require(num_cases >= 2 and all(mentions[name] == num_cases + 2 for name in outgoing))

    removed = outgoing.keys() | saved_to_scratch.keys()
    declarations = re.compile(r"    var (\w+): .+;")
    prefix = [line for line in lines[:loop]
              if not ((match := declarations.fullmatch(line)) and match[1] in removed)]
    return "\n".join(prefix + lines[loop:switch + 1] + rewritten + lines[end_switch:end_switch + 1]
                     + lines[resume:carry] + lines[end_carry + 1:]) + "\n"


def coalesce_loop_state(shader):
    """Return the input unchanged when Naga emits a different control-flow shape."""
    try:
        return _coalesce(shader)
    except (ValueError, KeyError, IndexError, StopIteration):
        return shader

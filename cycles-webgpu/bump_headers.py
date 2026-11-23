# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def prepare_bump_header(work: Path, cycles: Path) -> None:
    """Reuse the stock bump terminator without pulling in all BSDF dispatches."""
    source = (cycles / "kernel/closure/bsdf.h").read_text()
    start = source.index("ccl_device_inline float bump_shadowing_term(")
    end = source.index("ccl_device_inline float shift_cos_in(", start)
    (work / "bump_shadowing.h").write_text(
        "CCL_NAMESPACE_BEGIN\n\n" + source[start:end] + "CCL_NAMESPACE_END\n"
    )

# Material node GPU tests

The suite reads the browser build’s Blender RNA, constructs the same material graphs in
Chrome and desktop Blender, and verifies which backend renders each one. It records
options, available sockets, defaults, numeric bounds, and layouts for each enumerated
node mode. Registered helper, layout, world, light, and line-style nodes appear in the
catalog with their scope.

Run these commands from the web project directory. Use a packaged browser build, Node
24+, Chrome with hardware WebGPU, Python 3.10+, and a desktop Blender with a GPU-supported
Cycles device. On macOS the native default is Metal. Set the CHROME environment variable
to use another Chrome executable.

    python3 material-tests/run.py inventory --browser-build /path/to/browser/package \
      --source-revision "$(git -C ../blender rev-parse HEAD)" \
      --output /tmp/material-catalog.json

    python3 material-tests/run.py run --browser-build /path/to/browser/package \
      --native /path/to/Blender --catalog /tmp/material-catalog.json \
      --output /tmp/material-results --strategy modes

The native and browser revisions must match by default. An explicit
`--allow-different-revision` compares cases with matching graph fingerprints and
records both revisions in the report. Attachments, project files, and preferences from
your regular Blender session are outside the fixtures.

| Strategy | Coverage |
| --- | --- |
| `default` | Each available output with its default settings |
| `modes` | Defaults plus each available enum choice and boolean toggle, including nested settings |
| `options` | Modes plus representative slider values, linked inputs, texture storage, and ramp/curve edits |
| `pairwise` | Options plus pairs of discrete node settings |

Use `--match REGEX` to select cases by node or option in the case JSON, or
`--shards COUNT --shard-index INDEX` to divide the matrix starting at index zero.
The default fixture uses an area light and a constant world. The `point` and `two-points`
lighting fixtures use zero-radius point lights and a black world. `area-hdri` uses a
single area light and an unsampled HDRI. `mixed-hdri` uses an area light, a finite-radius
point light, and an importance-sampled HDRI with the Light Tree. Numeric tests use finite samples within the ranges recorded
in the catalog. Line-style nodes use a textured Freestyle stroke fixture on Eevee.
The volume fixtures exercise image-driven inputs and Accurate intersections.
The suite records invalid layouts, unsupported nodes, and cases requiring an
additional fixture with a reason.

Check Eevee volume image parity in Fast and Accurate modes with axial and oblique
mesh-volume views.

    python3 material-tests/check_volume.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/volume-results

Check Eevee's UDIM Image Texture Color and Alpha against desktop GPU renders
for Closest, Linear, Cubic, and Smart interpolation.

    python3 material-tests/check_udim.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/udim-results

The report includes a diffuse GPU control for each batch. Cycles records an actual
scheduler completion or an explicit CPU fallback in the browser and a GPU path
tracing device in desktop Blender. Eevee checks the graphics backend, shader errors,
and suspicious magenta output. The render fixture exports and checks a fresh
32 × 32 image per case. Results include the decoded RGBA pixel hash and the center
region's mean RGB. CPU fallback, shader failure, changed graphs, and missing evidence
have individual verdicts. Pixel hashes let you check exact image equality independently
of the GPU verdict.

For CI, `--fail-on-fallback` returns a failure when it finds browser CPU fallback,
GPU failure, or suspected shader error output against a native GPU render. See
`report.md` for grouped counts and `results.json` for individual graph settings,
backend evidence, and results. Each batch retains its native and browser logs.

Run the log and backend classifier tests with

    python3 -m unittest discover -s material-tests -v

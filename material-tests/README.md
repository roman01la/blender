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

## Cycles coverage from Blender source

See the [work plan](cycles-webgpu-work-plan.md) and [coverage ledger](source_coverage.json) for implementation priorities and verification.

Run the source audit without a browser or desktop build.

    python3 material-tests/run.py source-coverage --check --output /tmp/cycles-coverage.json

The audit reads the pinned sibling Blender source and expands its DeviceKernel,
ShaderNodeType, ClosureType, PassType, and kernel feature registries. The JSON
report includes each stock name, value, source location, WebGPU status, and
reviewed evidence. Pass `--blender-source` to compare another Blender checkout.

`--check` fails when a stock entry is added, removed, reordered, or assigned a
new value, when the ledger has stale or duplicate entries, or when an evidence
anchor disappears. Review the stock and WebGPU paths before updating the
registry signature and the related entries in `source_coverage.json`. A stock
kernel ID reused as a private WebGPU dispatch slot gives no coverage credit for
the original operation.

The ledger records ported paths, limited implementations, known unsupported
paths, stock CPU-only operations, internal entries, and unverified entries.
Use `--require-audited` to also fail on entries marked unverified. Ported and
limited statuses describe source paths. Render results from the material matrix
and focused fixtures record GPU execution and image agreement for the tested
settings and combinations.

Check Eevee volume image parity in Fast and Accurate modes with axial and oblique
mesh-volume views.

    python3 material-tests/check_volume.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/volume-results

Check Eevee's UDIM Image Texture Color and Alpha against desktop GPU renders
for Closest, Linear, Cubic, and Smart interpolation.

    python3 material-tests/check_udim.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/udim-results

Check Cycles Light Tree triangle lights against desktop CPU Cycles. The cases
cover front and back sampling, a tree with no lamps, transformed shared meshes,
and glossy emitter hits with and without multiple importance sampling.
Textured cases exercise camera-hidden image and procedural emitters, flipped
UVs, and transformed instances.
Use `--transparent-emission` to compare front and back sampled emission,
partially transparent emitters, and camera alpha on both sides. The latter
cases also cover directional sky emission and the flat light distribution.

    python3 material-tests/check_triangle_emitters.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/triangle-emitter-results

Check Cycles Light Falloff with point, area, and sun lights against desktop CPU
Cycles. The cases cover all three falloff outputs, near-light smoothing, and
the stock infinite-distance behavior for sun lights. Camera-visible sources
also exercise forward light hits. Use `--shadow-toggle` to compare Point and
Area lights with and without shadows from an off-camera blocker.
Use `--lamp-mis` to verify Area direct lighting and camera hits with the lamp's
multiple importance sampling setting disabled.

    python3 material-tests/check_light_falloff.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/light-falloff-results

Compare Cycles Ambient Occlusion, Bevel and Raycast with desktop Metal. The
fixtures exercise occluded and unoccluded geometry, local rays, and both hits
and misses. The tests combine queries with Noise, Gabor, advanced Voronoi, and
projected images, including graphs with all three procedural textures. They
check full-image error and confirm GPU execution.

    python3 material-tests/check_scene_queries.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/scene-query-results

Compare Cycles Transparent BSDF with desktop Metal. The cases cover image and
procedural tint, film alpha, colored shadows, and secondary Ambient Occlusion.

    python3 material-tests/check_transparency.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/transparency-results

Check mixed Transparent BSDF closures against desktop CPU Cycles. The fixtures
cover Lambert and metallic Principled, varying weights, film alpha, and shadows.

    python3 material-tests/check_mixed_transparency.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/mixed-transparency-results

Check the Cycles Ambient Occlusion film pass through opaque, tinted, varying,
and backface-dependent shadow materials. A paired render checks that enabling
the data pass preserves Combined output.

    python3 material-tests/check_film_ao.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/film-ao-results

Check external RGBA16 texture precision, filtering, and Glossy mixed with
Transparent against desktop CPU Cycles. The fixtures include the Classroom
wireframe material graph, film alpha and Normal, translucent shadows, and
guided OIDN.

    python3 material-tests/check_glossy_transparency.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/glossy-transparency-results

Compare ordered sharp and rough Glossy with Transparent against desktop CPU
Cycles. The fixtures cover camera-dependent selection, simultaneous weighted
lobes, varying tint, and the Glossy Color pass. Shadow fixtures cover cubic
alpha filtering, stock Ambient Occlusion defaults, and single-octave 3D F1
Voronoi Distance with and without normalization.
Add `--guided` for guided WebGPU and CPU OIDN.

    python3 material-tests/check_ordered_glossy_transparency.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/ordered-glossy-transparency-results

Compare Principled transmission, Glass, and Refraction with both caustics
switches on and off. The fixtures use a camera-visible sphere and a diffuse
receiver inside a camera-hidden sphere to exercise both ray visibility modes.

    python3 material-tests/check_glass_caustics.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/glass-caustics-results

Check ordered Transparent, Diffuse, and Translucent closures against desktop
CPU Cycles. The fixtures cover varying image alpha, shared Bump, object-random
image selection, transmitted lighting, film alpha and Normal, shadows, and
guided OIDN. A flat control checks that an off-camera Bump leaves film alpha
unchanged.

    python3 material-tests/check_paper_closures.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/paper-closure-results

Compare a Principled BSDF with image-linked Alpha and a separate Translucent
closure. The fixtures cover Add and Mix in both orders, constant Alpha,
Multiscatter GGX, the Normal pass, guided OIDN, and varying shadows.

    python3 material-tests/check_principled_translucent.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/principled-translucent-results

Compare mapped HDRI World shaders with desktop CPU Cycles. The cases cover
rotation, HSV adjustment, Light Tree and distribution sampling, illumination
with the world hidden from the camera, disabled world MIS, and guided OIDN.

    python3 material-tests/check_world_graph.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/world-graph-results

Compare weighted diffuse-only Principled mixtures against desktop CPU Cycles.
The fixtures cover branch order, varying weights, Blender’s per-closure cutoff,
and the fallback for independent normals. Use `--nested` for the four-branch cases.

    python3 material-tests/check_mixed_principled.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/mixed-principled-results

Check Diffuse and Glossy Bump normals against desktop CPU Cycles, including both
closure orders for GGX and Beckmann and an encoded Normal-pass comparison.
Use `--ordered` for nested four-closure fixtures and add `--guided` to compare
WebGPU tracing and guided denoising against desktop CPU Cycles and OIDN.
Use `--displacement-bump` for standalone GGX and Beckmann with both shader-wide
displacement and a local Bump normal. Its default point lights isolate the
materials. Use `--lighting area-hdri` to exercise area lighting with the Light
Tree disabled.
Use `--mixed-displacement-bump` for Lambert Diffuse and isotropic GGX or
Beckmann with a shared local normal and shader-wide displacement. The fixtures
cover both closure orders, linked and endpoint mix weights, the Glossy Color
pass, and the fallback when both closures independently supply a local normal.

    python3 material-tests/check_mixed_glossy.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/mixed-glossy-results

Check six Hair modes and both Christensen–Burley nodes against desktop CPU
Cycles with 128 samples and the mixed-light HDRI fixture.

    python3 material-tests/check_surface_scattering.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/surface-scattering-results

Check all five phases of Volume Scatter and Volume Coefficients plus
Principled Volume against desktop CPU Cycles. The comparison defaults to
512 samples and four volume bounces with an area light, a point light, and an
importance-sampled HDRI.
Pass `--lighting area-world` to use the default material audit lighting.

    python3 material-tests/check_volume_scattering.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/volume-scattering-results

Check intersecting homogeneous media, a camera inside two media, and a stock
VDB smoke grid with linear and cubic filtering. The denoising cases compare
with desktop CPU OIDN and observe the browser's GPU queue through tracing,
albedo and normal prefiltering, denoising, and film export.

    python3 material-tests/check_volume_transport.py --browser-build /path/to/browser/package \
      --native /path/to/Blender --output /tmp/volume-transport-results

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

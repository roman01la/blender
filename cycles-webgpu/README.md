# Cycles WebGPU kernel experiments

The camera experiment compiles two functions from the pinned Cycles source
into a WebGPU compute shader. `get_work_pixel` maps a GPU work index to a tile
pixel and sample. `camera_sample_perspective` generates a primary ray with
camera transforms and clipping. The probe dispatches a static perspective
camera at pixel centers and reads back each ray origin, direction, and
clipping interval. A native executable built from the same C++ checks the
GPU output.

The HLSL entry point provides resource bindings and a dispatch bounds check.
Clang compiles the Cycles C++ with Emscripten's headers, and LLVM links the
optimized camera code to the entry point before emitting Vulkan SPIR-V.
SPIRV-Tools removes unused binding-name constants and validates the normalized
module. The builder preserves signed float-to-integer conversions for the
browser port's Naga translator. LLVM keeps structured branches and loops in
BVH2 traversal and shading, including Cycles' Sobol sampler.

## Verify camera rays

Use LLVM 22.1.8, SPIRV-Tools v2026.4, the port's Emscripten 6.0.1 sysroot,
Node.js, and the shader translator Wasm from an editor build. Run these
commands from the workspace root while the Blender server is running.

```sh
python3 web/cycles-webgpu/build.py \
  --llvm-bin /path/to/llvm/bin \
  --spirv-bin /path/to/spirv-tools/bin \
  --emscripten-sysroot /path/to/emscripten/cache/sysroot \
  --translator /path/to/blender-shader-translator.wasm \
  --output /tmp/blender-cycles-webgpu

node web/cycles-webgpu/verify.mjs \
  /tmp/blender-cycles-webgpu/camera.wgsl \
  /tmp/blender-cycles-webgpu/reference.json
```

The test uses a disposable Chrome profile and the existing loopback server.
Set `CHROME` to the Chrome binary if it is outside the default macOS path.
Set `BLENDER_LOCAL_ORIGIN` if the server runs on an origin other than
`http://127.0.0.1:8002`. Expect `CYCLES_WEBGPU_PRIMARY_RAYS_PASS` with 392 rays
checked. The test covers both Cycles tile ordering modes, a translated and
rotated camera, clipping, and bounds on partial workgroups. Chrome verified
the results on an Apple Metal 3 adapter in a dedicated worker.

## Verify BVH2 traversal

The traversal experiment compiles the stock Cycles BVH2 kernel for static
triangles. It uses Cycles' aligned nodes, primitive visibility, ray clipping,
self-intersection checks, and static instance transforms. The LLVM pass
keeps storage accesses as 32-bit words and rewrites the compiler's control
flow into a single structured dispatch loop so the Vulkan backend can
translate the nested traversal.

The bundled fixture uses BVH2 buffers captured during a pinned Blender CPU
render of a synthetic scene with shared transformed meshes and overlapping
triangles. The expected results come from upstream Cycles
`scene_intersect` running natively with `ThreadKernelGlobalsCPU`. Chrome
checks each hit and miss, primitive and object ID, primitive type,
distance and barycentric coordinates. The fixture has 1,138 rays,
covering 882 hits across 189 triangles in three objects and 256 misses.

```sh
python3 web/cycles-webgpu/build.py --bvh2 \
  --llvm-bin /path/to/llvm/bin \
  --spirv-bin /path/to/spirv-tools/bin \
  --emscripten-sysroot /path/to/emscripten/cache/sysroot \
  --translator /path/to/blender-shader-translator.wasm \
  --output /tmp/blender-cycles-webgpu

mkdir -p /tmp/blender-cycles-webgpu/fixture
tar -xzf web/cycles-webgpu/fixtures/bvh2-cpu-capture.tar.gz \
  -C /tmp/blender-cycles-webgpu/fixture

node web/cycles-webgpu/verify.mjs --bvh2 \
  /tmp/blender-cycles-webgpu/bvh2.wgsl \
  /tmp/blender-cycles-webgpu/fixture/scene.bin \
  /tmp/blender-cycles-webgpu/fixture/rays.bin \
  /tmp/blender-cycles-webgpu/fixture/expected.bin
```

Expect `CYCLES_WEBGPU_BVH2_PASS`. The builder also produces
`bvh2-reference scene.bin rays.bin hits.bin` so you can compare GPU results
with the same adapted traversal compiled to native code for other trusted
BVH2 captures. All inputs and outputs contain little-endian 32-bit words.

- Scene header words 0–9 contain the signed root node index followed by
  absolute word offsets for aligned BVH nodes, leaves, primitive visibility,
  primitive index, primitive object, object node, object records, triangle
  vertex indices and triangle vertex positions. BVH nodes and leaves use
  their original four-word records. Objects contain the signed position
  offset and three inverse-transform float4 rows. Triangle indices and
  positions are packed triples.
- The first ray word gives the ray count. Each ray has 12 words for
  origin XYZ and minimum distance, direction XYZ and maximum distance,
  visibility mask, optional self object and primitive IDs, and a reserved
  word. Use `0xFFFFFFFF` for absent self IDs.
- Each result has eight words for hit flag, primitive ID, object ID,
  primitive type, distance, two barycentric coordinates and a reserved
  word. The float values are stored as raw IEEE 754 bits.

## Verify camera, intersection, and surface setup together

Build the `--bvh2`, `--primary-camera`, and `--surface` modes with the same
toolchain arguments and output directory as above. The camera writes the ray
buffer in the BVH2 input format. The intersection and flat triangle surface
dispatches consume the ray and hit buffers on the GPU in the same submission.
Afterward the verifier reads back the outputs and compares hit records with
native Cycles. It checks hit positions against the CPU rays and geometric
normals against the captured triangles and inverse transforms.

```sh
mkdir -p /tmp/blender-cycles-webgpu/surface-fixture
tar -xzf web/cycles-webgpu/fixtures/surface-cpu-capture.tar.gz \
  -C /tmp/blender-cycles-webgpu/surface-fixture

node web/cycles-webgpu/verify.mjs --surface \
  /tmp/blender-cycles-webgpu/primary-camera.wgsl \
  /tmp/blender-cycles-webgpu/bvh2.wgsl \
  /tmp/blender-cycles-webgpu/surface-fixture/scene.bin \
  /tmp/blender-cycles-webgpu/primary-reference.json \
  /tmp/blender-cycles-webgpu/bvh2-reference \
  /tmp/blender-cycles-webgpu/surface.wgsl \
  /tmp/blender-cycles-webgpu/surface-fixture/surface.bin
```

Expect `CYCLES_WEBGPU_SURFACE_PASS`. The fixture captures tilted and scaled
instances, multiple material IDs, front and back faces, and smooth triangles.
The surface pass uses stock Cycles triangle setup for flat triangles and
reports a separate status for unsupported hits. The `--primary` verifier
uses the same first five arguments to check camera generation and intersections.
Build `--primary-camera` to generate `sampled-primary-reference.json` as well.
Pass that file in place of `primary-reference.json` to the `--primary` verifier
to compare pixel-filtered rays and BVH2 hits with the native reference.
The sampled cases cover blue noise, Sobol-Burley, tabulated Sobol with
scrambling distance, and a fixed viewport jitter offset.

Camera tiles use the existing seven camera words followed by a visibility
mask. The camera fills the BVH2 ray count and records with that mask and
empty self IDs. The surface metadata header contains two absolute word
offsets. The first points to objects containing an object flags word and
three forward transform rows. The second points to one packed shader word
per triangle. The output has eight words per ray for world position XYZ,
status, oriented geometric normal XYZ, and packed shader. Status values are
0 for a miss, 1 for a front face, 2 for a back face, and 3 for an unsupported
surface. Misses and unsupported surfaces have a packed shader value of
`0xffffffff`.

## Verify direct lighting and progressive film

Build `--direct-light` with the same toolchain and output directory as the
camera, BVH2, and surface shaders. The direct-light pass uses Cycles' point
light sampler and diffuse BSDF and traces each opaque shadow ray through its
BVH2. The test supplies constant per-material albedos, one normalized point
light, and a constant background. Unsupported surface samples contribute
transparent black.

```sh
node web/cycles-webgpu/verify.mjs --lit \
  /tmp/blender-cycles-webgpu/primary-camera.wgsl \
  /tmp/blender-cycles-webgpu/bvh2.wgsl \
  /tmp/blender-cycles-webgpu/surface-fixture/scene.bin \
  /tmp/blender-cycles-webgpu/primary-reference.json \
  /tmp/blender-cycles-webgpu/bvh2-reference \
  /tmp/blender-cycles-webgpu/surface.wgsl \
  /tmp/blender-cycles-webgpu/surface-fixture/surface.bin \
  /tmp/blender-cycles-webgpu/direct-light.wgsl \
  /tmp/blender-cycles-webgpu/direct-light-reference \
  web/cycles-webgpu/film.wgsl

node web/cycles-webgpu/verify.mjs --film web/cycles-webgpu/film.wgsl
```

Expect `CYCLES_WEBGPU_LIT_FILM_PASS` and `CYCLES_WEBGPU_FILM_PASS`. The integrated
replay traces two consecutive sample ranges per tile and resolves both into
scene-linear RGBA using one persistent GPU film buffer. Camera, traversal,
surface, lighting, and film stages share GPU buffers, with verification
readback after both submissions. Chrome compares the film and per-ray
radiance with the native reference and checks for lit, shadowed, backlit, and
background samples. The film fixture holds camera rays at pixel centers across both submissions.
The reference runs the same Cycles lighting and traversal functions in a
native executable, with camera and primary hits supplied by the other CPU
reference tools.

The film-only test covers multiple interleaved samples per pixel, uneven
sample counts, HDR values, opacity, exposure, partial workgroups, and clear
and resize. The resolved pixels are premultiplied and scene-linear so the
display can apply Blender's view transform.

The direct-light pass binds the scene, primary rays, primary hits, surface
points, lighting table, and per-ray RGBA output at bindings 0–5 respectively.
The lighting table begins with three four-word records. The first contains
the point light position and the unsigned material count. The second
contains the RGB light strength, and the third contains the RGB background.
One RGB albedo and a padding word follow per material. Float values are
stored as IEEE 754 words.

The film's `accumulate` entry point uses bindings 0–3 for its parameters,
the camera pixel and sample indices, the lighting pass output, and the
persistent film. Each pixel uses four float sums and one unsigned sample
count. The `resolve` entry point uses bindings 0, 3, and 4 for the same
parameters, the film, and a row-major float RGBA output. For each film, clear
or allocate its storage before the first submission, and dispatch a resolve
after the contributions it should include.

## Thread-local surface closures

The core SVM build generates `local-closures.wgsl` alongside its split
shader. The standard Cycles WebGPU package includes both and selects fusion
automatically for eligible renders. Use `--local-closures` to rebuild just
the fused shader. It keeps core SVM closure
data thread-local through legacy opaque reflection lighting and path
continuation, with inline shadow traversal. It requires an adapter that can
run its 576-lane workgroup and bind ten storage buffers. The split WebGPU
path handles other scenes and devices, including Light Trees, guided
denoising, adaptive sampling, and additional surface passes.

For an A/B comparison, set `CYCLES_WEBGPU_FUSED_CLOSURES=0` in Blender's
process environment before rendering to use the split path.

Run the LLVM scalar-pointer regression with:

```sh
python3 web/cycles-webgpu/test_lowering.py --llvm-bin LLVM_BIN --spirv-bin SPIRV_TOOLS_BIN
```

## Use WebGPU in Cycles

Build the stock-scheduler shaders below, stage and build Blender, then pass
their directory to the browser packager.

```sh
python3 web/package_blender.py INSTALL_DIR BROWSER_DIR \
  --cycles-webgpu /tmp/blender-cycles-webgpu
```

In Blender Preferences under System → Cycles Render Devices, select WebGPU
and enable the browser device. Set the scene’s Cycles Device to GPU Compute.
The CLI device override also accepts `--cycles-device WEBGPU`.

Blender creates a `DEVICE_WEBGPU` session and uses `PathTraceWorkGPU` to
admit camera work tiles and select the next kernel from GPU queue counters.
The device marshals Cycles’ work tiles and compacts the slots for the selected
intersection, surface, or background stage. The optional background shader
evaluates the environment and forward area-light hits directly into the
retired path. Both modes reuse the same film kernels. The scheduler admits
paths across a whole sample batch and can refill the pool while earlier
paths are active. New paths use free GPU slots, and multiple samples can
accumulate into the same pixel using atomic updates. The Cycles compute device
requests the adapter's available limits. The path pool grows at sample batch
boundaries according to the work the stock scheduler admits and the size and
indexing limits of its buffers. Large linear dispatches extend across two
workgroup dimensions. SVM tries the largest supported one-dimensional
workgroup and retries at a smaller size if the driver rejects it.

Each queue snapshot clears, counts, and copies the counters in one command
buffer. The renderer waits for validation and mapping before using the
snapshot. It reuses the staging buffer between snapshots.

The deferred lighting shaders prepare light samples, run Cycles’ opaque BVH2 shadow traversal,
and continue paths in separate submissions. Earlier shader packages still
use the fused lighting and shadow shader.

The packed shader scene inputs and camera sampling tables live in GPU
buffers. Upload staging is released after the transfer. Blender’s original
host scene supports synchronization and CPU fallback. Completed paths
accumulate into a GPU film in Cycles’ native Combined and sample-count
layout. The renderer downloads the film for display and output, where
Cycles uses its existing pass conversion. Unsupported scenes and device
failures use the host path tracer. CPU fallback replays completed samples
from zero and repeats Cycles’ adaptive convergence checks at their original
sample boundaries. After a browser device loss, the next offline render
with a new Cycles session requests a fresh WebGPU device on the existing
compute worker.

## Build the scheduler shaders

Build the scheduler shaders with the same toolchain arguments and output
directory. Build the standalone SVM shader and native reference to test the
material evaluator in isolation.

```sh
python3 web/cycles-webgpu/build.py --stock-camera TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --bvh2-compact TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --surface-compact TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --surface-query TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-compact TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-compact --svm-core TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-compact --svm-core --svm-textures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-compact --svm-core --svm-textures --validated-svm-words TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-compact --svm-core --svm-textures --validated-svm-words --hardware-svm-textures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-query TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-query --extended-queries TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-query --extended-queries --ao-only-queries TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm-query --advanced-queries TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --sampled-emission TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --film-surface TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --film-ao TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --film-ao-prepare TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --subsurface-compact TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --absorption TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --absorption-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-interval TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter --volume-guides TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter-shadow --volume-guides TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter-shadow --volume-guides --shadow-materials TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-extinction-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter --volume-grid TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter-shadow --volume-grid TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-extinction-shadow --volume-grid TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter --volume-grid --volume-guides TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --volume-scatter-shadow --volume-grid --volume-guides TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --reflection-only --light-tree --volume-forward TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --light-tree --volume-forward TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background --light-tree --volume-forward TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background --light-tree --volume-forward --denoise-background TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background --light-tree TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --intersect-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --transparent-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --transparent-shadow --shadow-materials TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --reflection-only TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --reflection-only --point-distribution TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --reflection-only --light-tree TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --point-distribution TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --light-tree TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --hair-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --point-distribution --hair-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --defer-shadow --light-tree --hair-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --svm TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --path-update TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --path-update --standalone-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --path-update --reflection-only --standalone-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --wavefront-light --reflection-only TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --path-update --reflection-only TOOLCHAIN_ARGUMENTS
node web/cycles-webgpu/verify.mjs --svm \
  /tmp/blender-cycles-webgpu/svm.wgsl \
  /tmp/blender-cycles-webgpu/svm-reference.json
node web/cycles-webgpu/verify.mjs --svm-compact \
  /tmp/blender-cycles-webgpu/svm-compact.wgsl \
  /tmp/blender-cycles-webgpu/svm-reference.json
node web/cycles-webgpu/verify.mjs --svm-core \
  /tmp/blender-cycles-webgpu/svm-compact-core.wgsl \
  /tmp/blender-cycles-webgpu/svm-reference.json
```

The core shader leaves the extended operations to the full shader. The
`--svm-core` check accepts those lanes as unsupported and compares every lane
the core shader evaluates. The reference records a relative tolerance per lane.
The sky models allow 2 × 10⁻⁵ for the GPU's transcendental functions, and the
other lanes use 3 × 10⁻⁶.

Use a perspective or orthographic camera and the Combined pass with static flat
or smooth triangles. Both projections support depth of field with Cycles' stock
lens sampling. The surface stage uses Cycles’ packed vertex and corner normals,
specular normal correction, and geometric shadow terminator offset. The path loop honors Blender’s total, diffuse,
glossy, and transmission bounce limits independently, including their
defaults, and uses the configured Filter Glossy and indirect clamp. A render
slice can contain up to 2,097,152 pixels, which includes a full 1920 × 1080
frame. For larger offline WebGPU frames, Cycles caps the render tile size at
1024 pixels per side so its adaptive overscan fits inside the slice limit.
Turn off path guiding. The flat distribution uses the saved Light Sampling
Threshold when sampling constant light emission. In a browser build without
a denoiser, a denoising request logs a warning and
returns the noisy Combined pass. CPU Cycles handles scenes outside the
supported tracing subset.

The optional point-distribution shader uses the light distribution synchronized
by Blender's Light Manager and compiles Cycles' CDF and point-light samplers.
It supports multiple normalized, zero-radius point lights with constant
emission and Light Tree and World sampling disabled.
A black world meets the point-light path's visibility requirements. The
single rectangular area light path can use an unsampled equirectangular HDRI.

The optional Light Tree shaders use the stock synchronized tree to sample
normalized point lights, full-spread rectangular area lights, static triangle
meshes with constant or supported textured emission, and an untransformed
equirectangular HDRI. Triangle emitters reuse Cycles' mesh subtrees, triangle
sampler, and PDFs for direct samples and BSDF hits. The sampled-emission
shader uses Cycles' triangle shading setup and SVM to evaluate emission at
the light sample before shadow traversal.
With Light Tree disabled, the mixed-light shaders reuse Blender's flat CDF
for point and Sun lights, area lights, and static triangle emitters. Both
selectors support an HDRI importance map with Auto or Manual World sampling
and use the stock selector PDF for direct and forward MIS. Forward finite
light hits are evaluated nearest first up to the transparent bounce limit.
The light distribution also works with supported volume-only media.
Closed meshes can combine supported GGX transmission with homogeneous volume
closures. A transmitted path updates the medium stack after the surface bounce.
Volume Absorption and Volume Scatter can be combined with Add Shader on the
same boundary.

The compact query SVM runs Ambient Occlusion, Bevel and the five built-in
Raycast outputs against the resident static triangle BVH2. It uses Cycles'
stock ray queries, sampling sequences and triangle normals. Core material
nodes can use it. The optional texture variant also supports Noise, color nodes,
projected image textures, and cubic filtering. A separate shader adds Gabor
and advanced Voronoi to the texture variant. It shares the stock Noise fractal
functions across the output channels to reduce shader size.
Dynamic Raycast attribute outputs, Raycast evaluated through Bump, and Bevel
connected to a closure's Normal input also use CPU Cycles.

The `--subsurface-compact` shader adds Cycles' Random Walk, Skin, Legacy, and
Christensen–Burley methods to the editor's stock wavefront scheduler. It supports opaque
Principled materials and the Subsurface Scattering node with uniform
subsurface controls and three positive radii on static triangle meshes.
The standalone `--subsurface` build can be compared with
the native kernel using
`verify.mjs --subsurface subsurface.wgsl subsurface-reference.json`.

### Cycles denoising

Build the editor with [Open Image Denoise](../README.md#blender-dependencies).
With CPU denoising, the Cycles scheduler runs the stock RT filter on the
Wasm CPU. For tiled offline renders, Cycles assembles the full film before
denoising. Both CPU and WebGPU tracing support Color as the denoiser input.

Wasm SIMD128 uses OIDN's stock convolution blocking for up to four output
channel blocks. Cycles info logs report film assembly, full-frame denoising,
and result write times. The OIDN timings distinguish filter setup from
execution for color and each guide prefilter.

For Albedo or Albedo and Normal inputs on supported WebGPU scenes, build the
guide variants into the same output directory as the other Cycles shaders.

```sh
python3 web/cycles-webgpu/build.py --denoise-surface TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --denoise-surface --standalone-closures TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background --denoise-background TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --shade-background --denoise-background --light-tree TOOLCHAIN_ARGUMENTS
```

Pass that directory to `package_blender.py --cycles-webgpu`. The guide shaders
reuse Cycles' surface denoising feature weights. The GPU film accumulates
albedo and camera-space normal sums alongside Combined. Keep the view
layer's **Denoising Data** export disabled for WebGPU tracing. That option
also requests specular, depth, roughness, and motion passes.

To package the WebGPU denoiser, build its JavaScript bundle into the same
directory as the Cycles WGSL shaders. Run these commands from the project
root containing `web/`.

```sh
npm --prefix web/oidn-webgpu ci --ignore-scripts
node web/oidn-webgpu/build.mjs /path/to/cycles-shaders
```

The packager bundles the runtime with Blender's pinned OIDN model files from
the installation. The browser's factory General scene selects Cycles and
WebGPU when the Cycles shaders are packaged. It also selects **Denoise on
GPU** when the OIDN runtime and models are available. Saved scenes and
preferences retain their explicit selections.

For supported offline renders, Cycles assembles the raw film on WebGPU
and passes GPU buffers to OIDN. The film uses Cycles' pass offsets, sample
counts, and alpha. The filter runs autoexposure, image transforms, and the
FP32 U-Net on the tracing device. Cycles reads back the finished film for
its standard render result and export. Progress updates continue throughout
the render. Supported still-image F12 jobs run off the editor worker and
update Blender's Render Result display from a separate WebGPU preview.
The preview carries scene-linear RGBA to Blender's display shader and
leaves the accumulated film on the tracing device. The U-Net executor
comes from the [pinned OIDN WebGPU project](https://github.com/pissang/oidn-webgpu).

The background job covers Cycles stills, still-file output, and image
sequences. It runs Blender's stock CPU compositor on the finished Render
Result after Cycles exports the denoised film. The saved compositor device
setting is preserved. A compositor that references a scene requiring the
editor GPU context uses the editor worker, as do Freestyle and Grease
Pencil rendering.

The resident path denoises Combined using the stock guide passes. Shadow
catcher, other denoised passes, and custom OIDN weights use the host path.

### Adaptive sampling

Build the following shaders in the same output directory to enable adaptive
sampling on WebGPU. Packages can omit this group when they only need
nonadaptive WebGPU rendering.

```sh
python3 web/cycles-webgpu/build.py --adaptive-camera-slots TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --adaptive-camera-rays TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --adaptive-check TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --adaptive-filter-x TOOLCHAIN_ARGUMENTS
python3 web/cycles-webgpu/build.py --adaptive-filter-y TOOLCHAIN_ARGUMENTS
```

Cycles' render scheduler chooses the convergence checkpoints and noise
threshold. The translated Blender convergence and X/Y filter kernels update
the adaptive auxiliary pass in the GPU film. Each active pixel claims a
contiguous effective sample index before camera ray generation. Camera setup
stores Cycles' sample class for that index. When a path retires, one dispatch
atomically accumulates its Combined and adaptive auxiliary contributions. The
stock Debug Sample Count pass can display the per-pixel sample counts.

Build `--adaptive-film` to compare the retirement shader with the translated
stock film writers using `verify.mjs --film film.wgsl adaptive-film.wgsl`.

The optional reflection-only pair goes in the same output directory.
Cycles compiles that pair when it first sees a non-transmissive wavefront
scene and compiles the full pair when a scene uses transmission. Both pairs
share the same kernels for SVM, traversal, compaction, and film. Packages
with only the full wavefront pair still support both scene types.

Build the optional core SVM shader alongside the full shader to reduce pipeline
compilation time for scenes using the existing material subset. The scene encoder
selects the full shader when a used material needs one of the extended texture or
color operations.

Blender's synchronized SVM bytecode supplies the closure inputs for Diffuse
BSDF and Principled BSDF. For opaque matte Principled, set Specular IOR Level
to zero and leave Diffuse Roughness and the layer weights at zero.

Specular and metallic Principled support both **GGX** and the default
**Multiscatter GGX** distribution. A new Principled material works in a
supported scene with its default socket values. For variations, use an opaque,
isotropic material with non-delta roughness and a constant Metallic, IOR,
Specular IOR Level, and Specular Tint. Base Color may be constant or
image-textured. Leave Diffuse Roughness, transmission, coat, sheen,
emission strength, and thin-film thickness at zero for an opaque
reflective material. Delta reflection and unsupported layers use CPU Cycles.
Multiscatter calls Cycles'
`microfacet_ggx_preserve_energy()` with its directional and average energy
tables for closure evaluation and sampling.

GGX scenes require reflective caustics and glossy light contributions. Keep
the objects’ shadow terminator shading offsets at their defaults. With a
point light, use a black world or turn off both its glossy and diffuse ray
visibility.

For opaque diffuse and Principled materials, the compact renderer also
supports one rectangular area light with full spread and multiple importance
sampling enabled. The world may be constant or a Background node linked
to one untransformed floating-point equirectangular Environment Texture
with Closest or Linear interpolation and constant strength. Set the World
Sampling Method to None. The HDRI contributes when camera and bounced rays
miss, and Cycles’ area-light sampling and emitter hits contribute direct
light with multiple importance sampling.

Principled transmission supports GGX and Multiscatter GGX with isotropic roughness,
a nonmetallic base, constant Transmission Weight, Alpha of one, and Thin Wall and
dispersion turned off. Standalone Glass supports GGX, Multiscatter GGX, and Beckmann.
Standalone Refraction supports GGX and Beckmann. Both standalone nodes accept zero
roughness and linked roughness. IOR is constant and differs from one. The renderer
reuses Cycles' microfacet closures and regular and inverse-IOR energy tables.

Transmission works with a single rectangular area light, the point-light
distribution, and the supported Light Tree lights and world. Enable refractive
caustics and transmission lighting on the lights. A black world satisfies the
point distribution's world restrictions. Shadow caustics and transmission
combined with a scene volume use CPU Cycles.

Each camera path occupies a free GPU slot until it retires into the film.
A compute pass compacts slots for the kernel chosen by the stock scheduler. BVH traversal, surface
setup, SVM evaluation, lighting, and path continuation use those slot IDs.
Continuations reuse Cycles’ path-state transitions, ray visibility, diffuse and
GGX closure math, per-bounce random dimensions, and Russian roulette. Camera
admission computes the pixel RNG seed once per path, as in stock Cycles.
Closest and shadow traversal test the source triangle in its BVH coordinate
space before applying the stock ray offset. Mesh traversal and forward
area-light intersections use the same adjusted continuation origin.
Diffuse continuations use Cycles' concentric disk mapping and compute the
hemisphere height from the disk radius before trigonometric rounding.
Film accumulation uses the raster coordinates stored with each completed path.
Color and Value AOVs use the stock SVM pass offsets and accumulate directly
into the resident film on primary paths.
The optional film shaders write Normal, Diffuse Color, Glossy Color,
Transmission Color, and Ambient Occlusion. They use Cycles' alpha threshold,
closure weights, microfacet albedo tables, and AO sample sequence. AO supports
constant transparent shadows. Diffuse, glossy, and transmission direct and
indirect lighting passes keep the first surface's BSDF weights across bounces
and accumulate into the resident film.
True displacement uses Cycles' existing geometry preparation before uploading
the displaced BVH for WebGPU tracing. Bump-only displacement can trace on
WebGPU when Blender proves its three bump evaluations are identical.
Surface Bump uses Cycles' ray differentials for position, UV, Generated,
and supported packed attributes, including Pointiness. Opaque Diffuse and
isotropic Principled closures can use the resulting normal. Diffuse and
Translucent can also share a Mix Shader with an unperturbed normal.

Image Texture supports Color and Alpha, Flat, Box, Sphere and Tube projection,
Closest, Linear, Cubic and Smart filtering, and all four address modes.
Environment Texture supports equirectangular and mirror ball projection in
surface materials. The host packs RGB texels from four-channel float or byte
images and adds an alpha plane when an output or color unassociation needs it.
The SVM stage applies stock color unassociation and sRGB conversion after
filtering. The extended shader handles projected images and cubic filtering.
Tiled images and texture-driven Principled roughness use CPU Cycles.

For surface materials, Light Path and Light Falloff evaluate the stock SVM
instructions against the live ray and path. Geometry provides Position, Normal,
True Normal, Incoming, Parametric, and Backfacing. Camera Data, Fresnel,
Layer Weight, Object Info and Texture Coordinate read the synchronized
camera, object and path data. Mix Shader and Add Shader support diffuse
Lambert closures sharing the surface normal.
The standalone surface kernels support Translucent, Toon, Glossy, Metallic,
and both Sheen distributions. Microfiber Sheen reads Blender's synchronized
LTC lookup table. Cycles loads the corresponding continuation and denoising
guide kernels when the scene uses those closures. Hair Reflection and
Transmission and Principled Hair Chiang and Huang use Blender's closures on
static triangle meshes. Huang uses the stock projected ellipse for surface
transparency and shadow extinction.
Transparent BSDF traces primary rays and colored shadows with Blender's
transparent bounce limits. A varying tint evaluates SVM at each shadow hit.
Mix and Add Shader can combine Transparent BSDF with Lambert or reflective
Principled GGX. Spatially split BVHs and transparency with AOVs use CPU Cycles.
Ray Portal uses Cycles' closure setup, path transition, and shadow rule.

The optional volume shaders support overlapping bounded media with opaque
surface materials, disjoint point and area lights, and world lighting. They
reuse Cycles' homogeneous integration, phase functions, and segment-conditioned
Light Tree sampling. Static OpenVDB media use named Principled Volume inputs,
NanoVDB voxel data, and stock filtering and null-collision integration. Packed
scene buffers use the adapter's storage binding limit. The path supports up to eight
media. Guided Open Image Denoise uses volume albedo and normal and can retain
the film on WebGPU through export. Build `--volume-reference` to compare
volume coefficients and phase functions with the native reference.

Geometry also provides Tangent, Pointiness and Random Per Island. Attribute,
Color Attribute, Volume Info, Hair Info, Point Info and Particle Info follow
the stock attribute lookup. The host mirrors `find_attribute()` for every
object using a material. It marks an attribute that none of them provide as
`ATTR_STD_NOT_FOUND` in the packed bytecode, so the shader uses the node's
Missing inputs. Other attributes get one of up to 16 packed slots. Each slot
stores a mode and type word and three float4 corners per triangle, converted
as `attribute_data_fetch()` returns them. Byte colors are converted to the
working color space. The shader interpolates the corners like
`triangle_interpolate()` and applies the stock output conversions. Packed
per-triangle attributes need matching per-object values for shared geometry.
From Instancer reads the instance's synchronized UV and Generated values.
Particle Info reads Cycles' dummy particle and needs objects without
particle instancing.

Normal Map supports all spaces, the DirectX convention and both bases. Tangent
space reads the UV tangent and sign slots and a host-built slot of smooth
object-space corner normals. Original base needs geometry without undisplaced
normals. Tangent supports Radial and UV Map. Wireframe uses host-built triangle
positions and tracks Cycles ray differentials for pixel size. Pixel-size
Wireframe with subsurface scattering or sampled mesh emission uses CPU Cycles.
Vector Transform, Wavelength and IES Texture use the stock code. The packed
header carries the film's XYZ to RGB rows, the CIE
color matching table and the synchronized IES table. Sky Texture supports
Preetham, Hosek / Wilkie and both Nishita models in surface materials. The
Nishita lookup table uses the packed image slots.

The Wasm64 queue probe ran the static BVH2 shader in a Chrome worker and
matched the native fixture on all 1,138 rays. It also exercised named
entry points, sparse uniform bindings, partial-byte transfers, resize and
clear, and rejection of waits on the owner.

Broader scene support requires more kernels for lighting, SVM, geometry, and
render passes. Stock scene synchronization and CPU fallback use Blender’s
host `DeviceScene`.

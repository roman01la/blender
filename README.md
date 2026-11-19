# Blender browser port

Put this repository in the `web/` directory and clone
[Blender 5.4.0-alpha at 4831affe](https://projects.blender.org/blender/blender/src/commit/4831affec6ebefd54061c41d60ab26a0189def42)
into the sibling `blender/` directory. Run the commands from the workspace root with
Emscripten 6.0.1 active. They also require CMake 3.24 or newer, Ninja, Git LFS,
and host Python 3.13. The dependency build uses Meson and pkg-config for Rubber Band
and Blender’s pinned Cython for generating NumPy sources. It also uses Cargo
through rustup to build the shader translator with the pinned Rust toolchain.

## WebGPU, Wasm64, and pthreads

```sh
emcmake cmake -S web -B build/web64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m64 -DBLENDER_WEB_SMOKE_PTHREADS=ON
cmake --build build/web64
python3 web/serve.py build/web64
```

Open the loopback URL printed by the server in a browser with Wasm64 and WebGPU
support. The test waits asynchronously for a pthread to start, requests a canvas
surface and device through `GHOST_WebGPUCanvas`, then generates and renders
triangle vertices from the same GPU buffer in one queue submission. The test
also verifies buffer uploads with a partial trailing word, fills, copies,
zero clears, and asynchronous readbacks. Expect a shaded triangle and
“WebGPU commands completed.” The local server sends the cross-origin
isolation headers needed for pthreads. [Emscripten pthreads documentation](https://emscripten.org/docs/porting/pthreads.html)

## Browser windows and input

The staged Emscripten build selects the browser GHOST backend for interactive
sessions. It binds the page’s `<canvas id="gpu">` on the browser main
thread. It uses CSS pixels for cursor events and reports the device pixel
ratio separately so Blender can map coordinates to the framebuffer.

The backend creates a single canvas window and queues pointer, tablet,
keyboard, wheel, focus, and resize events through GHOST. It supports maximize
and restore and runs existing GHOST timers from the frame loop. The browser
provides the window frame, so the canvas starts with Blender's menus and
workspace tabs. Secondary windows, fullscreen, clipboard, and IME integration
are pending.

For WebGPU, call `GHOST_SystemWeb::requestWebGPU()` before requesting a
`GHOST_kDrawingContextTypeWebGPU` window or offscreen context. The readiness
callback lets the device request complete without blocking the browser.
Keep the canvas’s CSS width and height independent of its backing dimensions.
GHOST uses the logical size and device pixel ratio to size its backing
texture, and the browser presents submitted frames when control returns to
the event loop. [WebGPU canvas presentation](https://www.w3.org/TR/webgpu/#canvas-context)

Chrome verified the Wasm64 GHOST context by checking canvas pixels after
resize, device-scale changes, and reopening the window. A hidden canvas
with no drawable suspended presentation until restored. An offscreen
context sharing the device rendered into a texture and read back the
expected pixel.

Interactive editor startup prepares GHOST after parsing the GUI and force
options. It keeps the command-line arguments alive across the device request,
then resumes `WM_init` on the browser event loop. If WebGPU setup fails,
startup reports the error. Background commands skip the interactive device
request. The browser drives `WM_main` through animation frames. A Chrome
Wasm64 probe verified that the startup callback is deferred for a missing
canvas, a newly requested device, and a reused device. It then started the
frame loop and verified canvas pixels.

## Blender dependencies

Put host ISPC 1.30 on `PATH` for the Wasm64 dependency build.

```sh
python3.13 -m venv build/numpy-tools
build/numpy-tools/bin/python -m pip install Cython==3.0.11
emcmake cmake -S web/deps -B build/deps64 -G Ninja \
  -DCMAKE_C_FLAGS=-m64 -DCMAKE_CXX_FLAGS=-m64 \
  "-DBLENDER_WEB_NUMPY_BUILD_PYTHON=$PWD/build/numpy-tools/bin/python"
cmake --build build/deps64
```

The dependency build installs Blender’s pinned CPython, font, compression,
image and color libraries, FFTW, oneTBB, Eigen, Embree, Alembic,
OpenVDB/NanoVDB, Manifold, Jolt, OpenTimelineIO, Haru, Potrace, Ceres, Draco,
meshoptimizer, KTX, libsndfile with Xiph codecs, LAME, and Rubber Band into
`build/deps64/sysroot`. It also builds FFmpeg 8.1 with Wasm SIMD, pthreads,
and the pinned AOM, OpenJPEG, VP8/VP9, x264, x265, Theora, Opus, Vorbis,
and MP3 encoder libraries. Its x265 archive supplies 8-, 10-, and 12-bit
encoding. The geometry and shading packages include OpenSubdiv 3.7.0, GMP
6.3.0, and MaterialX 1.39.4.
It verifies source archives against Blender’s dependency manifest. The CPython
package includes the embedded archive, target headers, standard library, and
generated ABI configuration. CPython and the editor link the same Expat
archive from the Wasm sysroot. NumPy 2.3.4 installs its Python sources, Wasm
extensions, and C API headers in the target Python’s `site-packages` directory.
The checked build uses NumPy's bundled BLAS and LAPACK routines. Use a separate
build and install prefix for each Wasm architecture. The dependency archives
use position-independent code for linking into the editor alongside Python’s
dynamic modules. [NumPy cross-compilation](https://numpy.org/doc/2.3/building/cross_compilation.html) · [Emscripten dynamic linking](https://emscripten.org/docs/compiling/Dynamic-Linking.html)

The Wasm64 dependency build includes Blender's pinned Open Image Denoise
release and RT model weights. The editor picks it up from the dependency
sysroot. The Wasm CPU filter uses OIDN's overlapping tiles with a 512 MiB
memory budget. In a browser package with the Cycles shaders and OIDN runtime,
the factory General scene uses Cycles, WebGPU tracing, and WebGPU OIDN by
default. See [building the Cycles WebGPU denoiser](cycles-webgpu/README.md#cycles-denoising).

For native Wasm exceptions, configure the dependencies in a separate build
directory with `-DBLENDER_WEB_NATIVE_EXCEPTIONS=ON`. In the editor configuration
below, point `sysroot` at the resulting prefix and set
`wasm_flags="-m64 -pthread -msimd128 -fwasm-exceptions -sSUPPORT_LONGJMP=wasm"`.
Use those flags for both C and C++ so they use the same `setjmp` and `longjmp` ABI.
Pass `--native-exceptions` when building the Python child runtime from that
prefix. [Emscripten exception handling](https://emscripten.org/docs/porting/exceptions.html) · [Emscripten setjmp and longjmp](https://emscripten.org/docs/porting/setjmp-longjmp.html)

Configure the editor with `-DBLENDER_WEB_JSPI=ON` to use the browser's native
WebAssembly stack switching with these dependencies. Rebuild CPython from
the current dependency recipe so its Wasm64 function trampoline uses typed
Wasm calls. JSPI requires `WebAssembly.Suspending` and `WebAssembly.promising`.
Emscripten 6.0.1 treats JSPI as experimental. The default editor configuration
uses Asyncify. [Emscripten asynchronous code](https://emscripten.org/docs/porting/asyncify.html)

The image packages provide C and C++ APIs to the editor. The bundled
`OpenEXR`, `PyOpenColorIO`, and `OpenImageIO` Python extensions still need
Wasm module packaging and import verification. OpenTimelineIO has its C++
library installed. Its Python bindings also need browser packaging.

Chrome Wasm64 pthread checks cover Embree ray packets, masks, filter callbacks
and occlusion, Manifold mesh Booleans, and Jolt rigid-body contact and settling.
Jolt uses its CPU backend with Wasm SIMD enabled. Node.js Wasm64 checks cover
Blender’s animated-mesh Alembic fixture and a compressed OpenVDB round trip
with conversion to NanoVDB.

Chrome also passed OpenTimelineIO JSON and compressed bundle round trips,
compressed PDF generation with Haru, Potrace bitmap tracing, Draco and
meshoptimizer round trips, KTX2 Zstandard and UASTC texture round trips,
and a threaded Ceres solve using Eigen's sparse linear algebra.

Blender’s embedded Mantaflow ran on Wasm64 with pthreads in Chrome and Node.
The probe advected a 24×24×24 gas grid, wrote its density to an OpenVDB
cache, then cleared and reloaded it. The restored grid matched the simulated
grid within `1e-5`.

Blender’s bundled Audaspace ran in Chrome with Wasm64 pthreads. The check
decoded and sought through in-memory WAV, FLAC, Vorbis, and Opus files,
then used Rubber Band with FFTW for time stretch and pitch shift. LAME
produced MP3 frames. A muted headless test verified that a browser click
unlocked Audaspace’s OpenAL playback and that Web Audio scheduled stereo
samples.

FFmpeg passed Chrome Wasm64 pthread encode and decode checks for H.264 in
MP4, VP8 and 8-/10-bit VP9 in WebM, 10-/12-bit HEVC and AV1 in Matroska,
and Theora in Ogg. It also decoded AAC from a desktop-generated MP4.
Blender’s bundled Audaspace FFmpeg plugin opened the same MP4 from memory,
sought within its audio track, and read the expected tone.

Chrome also ran Blender’s exact geometry predicates with 64-bit GMP limbs.
OpenSubdiv’s CPU and TBB evaluators agreed on Catmull–Clark subdivision,
and Blender’s evaluator could load the GLSL patch basis source. Chrome verified
WebGPU subdivision on a 10-binding adapter in Edit Mode and Material Preview.
[WebGPU subdivision checks](../profiles/2026-11-18/webgpu-subdivision/README.md)

A MaterialX standard-surface graph survived an XML round trip and generated
GLSL, MSL, OSL, and MDL. MaterialX platform renderers and Python bindings are
pending.

## Stage and configure the editor

The staging command creates a separate checkout at the upstream revision and
reuses the Git LFS objects already available locally. It applies the Emscripten
CMake and frame-loop hooks there and adds the browser window and canvas
implementations to GHOST.

Use a host Node.js executable with Memory64 support for `makesdna`, `makesrna`,
`shader_tool`, `datatoc`, and `msgfmt`. These generators build for the target ABI
and access the checkout and build directory through Node. The Wasm64 check
used Node.js 24.14.0. If the activated SDK puts an older Node on `PATH`, replace
the `node64` assignment below with the path to a compatible Node. [Node.js 24](https://nodejs.org/en/blog/release/v24.0.0) · [Emscripten NODERAWFS](https://emscripten.org/docs/tools_reference/settings_reference.html#noderawfs)

```sh
python3 web/stage_blender.py build/blender-web-source
source_dir="$PWD/build/blender-web-source"
sysroot="$PWD/build/deps64/sysroot"
wasm_flags="-m64 -pthread -msimd128 -fexceptions"
node64="$(command -v node)"
emcmake cmake -S "$source_dir" -B build/blender64 -G Ninja \
  -C "$source_dir/build_files/cmake/config/blender_full.cmake" \
  -DWITH_LIBS_PRECOMPILED=OFF -DWITH_STRICT_BUILD_OPTIONS=ON \
  -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm64 \
  "-DCMAKE_CROSSCOMPILING_EMULATOR=$node64" \
  "-DCMAKE_C_FLAGS=$wasm_flags" "-DCMAKE_CXX_FLAGS=$wasm_flags" \
  "-DPYTHON_ROOT_DIR=$sysroot" \
  "-DCMAKE_PREFIX_PATH=$sysroot" "-DCMAKE_FIND_ROOT_PATH=$sysroot"
```

The strict full-editor configure check finds OpenSubdiv, GMP, and MaterialX
and reaches Cycles and test configuration. It reports missing dependencies
including SpaceNav, USD, OpenShadingLanguage, and OpenXR, along with native
audio device, Metal, and OptiX requirements. It also reports
disabling Cycles path guiding because OpenPGL is missing. Render tests look
for a host `oiiotool`.

A diagnostic build with unavailable dependencies disabled ran `makesdna`,
`makesrna`, `shader_tool`, `datatoc`, and `msgfmt` for Wasm64. It built the core
utility and RNA libraries and generated 50 translation catalogs. Blender’s
filesystem queries use `statvfs` on Emscripten. The OpenSubdiv GPU evaluator
compiles against Blender’s existing GPU buffers with its unused OpenGL
headers removed.

The diagnostic build compiled the Cycles CPU kernel with Wasm SIMD and used
the Emscripten configuration for Blender’s bundled glog. Node verified glog
output from the main thread and a pthread. The staged editor build finds
the Wasm NumPy headers and builds Audaspace’s Python module. The editor
link retains the symbols NumPy imports so embedded CPython can load its
extensions from the packaged filesystem. Dropping the desktop linker’s broad
Python export flag reduced the diagnostic Wasm export table from about 39 MiB
to about 15 KiB before optimization. A Wasm64 pthread probe in Chrome
and Node verified the CPython ABI, NumPy arrays, FFTs, linear solves, random
generation, and `.npy` serialization. It also loaded the full Audaspace Python
module, wrote a strided NumPy array to an animated audio property, and checked
the `float32` frames read back. The editor and thumbnailer install rules copy
their `.wasm` binaries beside the JavaScript launchers in portable and system
layouts.

The Wasm expression evaluator tracks math-domain and division-by-zero errors
during evaluation and constant folding. It preserves errors from intermediate
operations even when a later operation returns a finite value. Native builds
use the floating-point environment. All 154 expression tests passed in the
native test executable, Node, and Chrome. [WebAssembly floating-point operations](https://www.w3.org/TR/wasm-core/#floating-point-operations)

## Blender GPU backend

Blender selects its WebGPU backend after GHOST has prepared the device. Window
and offscreen GPU contexts use that device and report its negotiated limits.
The backend implements storage and uniform buffer uploads and clears. Storage
readback schedules a copy into a mapped staging buffer and requires the browser
event loop to complete the mapping. Viewport picking and gizmo occlusion
queries suspend their current Wasm call stack until mapping completes.
Blender starts the next frame after the suspended frame finishes. The canvas
backbuffer can clear a WebGPU color attachment and submit the commands at the
end of the frame.

Material thumbnail jobs start on the editor worker between draws because it
owns the WebGPU device. EEVEE reads the rendered color texture through a
staging buffer and resumes the preview job once the buffer maps.
Chrome verified that the Geometry Nodes Set Material picker generates thumbnails,
assigns the selected material, and keeps the editor responsive after selection.
The WebGPU volume shader uses base mip sampling when evaluating material nodes
inside the per-froxel loop, where screen-space derivatives can be undefined.

Chrome verified the actual Blender GPU API by selecting WebGPU, creating a
context, uploading and reading back a storage buffer, and checking a canvas
pixel after a color clear. The full editor also passed a viewport test that
selected, deselected, reselected, and moved the default cube.

Edit Mode selection uses separate triangle and line shaders and point-sprite
variants. Integer selection IDs are read from offscreen color attachments via
WebGPU staging buffers. The edit-face overlay uses a clip-space depth bias to
prevent coplanar flickering against the mesh. Workbench packs temporal
antialiasing weights into four-component slots so the GLSL and WGSL passes use
the same uniform layout. A Chrome editor test selected vertices, edges, and
faces individually and selected multiple faces with a box drag.
The editor compiles four shaders for default object picking and the selection
grid and background variants during startup. Overlay shaders for other
geometry compile when that geometry first contributes to a draw. The browser
keeps the loading indicator visible and the canvas input disabled until WebGPU
completes the first workspace frame.

In a fresh Chrome session, the first default-cube pick took 95 ms from pointer
down to GPU readback, down from 624 ms in the previous build. The click created
zero shader modules. A repeat pick took 73 ms. Chrome also verified that input
stays blocked while the browser waits for the first workspace frame.

The backend sends Blender’s generated GLSL through its pinned glslang revision
to SPIR-V, then translates it to WGSL with Naga 30.0.1. The browser loads Naga
as a separate Wasm32 module during startup. Blender’s own executable remains
Wasm64. The WebGPU GLSL generator assigns separate texture and sampler
bindings and lowers push constants into a uniform block.

Chrome compiled the actual vertex and fragment stages of `gpu_shader_text`,
rendered a glyph with them in a WebGPU render pipeline, and verified the canvas
pixels. [Naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)

The backend allocates vertex and index buffers on first GPU use. It supports
CPU uploads, device-only geometry, compressed 16-bit indices, index subranges,
and asynchronous readback. Aligned transfers use WebGPU buffer commands.
Transfers that touch only part of a four-byte word use a compute pass to keep
the other bytes intact. [WebGPU buffer copy commands](https://www.w3.org/TR/webgpu/#buffer-copy-commands)

A Blender GPU API probe in Chrome verified device-only vertex resizing,
vertex-to-storage copies, unaligned writes and copies into a 16-bit index
subrange, asynchronous readback, and indexed rendering.

Blender's batch API now records direct draws to the canvas. The backend
reuses render passes and caches pipelines by vertex layout and draw state.
Draw uniforms use dynamic offsets into reusable buffer pages, so changes to
shader uniforms between draws preserve the values used by each draw. Chrome
verified two indexed draws with different colors in one pass using Blender's
`gpu_shader_3D_uniform_color` and a 16-bit index subrange.

The backend allocates 2D, array, cube, and 3D textures for supported formats and
uploads full images or strided subregions. Draws bind sampled textures and
read-only storage buffers. The backend caches sampler objects and reuses bind
groups when their resources stay the same. Texture updates follow previously
recorded draws on the GPU queue. [WebGPU texture uploads](https://www.w3.org/TR/webgpu/#dom-gpuqueue-writetexture)

Chrome verified two differently colored strings from Blender's bundled Inter
font via `BLF_draw`. This exercises the glyph atlas uploads, glyph storage
buffer, and `gpu_shader_text` together.

Blender can render to offscreen color and depth/stencil attachments, then
composite the offscreen color into the canvas. Immediate-mode draws use
reusable vertex upload pages. The backend expands triangle fans and line loops
to WebGPU primitive topologies and reuses Blender's polyline expansion for
wide lines. [WebGPU render passes](https://www.w3.org/TR/webgpu/#render-pass)

Chrome verified immediate triangles and a triangle fan through
`GPUOffScreen`, including depth rejection, stencil masking, and the orientation
of the region composited into the canvas. The widget and polyline shader changes
remove the WGSL validation errors previously reported during startup.

The WebGPU shader wrapper emulates programmable point size and point
coordinates with instanced four-vertex triangle strips. Vertex buffers
provide one set of attributes per point and the shader applies the active
viewport scale. Blender's keyframe shader supplies the shape, outline,
and color. Chrome verified two different marker sizes and a nonzero batch
draw offset through the immediate and batch APIs. The direct canvas and
composited offscreen renders matched pixel-for-pixel. Indexed point draws still
require index-based attribute fetching. [WebGPU vertex step modes](https://www.w3.org/TR/webgpu/#enumdef-gpuvertexstepmode)

Blender's 1D textures and arrays use one-row WebGPU textures. The translator
lowers texel fetches, sampling, and dimension queries to that layout while
keeping Blender's logical array layer and mip coordinates. Chrome verified
uploads, a per-layer update followed by a GPU copy, filtered sampling at two
mip levels, and a plain 1D fetch. Sampling offsets and explicit gradients still
require lowering for this layout. [WGSL texture operations](https://www.w3.org/TR/WGSL/#texture-builtin-functions)

CPU uploads to depth-only textures use a depth pass with an R32Float
staging texture. The pass writes the requested mip, layers, and rectangle.
A one-texel mip uses a depth clear. Chrome verified a strided Depth32Float
upload to one array layer and a Depth16Unorm upload to mip level 1. Depth
testing still saw the earlier values outside the rectangle and in the other
layer. [WebGPU depth attachments](https://www.w3.org/TR/webgpu/#depth-stencil-attachment)

Draws can bind Blender uniform buffers alongside per-draw constants. The
translator preserves depth texture types separately from color texture types
that GLSL represents the same way. Chrome verified non-filtering depth
sampling and texel fetches, including a depth-only view of a packed
depth/stencil texture, alongside a color texture and a uniform buffer.
[WGSL depth textures](https://www.w3.org/TR/WGSL/#sampled-texture-types)

The backend allocates unsigned 8-, 16-, and 32-bit textures and reads 2D and
3D unsigned textures in shaders. Integer bindings use samplerless texel
fetches. Normalized sampling supports Blender's default nearest,
clamp-to-edge behavior at mip level 0. Chrome verified a 3D R8Uint subregion
upload, a clamped volume lookup through a shader helper, and exact 16- and
32-bit values. [WGSL texture loads](https://www.w3.org/TR/WGSL/#textureload)

Color textures and samplers stay paired through GLSL compilation so a shader
can pass them through helpers. The translator splits bindings and arguments
into WebGPU handles before emitting WGSL. It also lowers scalar and vector
signed modulo to integer arithmetic, including operands outside the exact
range of a 32-bit float. Unsigned texture LOD lookups use the same mip-0
sampling as the default unsigned sampler.
[SPIR-V image access](https://github.com/KhronosGroup/SPIRV-Guide/blob/main/chapters/image_accesses.md)

WebGPU records direct and indirect compute dispatches with writable storage
buffers, and reads GPU-generated indexed and non-indexed draw commands directly
from those buffers. Multi-draw records each command in the same render pass.
Chrome verified compute-generated dispatch and draw arguments, a nonzero
first instance, indexed base vertex, and the expected colors from the resulting
draws with no validation errors.
Native indirect draws with nonzero first-instance values use the adapter's
`indirect-first-instance` feature.
[WebGPU indirect draws](https://www.w3.org/TR/webgpu/#dom-gpurendercommandsmixin-drawindirect)

Compute shaders bind single-mip storage views of 2D, array, and 3D textures
in supported formats. Read and read/write access requires the corresponding
WGSL language feature. Chrome verified exact R32Float values from Blender
compute dispatches in 2D textures, two array layers, and two volume slices.
[WGSL storage texture access](https://www.w3.org/TR/WGSL/#storage-texture-types)

Material Preview uses EEVEE's deferred renderer. The WebGPU backend maps
packed color and normal targets to compatible texture formats, backs atomic
textures with storage buffers, and orders shadow light data copies on the GPU.
EEVEE light culling requires 1,024 invocations and 32 KiB of storage in one
workgroup. The backend requests the adapter's available limits. Chrome
verified that the default cube renders, updates to a newly assigned Principled
BSDF material, and continues drawing without WebGPU validation errors.

Browser startup prepares EEVEE's viewport shaders, the default mesh and studio
world materials, the Shading workspace reference spheres, and the Material
Preview OpenColorIO display transform. The loading indicator stays visible
through shader preparation and the first workspace frame. In Chrome, the first
Material Preview switch reached a GPU-completed frame in 344 ms, down from
2.7 seconds. The longest animation frame gap fell from 2.2 seconds to 100 ms.
The switch created five shader modules, down from 84.
Chrome also opened the Shading workspace, rendered its diffuse and metallic
reference spheres, and switched back to Layout without WebGPU errors.
The WebGPU backend makes EEVEE's surface prepass store the rasterized fragment
depth explicitly. In Chrome, the fixed-function prepass write differed from
the material pass by one representable float at some pixels. Writing the
fragment depth makes the passes agree and lets EEVEE use its original exact
depth comparison for opaque materials.

Blender's framebuffer API can disable sRGB conversion when drawing a node
editor. The WebGPU backend attaches a linear view of the sRGB texture for
those draws. Normal sampling still decodes the stored display-encoded colors
for compositing.

The translator splits storage-buffer struct reads into member loads in shaders
with atomics. Blender's draw-command generator can use `atomicLoad` for draw
counters while retaining the original storage layout.

Signed 10:10:10:2 vertex attributes stay packed in Blender's vertex buffers.
The backend fetches one 32-bit word and caches a vertex shader variant that
sign-extends and normalizes the components. Unsigned 10:10:10:2 attributes
use WebGPU's native format. Missing float and integer attributes use one
zero-stride buffer and WebGPU's component defaults. Chrome verified that
packed and ordinary draws using the same shader produce the expected values.
[WebGPU vertex formats](https://www.w3.org/TR/webgpu/#vertex-formats)

Named shader interfaces keep the original member access in GLSL and use
individual locations across the WGSL vertex-fragment boundary. Chrome
verified a named interface with flat, linear, and perspective members. The
backend implements Blender's single-source blend equations. Chrome checked
premultiplied additive and reverse-subtract blending with known pixel values.
Chrome also verified separate RGB and alpha revealage through dual-source
blending. This path requires the adapter's `dual-source-blending` feature.

Blender boolean, signed and unsigned integer, and float shader specialization
values select cached render and compute pipelines with WGSL overrides.
Chrome verified switching between three render variants and using two compute
variants in the same frame, including a float override passed as exact bits.
The mesh wireframe shader's negated condition now survives translation.
[WebGPU pipeline constants](https://www.w3.org/TR/webgpu/#dom-gpuprogrammablestage-constants)

Depth gather preserves the four ordered texels returned by WebGPU. Chrome
verified the gather in a vertex shader and a fragment shader alongside
ordinary depth sampling. [WGSL texture gather](https://www.w3.org/TR/WGSL/#texturegather)

Blender texel buffers read the existing vertex allocation as read-only storage
with a small format descriptor. Shader helpers can forward a texel-buffer
handle. Chrome verified float, signed, unsigned, and normalized inputs,
including exact unsigned values written by a compute pass and read by a draw
in the same frame. Texel views also accept tightly packed, equally sized
attributes laid out as consecutive texels, as used by Grease Pencil.

Bind-group layouts use the resources each translated shader entry point
actually accesses. Blender's geometry expansion binds its vertex and index
storage streams through the shared batch API. This lets the cursor's polyline
pass render with the rest of the stock workspace.

The browser selects WebGPU directly even when the startup preferences specify
a desktop backend. Rejected viewport grid lines send both endpoints outside the
same clip plane. This restores the floor grid and axes in the stock workspace.
The translator preserves explicit switch cases that share the default block,
so both triangles of Blender's widget backgrounds render. Chrome verified the
stock workspace, selected the camera in the Outliner, and used `G`, `X`, `1`,
and `Enter` to translate the default cube one meter.

Weight Paint computes contour gradients before branching on the interpolated
missing-group flag and samples the single-mip color ramp with an explicit level.
Chrome displayed the contours on a cube with two assigned vertex weights and
rendered the brush thumbnails in the asset shelf.
[WGSL derivative uniformity](https://www.w3.org/TR/WGSL/#derivative-uniformity)

WebGPU generates color mipmaps on the GPU in the existing command encoder.
Each pass reads the previous level and writes the next level of one layer,
with a pipeline cached per format. The filter covers the full source footprint
for odd sizes and averages sRGB textures in linear space. Chrome checked
readback pixels for even, odd, single-column, and sRGB mipmaps.
Partial updates currently regenerate the chain for the whole texture. 3D,
depth, and integer mipmap generation remain pending.
[WebGPU texture subresources](https://www.w3.org/TR/webgpu/#texture-subresource)

Indexed point sprites gather attributes into instance streams on the GPU.
The gather uses Blender's 16-bit or 32-bit indices, subrange offset, and base
index. It preserves the original vertex ID for the point shader. Chrome checked
both index widths and a nonzero base and drew vertex markers in Mesh Edit Mode.
Indirect point draws expand GPU-generated commands in one compute pass. The
vertex shader fetches directly from the original buffers and preserves vertex
and instance IDs. Chrome verified strided multidraw, both index widths,
positive and negative base vertices, and multiple instances.
[WebGPU vertex buffer layouts](https://www.w3.org/TR/webgpu/#dictdef-gpuvertexbufferlayout)

Mesh edit edges clip in homogeneous space before expansion. Armature endpoints
use indexed triangles, and rejected wire and handle primitives stay outside the
clip volume. Point clouds and Gaussian splats use repeated endpoints to break
procedural triangle strips. Chrome drew moved mesh edges, octahedral bones and
custom bone outlines, legacy Bézier handles, an animated motion path, and
disconnected point-cloud and Gaussian splat selection outlines. Hair Curves
generates native strip restart indices on the GPU. On devices with fewer than
11 storage bindings per compute stage, the interpolation pass packs curve
metadata to fit in eight bindings. Chrome on an eight-binding device rendered
strands, ribbons, cylinders, and all four spline types with zero rendering
validation diagnostics.

The Grease Pencil selection overlay expands the line strip into independent
segments and checks both endpoints. Its depth merge and anti-aliasing shaders
compile through the storage texture and dual-source blending paths.
Chrome drew a Grease Pencil stroke in Draw Mode with the asset shelf hidden.
Brush previews in the open asset shelf still need a browser test with the
updated preview scheduler.

Scissored framebuffer clears clip color, depth, and stencil writes to the
requested rectangle. Explicit load actions apply once, and discard takes
effect when the framebuffer scope ends. Chrome verified color, depth, and
stencil with a pixel test.
[WebGPU render pass attachments](https://www.w3.org/TR/webgpu/#render-pass-encoder-creation)

CPU uploads to packed depth/stencil textures, depth blits, depth and canvas
framebuffer readback, and general texture views remain pending.

## Package the browser runtime

After building the staged editor, install it and package the installed files
with Emscripten active.

```sh
cmake --install build/blender64 --prefix "$PWD/build/blender-install"
python3 web/package_blender.py build/blender-install build/blender-browser
python3 web/serve.py build/blender-browser
```

On macOS and Linux, the development launcher allows one listening server per
project checkout. If it is already running, the launcher reports its URL.
The page transfers its canvas to a dedicated editor worker and forwards input
and display changes to it. The worker loads Wasm and Blender's assets, owns the
WebGPU device, and starts additional pthreads for Blender's tasks. The page
keeps the loading indicator visible until the worker completes the first
workspace frame. Input and display changes wake the worker for the next
animation frame. When drawing stops, Blender wakes for its window and Python
application timers, with a periodic check for background notifications.

The packer reads either Blender’s portable layout or its system install layout.
It copies the editor launcher, Wasm module, and shader translator, and mounts the
versioned resource directory and any installed glTF bridge modules in the virtual filesystem
before `main()`. This keeps Python modules, scripts, and data at the paths
Blender’s portable resource lookup expects.
Chrome verified both layouts with a two-chunk Wasm64 pthread probe and checked
that the page reports a missing chunk. Chrome also ran the installed diagnostic
Blender build in background mode. It imported `bpy`, NumPy 2.3.4, and `pyexpat`,
ran a NumPy array calculation, and checked that all three glTF bridge files
were mounted. With `--threads 4`, it also evaluated a subdivision modifier to
96 faces, saved a `.blend` to the virtual filesystem, and reopened it. The
browser launcher prewarms workers for the reported hardware concurrency plus
a reserve for service tasks and OpenImageIO. Browser startup caps OpenImageIO
at two to four threads, loads recent-file data synchronously from the virtual
filesystem, and skips the native file-limit adjustment. Chrome reached the
stock workspace with no WGSL or pthread diagnostics.
[Emscripten pthreads](https://emscripten.org/docs/porting/pthreads.html)

Build the Python child runtime with the same Wasm64 dependency sysroot as
Blender, then pass its output directory to the browser packager.

```sh
python3 web/python-process/build.py build/web-deps/sysroot build/python-process
python3 web/package_blender.py build/blender-install build/blender-browser \
  --python-process build/python-process
```

The bundled runtime supplies the `spawn` start method for
`multiprocessing.Process` with picklable module-level targets. Each child
runs in its own CPython interpreter and virtual filesystem in a Web Worker.
The parent copies Python arguments and forwards child output to the browser
console. It supports process status, joins with timeouts, termination, and
exit codes. Chrome verified these operations from the full editor's Python
Console and in background mode with two concurrent children, independent
filesystems, and distinct exit codes.

`Queue`, `SimpleQueue`, and `Pipe` exchange pickled messages through a browser
broker. A pipe can also send raw bytes. Queues and pipe endpoints can be
passed to spawned children, and the broker releases their leases when a
worker exits. `Pool` reuses worker processes for `apply`, `map`, `starmap`,
and iterator tasks. Worker messages and exits wake the browser-thread result
dispatcher. Completion and error callbacks run when the dispatcher can enter
Python, including while the editor is idle. Chrome verified callback delivery
without periodic result reads or process status checks, including worker
recycling, errors, termination, and submission from inside a callback.

`multiprocessing.Manager()` starts a server in a dedicated Python Worker.
Built-in proxies share containers, queues, and synchronization objects across
workers. Manager-hosted pools, custom registration, and socket-based managers
remain pending. Referents live for the duration of the manager.

To include `SharedMemory`, `ShareableList`, and `SharedMemoryManager`, build
the optional shared-memory worker profile with position-independent CPython,
zlib, bzip2, and Expat archives from a clean build of `web/deps`.

```sh
python3 web/python-process/build.py build/web-deps/sysroot build/python-process \
  --shared-memory-sysroot build/web-deps/sysroot
```

The profile relocates each interpreter into a private 64 MiB arena in
Blender's Wasm memory. Shared segments expose ordinary writable Python
memoryviews, and remain valid until their last lease closes after unlink.
Chrome verified live writes in both directions through already-open views,
shared lists, and managed name cleanup from Blender's Python Console. A separate
Chrome test also ran four concurrent Python pthreads on one shared segment.
The browser session owns the name registry. A worker that is stopped before
it reports a clean exit keeps its arena and mapping leases until the browser
session ends, since termination can return before the worker stops writing.

OS file descriptors, process sentinels, and Blender's `bpy` API inside
children remain pending.

## Deploy to blender.rocks

This repository contains the browser port. Keep the upstream Blender checkout
as the sibling `blender/` directory at `4831affec6ebefd54061c41d60ab26a0189def42`.
The favicon is Blender’s transparent 48-pixel Windows app icon from that
checkout.

From the web repository, deploy a verified browser package with Node.js, Python
3.11 or later, and an authenticated Wrangler account. The following commands
assume the dedicated R2 bucket already exists. To provision it once, run
`npx wrangler@4.160.0 r2 bucket create blender-rocks-releases`.

```sh
npx wrangler@4.160.0 whoami
python3 cloudflare/deploy.py /path/to/blender-browser
```

Push the source commit to the public GitHub repository configured as `origin`
before deploying. The deployed page links to that exact commit. The deployer
uploads the Wasm and resource bundle to content-addressed R2 keys. Smaller
runtime files go to Workers Static Assets. Both paths send the isolation
headers required by Blender’s Wasm threads.

Use `--stage-only` to generate `cloudflare/dist` for
`npx wrangler@4.160.0 deploy --dry-run --config cloudflare/wrangler.jsonc`.

## License

This port of Blender is distributed under the [GNU GPL v3 or later](LICENSE).
Individual source files and bundled dependencies retain their stated licenses.

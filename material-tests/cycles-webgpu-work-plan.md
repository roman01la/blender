# Cycles WebGPU work plan

Use the [source coverage ledger](source_coverage.json) to choose backend work and the [material-test suite](README.md) and saved scenes to verify it. Start with the limited `NODE_CLOSURE_BSDF`, `NODE_MIX_CLOSURE`, and `CLOSURE_BSDF_PRINCIPLED_ID` entries. The common gap is evaluating several closures at one hit, each with its own weight, normal, and parameters.

## Implementation order

1. **Two diffuse-only Principled branches.** Start with a minimal weighted mix. Reuse Blender’s stock SVM closure setup and surface selection and evaluation from `intern/cycles/kernel/svm/closure.h` and `intern/cycles/kernel/integrator/surface_shader.h` in the pinned Blender checkout. Keep the WebGPU storage adapter generic so it can handle additional stock closure types.
2. **Surface shader queries in scenes with volume transport.** Continue from the verified AO and Raycast cases with separate volume objects. Check the original surface hit, path RNG, and BVH visibility after a boundary or scatter. Extend to Bevel and the existing differential and transparency routes.
3. **Diffuse and Glossy mixtures with more than two closures.** Preserve closure order, weighted sampling, and the normal of each lobe through direct lighting and continuation rays. Recheck the existing independently bumped pair before extending to nested mixes such as Classroom’s pencil.
4. **Diffuse, Translucent, and Transparent composition.** Extend ordered selection and mixture PDFs to Classroom’s paper material. Keep the shared Bump and the stock Translucent normal correction. Exercise linked alpha on camera and shadow rays.
5. **Shader-wide displacement with mixed reflective lobes.** Preserve the displacement normal and the shared closure normal for Classroom’s Diffuse and Beckmann plastic. Reuse the two-normal storage used for standalone displacement. Cover independent closure normals with ordered records when both branches supply their own normal.
6. **Transparent surfaces with emission.** Handle Classroom’s backface-dependent window shader using the stock shadow and sampled-emission paths. Compare per-hit extinction, visible emission, film alpha, and sampled triangle lighting before enabling the combination in saved scenes.
7. **Classroom analytic lights and Ambient Occlusion.** Preserve lamp-level Light Falloff, shadow and MIS settings. Run the saved Ambient Occlusion pass through the same varying shadow traversal as direct lighting.
8. **Remaining saved-scene closure mixtures.** Cover Principled plus Translucent, two independent opaque Principled branches with tangent-space normals, and camera-dependent sharp and rough Glossy mixed with Transparent. Use the compiled SVM program to identify live branches before extending the backend.

## Shared closure contract

Stock `svm_node_mix_closure` saturates the factor and multiplies its two branch weights by the enclosing mix weight. Each closure retains its own color weight and sampling weight. `surface_shader_bsdf_bssrdf_pick` selects a lobe in allocation order using the weighted random sample and remaps that sample within the chosen lobe. `_surface_shader_bsdf_eval_mis` evaluates the eligible lobes, accumulates their weighted values, and combines their PDFs using the sampling weights. Reuse the same closure set for lighting, continuation, film passes, and denoising guides. Collapsing identical diffuse lobes requires equivalent normal, roughness, cutoff, and sampling behavior.

| Saved scene | Dependency |
| --- | --- |
| Blender 3.3 Splash | `Material.062` uses Ambient Occlusion Color through a Color Ramp to mix two diffuse-only Principled branches. Alpha is opaque, the specular and other layers have zero weight, and each branch has its own color input. The weighted closure mix and volume plus query admission are separate gates. |
| Classroom | The `pencil` graph exercises nested Diffuse and Glossy mixes with Bump. It requires independent closure normals and closure ordering across more than two lobes. The independently bumped pair is a narrower regression check. |
| Monster Under the Bed | `Wood plinth` requires two full Principled closure sets. The existing Normal Map decoder resolves tangent, sign, and smooth-normal attributes for eligible surfaces. Verify that this per-corner basis, the normal transform and backface rules, and each branch’s resulting normal survive closure packing. |
| Sky Texture Demo | `Material` combines sharp GGX Glossy and rough Beckmann Glossy, with Fresnel driving glossy color. Light Path Is Camera Ray chooses between the network and Transparent. Stock SVM folds the sharp Beckmann Refraction branch out because its Mix Shader factor is zero. |

## Diffuse and Glossy closure records

The two-closure path now captures a Bump normal when SVM evaluates its closure. This preserves both normals through stack reuse. The fixed material record carries the Diffuse normal in its normal slot and the Glossy normal in its auxiliary slot. Direct lighting, continuation, film passes, and denoising guides each read the normal for their own lobe. A separate compatibility marker protects this packing. Shader-wide displacement uses the auxiliary slot and remains outside this combination. [SVM capture](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Shader selection](../blender-platform/cycles-webgpu/pipeline.cpp)

All ten [focused fixtures](check_mixed_glossy.py) completed 128 samples on WebGPU. Independent Bump pairs in both closure orders matched desktop CPU Cycles within 0.000101 normalized RGB RMSE for GGX and Beckmann. The signed Normal-pass output matched within 0.000087. An existing curved Beckmann scene improved from 0.090019 to 0.000076 RMSE against the same desktop reference. With OIDN enabled, the scene traced and denoised on WebGPU using the albedo and normal guides, with 0.000096 RMSE against CPU Cycles and CPU OIDN.

The ordered path stores up to four Lambert Diffuse and isotropic, nonsingular GGX or Beckmann Glossy closures. SVM captures each lobe in allocation order after applying its weight cutoff. Classroom’s pencil has two Lambert lobes and two GGX lobes with distinct roughnesses of 0.173205 and 0.447214, all sharing a Bump normal. The film and denoising guides use the same closure list as lighting and path continuation. The list sits after the AOV records in the existing material buffer, and its size counts toward the device’s path-pool capacity. [Material layout](../blender-platform/cycles-webgpu/material_layout.h) · [Stock BSDF calls](../cycles-webgpu/ggx_kernels.cc)

All eleven positive `--ordered --guided` fixtures completed 128 samples on WebGPU and matched desktop CPU Cycles within 0.000101 normalized RGB RMSE. They cover both allocation orders, distinct GGX roughnesses, mixed GGX and Beckmann, multiple Bump normals, and an explicit geometric normal. The fixtures compare the signed Normal pass and verify all three guided OIDN models run on WebGPU. The denoised comparison measured 0.000094 RMSE against desktop CPU Cycles and OIDN. Both deliberately unsupported controls fell back to CPU. [Ordered closure fixtures](check_mixed_glossy.py)

The four-lobe path currently accepts Lambert and isotropic reflection only. Rough Diffuse, anisotropy, singular Glossy, and combinations with other closure types still need separate coverage, as do shader displacement and AOV writes in the ordered graph. A separate 96-sample check rendered an ordered material beside a material writing color and value AOVs. Combined, color AOV, and value AOV all ran on WebGPU and stayed within 0.000087 normalized RGB RMSE, confirming that both AOV offsets still work with the new material records.

## Glossy with displacement and local Bump

Stock SVM writes the displacement bump normal to `ShaderData::N` before evaluating the surface closures. Standalone isotropic GGX and Beckmann reuse the two-normal material record already used by Glass. Light sampling, the MIS origin, and the Normal and AO fallbacks use the displaced shader normal. The BSDF and denoising guides use the closure-local normal with stock specular reflection correction. [Stock SVM normal write](../../blender/intern/cycles/kernel/svm/closure.h) · [Material evaluation](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Film](../cycles-webgpu/ggx_kernels.cc)

The [`--displacement-bump` fixtures](check_mixed_glossy.py) cover independent procedural heights, the shared Bump link used by Classroom, an encoded signed Normal pass, and guided OIDN. On the previous build all seven combined-Bump cases fell back at SVM `NODE_END`. Four single-Bump or flat controls stayed on WebGPU. With the fix, all eleven cases completed 128 samples on WebGPU within 0.000101 normalized RGB RMSE against desktop Cycles. Guided WebGPU OIDN matched CPU OIDN within 0.000080 RMSE.

The isolated original `dustbin_metal` material with its packed image completed 192 samples on WebGPU. It used an area light and a world background with the Light Tree disabled and matched desktop Cycles within 0.000023 normalized RGB RMSE. The displaced forward area-light MIS guard applies when the Light Tree samples an area emitter. The AO visibility ray still uses a mesh-normal offset.

## Shared Bump and displacement in Diffuse and Glossy

The Lambert and nonsingular isotropic GGX or Beckmann pair reuses the two-normal record when both lobes reference the same local Bump. The shader-wide displacement normal supplies light selection, MIS, and the Normal and AO fallbacks. The shared local input feeds the stock per-lobe setup, including the Glossy reflection correction. The existing continuation and denoising stages already read both normals. The SVM encoder and film shader each advertise support for this combination.

The [mixed displacement fixtures](check_mixed_glossy.py) passed all 16 supported cases at 128 samples on WebGPU. They cover both closure orders, GGX and Beckmann, a 0.06 mix, linked weight, both folded endpoints, and independent displacement height with a shared local normal. The signed Normal and Glossy Color differences were at most 0.000094 and 0.000036 normalized RGB RMSE. Guided WebGPU OIDN matched CPU OIDN at 0.0000792. The guard for independent local normals still falls back.

An isolated 473 × 67 crop of the original teacher desk uses the saved mesh, UVs, material, camera, world, Sun, corridor Point, and blackboard emissive mesh. The unsupported window emitters, Light Falloff lamps, and shadowless exterior Area were excluded on both devices. It completed 192 samples on WebGPU at 0.0000356 normalized RGB RMSE against desktop Cycles.

The pair needs two unfiltered closure slots and no volume boundary on the same material. When the mix reduces to Lambert, its single closure uses the same two normal records. Independent local normals with shader-wide displacement still require a third normal. [SVM admission](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Film](../cycles-webgpu/ggx_kernels.cc)

## External RGBA16 and Glossy mixed with Transparent

The material encoder admits stock `IMAGE_DATA_TYPE_USHORT4` and normalizes each channel before packing the same float texture records as existing images. The GPU uses the existing SVM texture fetch, filtering, and color conversion. The image-only fixtures verify that RGB and alpha retain values between 8-bit levels. All five cases completed 128 samples on WebGPU within 0.000087 normalized RGB RMSE against desktop Cycles. [Stock texture read](../../blender/intern/cycles/kernel/util/image.h) · [Material texture packing](../blender-platform/cycles-webgpu/svm_materials.h) · [Comparison fixtures](check_glossy_transparency.py)

A single nonsingular isotropic GGX or Beckmann lobe can share its record with an explicit Transparent closure. Continuation calls the stock closure picker in allocation order and uses the existing transparent transition to retain the previous MIS state. Direct-light MIS includes the Transparent sample weight in the mixture PDF. Shadow, film, and denoising stages use the stored extinction. The initial pair covers constant or linked weights. Anisotropy, local Bump, shader-wide displacement, same-shader media, and additional lobes need separate coverage. [Stock closure picker](../../blender/intern/cycles/kernel/integrator/surface_shader.h) · [Continuation](../cycles-webgpu/camera_kernels.cc) · [SVM](../blender-platform/cycles-webgpu/svm_material_eval.h)

All 14 focused fixtures completed 128 samples on WebGPU. The undenoised comparisons, including the Normal pass and tinted shadows, matched desktop Cycles within 0.000094 normalized RGB RMSE. Film alpha matched within 0.000123. Guided WebGPU OIDN matched CPU OIDN at 0.000150 RGB RMSE with all three model stages on WebGPU. All 16 existing [Lambert and Principled transparency comparisons](check_mixed_transparency.py) also passed on the new build.

The original `dustBin_wireframe` graph with its unchanged external PNG completed 192 samples on WebGPU. The area-lit fixture matched desktop CPU Cycles at 0.0000253 normalized RGB RMSE and 0.0000103 alpha RMSE. A second fixture with a diffuse shadow receiver matched at 0.0000346 RGB RMSE. The image-only build advanced the same external graph from the image guard to the closure gate, isolating the two fixes.

A separate emissive-mesh probe with the same texture triggered a Chrome GPU-process exit during pipeline compilation. The previous build reproduced the exit with the texture packed as FLOAT4. This remains a separate texture-emission pipeline issue.

## Diffuse, Translucent, and Transparent composition

The existing ordered closure records can carry Lambert Diffuse, Translucent, and one explicit Transparent closure. Each Diffuse and Translucent lobe keeps its normal input, and Translucent applies the stock specular reflection normal correction. The existing auxiliary record carries the accumulated Transparent extinction and identifies live Translucent lobes for light sampling. [Stock SVM setup](../../blender/intern/cycles/kernel/svm/closure.h) · [Material layout](../blender-platform/cycles-webgpu/material_layout.h)

The continuation uses stock BSDF sampling and combines lobe evaluations in allocation order. When Cycles samples Transparent, the Translucent lobe can contribute directional density along the unchanged ray. The sample keeps the previous MIS origin and still accounts for that density. For next-event estimation, Transparent adds its discrete weight to the mixture denominator. [Stock mixture evaluation](../../blender/intern/cycles/kernel/integrator/surface_shader.h) · [Ordered WebGPU sampling](../cycles-webgpu/ggx_kernels.cc)

The [paper fixtures](check_paper_closures.py) compare constant closure orders, image-linked alpha, shared image Bump, Object Info Random, signed Normal, tinted shadows, and guided OIDN. The previous build fell back on all eleven three-lobe fixtures and the bumped Diffuse and Translucent pair. The varying-shadow path evaluates the Transparent tint when the Bump output feeds only scattering normals. The off-camera Bump control also verifies that enabling the differential layout preserves the stock minimum transparent bounce count and leaves film alpha unchanged. The camera sampling packer detects transparent metadata in both the differential and volume layouts. An isolated render of the original Classroom material uses objects that exercise each of its three texture branches.

All 16 focused cases completed 128 samples on WebGPU. Raw image comparisons matched desktop Cycles within 0.000094 normalized RGB RMSE and 0.000123 alpha RMSE. Strong Bump matched the signed Normal pass at 0.000087 RMSE. The off-camera Bump control produced identical alpha to the flat case on both devices. Guided OIDN ran all three models on WebGPU and matched desktop CPU OIDN at 0.000203 RGB RMSE.

The original paper material completed 192 samples on WebGPU at 180 × 112 with an area light, a world background, Light Tree, and varying shadows. It matched desktop Cycles at 0.0000533 normalized RGB RMSE after correcting the transparent bounce settings. A separate 192 × 128 probe with the three original image branches and mixed lighting matched desktop at 0.0000322 RGB RMSE and 0.0000332 alpha RMSE over 192 samples.

The native sampling check compared 3,072 samples over all six closure allocation orders against stock Cycles setup, evaluation, and sampling. Maximum absolute error was 0.0000000075 across outgoing directions, throughput, Diffuse pass weight, and PDFs.

A separate [volume regression](check_volume_transport.py) completed both transparent film and varying transparent Bump cases at 256 samples on WebGPU. They matched desktop Cycles at 0.00344 and 0.00176 normalized RGB RMSE respectively. Transparent film alpha matched at 0.000487 RMSE.

## Transparent window emission and analytic lamps

Classroom’s window shaders choose Transparent extinction or sky-driven Emission using the actual triangle hit and its Backfacing input. A zero-weight Transparent candidate leaves an opaque emission hit. The shadow and sampled mesh-emission shaders retain the same hit-side context. The emitter guard checks only stock-synchronized sampled triangles that need ray differentials. Bump on other surfaces can use the main differential path alongside ordinary textured emitters. [SVM material interpreter](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Triangle comparisons](check_triangle_emitters.py)

Chrome’s GPU process exited while compiling the original sampled mesh-emission module. The sampled module now follows Blender’s surface-light feature mask and skips scattering payloads with the stock SVM helper, while retaining Alpha-weighted Principled emission. The shader shrank from 11.70 MB to 8.44 MB and retained its texture and procedural nodes. The 473 × 67 Classroom crop with the original teacher desk and window meshes completed 192 samples on WebGPU at 0.0000312 normalized RGB RMSE against desktop. This crop used the compatible subset of the saved lights. After lamp support landed, a second 192-sample crop restored all 10 saved analytic lights and three emissive meshes and matched desktop at 0.0000118 normalized RGB RMSE. [Stock SVM](../../blender/intern/cycles/kernel/svm/svm.h) · [Shader build](../cycles-webgpu/build.py)

The mixed lamp route preserves the stock cast-shadow flag. Point and Area fixtures with shadows on or off matched desktop within 0.000051 normalized RGB RMSE, with an off-camera blocker to make the shadow visible. The Area route also preserves the per-light MIS flag and the synchronized integrator flag that enables analytic-light intersections. Six cases matched within 0.000051 normalized RGB RMSE. Four camera comparisons were pixel-identical, including the stock behavior when a second traceable MIS lamp enables camera intersections. [Stock ray setup](../../blender/intern/cycles/kernel/light/sample.h) · [Lamp comparisons](check_light_falloff.py)

Light Falloff calls the stock SVM routine using the offset shadow ray length in next-event estimation and the hit distance for forward lamp shading. The 15 focused Point, Area, and Sun cases matched desktop within 0.000062 normalized RGB RMSE over 128 samples. An Area fixture with homogeneous scattering and threshold zero traced all 192 samples on WebGPU at 0.00233 RMSE. [Lamp encoder](../blender-platform/cycles-webgpu/scene.cpp) · [Stock Light Falloff](../../blender/intern/cycles/kernel/svm/light_path.h)

## Ambient Occlusion data pass

The AO pass prepares the stock cosine ray and uses its shadow offset and camera throughput. With varying transparency, it reuses the direct-light shadow records after direct lighting has consumed them. The same SVM shadow kernel reads and writes the AO pass buffer instead of direct radiance. The AO clamp runs after applying visibility. Opaque and constant transparency maintain the inline AO path. This support is for the AO film pass on surface paths. [AO preparation](../cycles-webgpu/film_ao_dispatch.hlsl) · [Varying traversal](../cycles-webgpu/bvh2_dispatch.hlsl) · [AO comparisons](check_film_ao.py)

All nine AO fixtures completed 128 samples on WebGPU. Opaque, constant tint, varying scalar and colored extinction, and both sides of a transparent emitter matched desktop within 0.000094 normalized RGB RMSE. Combined with AO enabled was pixel-identical to the Combined-only control on both devices.

## Remaining closure compositions

The compact Principled and Translucent path keeps the existing Principled alpha and GGX records and stores the independent weighted Translucent contribution in the unused transmission vector. The packed order flag preserves SVM allocation order. It uses the stock Translucent setup, sampling, and evaluation alongside the stock GGX routines. The focused comparison passed all 12 cases on WebGPU at 128 samples. The original `branch_1.001` graph and packed image rendered on a smooth sphere at 192 samples with 0.0000298 normalized RGB RMSE and 0.0000185 alpha RMSE against desktop Cycles. [Focused suite](check_principled_translucent.py)

The ordered Glossy path now admits singular isotropic GGX and Beckmann and a single Transparent lobe alongside Glossy. Native comparison exercised 6,144 samples across all closure orders, both distributions, and glossy blur on and off. The 11 focused comparisons passed on WebGPU at 128 samples. A 192-sample crop of the original Sky head, split normals, saved camera and world matched desktop Cycles at 0.0000290 normalized RGB RMSE. [Focused suite](check_ordered_glossy_transparency.py)

Sky’s flag material uses a flat 16-bit image with cubic filtering to drive shadow extinction. Both varying shadow shader variants now compile the same cubic texture helper as the surface path. The expanded suite passed all 13 cases at 128 samples on WebGPU. The cubic case matched desktop with 0.0000613 normalized RGB RMSE and reproduced the 0.0112818 render difference between cubic and linear that desktop produced. [Image interpolation](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Focused suite](check_ordered_glossy_transparency.py)

Sky also uses compact Voronoi F1 Distance and shader queries in varying-shadow materials. Shadow evaluation follows Cycles' feature mask for query outputs, reuses 3D Euclidean F1 distance and normalization, and preserves full queries on camera and continuation hits. The reduced surface shader now handles these compact cases. [Shadow eligibility](../blender-platform/cycles-webgpu/svm_materials.h) · [Focused suite](check_ordered_glossy_transparency.py)

With both Cycles caustics settings disabled, the compact surface shader uses the stock diffuse-ray gate. It omits the Glass closure and applies its lower-layer attenuation on diffuse rays. The two matched Principled comparisons passed on WebGPU at 0.0000613 normalized RGB RMSE and reproduced the native enabled-to-disabled difference of 0.0051725. Asymmetric settings remain guarded. [Focused suite](check_glass_caustics.py)

The opaque paired Principled path fits two branch descriptors in the existing eight-vector ordered tail. The first slice covers two isotropic dielectric GGX or Multiscatter GGX branches with their own mapped normals and varying color, roughness, or specular level. Each branch retains its stock allocation order and share of the mixture PDF. All 12 supported focused cases traced on WebGPU with a maximum normalized RGB RMSE of 0.000107 against desktop. This includes the Normal and Glossy Color passes and guided OIDN. The coat case verified the compatibility guard. Metal, refraction, and subsurface mixtures require additional coverage. [Paired SVM packing](../blender-platform/cycles-webgpu/ordered_principled_svm.h) · [Focused suite](check_two_principled.py)

Monster also uses analytic Area lamps with a distance-dependent emission mix. The lamp encoder accepts a stock Ray Length input, a constant divisor, and a normalized Smootherstep Map Range. It packs the constant emission and mix orientation into the existing lamp records. Next-event estimation supplies the offset shadow segment length and forward hits supply the intersection distance. All eight focused comparisons completed on WebGPU and matched desktop Cycles within 0.000051 normalized RGB RMSE, including both mix orientations, the Light Tree, and the flat distribution. The original Wood plinth, packed textures, and nine Area lamp graphs matched the desktop render at 0.0000734 normalized RGB RMSE in a 1,024 × 93 crop at 192 samples. [Lamp packing](../blender-platform/cycles-webgpu/scene.cpp) · [Focused suite](check_light_falloff.py)

Blender 3.3 Splash applies a Mapping rotation and HSV and value adjustments before its World Background closure. The mixed-light kernels evaluate the synchronized stock SVM bytecode over the environment image and retain the stock importance distribution. The first slice supports one floating-point environment image and one Background closure with eligible coordinate, color, and scalar operations. All nine focused cases traced on WebGPU, including Light Tree, the flat distribution, a single Area light, sampling disabled, and WebGPU OIDN. The unchanged World graph and its packed HDRI matched desktop at 0.0000225 normalized RGB RMSE in a 192-sample fixture. [World evaluator](../blender-platform/cycles-webgpu/svm_world_eval.h) · [Focused suite](check_world_graph.py)

For materials that combine true displacement and bump, stock scene sync displaces the mesh and builds the BVH. The SVM surface stage reads the original position and normal for the residual bump block, evaluates the stock differential, and restores the hit position. All eight focused comparisons passed on WebGPU, including smooth and flat Normal, both Normal Map bases, and true-only and bump-only controls. Their maximum normalized RGB RMSE against desktop was 0.000462. A 165 × 177 crop of the original `Old white plastic` mask with the saved images and nine Area lamps matched desktop at 0.0008932 over 192 samples. [Displacement adapter](../blender-platform/cycles-webgpu/displacement_bump_svm.h) · [Focused suite](check_both_displacement.py)

The previous Monster admission stopped at `Fur 01`, SVM `NODE_END`, word 6053. Its two opaque dielectric Multiscatter Principled branches have linked anisotropy, a tangent Normal Map on the first branch, and Bump, linked Specular IOR Level, and weight-0.125 dielectric Coat on the second. The ordered path now retains each branch's tangent and anisotropy aspect, sets up the white Coat with stock dielectric GGX and lookup tables, and applies stock attenuation before setting up the lower lobes. The original material and nine lamps completed 192 samples on WebGPU in a 156 × 103 crop at 0.000178 normalized RGB RMSE against desktop. [Stock layering](../../blender/intern/cycles/kernel/svm/closure.h) · [Stock Coat](../../blender/intern/cycles/kernel/closure/bsdf_microfacet.h) · [Focused suite](check_two_principled.py)

The next admission stopped at `Fabric sheet`, SVM `NODE_END`, word 7286. The Fabric materials mix two Principled branches and use weight-0.2 white Sheen with roughness 0.5 on one branch. For eligible isotropic branches without an active Coat on the same branch, the SVM shader uses Blender's Sheen setup and LTC table. It stores the fitted transform and attenuates the lower layers. Direct lighting, continuation, film and denoising guides retain the diffuse Sheen lobe. The original `pyjamas` material and nine lamps completed 192 samples on WebGPU in a 252 × 366 crop at 0.001206 normalized RGB RMSE. Removing the native Sheen alone changed the reference by 0.01020 RGB RMSE. The focused suite passed 21 Fur and Coat cases, eight Sheen cases, and the colored-Coat compatibility guard. [Stock Sheen](../../blender/intern/cycles/kernel/closure/bsdf_sheen.h) · [Paired fixtures](check_two_principled.py)

`hands` and the `monster` skin use Coat alongside Legacy Random Walk subsurface scattering. The single opaque isotropic path keeps the walk's IOR, method, and weight and applies stock Coat attenuation and sampling before selecting the remaining lobes. The original hand and skin crops completed all 192 samples on WebGPU at 0.00783 and 0.02950 normalized RGB RMSE. At the same resolution and sample count, removing Coat alone leaves 0.02784 RMSE, and removing subsurface alone lowers it to 0.000410. The controls localize much of the discrepancy to Legacy Random Walk.

The synchronized nail geometry evaluates a `Metal scratched` mix with linked Metallic and a second coated Metallic branch. The ordered path uses stock F82 reflection and keeps separate multiscatter energy factors for the metal and dielectric specular lobes. It supports an isotropic metallic branch alongside the admitted layers. All eight focused metallic cases passed on WebGPU, including Glossy Color and OIDN. The original nail with both modifiers, images and nine lamps completed 192 WebGPU samples at 0.000142 normalized RGB RMSE.

The eyes and teeth emit through a white Coat. Main-path and sampled emission use the stock dielectric albedo lookup and apply the same Coat attenuation as desktop. All six desktop comparisons passed, including visible emission, next-event estimation with the Light Tree, and the flat light distribution. The maximum normalized mean RGB difference was 0.000016. [Coated emission comparisons](check_triangle_emitters.py)

## Volume transport and surface queries

The scene encoder admits `CYCLES_WEBGPU_SVM_QUERIES` with eligible volumes. The volume and surface query paths use separate storage. In the packed points buffer, `surface-query.wgsl` appends the original eight-word intersection after the position, normal, geometric, and shadow data. The query SVM reads that copy and binds the BVH in the usual hit binding. It uses the appended camera sampling data and the path’s pixel seed, sample number, RNG offset, and minimum ray PDF. The volume stack, boundary state, scatter count, and transmittance are stored in the volume dispatch buffers. [Scene admission](../blender-platform/cycles-webgpu/scene.cpp) · [Surface packing](../cycles-webgpu/surface_dispatch.hlsl) · [Query bindings](../cycles-webgpu/svm_dispatch.hlsl)

The stock query adapters share the surface hit and BVH and have distinct contracts.

- **Ambient Occlusion.** Stock `svm_ao` branches the current path RNG on `PRNG_SURFACE_AO` and traces from the shading point using its object and primitive as self IDs. It counts binary hits through `PATH_RAY_VISIBILITY_SHADOW_OPAQUE` or uses object-local traversal. The stock visibility masks govern which surfaces occlude an AO ray. Volume extinction belongs to the lighting transport stage.
- **Bevel.** Stock `svm_bevel` also depends on the shading and geometric normals, minimum ray PDF, and the `PRNG_SURFACE_BEVEL` sequence. It finds normals using local object traversal and preserves the shading normal when the minimum ray PDF is below 8. Cover both active beveling after volume transmission and this bypass after a blurry scatter, including a linked input normal.
- **Raycast.** Stock `svm_raycast` uses the supplied position, direction, and distance, then returns the closest hit using local traversal or `PATH_RAY_VISIBILITY_RAYCAST`. The current WebGPU decoder accepts its standard hit outputs and excludes attribute requests and the bump-filter-width path. [Stock-query adapter](../cycles-webgpu/scene_query_kernels.h) · [SVM decoder](../blender-platform/cycles-webgpu/svm_material_eval.h)

### Focused comparison

An isolated build that enabled the combined path completed the following 128 × 96 renders at 128 fixed samples on WebGPU. Errors are normalized RGB RMSE against desktop Cycles.

| Case | RMSE |
| --- | ---: |
| AO control without volume | 0.00000909 |
| AO with absorption | 0.00000962 |
| AO with scattering | 0.000911 |
| Raycast with scattering | 0.001785 |

The implementation validates the volume-differential marker on the surface shader selected for dispatch, including the query variant. A 128-sample test of AO and Bump with scattering finished on WebGPU at 0.000936 normalized RGB RMSE against desktop Cycles. A Bevel output feeding emission with scattering also finished on WebGPU at 0.002135 RMSE. The [volume regression suite](check_volume_transport.py) adds AO with scattering and AO plus Bump with scattering. Both completed on WebGPU at 192 samples and matched desktop within 0.0042 RGB RMSE.

### Composition checks

- Surface queries must read the live triangle hit after volume classification, including volume-only boundary passes and ordinary hits after a scatter. Match stock `path_state_volume_next` and `path_state_next` when reconstructing the query RNG offset from main bounces and extra steps.
- Exercise the query-hit tail alongside the tangent tail and volume ray differentials in the same dispatch. Test the depth and direction after both a transmitted segment and a scatter. [Shader selection](../blender-platform/cycles-webgpu/pipeline.cpp) · [Variant build](../cycles-webgpu/build.py)
- Keep BVH queries on the main surface path. The varying transparent-shadow evaluator follows Blender’s shadow feature mask and uses the stock fallback values for AO, Bevel and Raycast. It guards unsupported path-depth queries. Test query-driven opacity separately from sampled emissive graphs.

Blender 3.3 Splash uses Ambient Occlusion in textured materials with box projection. Chrome’s GPU process exited while compiling the general textured query shader, before compiling lighting. The host now uses the AO-only textured variant when the scene’s eligible material bytecode uses AO and has no Bevel or Raycast. It retains Cycles’ AO kernel, inputs, image projection, and main-path hit state. All six focused CPU comparisons passed, with maximum normalized RGB RMSE of 0.000123. Run the projection fixtures with `--ao-textures`. [SVM selection](../blender-platform/cycles-webgpu/pipeline.cpp) · [Projection fixtures](check_scene_queries.py)

Extend the fixture with an AO receiver inside the medium and a query radius that reaches the volume boundary. Compare positive density and a zero-density control, and capture both the AO-driven output and Combined image. Confirm completed WebGPU surface and volume stages, then check the saved 3.3 Splash. Keep the ledger at `limited` and add the exact combinations that pass.

## Saved-scene baseline

The five scenes in `blender_test` were compared at their saved frames and resolutions with eight fixed samples.

| Scene | Baseline WebGPU result |
| --- | --- |
| Blender 3.5 Splash | Tracing and OIDN completed on WebGPU at 1327 × 1250. Two browser runs produced identical pixels. Normalized RGB RMSE against desktop Cycles was 0.00208. |
| Blender 3.3 Splash | CPU fallback at `Material.062`, where Ambient Occlusion drives a mix of two diffuse-only Principled shaders in a scene with volumes. |
| Classroom | CPU fallback at `pencil`, which mixes two Diffuse and two Glossy closures with Bump. |
| Monster Under the Bed | CPU fallback at `Wood plinth`, which combines two Principled shaders and a linked tangent-space Normal Map. |
| Sky Texture Demo | CPU fallback at `Material`, which combines two Glossy lobes and Beckmann Refraction with camera-dependent transparency. |

### Combined-build progress

The initial combined build passed `Material.062` with the two-branch diffuse-only Principled mix and volume plus query admission. It fell back at `castle.006` at SVM `NODE_CLOSURE_BSDF` word 18879, where four diffuse-only Principled branches are combined under three nested Mix Shaders. They share the default closure normal. A Normal Map drives a color input through RGB Curves. The initial two-branch limit rejected the third diffuse closure.

The expanded weighted Lambert build passes both materials and falls back at `branch_1.001` at `NODE_CLOSURE_BSDF` word 29836. This material adds a standalone Translucent BSDF to a Principled BSDF with GGX reflection and image-linked Base Color and Alpha. The image color also drives the Translucent lobe. Stock Cycles creates Transparent extinction from the Principled Alpha and applies Alpha to that branch's remaining lobes. The Add Shader keeps the Translucent lobe independent. The decoder rejects this closure combination in either order. [SVM closure admission](../blender-platform/cycles-webgpu/svm_material_eval.h)

The [focused fixture suite](check_mixed_principled.py) verified both branch orders, varying weights, and Blender’s per-closure cutoff. All five four-branch fixtures completed 96 samples on WebGPU and matched desktop within 0.000062 normalized RGB RMSE. A separate 128-sample four-branch AO and volume fixture traced and denoised on WebGPU and matched desktop CPU Cycles and OIDN at 0.000950 RMSE.

The ordered-closure build admitted Classroom’s `pencil` and exposed a fallback at `dustbin_metal`, a Beckmann Glossy material with both shader-wide displacement bump and a closure-local Bump. The next build admitted both materials and reached `dustBin_wireframe` on `NODE_TEX_IMAGE` word 22395. The material uses an external 16-bit RGBA PNG, which stock Cycles loads as `IMAGE_DATA_TYPE_USHORT4`. Packing the same image in an isolated copy made Blender supply `FLOAT4` and advanced the rejection to the image-driven Glossy and Transparent mix. With the USHORT4 change alone, the same external-image fixture advances to that closure gate without repacking. [Stock image metadata](../../blender/intern/cycles/util/image_metadata.cpp) · [WebGPU image admission](../blender-platform/cycles-webgpu/svm_materials.h)

The build with `dustBin_wireframe` admitted first reached `crinkledPaper_paper` at `NODE_CLOSURE_BSDF` word 22723. Stock SVM allocates Transparent, Diffuse, then Translucent. Object Info Random selects one of three paper images. The selected alpha drives the outer mix, and the two inner lobes share the selected color and image-derived Bump. The ordered Transparent and Translucent build passes this material. [Closure decoder](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Shadow eligibility](../blender-platform/cycles-webgpu/svm_materials.h)

Before mixed displacement support, the saved 1920 × 1080 Classroom scene first fell back at `teacherDesk_plastic`, SVM `NODE_END` word 36027. A 0.06 mix combines Lambert Diffuse and isotropic Beckmann Glossy. Both closures share a UV Noise Bump, and the same Bump output drives object-space displacement Height through a Math Multiply. The mixed closure path retains the shader-wide displacement normal for light sampling and MIS and gives both lobes the shared Bump normal. The stock Diffuse and Beckmann routines apply their own normal correction to that shared input. [Displacement admission](../blender-platform/cycles-webgpu/svm_material_eval.h)

The mixed displacement build passes `teacherDesk_plastic` and first falls back at `dayLight_portal`, SVM `NODE_END` word 38766. Geometry Backfacing selects white Transparent on the front and a strength-20 Emission on the back. A Reflection vector drives a Hosek–Wilkie Sky Texture through RGB Curves and Hue/Saturation. The `hallWindow` and `windows` meshes share the material. The prior SVM decoder rejected a program containing both explicit Transparent and Emission, even when the branches were mutually exclusive at a hit.

The window build passes both hit sides, transparent film, flat light distribution and Light Tree, and Bump on an unrelated receiver. All ten supported fixtures completed 128 samples on WebGPU with mean RGB error up to 0.000015. Camera alpha matched exactly. The sampled pixel-Wireframe differential guard fell back as intended, while the camera-only Wireframe case stayed on WebGPU. [SVM admission](../blender-platform/cycles-webgpu/svm_material_eval.h) · [Hit context](../cycles-webgpu/bvh2_kernels.cc)

The follow-up Classroom build clears the window shader, analytic Light Falloff, the shadowless and MIS-disabled exterior Area light, and the varying-shadow AO pass. The original saved scene completed all eight fixed samples at 1920 × 1080 on WebGPU with the saved compositor and pass configuration in two Chrome sessions. Both exports had identical pixels and matched the desktop CPU reference at 0.0111108 normalized RGB RMSE. Render and pipeline setup took 113.9 and 97.8 seconds respectively.

The mixed displacement build’s Classroom CPU fallback export matched desktop at 0.0109405 normalized RGB RMSE. All five saved scenes were rerendered on the final AO build. The other four scenes retained their tracing and denoising devices.

| Scene | Tracing | RGB RMSE against desktop |
| --- | --- | ---: |
| Classroom | WebGPU tracing, saved AO and compositor | 0.0111108 |
| Blender 3.3 Splash | CPU fallback at `branch_1.001`, closure word 29836 | 0.0032511 |
| Blender 3.5 Splash | WebGPU tracing and OIDN | 0.0015748 |
| Monster Under the Bed | CPU fallback at `Wood plinth`, closure word 4046 | 0.0099355 |
| Sky Texture Demo | CPU fallback at `Material`, closure word 23190, with WebGPU OIDN | 0.0005671 |

The subsequent closure-composition build rerendered Classroom and Blender 3.5 Splash with all eight fixed samples on WebGPU. Classroom matched desktop at 0.0111032 normalized RGB RMSE, and the 3.5 Splash traced and denoised on WebGPU with an RMSE of 0.00157482.

The original Blender 3.3 Splash completed all eight fixed samples at 2350 × 1000 on WebGPU with zero fallback. Its saved CPU OIDN and compositor completed and the exported image matched desktop at 0.00318211 normalized RGB RMSE.

Sky Texture Demo completed all eight samples at 1064 × 446, frame 48, on WebGPU. It kept the saved Depth pass and volume denoising enabled. Chrome reported all three OIDN networks GPU-resident and read the film for export. The resulting image matched desktop at 0.0076576 normalized RGB RMSE. Focused Depth exports matched pixel-for-pixel through mixed Glass and absorbing Transparent boundaries. With scattering and varying Transparent, 3968 of 4096 pixels matched at sample zero. Classroom also rerendered on this build at 1920 × 1080 with zero fallback and an RMSE of 0.0111106. [Depth fixtures](check_volume_transport.py)

Blender 3.5 Splash rerendered on the same build at 1327 × 1250 with all eight samples and OIDN on WebGPU. Its normalized RGB RMSE remained 0.00157482.

After adding the paired layered closures and the single Coat path, the original Monster completed all eight samples on WebGPU at 1024 × 1024 with the saved compositor enabled and denoising disabled. The normalized RGB RMSE was 0.06711. An eight-sample desktop run with another seed differed from the reference by 0.16837, and an eight-pixel Gaussian reduced the CPU-to-WebGPU difference to 0.00503.

The integrated build repeated the full Monster render at 64 fixed samples. Every sample completed on WebGPU without CPU replay, and the export matched the same-seed desktop reference at 0.03534 normalized RGB RMSE. With an eight-pixel Gaussian applied for comparison, the difference was 0.00462. The alternate-seed 192-sample desktop skin control differed from the original desktop skin by 0.04919 RGB RMSE. The comparison exports preserve the original saved compositor and denoising settings.

The integrated build also rerendered Blender 3.5 Splash with WebGPU tracing and OIDN. Its normalized RGB RMSE was 0.00157481.

## Verification and ledger updates

For each change, check a small fixture against desktop CPU Cycles first. Use fixed seeds and sample counts, cover both branch orders and zero and full mix weights, and compare closure weights, normals, and sampling PDFs where the fixture can expose them. Collect the browser scheduler log and an image comparison. A WebGPU pass requires completed GPU sample batches with zero CPU fallbacks.

Rerender the five saved scenes after each backend change at the same settings. Keep the saved frame, camera, materials, lighting, resolution, compositor, and denoising settings. Use eight fixed samples, disable adaptive sampling and time limits for this baseline, and record tracing and denoising devices separately. If a scene still falls back, record its first remaining blocker.

Update the affected ledger groups with source anchors, reproducible fixture evidence, and the exact supported combinations. Keep partial implementations marked `limited` and record their remaining boundaries. Run the source check from the web project directory after ledger updates.

```sh
python3 material-tests/run.py source-coverage --check --output /tmp/cycles-coverage.json
```

Use `--require-audited` to check the remaining review backlog. See the [coverage guide](README.md#cycles-coverage-from-blender-source) for the audit rules.

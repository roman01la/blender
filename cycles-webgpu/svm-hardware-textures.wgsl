// SPDX-FileCopyrightText: 2017-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

@group(0) @binding(8) var cycles_svm_atlas: texture_2d<f32>;
@group(0) @binding(9) var cycles_svm_atlas_sampler: sampler;

fn cycles_svm_atlas_coord(value: f32, extension: u32) -> f32 {
  if (extension == 0u) {
    return fract(value);
  }
  if (extension == 3u) {
    return 1.0 - abs(2.0 * fract(0.5 * value) - 1.0);
  }
  return value;
}

fn cycles_svm_atlas_weights(t: f32) -> vec4<f32> {
  // B-spline basis from Cycles kernel/device/gpu/image.h.
  return (1.0 / 6.0) * vec4<f32>(t * (t * (-t + 3.0) - 3.0) + 1.0,
                               t * t * (3.0 * t - 6.0) + 4.0,
                               t * (t * (-3.0 * t + 3.0) + 3.0) + 1.0,
                               t * t * t);
}

fn cycles_svm_atlas_linear(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(cycles_svm_atlas, cycles_svm_atlas_sampler, uv, 0.0);
}

fn _Z26cycles_svm_hardware_samplejjjjff(image: u32, width: u32, height: u32, extension: u32,
                                       u: f32, v: f32) -> vec4<f32> {
  let atlas_size = textureDimensions(cycles_svm_atlas);
  if (image >= atlas_size.x || !all(abs(vec2<f32>(u, v)) <= vec2<f32>(1.0e5))) {
    return vec4<f32>(2.0e6, 0.0, 0.0, 0.0);
  }
  // The metadata row stores the interior pixel origin. A zero x marks
  // an image that must use the storage-buffer sampler.
  let origin = textureLoad(cycles_svm_atlas, vec2<i32>(i32(image), 0), 0).xy;
  if (origin.x == 0.0) {
    return vec4<f32>(2.0e6, 0.0, 0.0, 0.0);
  }

  let size = vec2<f32>(f32(width), f32(height));
  let pixel = size * vec2<f32>(cycles_svm_atlas_coord(u, extension),
                               cycles_svm_atlas_coord(v, extension));
  let lo = vec2<f32>(-1.5);
  let hi = size + vec2<f32>(1.5);
  if (extension == 2u && (any(pixel < lo) || any(pixel > hi))) {
    return vec4<f32>(0.0);
  }
  let pt = clamp(pixel, lo, hi) - vec2<f32>(0.5);
  let base = floor(pt);
  let frac = pt - base;
  let wx = cycles_svm_atlas_weights(frac.x);
  let wy = cycles_svm_atlas_weights(frac.y);
  let gx = vec2<f32>(wx.x + wx.y, wx.z + wx.w);
  let gy = vec2<f32>(wy.x + wy.y, wy.z + wy.w);

  // Stock Cycles' four bilinear bicubic lookups. Limit every lookup to the
  // border texel centers to keep it inside the image's two-texel gutter.
  let look_x = vec2<f32>(base.x - 0.5 + wx.y / gx.x, base.x + 1.5 + wx.w / gx.y);
  let look_y = vec2<f32>(base.y - 0.5 + wy.y / gy.x, base.y + 1.5 + wy.w / gy.y);
  let x = (origin.x + clamp(look_x, vec2<f32>(lo.x), vec2<f32>(hi.x))) / f32(atlas_size.x);
  let y = (origin.y + clamp(look_y, vec2<f32>(lo.y), vec2<f32>(hi.y))) / f32(atlas_size.y);
  return gy.x * (gx.x * cycles_svm_atlas_linear(vec2<f32>(x.x, y.x)) +
                 gx.y * cycles_svm_atlas_linear(vec2<f32>(x.y, y.x))) +
         gy.y * (gx.x * cycles_svm_atlas_linear(vec2<f32>(x.x, y.y)) +
                 gx.y * cycles_svm_atlas_linear(vec2<f32>(x.y, y.y)));
}

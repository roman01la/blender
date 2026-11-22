/* SPDX-License-Identifier: Apache-2.0 */

uint4 cycles_svm_scene_context(uint field, uint object, uint packed_shader)
{
  if (field >= CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA && field < CYCLES_WEBGPU_SVM_WORLD_TO_NDC_END) {
    const uint row = CYCLES_WEBGPU_CAMERA_SVM_OFFSET + 4 * (field - CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA);
    return uint4(camera_words[row], camera_words[row + 1], camera_words[row + 2], camera_words[row + 3]);
  }
  if (field >= CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD && field < CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD_END) {
    const uint row = CYCLES_WEBGPU_CAMERA_TO_WORLD_OFFSET + 4 * (field - CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD);
    return uint4(camera_words[row], camera_words[row + 1], camera_words[row + 2], camera_words[row + 3]);
  }
  if (field >= CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM) {
    if (field < CYCLES_WEBGPU_SVM_OBJECT_INVERSE) {
      const uint row = metadata_words[0] + 13 * object + 1 +
                       4 * (field - CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
      return uint4(metadata_words[row], metadata_words[row + 1],
                   metadata_words[row + 2], metadata_words[row + 3]);
    }
    const uint base = metadata_words[CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET] +
                      CYCLES_WEBGPU_SURFACE_OBJECT_WORDS * object;
    if (field <= CYCLES_WEBGPU_SVM_OBJECT_COLOR) {
      const uint row = base + 4 * (field - CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
      return uint4(metadata_words[row], metadata_words[row + 1],
                   metadata_words[row + 2], metadata_words[row + 3]);
    }
    if (field == CYCLES_WEBGPU_SVM_MATERIAL_INDEX) {
      const uint shader = packed_shader & CYCLES_WEBGPU_SVM_SHADER_ID_MASK;
      const uint shader_pass = metadata_words[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] + shader;
      const int pass_id = __builtin_bit_cast(int, metadata_words[shader_pass]);
      return uint4(0, 0, 0, __builtin_bit_cast(uint, float(pass_id)));
    }
    if (field == CYCLES_WEBGPU_SVM_DUPLI_GENERATED) {
      return uint4(metadata_words[base + 18], metadata_words[base + 19],
                   metadata_words[base + 20], 0);
    }
    if (field == CYCLES_WEBGPU_SVM_DUPLI_UV) {
      return uint4(metadata_words[base + 21], metadata_words[base + 22], 0, 0);
    }
    return uint4(0, 0, 0, metadata_words[base + (field == CYCLES_WEBGPU_SVM_OBJECT_INDEX ? 16 : 17)]);
  }
  return uint4(0, 0, 0, 0);
}

/* SPDX-License-Identifier: Apache-2.0 */

#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
float4 cycles_material_lobe(uint index)
{
  return materials[index];
}
#endif

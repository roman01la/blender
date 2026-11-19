/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

// Clang supplies the resource templates. The local header supplies vector aliases
// on distributions that ship the HLSL frontend without its default headers.
typedef unsigned int uint3 __attribute__((ext_vector_type(3)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint;

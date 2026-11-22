/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

using VolumeFloat4 = float __attribute__((ext_vector_type(4)));
using VolumeUInt4 = unsigned int __attribute__((ext_vector_type(4)));

void cycles_volume_coefficients(VolumeUInt4 options,
                                VolumeFloat4 weight,
                                VolumeUInt4 data0,
                                VolumeUInt4 data1,
                                VolumeUInt4 data2,
                                VolumeUInt4 data3,
                                VolumeUInt4 data4,
                                VolumeFloat4 incoming,
                                VolumeFloat4 random,
                                VolumeFloat4 &extinction,
                                VolumeFloat4 &scattering,
                                VolumeFloat4 &transmittance,
                                VolumeFloat4 &outgoing,
                                VolumeFloat4 &emission);

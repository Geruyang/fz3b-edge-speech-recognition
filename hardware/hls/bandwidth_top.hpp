#pragma once

#include <ap_int.h>
#include <cstdint>

using BandwidthBeat = ap_uint<128>;

// Input regions and disjoint scratch are beat-indexed in one mapped image.
// write_enable=0 is the same-bitstream read-only reference; 1 copies every
// consumed 128-bit beat to write_base + input index on its
// owning physical HP path; result bytes describe reads, not total R+W traffic.
// Results contain
// the consumed byte count in bits [127:64] and the signature in [63:0].
void bandwidth_top(BandwidthBeat* hp0, BandwidthBeat* hp1,
                   BandwidthBeat* hp2, BandwidthBeat* hp3,
                   unsigned active_hp_mask, std::uint32_t tiles, std::uint32_t scale_base,
                   std::uint32_t write_base, unsigned write_enable,
                   BandwidthBeat& result0, BandwidthBeat& result1,
                   BandwidthBeat& result2, BandwidthBeat& result3);

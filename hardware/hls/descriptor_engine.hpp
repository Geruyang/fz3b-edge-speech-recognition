#pragma once

#include "../../driver/fzspeech_descriptor.h"

namespace edge_speech::pl {

struct BurstStep {
    fz_u64 address;
    fz_u32 bytes;
    fz_u32 port;
};

// One 128-bit INCR step. Check the SG and tile bounds again at the PL boundary.
bool next_descriptor_burst(const fz_dma_span& span, fz_u64 span_offset, fz_u64 tile_remaining,
                           fz_u64 dma_mask, fz_u32 active_mask, BurstStep& burst);

} // namespace edge_speech::pl

// HLS top for the combinational burst bounds step.
bool fzspeech_burst_step(const fz_dma_span& span, fz_u64 span_offset, fz_u64 tile_remaining,
                         fz_u64 dma_mask, fz_u32 active_mask, edge_speech::pl::BurstStep& burst);

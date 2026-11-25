#include "descriptor_engine.hpp"

namespace edge_speech::pl {

bool next_descriptor_burst(const fz_dma_span& span, fz_u64 span_offset, fz_u64 tile_remaining,
                           fz_u64 dma_mask, fz_u32 active_mask, BurstStep& burst) {
    if (!fz_aligned(span.logical_offset | span.dma_address | span.bytes | span_offset | tile_remaining) ||
        span_offset >= span.bytes || !tile_remaining ||
        span.bytes > static_cast<fz_u64>(-1) - span.logical_offset ||
        span.dma_address > dma_mask || span.bytes - 1 > dma_mask - span.dma_address ||
        span.port >= 4 || !(active_mask & (1U << span.port)) ||
        (active_mask != 0x9 && active_mask != 0xb && active_mask != 0xf))
        return false;

    const fz_u64 address = span.dma_address + span_offset;
    const fz_u64 span_remaining = span.bytes - span_offset;
    const fz_u64 remaining = span_remaining < tile_remaining ? span_remaining : tile_remaining;
    burst = {address, fz_burst_bytes(address, remaining), span.port};
    return true;
}

} // namespace edge_speech::pl

bool fzspeech_burst_step(const fz_dma_span& span, fz_u64 span_offset, fz_u64 tile_remaining,
                         fz_u64 dma_mask, fz_u32 active_mask, edge_speech::pl::BurstStep& burst) {
    burst = {};
    return edge_speech::pl::next_descriptor_burst(span, span_offset, tile_remaining, dma_mask, active_mask, burst);
}

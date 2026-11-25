#ifndef FZSPEECH_DESCRIPTOR_H
#define FZSPEECH_DESCRIPTOR_H

#include "uapi/fzspeech.h"

/* Kernel-generated DMA span. The provider writes fields little-endian to PL. */
struct fz_dma_span {
    fz_u64 logical_offset;
    fz_u64 dma_address;
    fz_u64 bytes;
    fz_u32 transfer_index;
    fz_u32 port;
};

#define FZ_MAX_DMA_SPANS (FZ_MAX_CHAIN_BYTES / sizeof(struct fz_dma_span))

static inline fz_u32 fz_span_limit(fz_u32 transfer_count)
{
    return (FZ_MAX_CHAIN_BYTES - sizeof(struct fz_chain) -
            transfer_count * sizeof(struct fz_transfer)) / sizeof(struct fz_dma_span);
}

static inline int fz_range(fz_u64 offset, fz_u64 bytes, fz_u64 limit)
{
    return bytes && offset < limit && bytes <= limit - offset;
}

static inline int fz_aligned(fz_u64 value)
{
    return !(value & (FZ_BEAT_BYTES - 1U));
}

static inline fz_u32 fz_profile_mask(fz_u32 id)
{
    switch (id) {
    case 2: return 0x9;
    case 3: return 0xb;
    case 4: return 0xf;
    default: return 0;
    }
}

static inline fz_u32 fz_pair_port(fz_u32 mask, fz_u32 pair)
{
    if (pair >= 8)
        return 4;
    switch (mask) {
    case 0x9: return pair < 4 ? 0 : 3;
    case 0xb: return pair < 3 ? 0 : (pair < 5 ? 1 : 3);
    case 0xf: return pair / 2;
    default: return 4;
    }
}

static inline fz_u32 fz_burst_bytes(fz_u64 address, fz_u64 remaining)
{
    fz_u32 boundary = 4096U - (fz_u32)(address & 4095U);
    fz_u32 maximum = boundary < FZ_MAX_BURST_BYTES ? boundary : FZ_MAX_BURST_BYTES;
    return remaining < maximum ? (fz_u32)remaining : maximum;
}

#endif

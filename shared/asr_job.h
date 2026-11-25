/* SPDX-License-Identifier: GPL-2.0 */
/* Driver-private LE descriptor encoder for the selected streaming Zipformer. */
#ifndef FZ_ASR_JOB_H
#define FZ_ASR_JOB_H

#include "../driver/fzspeech_job.h"
#include "asr_job_table.h"

#define FZ_ASR_JOB_VERSION 3U
#define FZ_ASR_CALL_COUNT 51U
#define FZ_ASR_STATE_NONE 65535U
#define FZ_ASR_DMA_MASK 0x3fffffffULL
#define FZ_ASR_REGION_COUNT 9U
#define FZ_ASR_REGION_BASE(id) ((fz_u64)(id) << 20)

#define FZ_ASR_DUAL_LAYOUT_SHA256_HEX "8289164bf96e949ad495689b70161533edc262dce6bae1d80857d5c4def1d89d"
static const unsigned char fz_asr_dual_layout_sha256[32] = {
    0x82,0x89,0x16,0x4b,0xf9,0x6e,0x94,0x9a,0xd4,0x95,0x68,0x9b,0x70,0x16,0x15,0x33,
    0xed,0xc2,0x62,0xdc,0xe6,0xba,0xe1,0xd8,0x08,0x57,0xd5,0xc4,0xde,0xf1,0xd8,0x9d
};

enum fz_asr_operation {
    FZ_ASR_FFN, FZ_ASR_CONV, FZ_ASR_EMBEDDING,
    FZ_ASR_FFN_HIDDEN_LEFT, FZ_ASR_FFN_HIDDEN_RIGHT, FZ_ASR_FFN_OUTPUT_LEFT, FZ_ASR_FFN_OUTPUT_RIGHT
};

enum fz_asr_region {
    FZ_ASR_WEIGHTS = 1, FZ_ASR_SCALES, FZ_ASR_BIAS, FZ_ASR_INPUT,
    FZ_ASR_OUTPUT, FZ_ASR_OLD, FZ_ASR_WORKING, FZ_ASR_SCRATCH, FZ_ASR_CODES
};

static inline fz_u32 fz_asr_region_access(unsigned region_id)
{
    switch (region_id) {
    case FZ_ASR_WEIGHTS: case FZ_ASR_SCALES: case FZ_ASR_BIAS:
    case FZ_ASR_INPUT: case FZ_ASR_OLD: return FZ_ACCESS_READ;
    case FZ_ASR_OUTPUT: case FZ_ASR_WORKING: return FZ_ACCESS_WRITE;
    case FZ_ASR_SCRATCH: case FZ_ASR_CODES: return FZ_ACCESS_READ_WRITE;
    default: return 0;
    }
}

/* An absent region has no descriptor, mapping or valid HLS pointer. */
static inline fz_u32 fz_asr_region_bytes(unsigned call_id, unsigned region_id)
{
    unsigned op, rows;
    if (call_id >= FZ_ASR_CALL_COUNT) return 0;
    op = fz_asr_calls[call_id].operation;
    rows = fz_asr_calls[call_id].rows;
    switch (region_id) {
    case FZ_ASR_WEIGHTS:
        if (op == FZ_ASR_EMBEDDING) return 688128;
        return op == FZ_ASR_FFN ? 393216 : 204800;
    case FZ_ASR_SCALES: case FZ_ASR_BIAS:
        return op == FZ_ASR_EMBEDDING ? 1696 : 4096;
    case FZ_ASR_INPUT:
        return op == FZ_ASR_EMBEDDING ? 12480 : rows * 1024;
    case FZ_ASR_OUTPUT:
        return rows * 1024;
    case FZ_ASR_OLD: case FZ_ASR_WORKING:
        return op == FZ_ASR_CONV ? 30720 : 0;
    case FZ_ASR_SCRATCH:
        if (op == FZ_ASR_CONV) return (30 + 2 * rows) * 1024;
        return op == FZ_ASR_FFN ? rows * 768 * 4 : 0;
    case FZ_ASR_CODES:
        if (op == FZ_ASR_CONV) return (30 + rows) * 256;
        return op == FZ_ASR_FFN ? rows * 768 : 0;
    default: return 0;
    }
}

static inline int fz_asr_operation_allowed(unsigned call_id, unsigned operation, int dual, unsigned lane)
{
    unsigned full;

    if (call_id >= FZ_ASR_CALL_COUNT || lane > 1) return 0;
    full = fz_asr_calls[call_id].operation;
    if (operation == full) return 1;
    return dual && full == FZ_ASR_FFN &&
           operation >= FZ_ASR_FFN_HIDDEN_LEFT && operation <= FZ_ASR_FFN_OUTPUT_RIGHT &&
           ((operation - FZ_ASR_FFN_HIDDEN_LEFT) & 1U) == lane;
}

/* Parameters retain the original authenticated allocation. The HLS kernel
 * selects its immutable subranges with the operation number.
 */
static inline fz_u32 fz_asr_operation_region_bytes(unsigned call_id, unsigned operation, unsigned region)
{
    unsigned rows, hidden;

    if (operation <= FZ_ASR_EMBEDDING)
        return fz_asr_region_bytes(call_id, region);
    if (call_id >= FZ_ASR_CALL_COUNT || fz_asr_calls[call_id].operation != FZ_ASR_FFN ||
        operation > FZ_ASR_FFN_OUTPUT_RIGHT)
        return 0;
    rows = fz_asr_calls[call_id].rows;
    hidden = operation <= FZ_ASR_FFN_HIDDEN_RIGHT;
    switch (region) {
    case FZ_ASR_INPUT: return rows * (hidden ? 1024U : 3072U);
    case FZ_ASR_OUTPUT: case FZ_ASR_SCRATCH: return rows * (hidden ? 1536U : 512U);
    case FZ_ASR_CODES: return rows * (hidden ? 256U : 768U);
    default: return fz_asr_region_bytes(call_id, region);
    }
}

static inline int fz_asr_dma_range(fz_u64 address, fz_u64 bytes)
{
    return bytes && address <= FZ_ASR_DMA_MASK && bytes - 1 <= FZ_ASR_DMA_MASK - address;
}

static inline int fz_asr_overlap(fz_u64 a, fz_u64 a_bytes, fz_u64 b, fz_u64 b_bytes)
{
    /* Both positive spans have already passed the 30-bit aperture check. */
    return a < b + b_bytes && b < a + a_bytes;
}

/* Caller authenticates immutable OEM parameter bytes before DMA mapping.
 * On failure neither packet nor *written is modified. Mapping lifetime,
 * sync-for-device and completion/ACK checks belong to the owning lease. */
static inline int fz_asr_job_encode_lane(unsigned char *packet, fz_u32 capacity,
        const struct fz_job_meta *meta, fz_u32 call_id, fz_u32 operation,
        fz_u32 rows, fz_u32 state_slot, fz_u32 sequence,
        const unsigned char parameters_sha256[32], const struct fz_job_span *spans,
        fz_u32 count, fz_u64 descriptor_dma, fz_u64 descriptor_capacity, fz_u32 *written,
        int dual, unsigned lane)
{
    fz_u32 i, j, bytes, region = 1, used = 0, extent;
    const unsigned char *capability = dual ? fz_asr_dual_layout_sha256 : fz_asr_layout_sha256;
    const struct fz_asr_call *call;
    if (!packet || !meta || !parameters_sha256 || !spans || !written ||
        !count || count > FZ_JOB_MAX_SPANS || !meta->generation || !meta->chain_id ||
        !meta->profile.epoch || !sequence || !(meta->profile.mask & 1U) ||
        fz_profile_mask(meta->profile.id) != meta->profile.mask)
        return FZ_JOB_INVALID;
    if (!fz_asr_operation_allowed(call_id, operation, dual, lane)) return FZ_JOB_UNSUPPORTED;
    call = &fz_asr_calls[call_id];
    if (call->rows != rows || call->state_slot != state_slot)
        return FZ_JOB_UNSUPPORTED;
    for (i = 0; i < 32; ++i)
        if (parameters_sha256[i] != call->parameters_sha256[i]) return FZ_JOB_UNSUPPORTED;
    bytes = FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * count;
    if (capacity < bytes || descriptor_capacity < bytes) return FZ_JOB_CAPACITY;
    if (!fz_aligned(descriptor_dma) || !fz_asr_dma_range(descriptor_dma, descriptor_capacity))
        return FZ_JOB_INVALID;
    for (i = 0; i < count; ++i) {
        const struct fz_job_span *span = &spans[i];
        while (region <= FZ_ASR_REGION_COUNT && !fz_asr_operation_region_bytes(call_id, operation, region)) ++region;
        extent = fz_asr_operation_region_bytes(call_id, operation, region);
        if (!extent || span->logical_offset != FZ_ASR_REGION_BASE(region) + used ||
            span->port != 0 || span->access != fz_asr_region_access(region) ||
            span->bytes > extent - used || !fz_aligned(span->dma_address | span->bytes) ||
            !fz_asr_dma_range(span->dma_address, span->bytes) ||
            fz_asr_overlap(span->dma_address, span->bytes, descriptor_dma, descriptor_capacity))
            return FZ_JOB_INVALID;
        for (j = 0; j < i; ++j)
            if (fz_asr_overlap(span->dma_address, span->bytes, spans[j].dma_address, spans[j].bytes))
                return FZ_JOB_INVALID;
        used += (fz_u32)span->bytes;
        if (used == extent) { ++region; used = 0; }
    }
    while (region <= FZ_ASR_REGION_COUNT && !fz_asr_operation_region_bytes(call_id, operation, region)) ++region;
    if (region <= FZ_ASR_REGION_COUNT || used) return FZ_JOB_INVALID;
    fz_job_put32(packet, FZ_JOB_MAGIC);
    fz_job_put32(packet + 4, FZ_ASR_JOB_VERSION);
    fz_job_put32(packet + 8, bytes);
    fz_job_put32(packet + 12, count);
    fz_job_put64(packet + 16, meta->generation);
    fz_job_put64(packet + 24, meta->chain_id);
    fz_job_put64(packet + 32, meta->profile.epoch);
    fz_job_put32(packet + 40, meta->profile.id);
    fz_job_put32(packet + 44, meta->profile.mask);
    for (i = 0; i < 32; ++i) {
        packet[48+i] = capability[i];
        packet[80+i] = call->parameters_sha256[i];
    }
    fz_job_put32(packet + 112, operation);
    fz_job_put32(packet + 116, rows);
    fz_job_put32(packet + 120, call_id | (state_slot << 16));
    fz_job_put32(packet + 124, sequence);
    for (i = 0; i < count; ++i) {
        unsigned char *entry = packet + FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * i;
        fz_job_put64(entry, spans[i].logical_offset);
        fz_job_put64(entry + 8, spans[i].dma_address);
        fz_job_put64(entry + 16, spans[i].bytes);
        fz_job_put32(entry + 24, spans[i].access);
        fz_job_put32(entry + 28, spans[i].port);
    }
    *written = bytes;
    return FZ_JOB_OK;
}

static inline int fz_asr_job_encode(unsigned char *packet, fz_u32 capacity,
        const struct fz_job_meta *meta, fz_u32 call_id, fz_u32 operation,
        fz_u32 rows, fz_u32 state_slot, fz_u32 sequence,
        const unsigned char parameters_sha256[32], const struct fz_job_span *spans,
        fz_u32 count, fz_u64 descriptor_dma, fz_u64 descriptor_capacity, fz_u32 *written)
{
    return fz_asr_job_encode_lane(packet, capacity, meta, call_id, operation, rows, state_slot, sequence,
                                  parameters_sha256, spans, count, descriptor_dma, descriptor_capacity, written, 0, 0);
}
#endif

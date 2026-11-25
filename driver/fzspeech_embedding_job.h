/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_EMBEDDING_JOB_H
#define FZSPEECH_EMBEDDING_JOB_H

#include "fzspeech_job.h"

/* Private wire ABI for the frozen normal-mode FZASR015 Embedding.
 * The caller supplies DMA-mapped spans and the extent of its own descriptor
 * allocation. Nothing in the packet is published until validation succeeds. */
static inline int fz_embedding_job_encode(unsigned char *packet, fz_u32 capacity,
        const struct fz_job_meta *meta, const struct fz_job_span *spans, fz_u32 count,
        fz_u64 dma_mask, fz_u64 descriptor_dma, fz_u64 descriptor_capacity, fz_u32 *written)
{
    static const fz_u32 region_bytes[] = {688128, 1696, 1696, 12480, 16384};
    static const unsigned char layout[32] = {
        0x13,0x37,0xd2,0x6a,0x7c,0xe0,0x36,0x96,0x77,0x8c,0x6c,0x5d,0x8d,0x7a,0xf2,0x03,
        0xc6,0x9c,0x84,0x87,0x29,0x05,0x0e,0x23,0xda,0xb9,0x07,0x84,0x24,0xc2,0x64,0xd3};
    static const unsigned char pack[32] = {
        0x67,0x30,0x5a,0x7e,0x68,0x86,0x6b,0xda,0x99,0xca,0xb9,0x55,0x4d,0xcb,0xf7,0xdd,
        0x0c,0x37,0x59,0x20,0x58,0xc9,0x24,0x2d,0x10,0xd8,0x96,0x33,0x25,0x9e,0x4c,0x1b};
    fz_u32 i, j, bytes, region = 0;
    fz_u64 used = 0, descriptor_last;
    if (!packet || !meta || !spans || !written || !count || count > FZ_JOB_MAX_SPANS ||
        !meta->generation || !meta->chain_id || !meta->profile.epoch ||
        !(meta->profile.mask & 1U) || fz_profile_mask(meta->profile.id) != meta->profile.mask)
        return FZ_JOB_INVALID;
    bytes = FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * count;
    if (capacity < bytes || descriptor_capacity < bytes) return FZ_JOB_CAPACITY;
    if (!fz_aligned(descriptor_dma) || descriptor_dma > dma_mask ||
        descriptor_capacity - 1 > dma_mask - descriptor_dma) return FZ_JOB_INVALID;
    descriptor_last = descriptor_dma + (descriptor_capacity - 1);
    for (i = 0; i < count; ++i) {
        const struct fz_job_span *s = &spans[i];
        fz_u64 last;
        if (region >= 5 || s->logical_offset != ((fz_u64)(region + 1) << 20) + used ||
            !s->bytes || s->bytes > region_bytes[region] - used ||
            !fz_aligned(s->dma_address | s->bytes) || s->port != 0 ||
            s->access != (region == 4 ? FZ_ACCESS_WRITE : FZ_ACCESS_READ) ||
            s->dma_address > dma_mask || s->bytes - 1 > dma_mask - s->dma_address)
            return FZ_JOB_INVALID;
        last = s->dma_address + (s->bytes - 1);
        if (s->dma_address <= descriptor_last && descriptor_dma <= last) return FZ_JOB_INVALID;
        for (j = 0; j < i; ++j) {
            const struct fz_job_span *prior = &spans[j];
            if (prior->dma_address <= last &&
                s->dma_address <= prior->dma_address + (prior->bytes - 1)) return FZ_JOB_INVALID;
        }
        used += s->bytes;
        if (used == region_bytes[region]) { ++region; used = 0; }
    }
    if (region != 5) return FZ_JOB_INVALID;
    fz_job_put32(packet, FZ_JOB_MAGIC);
    fz_job_put32(packet + 4, 2);
    fz_job_put32(packet + 8, bytes);
    fz_job_put32(packet + 12, count);
    fz_job_put64(packet + 16, meta->generation);
    fz_job_put64(packet + 24, meta->chain_id);
    fz_job_put64(packet + 32, meta->profile.epoch);
    fz_job_put32(packet + 40, meta->profile.id);
    fz_job_put32(packet + 44, meta->profile.mask);
    for (i = 0; i < 32; ++i) { packet[48 + i] = layout[i]; packet[80 + i] = pack[i]; }
    fz_job_put32(packet + 112, 2);
    fz_job_put32(packet + 116, 16);
    fz_job_put64(packet + 120, 0);
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
#endif

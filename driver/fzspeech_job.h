/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_JOB_H
#define FZSPEECH_JOB_H

#include "fzspeech_descriptor.h"

#define FZ_JOB_MAGIC 0x465a5347U
#define FZ_JOB_VERSION 1U
#define FZ_JOB_HEADER_BYTES 128U
#define FZ_JOB_ENTRY_BYTES 32U
#define FZ_JOB_MAX_SPANS 256U
#define FZ_JOB_MAX_PACKET_BYTES (FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * FZ_JOB_MAX_SPANS)
#define FZ_JOB_PAYLOAD_BYTES 17408U
#define FZ_JOB_VALUES_BYTES 16384U
#define FZ_JOB_TILES 4U
#define FZ_JOB_SCALE_BASE 1024U
#define FZ_JOB_WRITE_BASE 1088U
#define FZ_JOB_OK 0
#define FZ_JOB_INVALID 1
#define FZ_JOB_UNSUPPORTED 2
#define FZ_JOB_CAPACITY 3

struct fz_job_span {
    fz_u64 logical_offset, dma_address, bytes;
    fz_u32 access, port;
};

/* Private provenance; not sent as a user or hardware DMA address. The hardware
 * failed-span index refers to this array after canonical sorting. */
struct fz_job_part {
    struct fz_job_span span;
    fz_u32 transfer_index, dma_span_index;
    fz_u64 job_id;
};

struct fz_job_meta {
    fz_u64 generation, chain_id;
    struct fz_profile profile;
    fz_u32 write_enable;
};

static inline fz_u32 fz_job_get32(const unsigned char *p)
{
    return (fz_u32)p[0] | ((fz_u32)p[1] << 8) | ((fz_u32)p[2] << 16) | ((fz_u32)p[3] << 24);
}

static inline fz_u64 fz_job_get64(const unsigned char *p)
{
    return (fz_u64)fz_job_get32(p) | ((fz_u64)fz_job_get32(p + 4) << 32);
}

static inline void fz_job_put32(unsigned char *p, fz_u32 value)
{
    unsigned int n;
    for (n = 0; n < 4; ++n) p[n] = (unsigned char)(value >> (8 * n));
}

static inline void fz_job_put64(unsigned char *p, fz_u64 value)
{
    fz_job_put32(p, (fz_u32)value);
    fz_job_put32(p + 4, (fz_u32)(value >> 32));
}

static inline fz_u32 fz_job_crc32(const void *data, fz_u32 bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    fz_u32 crc = 0xffffffffU, n, bit;
    for (n = 0; n < bytes; ++n) {
        crc ^= p[n];
        for (bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc ^ 0xffffffffU;
}

static inline fz_u64 fz_job_expected_read_bytes(fz_u32 mask, fz_u32 port)
{
    fz_u32 pair;
    fz_u64 bytes = 0;
    for (pair = 0; pair < 8; ++pair)
        if (fz_pair_port(mask, pair) == port) bytes += FZ_JOB_PAYLOAD_BYTES / 8;
    return bytes;
}

static inline int fz_job_snapshot_matches(const struct fz_transport_stats *stats,
                                         const struct fz_job_meta *meta, fz_u32 held_allowed)
{
    fz_u32 port;
    if (!stats || !meta || !meta->generation || !meta->chain_id || !meta->profile.epoch ||
        !meta->profile.mask || fz_profile_mask(meta->profile.id) != meta->profile.mask ||
        meta->write_enable > 1 || stats->generation != meta->generation ||
        (stats->chain_id && (!held_allowed || stats->chain_id != meta->chain_id)) ||
        stats->profile.id != meta->profile.id || stats->profile.mask != meta->profile.mask ||
        stats->profile.epoch != meta->profile.epoch ||
        (stats->status != FZ_STATUS_PENDING && stats->status != FZ_STATUS_SUCCESS) ||
        !(stats->flags & 1U) || stats->fifo_empty_mask != 15U ||
        (stats->read_limit != 8 && stats->read_limit != 16) ||
        (stats->write_limit != 8 && stats->write_limit != 16)) return FZ_JOB_INVALID;
    for (port = 0; port < 4; ++port)
        if (stats->ports[port].read_outstanding || stats->ports[port].write_outstanding ||
            stats->ports[port].pending_flags || stats->ports[port].read_peak > 16 ||
            stats->ports[port].write_peak > 16) return FZ_JOB_INVALID;
    return FZ_JOB_OK;
}

/* Exact physical-counter proof for this fixed operation. Metadata is fetched
 * once on HP0; HLS payload counts remain distinct. Lifetime maxima can exceed
 * a newly lowered limit, but cannot newly exceed a stable limit in this job. */
static inline int fz_job_verify_counters(const struct fz_transport_stats *before,
                                        const struct fz_transport_stats *after,
                                        const struct fz_job_meta *meta, fz_u32 spans)
{
    fz_u32 port, packet_bytes = FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * spans;
    fz_u64 reads, writes;
    const struct fz_port_stats *a, *b;
    if (!before || !after || !meta) return FZ_JOB_INVALID;
    if (!spans || spans > FZ_JOB_MAX_SPANS || spans < (meta->write_enable ? 128U : 64U) ||
        fz_job_snapshot_matches(before, meta, 0) || fz_job_snapshot_matches(after, meta, 1) ||
        before->counter_epoch != after->counter_epoch || before->snapshot_sequence == after->snapshot_sequence ||
        before->read_limit != after->read_limit || before->write_limit != after->write_limit)
        return FZ_JOB_INVALID;
    for (port = 0; port < 4; ++port) {
        a = &before->ports[port]; b = &after->ports[port];
        reads = fz_job_expected_read_bytes(meta->profile.mask, port);
        writes = meta->write_enable ? reads : 0;
        if (port == 0) reads += packet_bytes;
        if (b->read_bytes < a->read_bytes || b->write_bytes < a->write_bytes ||
            b->read_bytes - a->read_bytes != reads || b->write_bytes - a->write_bytes != writes ||
            b->ar_stall < a->ar_stall || b->r_stall < a->r_stall || b->aw_stall < a->aw_stall ||
            b->w_stall < a->w_stall || b->b_stall < a->b_stall ||
            b->read_peak < a->read_peak || b->write_peak < a->write_peak ||
            b->read_peak > (a->read_peak > before->read_limit ? a->read_peak : before->read_limit) ||
            b->write_peak > (a->write_peak > before->write_limit ? a->write_peak : before->write_limit) ||
            (reads && !b->read_peak) || (writes && !b->write_peak)) return FZ_JOB_INVALID;
    }
    return FZ_JOB_OK;
}

/* The retained HLS probe has exactly 32 value blocks and 32 scale blocks. */
static inline int fz_job_block(fz_u64 offset, fz_u64 bytes, fz_u32 *block, fz_u32 *pair)
{
    fz_u64 within;
    if (offset < FZ_JOB_VALUES_BYTES) {
        if (bytes != 512 || (offset & 511U)) return FZ_JOB_UNSUPPORTED;
        *block = (fz_u32)(offset / 512);
        *pair = *block % 8;
    } else {
        within = offset - FZ_JOB_VALUES_BYTES;
        if (bytes != 32 || (within & 31U) || within >= 1024) return FZ_JOB_UNSUPPORTED;
        *block = 32 + (fz_u32)(within / 32);
        *pair = (fz_u32)(within / 32) % 8;
    }
    return *block < 64 ? FZ_JOB_OK : FZ_JOB_UNSUPPORTED;
}

/* Userspace convenience for the existing SUBMIT_CHAIN ABI. This does not add
 * physical addresses to that ABI or change generic transfer validation. */
static inline int fz_job_build_transfers(fz_u64 input_handle, fz_u64 input_offset,
                                        fz_u64 output_handle, fz_u64 output_offset,
                                        fz_u32 write_enable, fz_u64 first_job,
                                        struct fz_transfer *entries, fz_u32 capacity,
                                        fz_u32 *count)
{
    fz_u32 block, n = 0, needed = write_enable ? 128 : 64;
    fz_u64 offset, bytes;
    if (!entries || !count || !input_handle || write_enable > 1 || !first_job ||
        (write_enable && (!output_handle || input_handle == output_handle)) ||
        !fz_aligned(input_offset | output_offset) ||
        input_offset > (fz_u64)-1 - FZ_JOB_PAYLOAD_BYTES ||
        (write_enable && output_offset > (fz_u64)-1 - FZ_JOB_PAYLOAD_BYTES) ||
        first_job > (fz_u64)-1 - (needed - 1)) return FZ_JOB_INVALID;
    if (capacity < needed) return FZ_JOB_CAPACITY;
    for (block = 0; block < 64; ++block) {
        offset = block < 32 ? (fz_u64)block * 512 : FZ_JOB_VALUES_BYTES + (fz_u64)(block - 32) * 32;
        bytes = block < 32 ? 512 : 32;
        entries[n].job_id = first_job + n;
        entries[n].handle = input_handle;
        entries[n].offset = input_offset + offset;
        entries[n].bytes = bytes;
        entries[n].pair = block % 8;
        entries[n].access = FZ_ACCESS_READ;
        entries[n].depends_on = FZ_NO_DEPENDENCY;
        entries[n].reserved = 0;
        ++n;
        if (write_enable) {
            entries[n].job_id = first_job + n;
            entries[n].handle = output_handle;
            entries[n].offset = output_offset + offset;
            entries[n].bytes = bytes;
            entries[n].pair = block % 8;
            entries[n].access = FZ_ACCESS_WRITE;
            entries[n].depends_on = n - 1;
            entries[n].reserved = 0;
            ++n;
        }
    }
    *count = n;
    return FZ_JOB_OK;
}

/* Convert core-validated mapped SG spans to the existing HLS logical image.
 * Both source and destination retain their independent DMA segmentation and
 * guard offsets. Caller supplies private persistent storage, not kernel stack. */
static inline int fz_job_plan(const struct fz_chain *chain, const struct fz_transfer *entries,
                             const struct fz_dma_span *mapped, fz_u32 span_count,
                             struct fz_job_part *parts, fz_u32 capacity, struct fz_job_meta *meta)
{
    fz_u64 input_base = (fz_u64)-1, output_base = (fz_u64)-1;
    fz_u64 input_handle = 0, output_handle = 0, last_job = 0;
    fz_u64 progress[128] = {0};
    fz_u32 read_index[64];
    unsigned char write_seen[64] = {0};
    fz_u32 i, j, reads = 0, writes = 0, block, pair;
    fz_u64 base, logical;
    struct fz_job_part saved;
    const struct fz_transfer *entry;
    const struct fz_dma_span *span;
    if (!chain || !entries || !mapped || !parts || !meta || !chain->generation || !chain->chain_id ||
        !chain->profile.epoch || !chain->profile.mask ||
        fz_profile_mask(chain->profile.id) != chain->profile.mask || chain->reserved ||
        chain->descriptor_version != FZ_DESCRIPTOR_VERSION ||
        (chain->descriptor_count != 64 && chain->descriptor_count != 128) ||
        chain->chain_bytes != sizeof(*chain) + chain->descriptor_count * sizeof(*entries) ||
        !span_count) return FZ_JOB_INVALID;
    if (span_count > capacity || span_count > FZ_JOB_MAX_SPANS) return FZ_JOB_CAPACITY;
    for (i = 0; i < 64; ++i) read_index[i] = FZ_NO_DEPENDENCY;
    for (i = 0; i < chain->descriptor_count; ++i) {
        entry = &entries[i];
        if (!entry->handle || entry->reserved || entry->job_id <= last_job ||
            entry->offset > (fz_u64)-1 - entry->bytes ||
            !fz_aligned(entry->offset | entry->bytes)) return FZ_JOB_INVALID;
        last_job = entry->job_id;
        if (entry->access == FZ_ACCESS_READ) {
            if (input_handle && input_handle != entry->handle) return FZ_JOB_UNSUPPORTED;
            input_handle = entry->handle;
            if (entry->offset < input_base) input_base = entry->offset;
            ++reads;
        } else if (entry->access == FZ_ACCESS_WRITE) {
            if (output_handle && output_handle != entry->handle) return FZ_JOB_UNSUPPORTED;
            output_handle = entry->handle;
            if (entry->offset < output_base) output_base = entry->offset;
            ++writes;
        } else return FZ_JOB_UNSUPPORTED;
    }
    if (reads != 64 || (writes && writes != 64) || input_handle == output_handle)
        return FZ_JOB_UNSUPPORTED;
    for (i = 0; i < chain->descriptor_count; ++i) {
        entry = &entries[i];
        base = entry->access == FZ_ACCESS_READ ? input_base : output_base;
        logical = entry->offset - base;
        if (!fz_range(logical, entry->bytes, FZ_JOB_PAYLOAD_BYTES) ||
            fz_job_block(logical, entry->bytes, &block, &pair) || entry->pair != pair)
            return FZ_JOB_UNSUPPORTED;
        if (entry->access == FZ_ACCESS_READ) {
            if (read_index[block] != FZ_NO_DEPENDENCY || entry->depends_on != FZ_NO_DEPENDENCY)
                return FZ_JOB_UNSUPPORTED;
            read_index[block] = i;
        } else {
            if (write_seen[block] || read_index[block] == FZ_NO_DEPENDENCY ||
                entry->depends_on != read_index[block]) return FZ_JOB_UNSUPPORTED;
            write_seen[block] = 1;
        }
    }
    for (i = 0; i < span_count; ++i) {
        span = &mapped[i];
        if (span->transfer_index >= chain->descriptor_count) return FZ_JOB_INVALID;
        entry = &entries[span->transfer_index];
        if (!span->bytes || progress[span->transfer_index] >= entry->bytes ||
            span->bytes > entry->bytes - progress[span->transfer_index] ||
            span->logical_offset != entry->offset + progress[span->transfer_index] ||
            span->port != fz_pair_port(chain->profile.mask, entry->pair)) return FZ_JOB_INVALID;
        progress[span->transfer_index] += span->bytes;
        base = entry->access == FZ_ACCESS_READ ? input_base : output_base;
        parts[i].span.logical_offset = span->logical_offset - base +
            (entry->access == FZ_ACCESS_WRITE ? FZ_JOB_PAYLOAD_BYTES : 0);
        parts[i].span.dma_address = span->dma_address;
        parts[i].span.bytes = span->bytes;
        parts[i].span.access = entry->access;
        parts[i].span.port = span->port;
        parts[i].transfer_index = span->transfer_index;
        parts[i].dma_span_index = i;
        parts[i].job_id = entry->job_id;
    }
    for (i = 0; i < chain->descriptor_count; ++i)
        if (progress[i] != entries[i].bytes) return FZ_JOB_INVALID;
    for (i = 1; i < span_count; ++i) {
        saved = parts[i]; j = i;
        while (j && parts[j - 1].span.logical_offset > saved.span.logical_offset) {
            parts[j] = parts[j - 1]; --j;
        }
        parts[j] = saved;
    }
    meta->generation = chain->generation; meta->chain_id = chain->chain_id;
    meta->profile = chain->profile; meta->write_enable = writes ? 1 : 0;
    return FZ_JOB_OK;
}

static inline int fz_job_encode(void *packet, fz_u32 capacity, const struct fz_job_meta *meta,
                               const struct fz_job_part *parts, fz_u32 count,
                               fz_u64 dma_mask, fz_u32 *packet_bytes)
{
    static const unsigned char layout[32] = {0x52,0xd8,0xad,0x6c,0x73,0x81,0xdd,0x75,0x33,0x55,0x3c,0x3b,0x9a,0x98,0x00,0xe9,0x3a,0x4a,0x48,0xbe,0x6e,0xac,0x9f,0x5e,0xff,0x75,0x7a,0x4e,0xb2,0xf0,0x3c,0x22};
    static const unsigned char pack[32] = {0x9e,0xf5,0x60,0x94,0xec,0x79,0x5a,0xce,0xbb,0x3f,0x07,0x29,0x1a,0x56,0x9f,0x19,0x11,0xa6,0x7c,0x1e,0x94,0x65,0x92,0xb7,0x61,0x0f,0x66,0x2d,0x75,0x95,0x4f,0x7e};
    unsigned char *p = (unsigned char *)packet;
    fz_u32 bytes, i, j, pair, access;
    fz_u64 next = 0, local, boundary, end;
    const struct fz_job_span *span;
    const struct fz_job_span *other;
    if (!p || !meta || !parts || !packet_bytes || !count || count > FZ_JOB_MAX_SPANS ||
        !meta->generation || !meta->chain_id || !meta->profile.epoch || meta->write_enable > 1 ||
        !meta->profile.mask || fz_profile_mask(meta->profile.id) != meta->profile.mask)
        return FZ_JOB_INVALID;
    bytes = FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * count;
    if (capacity < bytes) return FZ_JOB_CAPACITY;
    for (i = 0; i < count; ++i) {
        span = &parts[i].span;
        access = next < FZ_JOB_PAYLOAD_BYTES ? FZ_ACCESS_READ : FZ_ACCESS_WRITE;
        local = next % FZ_JOB_PAYLOAD_BYTES;
        if (local < FZ_JOB_VALUES_BYTES) {
            boundary = 512 - local % 512;
            pair = (fz_u32)(local / 512) % 8;
        } else {
            boundary = 32 - (local - FZ_JOB_VALUES_BYTES) % 32;
            pair = (fz_u32)((local - FZ_JOB_VALUES_BYTES) / 32) % 8;
        }
        end = (fz_u64)FZ_JOB_PAYLOAD_BYTES * (meta->write_enable ? 2 : 1);
        if (next >= end || span->logical_offset != next || !span->bytes ||
            span->bytes > boundary || span->bytes > end - next || span->access != access ||
            span->port != fz_pair_port(meta->profile.mask, pair) ||
            !fz_aligned(span->logical_offset | span->dma_address | span->bytes) ||
            span->dma_address > dma_mask || span->bytes - 1 > dma_mask - span->dma_address)
            return FZ_JOB_INVALID;
        for (j = 0; j < i; ++j) {
            other = &parts[j].span;
            if (((span->access | other->access) & FZ_ACCESS_WRITE) &&
                (span->dma_address <= other->dma_address ?
                 other->dma_address - span->dma_address < span->bytes :
                 span->dma_address - other->dma_address < other->bytes))
                return FZ_JOB_INVALID;
        }
        next += span->bytes;
    }
    if (next != (fz_u64)FZ_JOB_PAYLOAD_BYTES * (meta->write_enable ? 2 : 1)) return FZ_JOB_INVALID;
    fz_job_put32(p, FZ_JOB_MAGIC); fz_job_put32(p + 4, FZ_JOB_VERSION);
    fz_job_put32(p + 8, bytes); fz_job_put32(p + 12, count);
    fz_job_put64(p + 16, meta->generation); fz_job_put64(p + 24, meta->chain_id);
    fz_job_put64(p + 32, meta->profile.epoch); fz_job_put32(p + 40, meta->profile.id);
    fz_job_put32(p + 44, meta->profile.mask);
    for (i = 0; i < 32; ++i) {p[48 + i] = layout[i]; p[80 + i] = pack[i];}
    fz_job_put32(p + 112, FZ_JOB_TILES); fz_job_put32(p + 116, FZ_JOB_SCALE_BASE);
    fz_job_put32(p + 120, FZ_JOB_WRITE_BASE); fz_job_put32(p + 124, meta->write_enable);
    for (i = 0; i < count; ++i) {
        unsigned char *entry = p + FZ_JOB_HEADER_BYTES + i * FZ_JOB_ENTRY_BYTES;
        span = &parts[i].span;
        fz_job_put64(entry, span->logical_offset); fz_job_put64(entry + 8, span->dma_address);
        fz_job_put64(entry + 16, span->bytes); fz_job_put32(entry + 24, span->access);
        fz_job_put32(entry + 28, span->port);
    }
    *packet_bytes = bytes;
    return FZ_JOB_OK;
}

#endif

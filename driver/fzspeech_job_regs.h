/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_JOB_REGS_H
#define FZSPEECH_JOB_REGS_H

#include "fzspeech_job.h"
#include "fzspeech_transport_regs.h"

#define FZ_SQ_ID 0x400U
#define FZ_SQ_ABI 0x404U
#define FZ_SQ_CAP 0x408U
#define FZ_SQ_DMA_BITS 0x40cU
#define FZ_SQ_COMMAND 0x410U
#define FZ_SQ_RESULT 0x414U
#define FZ_SQ_FLAGS 0x418U
#define FZ_SQ_STATUS 0x41cU
#define FZ_SQ_PACKET_LO 0x420U
#define FZ_SQ_PACKET_BYTES 0x428U
#define FZ_SQ_SPAN_COUNT 0x42cU
#define FZ_SQ_GENERATION_REQ_LO 0x430U
#define FZ_SQ_CHAIN_REQ_LO 0x438U
#define FZ_SQ_EPOCH_REQ_LO 0x440U
#define FZ_SQ_PROFILE_ID_REQ 0x448U
#define FZ_SQ_PROFILE_MASK_REQ 0x44cU
#define FZ_SQ_PACKET_CRC 0x450U
#define FZ_SQ_ADMISSION 0x454U
#define FZ_SQ_ACK_CHAIN_LO 0x458U
#define FZ_SQ_ACK_GENERATION_LO 0x460U
#define FZ_SQ_IRQ_ENABLE 0x468U
#define FZ_SQ_GENERATION_LO 0x470U
#define FZ_SQ_CHAIN_LO 0x478U
#define FZ_SQ_COMPLETED_GENERATION_LO 0x480U
#define FZ_SQ_COMPLETED_CHAIN_LO 0x488U
#define FZ_SQ_COMPLETED_EPOCH_LO 0x490U
#define FZ_SQ_COMPLETED_PROFILE_ID 0x498U
#define FZ_SQ_COMPLETED_PROFILE_MASK 0x49cU
#define FZ_SQ_COMPLETED_STATUS 0x4a0U
#define FZ_SQ_COMPLETED_ERROR 0x4a4U
#define FZ_SQ_FAILED_SPAN 0x4a8U
#define FZ_SQ_DONE_SEQUENCE 0x4acU
#define FZ_SQ_RESULT_BASE(port) (0x500U + 0x20U * (port))
#define FZ_SQ_MIN_MMIO_BYTES 0x570U
#define FZ_SQ_DOORBELL 1U
#define FZ_SQ_ACK 2U
#define FZ_SQ_SET_GENERATION 4U
#define FZ_SQ_ACTIVE 1U
#define FZ_SQ_DONE_VALID 2U
#define FZ_SQ_FAULT 4U
#define FZ_SQ_ADMISSION_FLAG 8U

static inline void fz_sq_write64(const struct fz_transport_register_io *io, fz_u32 offset, fz_u64 value)
{
    io->write(io->context, offset, (fz_u32)value);
    io->write(io->context, offset + 4, (fz_u32)(value >> 32));
}

static inline int fz_sq_probe(const struct fz_transport_register_io *io, fz_u32 dma_bits)
{
    return fz_tr_probe(io) && dma_bits && dma_bits <= 64 &&
        io->read(io->context, FZ_SQ_ID) == FZ_JOB_MAGIC &&
        io->read(io->context, FZ_SQ_ABI) == FZ_JOB_VERSION &&
        (io->read(io->context, FZ_SQ_CAP) & 7U) == 7U &&
        io->read(io->context, FZ_SQ_DMA_BITS) == dma_bits;
}

static inline int fz_sq_set_admission(const struct fz_transport_register_io *io, fz_u32 enabled)
{
    if (enabled > 1 || (enabled && (io->read(io->context, FZ_SQ_FLAGS) & FZ_SQ_FAULT)))
        return FZ_TR_IO_ERROR;
    io->write(io->context, FZ_SQ_ADMISSION, enabled);
    return io->read(io->context, FZ_SQ_ADMISSION) == enabled ? FZ_TR_OK : FZ_TR_IO_ERROR;
}

static inline int fz_sq_set_generation(const struct fz_transport_register_io *io, fz_u64 generation)
{
    if (!generation || io->read(io->context, FZ_TR_RUN) ||
        io->read(io->context, FZ_SQ_ADMISSION) ||
        (io->read(io->context, FZ_SQ_FLAGS) & (FZ_SQ_ACTIVE | FZ_SQ_DONE_VALID | FZ_SQ_FAULT)) ||
        generation <= fz_tr_read64(io, FZ_SQ_GENERATION_LO)) return FZ_TR_INVALID;
    fz_sq_write64(io, FZ_SQ_GENERATION_REQ_LO, generation);
    io->write(io->context, FZ_SQ_COMMAND, FZ_SQ_SET_GENERATION);
    if (io->read(io->context, FZ_SQ_RESULT) != FZ_TR_OK ||
        fz_tr_read64(io, FZ_SQ_GENERATION_LO) != generation ||
        (io->read(io->context, FZ_SQ_FLAGS) & (FZ_SQ_ACTIVE | FZ_SQ_DONE_VALID | FZ_SQ_FAULT)))
        return FZ_TR_IO_ERROR;
    return FZ_TR_OK;
}

/* Optional witness from the same ordered accesses as the command. */
struct fz_sq_observation {
    fz_u32 written;
    const char *failed_check;
    fz_u64 expected, observed;
};

static inline int fz_sq_check(struct fz_sq_observation *note, const char *check,
                              fz_u64 expected, fz_u64 observed)
{
    if (expected == observed) return 1;
    if (note) {
        note->failed_check = check;
        note->expected = expected;
        note->observed = observed;
    }
    return 0;
}

static inline int fz_sq_publish_observed(const struct fz_transport_register_io *io, fz_u64 packet_dma,
                                fz_u32 bytes, fz_u32 count, fz_u32 crc,
                                const struct fz_job_meta *meta, struct fz_sq_observation *note)
{
    fz_u32 flags;
    const fz_u32 active = FZ_SQ_ACTIVE | FZ_SQ_DONE_VALID | FZ_SQ_FAULT;
    if (!fz_sq_check(note, "descriptor-valid", 1, meta && count && count <= FZ_JOB_MAX_SPANS &&
                     fz_aligned(packet_dma) && bytes == FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * count) ||
        !fz_sq_check(note, "run-nonzero", 1, !!io->read(io->context, FZ_TR_RUN)) ||
        !fz_sq_check(note, "admission-nonzero", 1, !!io->read(io->context, FZ_SQ_ADMISSION)) ||
        !fz_sq_check(note, "generation", meta->generation, fz_tr_read64(io, FZ_SQ_GENERATION_LO)) ||
        !fz_sq_check(note, "flags-before", 0, io->read(io->context, FZ_SQ_FLAGS) & active))
        return FZ_TR_INVALID;
    fz_sq_write64(io, FZ_SQ_PACKET_LO, packet_dma);
    io->write(io->context, FZ_SQ_PACKET_BYTES, bytes);
    io->write(io->context, FZ_SQ_SPAN_COUNT, count);
    fz_sq_write64(io, FZ_SQ_GENERATION_REQ_LO, meta->generation);
    fz_sq_write64(io, FZ_SQ_CHAIN_REQ_LO, meta->chain_id);
    fz_sq_write64(io, FZ_SQ_EPOCH_REQ_LO, meta->profile.epoch);
    io->write(io->context, FZ_SQ_PROFILE_ID_REQ, meta->profile.id);
    io->write(io->context, FZ_SQ_PROFILE_MASK_REQ, meta->profile.mask);
    io->write(io->context, FZ_SQ_PACKET_CRC, crc);
    io->write(io->context, FZ_SQ_COMMAND, FZ_SQ_DOORBELL);
    if (note) note->written = 1;
    flags = io->read(io->context, FZ_SQ_FLAGS);
    if (!fz_sq_check(note, "command-result", FZ_TR_OK, io->read(io->context, FZ_SQ_RESULT)) ||
        !fz_sq_check(note, "active-or-done-or-fault", 1, !!(flags & active)) ||
        !fz_sq_check(note, "generation", meta->generation, fz_tr_read64(io, FZ_SQ_GENERATION_LO)) ||
        !fz_sq_check(note, "chain", meta->chain_id, fz_tr_read64(io, FZ_SQ_CHAIN_LO)))
        return FZ_TR_IO_ERROR;
    return FZ_TR_OK;
}

static inline int fz_sq_publish(const struct fz_transport_register_io *io, fz_u64 packet_dma,
                                fz_u32 bytes, fz_u32 count, fz_u32 crc,
                                const struct fz_job_meta *meta)
{
    return fz_sq_publish_observed(io, packet_dma, bytes, count, crc, meta, 0);
}

/* Completion and result words are held until identity-matched ACK. */
static inline int fz_sq_receipt(const struct fz_transport_register_io *io, struct fz_probe_receipt *receipt)
{
    fz_u32 before, after, attempt, port, flags;
    for (attempt = 0; attempt < 3; ++attempt) {
        before = io->read(io->context, FZ_SQ_DONE_SEQUENCE);
        flags = io->read(io->context, FZ_SQ_FLAGS);
        if (!(flags & (FZ_SQ_DONE_VALID | FZ_SQ_FAULT))) return FZ_TR_BUSY;
        receipt->generation = fz_tr_read64(io, FZ_SQ_COMPLETED_GENERATION_LO);
        receipt->chain_id = fz_tr_read64(io, FZ_SQ_COMPLETED_CHAIN_LO);
        receipt->profile.epoch = fz_tr_read64(io, FZ_SQ_COMPLETED_EPOCH_LO);
        receipt->profile.id = io->read(io->context, FZ_SQ_COMPLETED_PROFILE_ID);
        receipt->profile.mask = io->read(io->context, FZ_SQ_COMPLETED_PROFILE_MASK);
        receipt->status = io->read(io->context, FZ_SQ_COMPLETED_STATUS);
        receipt->error = io->read(io->context, FZ_SQ_COMPLETED_ERROR);
        receipt->failed_span = io->read(io->context, FZ_SQ_FAILED_SPAN);
        receipt->done_sequence = before;
        receipt->failed_transfer = receipt->failed_dma_span = FZ_NO_DEPENDENCY;
        receipt->failed_job_id = 0;
        receipt->packet_bytes = receipt->span_count = 0;
        for (port = 0; port < 4; ++port) {
            receipt->signature[port] = fz_tr_read64(io, FZ_SQ_RESULT_BASE(port));
            receipt->read_bytes[port] = fz_tr_read64(io, FZ_SQ_RESULT_BASE(port) + 8);
        }
        after = io->read(io->context, FZ_SQ_DONE_SEQUENCE);
        if (before == after && flags == io->read(io->context, FZ_SQ_FLAGS)) {
            if (!receipt->generation || !receipt->chain_id || !receipt->profile.epoch ||
                !receipt->profile.mask || fz_profile_mask(receipt->profile.id) != receipt->profile.mask ||
                receipt->status > FZ_STATUS_UNKNOWN || receipt->status == FZ_STATUS_PENDING ||
                receipt->error > 7) return FZ_TR_IO_ERROR;
            return FZ_TR_OK;
        }
    }
    return FZ_TR_IO_ERROR;
}

static inline int fz_sq_ack_observed(const struct fz_transport_register_io *io, fz_u64 generation,
                                     fz_u64 chain, struct fz_sq_observation *note)
{
    fz_u32 flags = io->read(io->context, FZ_SQ_FLAGS);
    const fz_u32 mask = FZ_SQ_ACTIVE | FZ_SQ_DONE_VALID | FZ_SQ_FAULT;
    if (!fz_sq_check(note, "flags-before", FZ_SQ_DONE_VALID, flags & mask) ||
        !fz_sq_check(note, "completed-generation", generation, fz_tr_read64(io, FZ_SQ_COMPLETED_GENERATION_LO)) ||
        !fz_sq_check(note, "completed-chain", chain, fz_tr_read64(io, FZ_SQ_COMPLETED_CHAIN_LO)))
        return FZ_TR_INVALID;
    fz_sq_write64(io, FZ_SQ_ACK_CHAIN_LO, chain);
    fz_sq_write64(io, FZ_SQ_ACK_GENERATION_LO, generation);
    io->write(io->context, FZ_SQ_COMMAND, FZ_SQ_ACK);
    if (note) note->written = 1;
    if (!fz_sq_check(note, "command-result", FZ_TR_OK, io->read(io->context, FZ_SQ_RESULT)) ||
        !fz_sq_check(note, "flags-after", 0, io->read(io->context, FZ_SQ_FLAGS) & mask) ||
        !fz_sq_check(note, "chain-after", 0, fz_tr_read64(io, FZ_SQ_CHAIN_LO)))
        return FZ_TR_IO_ERROR;
    return FZ_TR_OK;
}

static inline int fz_sq_ack(const struct fz_transport_register_io *io, fz_u64 generation, fz_u64 chain)
{
    return fz_sq_ack_observed(io, generation, chain, 0);
}

#endif

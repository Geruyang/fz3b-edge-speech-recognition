/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_TRANSPORT_REGS_H
#define FZSPEECH_TRANSPORT_REGS_H

#include "fzspeech_descriptor.h"

#define FZ_TR_CORE_ID 0x465a5452U
#define FZ_TR_ABI 1U
#define FZ_TR_CAPS 0x0fU
#define FZ_TR_ID 0x000U
#define FZ_TR_VERSION 0x004U
#define FZ_TR_CAPABILITIES 0x008U
#define FZ_TR_RUN 0x00cU
#define FZ_TR_COMMAND 0x010U
#define FZ_TR_RESULT 0x014U
#define FZ_TR_REQUEST_ID 0x018U
#define FZ_TR_REQUEST_MASK 0x01cU
#define FZ_TR_EXPECTED_EPOCH_LO 0x020U
#define FZ_TR_EXPECTED_EPOCH_HI 0x024U
#define FZ_TR_REQUEST_READ_LIMIT 0x028U
#define FZ_TR_REQUEST_WRITE_LIMIT 0x02cU
#define FZ_TR_SNAPSHOT_SEQUENCE 0x040U
#define FZ_TR_PROFILE_ID 0x044U
#define FZ_TR_PROFILE_MASK 0x048U
#define FZ_TR_PROFILE_EPOCH_LO 0x04cU
#define FZ_TR_GENERATION_LO 0x054U
#define FZ_TR_CHAIN_LO 0x05cU
#define FZ_TR_STATUS 0x064U
#define FZ_TR_FLAGS 0x068U
#define FZ_TR_FIFO_EMPTY_MASK 0x06cU
#define FZ_TR_COUNTER_EPOCH_LO 0x070U
#define FZ_TR_READ_LIMIT 0x078U
#define FZ_TR_WRITE_LIMIT 0x07cU
#define FZ_TR_PORT_BASE(port) (0x100U + 0x80U * (port))
#define FZ_TR_PORT_READ_OUTSTANDING 0x00U
#define FZ_TR_PORT_WRITE_OUTSTANDING 0x04U
#define FZ_TR_PORT_READ_PEAK 0x08U
#define FZ_TR_PORT_WRITE_PEAK 0x0cU
#define FZ_TR_PORT_READ_BYTES 0x10U
#define FZ_TR_PORT_WRITE_BYTES 0x18U
#define FZ_TR_PORT_AR_STALL 0x20U
#define FZ_TR_PORT_R_STALL 0x28U
#define FZ_TR_PORT_AW_STALL 0x30U
#define FZ_TR_PORT_W_STALL 0x38U
#define FZ_TR_PORT_B_STALL 0x40U
#define FZ_TR_PORT_PENDING 0x48U
#define FZ_TR_CMD_SNAPSHOT 1U
#define FZ_TR_CMD_APPLY 2U
#define FZ_TR_FLAG_ENGINE_IDLE 1U
#define FZ_TR_FLAG_ISSUANCE_STOPPED 2U
#define FZ_TR_PROTOCOL_FAULT 8U
#define FZ_TR_OK 0
#define FZ_TR_BUSY 1
#define FZ_TR_MISMATCH 2
#define FZ_TR_INVALID 3
#define FZ_TR_IO_ERROR 4

/* read/write must be ordered MMIO accesses. A write returns only once accepted
 * by cfg/AXI-Lite; all commands are serialized with descriptor publication.
 */
struct fz_transport_register_io {
    void *context;
    fz_u32 (*read)(void *context, fz_u32 offset);
    void (*write)(void *context, fz_u32 offset, fz_u32 value);
};

static inline fz_u64 fz_tr_read64(const struct fz_transport_register_io *io, fz_u32 offset)
{
    fz_u64 low = io->read(io->context, offset);
    return low | ((fz_u64)io->read(io->context, offset + 4) << 32);
}

static inline int fz_tr_limit_valid(fz_u32 value)
{
    return value == 8 || value == 16;
}

static inline int fz_tr_probe(const struct fz_transport_register_io *io)
{
    return io && io->read && io->write &&
        io->read(io->context, FZ_TR_ID) == FZ_TR_CORE_ID &&
        io->read(io->context, FZ_TR_VERSION) == FZ_TR_ABI &&
        (io->read(io->context, FZ_TR_CAPABILITIES) & FZ_TR_CAPS) == FZ_TR_CAPS;
}

static inline int fz_tr_snapshot(const struct fz_transport_register_io *io,
                                 struct fz_transport_stats *stats)
{
    unsigned int attempt, poll, port;
    fz_u32 before, sequence;

    for (attempt = 0; attempt < 3; ++attempt) {
        before = io->read(io->context, FZ_TR_SNAPSHOT_SEQUENCE);
        io->write(io->context, FZ_TR_COMMAND, FZ_TR_CMD_SNAPSHOT);
        for (poll = 0; poll < 128; ++poll) {
            sequence = io->read(io->context, FZ_TR_SNAPSHOT_SEQUENCE);
            if (sequence != before)
                break;
        }
        if (poll == 128)
            return FZ_TR_IO_ERROR;
        stats->snapshot_sequence = sequence;
        stats->profile.id = io->read(io->context, FZ_TR_PROFILE_ID);
        stats->profile.mask = io->read(io->context, FZ_TR_PROFILE_MASK);
        stats->profile.epoch = fz_tr_read64(io, FZ_TR_PROFILE_EPOCH_LO);
        stats->generation = fz_tr_read64(io, FZ_TR_GENERATION_LO);
        stats->chain_id = fz_tr_read64(io, FZ_TR_CHAIN_LO);
        stats->status = io->read(io->context, FZ_TR_STATUS);
        stats->flags = io->read(io->context, FZ_TR_FLAGS);
        stats->fifo_empty_mask = io->read(io->context, FZ_TR_FIFO_EMPTY_MASK);
        stats->counter_epoch = fz_tr_read64(io, FZ_TR_COUNTER_EPOCH_LO);
        stats->read_limit = io->read(io->context, FZ_TR_READ_LIMIT);
        stats->write_limit = io->read(io->context, FZ_TR_WRITE_LIMIT);
        for (port = 0; port < 4; ++port) {
            struct fz_port_stats *p = &stats->ports[port];
            fz_u32 base = FZ_TR_PORT_BASE(port);
            p->read_outstanding = io->read(io->context, base + FZ_TR_PORT_READ_OUTSTANDING);
            p->write_outstanding = io->read(io->context, base + FZ_TR_PORT_WRITE_OUTSTANDING);
            p->read_peak = io->read(io->context, base + FZ_TR_PORT_READ_PEAK);
            p->write_peak = io->read(io->context, base + FZ_TR_PORT_WRITE_PEAK);
            p->read_bytes = fz_tr_read64(io, base + FZ_TR_PORT_READ_BYTES);
            p->write_bytes = fz_tr_read64(io, base + FZ_TR_PORT_WRITE_BYTES);
            p->ar_stall = fz_tr_read64(io, base + FZ_TR_PORT_AR_STALL);
            p->r_stall = fz_tr_read64(io, base + FZ_TR_PORT_R_STALL);
            p->aw_stall = fz_tr_read64(io, base + FZ_TR_PORT_AW_STALL);
            p->w_stall = fz_tr_read64(io, base + FZ_TR_PORT_W_STALL);
            p->b_stall = fz_tr_read64(io, base + FZ_TR_PORT_B_STALL);
            p->pending_flags = io->read(io->context, base + FZ_TR_PORT_PENDING);
            p->reserved = 0;
        }
        if (sequence != io->read(io->context, FZ_TR_SNAPSHOT_SEQUENCE))
            continue;
        if ((!stats->profile.mask ?
             (stats->profile.id || stats->profile.epoch || !(stats->flags & FZ_TR_FLAG_ISSUANCE_STOPPED)) :
             fz_profile_mask(stats->profile.id) != stats->profile.mask) ||
            !fz_tr_limit_valid(stats->read_limit) || !fz_tr_limit_valid(stats->write_limit) ||
            (stats->flags & ~3U) || (stats->fifo_empty_mask & ~15U) ||
            stats->status > FZ_STATUS_UNKNOWN)
            return FZ_TR_IO_ERROR;
        for (port = 0; port < 4; ++port) {
            const struct fz_port_stats *p = &stats->ports[port];
            if ((p->pending_flags & ~15U) || p->read_outstanding > stats->read_limit ||
                p->write_outstanding > stats->write_limit)
                return FZ_TR_IO_ERROR;
        }
        return FZ_TR_OK;
    }
    return FZ_TR_IO_ERROR;
}

static inline int fz_tr_retired(const struct fz_transport_stats *stats)
{
    unsigned int port;
    if (!(stats->flags & FZ_TR_FLAG_ENGINE_IDLE) || stats->fifo_empty_mask != 15U ||
        stats->status == FZ_STATUS_UNKNOWN)
        return 0;
    for (port = 0; port < 4; ++port) {
        if (stats->ports[port].read_outstanding || stats->ports[port].write_outstanding ||
            stats->ports[port].pending_flags)
            return 0;
    }
    return 1;
}

static inline int fz_tr_drained(const struct fz_transport_stats *stats)
{
    return (stats->flags & FZ_TR_FLAG_ISSUANCE_STOPPED) && fz_tr_retired(stats);
}

/* Caller freezes RUN and owns publication serialization. Hardware repeats the
 * drain and epoch checks atomically at APPLY; software never substitutes for it.
 */
static inline int fz_tr_apply(const struct fz_transport_register_io *io,
                              const struct fz_profile *profile,
                              fz_u32 read_limit, fz_u32 write_limit)
{
    fz_u32 result;
    struct fz_transport_stats after;
    if (!profile->mask || fz_profile_mask(profile->id) != profile->mask ||
        profile->epoch == (fz_u64)-1 || !fz_tr_limit_valid(read_limit) ||
        !fz_tr_limit_valid(write_limit))
        return FZ_TR_INVALID;
    io->write(io->context, FZ_TR_REQUEST_ID, profile->id);
    io->write(io->context, FZ_TR_REQUEST_MASK, profile->mask);
    io->write(io->context, FZ_TR_EXPECTED_EPOCH_LO, (fz_u32)profile->epoch);
    io->write(io->context, FZ_TR_EXPECTED_EPOCH_HI, (fz_u32)(profile->epoch >> 32));
    io->write(io->context, FZ_TR_REQUEST_READ_LIMIT, read_limit);
    io->write(io->context, FZ_TR_REQUEST_WRITE_LIMIT, write_limit);
    io->write(io->context, FZ_TR_COMMAND, FZ_TR_CMD_APPLY);
    result = io->read(io->context, FZ_TR_RESULT);
    if (result == FZ_TR_BUSY || result == FZ_TR_MISMATCH || result == FZ_TR_INVALID)
        return result;
    if (result != FZ_TR_OK || fz_tr_snapshot(io, &after) || !fz_tr_drained(&after) ||
        after.profile.id != profile->id || after.profile.mask != profile->mask ||
        after.profile.epoch != profile->epoch + 1 || after.read_limit != read_limit ||
        after.write_limit != write_limit)
        return FZ_TR_IO_ERROR;
    return FZ_TR_OK;
}

#endif

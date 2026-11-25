/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_MMIO_H
#define FZSPEECH_MMIO_H

#include <linux/errno.h>
#include <linux/io.h>
#include "fzspeech_transport.h"
#include "fzspeech_transport_regs.h"

/* Embed this in a provider-owned object. These helpers bind the actual RTL
 * control bank only: platform resources, descriptor publication and sequencer
 * generation/chain/status inputs must still be supplied by that provider.
 */
struct fz_mmio_transport {
    void __iomem *registers;
};

static inline fz_u32 fz_mmio_read(void *context, fz_u32 offset)
{
    struct fz_mmio_transport *mmio = context;
    return readl((char __iomem *)mmio->registers + offset);
}

static inline void fz_mmio_write(void *context, fz_u32 offset, fz_u32 value)
{
    struct fz_mmio_transport *mmio = context;
    writel(value, (char __iomem *)mmio->registers + offset);
}

static inline struct fz_transport_register_io fz_mmio_io(void *context)
{
    struct fz_transport_register_io io = {context, fz_mmio_read, fz_mmio_write};
    return io;
}

static inline int fz_mmio_errno(int result)
{
    switch (result) {
    case FZ_TR_OK: return 0;
    case FZ_TR_BUSY: return -EBUSY;
    case FZ_TR_MISMATCH: return -ESTALE;
    case FZ_TR_INVALID: return -EINVAL;
    default: return -EIO;
    }
}

/* The physical provider must call probe before attaching these callbacks. */
static inline int fz_mmio_probe(void *context)
{
    struct fz_transport_register_io io = fz_mmio_io(context);
    return fz_tr_probe(&io) ? 0 : -ENODEV;
}

static inline int fz_mmio_statistics(void *context, struct fz_transport_stats *stats)
{
    struct fz_transport_register_io io = fz_mmio_io(context);
    return fz_mmio_errno(fz_tr_snapshot(&io, stats));
}

static inline int fz_mmio_snapshot(void *context, struct fz_transport_snapshot *snapshot)
{
    struct fz_transport_stats stats;
    unsigned int port;
    int rc = fz_mmio_statistics(context, &stats);
    if (rc)
        return rc;
    if (stats.status == FZ_STATUS_UNKNOWN)
        return -EIO;
    for (port = 0; port < 4; ++port) {
        if (stats.ports[port].pending_flags & FZ_TR_PROTOCOL_FAULT)
            return -EIO;
    }
    snapshot->profile = stats.profile;
    snapshot->generation = stats.generation;
    snapshot->chain_id = stats.chain_id;
    snapshot->status = stats.status;
    snapshot->fifo_empty_mask = stats.fifo_empty_mask;
    snapshot->engine_idle = !!(stats.flags & FZ_TR_FLAG_ENGINE_IDLE);
    snapshot->issuance_stopped = !!(stats.flags & FZ_TR_FLAG_ISSUANCE_STOPPED);
    for (port = 0; port < 4; ++port) {
        snapshot->read_outstanding[port] = stats.ports[port].read_outstanding;
        snapshot->write_outstanding[port] = stats.ports[port].write_outstanding;
        snapshot->pending_flags[port] = stats.ports[port].pending_flags;
    }
    return 0;
}

static inline int fz_mmio_set_run(void *context, bool enabled)
{
    fz_mmio_write(context, FZ_TR_RUN, enabled ? 1 : 0);
    return fz_mmio_read(context, FZ_TR_RUN) == (enabled ? 1U : 0U) ? 0 : -EIO;
}

static inline int fz_mmio_configure(void *context, const struct fz_profile *request,
                                    fz_u32 read_limit, fz_u32 write_limit)
{
    struct fz_transport_register_io io = fz_mmio_io(context);
    return fz_mmio_errno(fz_tr_apply(&io, request, read_limit, write_limit));
}

static inline int fz_mmio_apply_profile(void *context, const struct fz_profile *request)
{
    struct fz_transport_stats stats;
    int rc = fz_mmio_statistics(context, &stats);
    return rc ? rc : fz_mmio_configure(context, request, stats.read_limit, stats.write_limit);
}

#endif

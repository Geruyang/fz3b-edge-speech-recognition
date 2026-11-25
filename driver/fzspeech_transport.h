/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_TRANSPORT_H
#define FZSPEECH_TRANSPORT_H

#include <linux/device.h>
#include <linux/module.h>

#include "fzspeech_descriptor.h"

#define FZ_TRANSPORT_VERSION 2U

/* All methods are serialized, run in process context, and must be bounded.
 * Snapshot is a coherent observation of all four physical ports and of the
 * descriptor sequencer. A stale cached value is not evidence of drain.
 */
struct fz_transport_snapshot {
	struct fz_profile profile;
	fz_u64 generation;
	fz_u64 chain_id;
	fz_u32 status;
	fz_u32 read_outstanding[4];
	fz_u32 write_outstanding[4];
	fz_u32 pending_flags[4];
	fz_u32 fifo_empty_mask;
	bool engine_idle;
	bool issuance_stopped;
};

struct fz_transport {
	fz_u32 abi_version;
	struct device *dma_dev;
	struct module *owner;
	void *context;
	fz_u64 max_buffer_bytes;
	fz_u64 budget_bytes;
	fz_u32 dma_address_bits;
	fz_u32 max_dma_spans;
	fz_u32 capabilities;
	/* Optional v2 sequencer lifecycle. Set generation while admission/RUN are
	 * stopped; prepare immutable private descriptors before streaming DMA sync;
	 * acknowledge only a fully retired, identity-matched completion. */
	int (*prepare_generation)(void *context, fz_u64 generation);
	int (*plan)(void *context, const struct fz_chain *chain,
		    const struct fz_transfer *transfers, const struct fz_dma_span *spans,
		    fz_u32 span_count);
	int (*retire)(void *context, fz_u64 generation, fz_u64 chain_id);
	int (*receipt)(void *context, struct fz_probe_receipt *receipt);
	void (*detached)(void *context);
	/* Stop new chains and allow already-issued transactions to retire. */
	int (*set_run)(void *context, bool enabled);
	int (*snapshot)(void *context, struct fz_transport_snapshot *status);
	/* Enforce idle/drained and expected epoch atomically in hardware. */
	/* -EBUSY/-ESTALE/-EINVAL mean rejected without changing configuration;
	 * all other failures are ambiguous and cause quarantine. */
	int (*apply_profile)(void *context, const struct fz_profile *request);
	/* Optional dynamic-limit/statistics extension. Both callbacks are required
	 * to advertise FZ_CAP_DYNAMIC_LIMITS. Configure applies the full profile
	 * and independent per-port read/write limits as one epoch transaction.
	 */
	int (*configure)(void *context, const struct fz_profile *request,
			 fz_u32 read_limit, fz_u32 write_limit);
	int (*statistics)(void *context, struct fz_transport_stats *stats);
	/* Success means published. A failure may have rung the doorbell.
	 * Copy descriptor metadata into provider-owned DMA storage before return.
	 * Honor sequence/dependency fencing and cap each burst at the span,
	 * transfer, 4 KiB and BL16 boundaries. Never bypass the active HP mask.
	 */
	int (*publish)(void *context, const struct fz_chain *chain,
		       const struct fz_transfer *transfers, const struct fz_dma_span *spans,
		       fz_u32 span_count);
};

/* Use only with a platform integration whose MMIO, IRQ, clock lease,
 * noncoherent DMA domain and physical drains have been qualified. The
 * provider must retain those resources until detach succeeds. Do not
 * call a teardown path whose caller ignores a failed detach.
 */
int fzspeech_attach_transport(const struct fz_transport *transport);
int fzspeech_detach_transport(const struct fz_transport *transport);
void fzspeech_quarantine_transport(const struct fz_transport *transport);

#endif

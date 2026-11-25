#ifndef FZSPEECH_UAPI_H
#define FZSPEECH_UAPI_H

#ifdef _WIN32
#include <stdint.h>
typedef uint32_t fz_u32;
typedef uint64_t fz_u64;
#else
#include <linux/ioctl.h>
#include <linux/types.h>
typedef __u32 fz_u32;
typedef __u64 fz_u64;
#endif

#define FZ_ABI_VERSION 1U
#define FZ_DESCRIPTOR_VERSION 1U
#define FZ_MAX_DESCRIPTORS 256U
#define FZ_MAX_CHAIN_BYTES 65536U
#define FZ_MAX_BUFFER_BYTES (1ULL << 31)
#define FZ_MAX_IO_BYTES 65536U
#define FZ_BEAT_BYTES 16U
#define FZ_MAX_BURST_BYTES 256U
#define FZ_NO_DEPENDENCY ((fz_u32)-1)
#define FZ_ACCESS_READ 1U
#define FZ_ACCESS_WRITE 2U
#define FZ_ACCESS_READ_WRITE 3U
#define FZ_STATUS_PENDING 0U
#define FZ_STATUS_SUCCESS 1U
#define FZ_STATUS_FAILED 2U
#define FZ_STATUS_UNKNOWN 3U
#define FZ_STATUS_TIMEOUT 4U
#define FZ_STATUS_IDLE 5U

struct fz_profile {
    fz_u64 epoch;
    fz_u32 id;
    fz_u32 mask;
};

struct fz_info {
    fz_u64 generation;
    struct fz_profile profile;
    fz_u64 max_buffer_bytes;
    fz_u64 budget_bytes;
    fz_u32 abi_version;
    fz_u32 descriptor_version;
    fz_u32 dma_address_bits;
    fz_u32 supported_hp_mask;
    fz_u32 max_descriptors;
    fz_u32 max_chain_bytes;
    fz_u32 max_dma_spans;
    fz_u32 quarantined;
    fz_u32 capabilities;
    fz_u32 reserved;
};

#define FZ_CAP_SG_TRANSFER 1U
#define FZ_CAP_ALL_PORT_DRAIN 2U
#define FZ_CAP_TRANSPORT_STATS 4U
#define FZ_CAP_DYNAMIC_LIMITS 8U
#define FZ_CAP_BANK16_PROBE 16U
#define FZ_CAP_JOB_RECEIPT 32U

struct fz_buffer {
    fz_u64 generation;
    fz_u64 handle;
    fz_u64 bytes;
    fz_u32 access;
    fz_u32 reserved;
};

struct fz_io {
    fz_u64 generation;
    fz_u64 handle;
    fz_u64 offset;
    fz_u64 bytes;
    /* User pointer for synchronous CPU copy. Never interpreted as a DMA address. */
    fz_u64 user_data;
};

struct fz_transfer {
    fz_u64 job_id;
    fz_u64 handle;
    fz_u64 offset;
    fz_u64 bytes;
    fz_u32 pair;
    fz_u32 access;
    fz_u32 depends_on;
    fz_u32 reserved;
};

struct fz_chain {
    fz_u64 generation;
    fz_u64 chain_id;
    struct fz_profile profile;
    fz_u32 descriptor_version;
    fz_u32 descriptor_count;
    fz_u32 chain_bytes;
    fz_u32 reserved;
    /* Followed immediately by descriptor_count struct fz_transfer entries. */
};

struct fz_wait {
    fz_u64 generation;
    fz_u64 chain_id;
    struct fz_profile profile;
    fz_u32 timeout_ms;
    fz_u32 status;
};

/* Changing either limit is a drained configuration transaction. The expected
 * profile epoch is consumed exactly once; success returns the new epoch.
 */
struct fz_limits {
    fz_u64 generation;
    struct fz_profile profile;
    fz_u32 read_limit;
    fz_u32 write_limit;
};

struct fz_port_stats {
    fz_u64 read_bytes, write_bytes;
    fz_u64 ar_stall, r_stall, aw_stall, w_stall, b_stall;
    fz_u32 read_outstanding, write_outstanding;
    fz_u32 read_peak, write_peak;
    fz_u32 pending_flags, reserved;
};

struct fz_transport_stats {
    fz_u64 generation, chain_id, counter_epoch;
    struct fz_profile profile;
    struct fz_port_stats ports[4];
    fz_u32 snapshot_sequence, status;
    fz_u32 read_limit, write_limit;
    fz_u32 flags, fifo_empty_mask;
};

/* Diagnostic receipt for the retained FZBANK16 HLS probe. Failed indices refer
 * to the provider's canonical private span/transfer manifest, never a user DMA
 * address. The completed receipt survives the hardware ACK until the next job. */
struct fz_probe_receipt {
    fz_u64 generation, chain_id;
    struct fz_profile profile;
    fz_u64 signature[4], read_bytes[4];
    fz_u64 failed_job_id;
    fz_u32 status, error, failed_span, failed_transfer, failed_dma_span, done_sequence;
    fz_u32 packet_bytes, span_count;
};

#ifndef _WIN32
#define FZ_IOCTL_MAGIC 0xf5
#define FZ_GET_INFO _IOR(FZ_IOCTL_MAGIC, 0, struct fz_info)
#define FZ_PREPARE _IOR(FZ_IOCTL_MAGIC, 1, struct fz_info)
#define FZ_SET_PORT_PROFILE _IOWR(FZ_IOCTL_MAGIC, 2, struct fz_profile)
#define FZ_ALLOC_BUFFER _IOWR(FZ_IOCTL_MAGIC, 3, struct fz_buffer)
#define FZ_FREE_BUFFER _IOW(FZ_IOCTL_MAGIC, 4, struct fz_buffer)
#define FZ_WRITE_BUFFER _IOW(FZ_IOCTL_MAGIC, 5, struct fz_io)
#define FZ_READ_BUFFER _IOW(FZ_IOCTL_MAGIC, 6, struct fz_io)
#define FZ_SUBMIT_CHAIN _IOW(FZ_IOCTL_MAGIC, 7, struct fz_chain)
#define FZ_WAIT _IOWR(FZ_IOCTL_MAGIC, 8, struct fz_wait)
#define FZ_QUIESCE _IOWR(FZ_IOCTL_MAGIC, 9, struct fz_wait)
#define FZ_GET_TRANSPORT_STATS _IOR(FZ_IOCTL_MAGIC, 10, struct fz_transport_stats)
#define FZ_SET_TRANSPORT_LIMITS _IOWR(FZ_IOCTL_MAGIC, 11, struct fz_limits)
#define FZ_GET_JOB_RECEIPT _IOWR(FZ_IOCTL_MAGIC, 12, struct fz_probe_receipt)
/* Explicit safe provider detach. Requires the owner generation; drains and
 * closes this fd's device lease. A subsequent ioctl on that fd is rejected. */
#define FZ_RELEASE_DEVICE _IOW(FZ_IOCTL_MAGIC, 13, struct fz_wait)
#endif
#endif

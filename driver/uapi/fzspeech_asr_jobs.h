/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef FZSPEECH_ASR_JOBS_UAPI_H
#define FZSPEECH_ASR_JOBS_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define FZ_ASR_DEVICE_NAME "fzspeech-asr-jobs"
#define FZ_ASR_USERS_ABI 3U
#define FZ_ASR_DUAL_USERS_ABI 4U
#define FZ_ASR_LANE0_DEVICE_NAME "fzspeech-asr-lane0"
#define FZ_ASR_LANE1_DEVICE_NAME "fzspeech-asr-lane1"
#define FZ_ASR_CHUNK_RESET 1U
#define FZ_ASR_RECEIPT_IRQ 1U
#define FZ_ASR_RECEIPT_ACK 2U
#define FZ_ASR_RECEIPT_DRAIN 4U
#define FZ_ASR_RECEIPT_CPU_SYNC 8U
#define FZ_ASR_RECEIPT_GUARDS 16U
#define FZ_ASR_RECEIPT_ALL 31U

struct fz_asr_info {
	__u32 abi, wire_version, dma_bits, reserved;
	__u8 capability_sha256[32];
};

struct fz_asr_chunk {
	__u64 generation, epoch, chunk_id;
	__u32 flags, reserved;
};

struct fz_asr_lane_info {
	__u64 platform_generation;
	__u32 lane;
	__u32 lane_count;
};

/* The kernel copies CPU input and owns every DMA allocation. Only return 0
 * publishes provisional outputs. COMMIT_CHUNK promotes working history. */
struct fz_asr_request {
	__u64 generation, epoch, chunk_id;
	__u64 input_ptr, output_ptr, old_history_ptr, working_history_ptr;
	__u32 operation, rows, call_id;
	__u32 input_bytes, output_bytes, history_bytes;
	__u32 reserved[2];
};

struct fz_asr_receipt {
	__u64 generation, epoch, chunk_id, chain_id;
	__u64 prepare_ns, device_ns, retire_ns;
	__u32 sequence, operation, rows, call_id;
	__u32 irq_events, flags, packet_bytes, span_count;
	__u32 packet_crc32, hw_done_sequence;
};

struct fz_asr_submit {
	struct fz_asr_request request;
	struct fz_asr_receipt receipt;
};

#define FZ_ASR_GET_INFO _IOR(0xe7, 19, struct fz_asr_info)
#define FZ_ASR_BEGIN_CHUNK _IOWR(0xe7, 20, struct fz_asr_chunk)
#define FZ_ASR_SUBMIT_JOB _IOWR(0xe7, 21, struct fz_asr_submit)
#define FZ_ASR_COMMIT_CHUNK _IOW(0xe7, 22, struct fz_asr_chunk)
#define FZ_ASR_ABORT_CHUNK _IO(0xe7, 23)
#define FZ_ASR_CANCEL_JOB _IO(0xe7, 24)
#define FZ_ASR_GET_LANE _IOR(0xe7, 25, struct fz_asr_lane_info)

#endif

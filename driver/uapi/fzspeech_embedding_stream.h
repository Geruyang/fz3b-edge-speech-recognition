/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef FZSPEECH_EMBEDDING_STREAM_H
#define FZSPEECH_EMBEDDING_STREAM_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define FZ_EMBED_STREAM_DEVICE "/dev/fzspeech-embedding-stream"
#define FZ_EMBED_STREAM_INPUT_BYTES 12480U
#define FZ_EMBED_STREAM_OUTPUT_BYTES 16384U
#define FZ_EMBED_STREAM_EVIDENCE_MAGIC 0x56455a46U
#define FZ_EMBED_STREAM_EVIDENCE_VERSION 1U
#define FZ_EMBED_STREAM_EVIDENCE_BYTES 45392U
#define FZ_EMBED_STREAM_REGISTER_BYTES 4176U
#define FZ_EMBED_STREAM_PACKET_CAPACITY 8320U
#define FZ_EMBED_STREAM_GUARD_BYTES 16384U
#define FZ_EMBED_STREAM_EVIDENCE_ACK 1U
#define FZ_EMBED_STREAM_EVIDENCE_CPU_SYNC 2U
#define FZ_EMBED_STREAM_EVIDENCE_DRAIN 4U
#define FZ_EMBED_STREAM_EVIDENCE_GUARDS 8U
#define FZ_EMBED_STREAM_EVIDENCE_READONLY 16U

/* A request is visible to the accelerator only after the complete input
 * has been copied. Trust output_ptr only on return 0. Chain IDs start at
 * one and increase by one for this file's single hardware lease. */
struct fz_embedding_stream_request {
	__u64 chain_id;
	__u64 input_ptr;
	__u64 output_ptr;
	__u64 generation;
	__u64 epoch;
	__u32 input_bytes;
	__u32 output_bytes;
	__u32 sequence;
	__u32 reserved;
};

/* Read the latest result committed to this open file. Every identity field
 * is required. ENODATA means no successful SUBMIT; ESTALE means another
 * identity is current. Neither rejection copies to the destination. */
struct fz_embedding_stream_evidence_request {
	__u64 generation;
	__u64 epoch;
	__u64 chain_id;
	__u32 sequence;
	__u32 reserved;
	__u64 evidence_ptr;
	__u32 evidence_bytes;
	__u32 reserved2;
};

/* All scalar fields and register words are little-endian on FZ3B.
 * Register snapshots are before publish, held before ACK, and after drain.
 * Only the first packet_bytes in the packet slot belong to the descriptor. */
struct fz_embedding_stream_evidence_header {
	__u32 magic;
	__u32 version;
	__u32 evidence_bytes;
	__u32 packet_bytes;
	__u64 generation;
	__u64 epoch;
	__u64 chain_id;
	__u64 descriptor_dma;
	__u32 descriptor_bytes;
	__u32 sequence;
	__u32 profile_id;
	__u32 profile_mask;
	__u32 irq;
	__u32 irq_events;
	__u32 flags;
	__u32 span_count;
	__u32 packet_crc32;
	__u32 dma_bits;
	__u8 output_sha256[32];
	__u32 registers_bytes;
	__u32 guard_bytes;
};

struct fz_embedding_stream_evidence {
	struct fz_embedding_stream_evidence_header header;
	__u8 registers[FZ_EMBED_STREAM_REGISTER_BYTES];
	__u8 packet[FZ_EMBED_STREAM_PACKET_CAPACITY];
	__u8 guards_before[FZ_EMBED_STREAM_GUARD_BYTES];
	__u8 guards_after[FZ_EMBED_STREAM_GUARD_BYTES];
};

#define FZ_EMBED_STREAM_SUBMIT _IOWR(0xe7, 1, struct fz_embedding_stream_request)
#define FZ_EMBED_STREAM_CANCEL _IO(0xe7, 2)
#define FZ_EMBED_STREAM_EVIDENCE _IOW(0xe7, 3, struct fz_embedding_stream_evidence_request)

#endif

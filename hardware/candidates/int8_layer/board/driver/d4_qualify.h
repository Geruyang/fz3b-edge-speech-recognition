/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZ_ASR_D4_QUALIFY_H
#define FZ_ASR_D4_QUALIFY_H

#define D4_FRAME_LEN 48U
#define D4_CAPTURE_COUNT 4U
#define D4_CLOCKS 4U
#define D4_COUNT_OFFSET 8U
#define D4_ID_OFFSET 32U
#define D4_ID_BYTES 12U
#define D4_CRC_OFFSET 44U
#define D4_GOOD_FLAGS 0x1fU

struct d4_frame {
	unsigned char bytes[D4_FRAME_LEN];
	unsigned long long lo_ns, hi_ns;
};

static inline unsigned int d4_get32(const unsigned char *p)
{
	return (unsigned int)p[0] | (unsigned int)p[1] << 8 |
	       (unsigned int)p[2] << 16 | (unsigned int)p[3] << 24;
}

static inline unsigned int d4_crc(const unsigned char *p)
{
	unsigned int crc = ~0U;
	unsigned int offset, bit;

	for (offset = 0; offset < D4_CRC_OFFSET; offset++) {
		crc ^= p[offset];
		for (bit = 0; bit < 8; bit++)
			crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320U : 0);
	}
	return ~crc;
}

static inline int d4_frame_ok(const struct d4_frame *f, unsigned char profile,
			      const unsigned char *identity, unsigned int expected_status)
{
	unsigned int i;
	const unsigned char *p = f->bytes;

	if (p[0] != 0xd4 || p[1] != 2 || p[2] != D4_GOOD_FLAGS || p[3] != profile ||
	    d4_get32(p + 28) != expected_status || d4_get32(p + D4_CRC_OFFSET) != d4_crc(p) ||
	    !f->lo_ns || f->hi_ns < f->lo_ns)
		return 0;
	for (i = 0; i < D4_ID_BYTES; i++)
		if (p[D4_ID_OFFSET + i] != identity[i])
			return 0;
	return 1;
}

static inline int d4_next(const struct d4_frame *a, const struct d4_frame *b)
{
	const unsigned int seq = d4_get32(a->bytes + 4);

	return seq && seq != ~0U && d4_get32(b->bytes + 4) == seq + 1U &&
	       d4_get32(a->bytes + 24) == d4_get32(b->bytes + 24) && b->lo_ns > a->hi_ns;
}

static inline int d4_check_rates(const struct d4_frame *a, const struct d4_frame *b,
				 const unsigned int nominal[D4_CLOCKS])
{
	unsigned long long span_min, span_max;
	unsigned int i;

	if (!d4_next(a, b) || b->hi_ns < b->lo_ns)
		return 0;
	span_min = b->lo_ns - a->hi_ns;
	span_max = b->hi_ns - a->lo_ns;
	/* At <= 1 GHz the full interval contains strictly fewer than 2^32 edges. */
	if (span_min < 900000000ULL || span_max > 1500000000ULL ||
	    span_max - span_min > 2000000ULL)
		return 0;
	for (i = 0; i < D4_CLOCKS; i++) {
		unsigned int edges = d4_get32(b->bytes + D4_COUNT_OFFSET + i * 4) -
				     d4_get32(a->bytes + D4_COUNT_OFFSET + i * 4);
		unsigned long long scaled = (unsigned long long)edges * 1000000000ULL;
		unsigned long long low = (99ULL * nominal[i]) / 100;
		unsigned long long high = (101ULL * nominal[i]) / 100;

		if (!edges || scaled < low * span_max || scaled > high * span_min)
			return 0;
	}
	return 1;
}

#endif

#pragma once
#include <cstdint>

namespace fzint {
struct Ratio { uint32_t d, reciprocal; };

// Compute floor(2^32/d) once per row/tensor. A unit divisor has a separate path.
inline Ratio ratio(uint32_t d) {
#pragma HLS INLINE off
    uint64_t remainder=1;
    uint32_t numerator=0;
    if (d!=1) {
        for (unsigned bit=0;bit<32;++bit) {
#pragma HLS PIPELINE II=1
            remainder <<= 1;
            numerator <<= 1;
            if (remainder>=d) { remainder-=d; ++numerator; }
        }
    }
    return {d,numerator};
}

inline uint32_t nearest(uint32_t q,uint32_t rem,uint32_t d) {
    return q+(rem>d-rem || (rem==d-rem && (q&1U)));
}

// For a<2^32 the reciprocal estimate undershoots by at most one.
inline int32_t scale_code(int32_t value,Ratio r) {
    if (r.d==1) return value;
    const uint32_t a=value<0 ? 0U-uint32_t(value) : uint32_t(value);
    uint32_t q=uint32_t((uint64_t(a)*r.reciprocal)>>32);
    uint32_t rem=a-q*r.d;
    if (rem>=r.d) { ++q; rem-=r.d; }
    q=nearest(q,rem,r.d);
    return value<0 ? -int32_t(q) : int32_t(q);
}

// Wide pooling sums, normalized values and weighted-V numerators have a wider
// contract. Use a single iterated remainder, preserving ties and INT64_MIN.
inline int64_t wide_ratio(int64_t value,uint32_t d) {
#pragma HLS INLINE off
    if (d==1) return value;
    uint64_t q=value<0 ? uint64_t(0)-uint64_t(value) : uint64_t(value);
    uint64_t rem=0;
    for (int i=0;i<64;++i) {
#pragma HLS PIPELINE II=1
        rem=(rem<<1)|(q>>63);
        q<<=1;
        if (rem>=d) {rem-=d; ++q;}
    }
    q+=rem>d-rem || (rem==d-rem && (q&1));
    return value<0 ? -int64_t(q) : int64_t(q);
}
}

#pragma once
#include "scaling.hpp"

namespace fzint {
#ifdef FZ_FAST_FLOW
struct RowRatio { uint32_t d, inverse, shift; };

inline RowRatio row_ratio(uint32_t d) {
#pragma HLS INLINE off
    const unsigned s=31-__builtin_clz(d);
    uint64_t r=uint64_t(1)<<s;
    uint32_t inv=0;
    if (r==d) inv=1U<<31;
    else {
        for (int i=0;i<31;++i) {
#pragma HLS PIPELINE II=1
            r<<=1; inv<<=1;
            if(r>=d) {r-=d;++inv;}
        }
    }
    return {d,inv,s};
}

inline uint64_t unsigned_magnitude(int64_t value) {
    return value<0 ? uint64_t(0)-uint64_t(value) : uint64_t(value);
}
inline bool bounded_ratio(uint64_t a,RowRatio scale) {
    return scale.d==1 || (a>>scale.shift)<=UINT32_MAX;
}
inline int64_t apply_bounded_ratio(int64_t value,RowRatio scale) {
    // h=floor(|x|/2^s), inv=floor(2^(31+s)/d). For h<2^32 the
    // absolute underestimate before flooring is <1+h/2^31 <3.
    if(scale.d==1) return value;
    const uint64_t a=unsigned_magnitude(value);
    const uint64_t h=a>>scale.shift;
    uint32_t q=uint32_t((uint64_t(uint32_t(h))*scale.inverse)>>31);
    uint64_t r=(a-uint64_t(q)*scale.d)&((uint64_t(1)<<34)-1);
    for(int i=0;i<3;++i) {
#pragma HLS UNROLL
        if(r>=scale.d){r-=scale.d; ++q;}
    }
    const uint64_t rounded=uint64_t(q)+(r>scale.d-r || (r==scale.d-r && (q&1U)));
    return value<0 ? -int64_t(rounded) : int64_t(rounded);
}
inline int64_t apply_row_ratio(int64_t value,RowRatio scale) {
#pragma HLS INLINE off
    return bounded_ratio(unsigned_magnitude(value),scale) ? apply_bounded_ratio(value,scale)
                                                          : wide_ratio(value,scale.d);
}
#else
inline uint32_t row_ratio(uint32_t d){return d;}
inline int64_t apply_row_ratio(int64_t n,uint32_t d){return wide_ratio(n,d);}
#endif
}

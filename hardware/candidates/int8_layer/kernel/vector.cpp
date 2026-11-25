#include "row_ratio.hpp"

namespace fzint {
int32_t tensor_scale(const int32_t* src, int size) {
#pragma HLS INLINE off
    int64_t peak = 0;
    for (int i=0; i<size; ++i) {
#pragma HLS PIPELINE II=1
        const int64_t v = mag(src[i]);
        if (v > peak) peak = v;
    }
    return scale_for(peak);
}
int32_t quantized_copy(int32_t* dst, const int32_t* src, int size) {
#pragma HLS INLINE off
    const int32_t scale = tensor_scale(src, size);
    const auto divisor = ratio(scale);
#ifdef FZ_CACHED_SCALE
    uint32_t peak=0;
#endif
    for (int i=0; i<size; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = scale_code(src[i], divisor) * scale;
#ifdef FZ_CACHED_SCALE
        const uint32_t value=uint32_t(mag(dst[i]));
        if(value>peak) peak=value;
#endif
    }
#ifdef FZ_CACHED_SCALE
    return scale_for(peak);
#else
    return 0;
#endif
}
static int32_t sigmoid(int64_t x) {
    static constexpr int table[33] = {16384,18421,20397,22255,23955,25471,26790,27917,28862,
        29644,30282,30799,31214,31545,31807,32015,32179,32307,32408,32487,32549,32597,
        32635,32664,32687,32705,32719,32730,32738,32745,32750,32754,32757};
    int64_t a = x < 0 ? -x : x;
    if (a > 8*65536) a = 8*65536;
    int j = int(a >> 14);
    if (j > 31) j = 31;
    int v = table[j] + int(divide((a-j*16384)*(table[j+1]-table[j]), 16384));
    return x < 0 ? 32768-v : v;
}
int32_t swish(int32_t x) {
    return int32_t(divide(int64_t(x)*sigmoid(int64_t(x)-65536), 32768));
}
#ifdef FZ_FUSED_COPY
// Find the exact peak while producing each value. Both residual destinations
// receive the same rounded word in the second pass.
void finish_projection(bool activation,int b,int size,Work& w) {
#pragma HLS INLINE off
    uint32_t peak=0;
    for(int i=0;i<size;++i) {
        FZ_ELEMENT_PIPELINE
        int32_t raw=w.raw[b][i];
        const int32_t value=activation ? swish(raw) : w.x[b][i]+raw;
        w.raw[b][i]=value;
        const uint32_t magnitude=uint32_t(mag(value));
        if(magnitude>peak) peak=magnitude;
    }
    const int32_t scale=scale_for(peak);
    const auto divisor=ratio(scale);
    uint32_t stored_peak=0;
    for(int i=0;i<size;++i) {
        FZ_ELEMENT_PIPELINE
        const int32_t value=scale_code(w.raw[b][i],divisor)*scale;
        w.feed[b][i]=value;
        if(!activation)w.x[b][i]=value;
        const uint32_t magnitude=uint32_t(mag(value));
        if(magnitude>stored_peak)stored_peak=magnitude;
    }
    w.feed_scale[b]=scale_for(stored_peak);
}
#endif
void convolution_input(int op, int b, int rows, History& h, Work& w) {
#pragma HLS INLINE off
    int slot = op == c1_in ? 0 : 1;
    int32_t* dst = w.feed[b];
    int32_t* gate = dst + 30*channels;
#ifdef FZ_FUSED_COPY
    uint32_t old_peak=0,gate_peak=0,stored_peak=0;
#endif
    for (int j=0; j<30*channels; ++j) {
        FZ_ELEMENT_PIPELINE
        dst[j] = h.conv[slot][j/256][j%256] * h.scale[3+slot][j/256];
#ifdef FZ_FUSED_COPY
        const uint32_t x=uint32_t(mag(dst[j]));
        if(x>old_peak)old_peak=x;
#endif
    }
    for (int j=0; j<rows*channels; ++j) {
        FZ_ELEMENT_PIPELINE
        int i = (j/256)*512 + j%256;
        gate[j] = int32_t(divide(int64_t(w.raw[b][i])*sigmoid(w.raw[b][i+256]), 32768));
#ifdef FZ_FUSED_COPY
        const uint32_t x=uint32_t(mag(gate[j]));
        if(x>gate_peak)gate_peak=x;
#endif
    }
#ifdef FZ_FUSED_COPY
    const int32_t s=scale_for(gate_peak);
#else
    int32_t s = tensor_scale(gate, rows*channels);
#endif
    const auto divisor=ratio(s);
    for (int r=0; r<30; ++r) {
        int src = rows+r;
        bool fresh = src >= 30;
        h.scale[3+slot][r] = fresh ? s : h.scale[3+slot][src];
        for (int c=0; c<channels; ++c) {
#pragma HLS PIPELINE II=1
            if (fresh) {
                int j = (src-30)*256+c;
                int8_t q = int8_t(scale_code(gate[j], divisor));
                gate[j] = q*s;
#ifdef FZ_FUSED_COPY
                const uint32_t x=uint32_t(mag(gate[j]));
                if(x>stored_peak)stored_peak=x;
#endif
                h.conv[slot][r][c] = q;
            } else h.conv[slot][r][c] = h.conv[slot][src][c];
        }
    }
#ifdef FZ_FUSED_COPY
    w.feed_scale[b]=scale_for(old_peak>stored_peak?old_peak:stored_peak);
#endif
}
void prefix(int b, int rows, History& h, Work& w) {
#pragma HLS INLINE off
    for (int r=0; r<rows; ++r) {
        const auto count_ratio=row_ratio(uint32_t(h.count+r+1));
#ifdef FZ_STREAM_FLOW
        bool bounded=true;
        for (int c=0;c<channels;++c) {
            FZ_ELEMENT_PIPELINE
            h.sum[c]+=w.x[b][r*channels+c];
            if (!bounded_ratio(unsigned_magnitude(h.sum[c]),count_ratio)) bounded=false;
        }
        if (bounded) {
            for (int c=0;c<channels;++c) {
                FZ_ELEMENT_PIPELINE
                w.feed[b][r*channels+c]=int32_t(apply_bounded_ratio(h.sum[c],count_ratio));
            }
        } else {
            for (int c=0;c<channels;++c)
                w.feed[b][r*channels+c]=int32_t(apply_row_ratio(h.sum[c],count_ratio));
        }
#else
        for (int c=0; c<channels; ++c) {
#ifdef FZ_AREA_REVISION
// A pipelined caller forces Vitis to unroll the nested 64-step divider.
#pragma HLS PIPELINE off
#else
#pragma HLS PIPELINE II=1
#endif
            h.sum[c] += w.x[b][r*channels+c];
            w.feed[b][r*channels+c] = int32_t(apply_row_ratio(h.sum[c],count_ratio));
        }
#endif
    }
    h.count += rows;
#ifdef FZ_CACHED_SCALE
    w.feed_scale[b]=
#endif
    quantized_copy(w.feed[b], w.feed[b], rows*channels);
}
static uint32_t isqrt(uint64_t x) {
    uint64_t root = 0;
    for (int i=31; i>=0; --i) {
        uint64_t bit = uint64_t(1) << (2*i);
        if (x >= root+bit) { x -= root+bit; root = (root>>1)+bit; }
        else root >>= 1;
    }
    return uint32_t(root);
}
void normalize(int b, int rows, const Params& p, const Boundary& in, Boundary& out, Work& w) {
#pragma HLS INLINE off
    for (int r=0; r<rows; ++r) {
        int64_t square = 0;
#ifdef FZ_STREAM_FLOW
        uint64_t peak=0;
#endif
        for (int c=0; c<channels; ++c) {
            FZ_ELEMENT_PIPELINE
            int64_t x = w.x[b][r*256+c];
            square += x*x;
#ifdef FZ_STREAM_FLOW
            uint64_t magnitude=unsigned_magnitude(x);
            if (magnitude>peak) peak=magnitude;
#endif
        }
        uint32_t root = isqrt(divide(square, 256) + int64_t(p.epsilon)*65536);
        const auto root_ratio=row_ratio(root);
#ifdef FZ_STREAM_FLOW
        if (bounded_ratio(peak*65536,root_ratio)) {
            for (int c=0;c<channels;++c) {
                FZ_ELEMENT_PIPELINE
                int32_t start=in.codes[b][r][c]*in.scale[b];
                int64_t norm=apply_bounded_ratio(int64_t(w.x[b][r*256+c])*65536,root_ratio);
                w.feed[b][r*256+c]=start+int32_t(divide(int64_t(p.beta)*(norm-start),65536));
            }
        } else
#endif
        for (int c=0; c<channels; ++c) {
            const int32_t start = in.codes[b][r][c]*in.scale[b];
            int64_t norm = apply_row_ratio(int64_t(w.x[b][r*256+c])*65536, root_ratio);
            w.feed[b][r*256+c] = start + int32_t(divide(int64_t(p.beta)*(norm-start),65536));
        }
    }
    int32_t s = tensor_scale(w.feed[b], rows*channels);
    const auto divisor=ratio(s);
    out.scale[b] = s;
    for (int j=0; j<rows*channels; ++j) {
        FZ_ELEMENT_PIPELINE
        out.codes[b][j/256][j%256] = int8_t(scale_code(w.feed[b][j],divisor));
    }
}
}

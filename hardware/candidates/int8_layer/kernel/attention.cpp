#include "row_ratio.hpp"

namespace fzint {
static int32_t slice_scale(const int32_t* raw, int rows, int stride, int offset, int width) {
    int64_t peak = 0;
    for (int r=0; r<rows; ++r)
        for (int d=0; d<width; ++d) {
            FZ_ELEMENT_PIPELINE
            int64_t a = mag(raw[r*stride+offset+d]);
            if (a>peak) peak=a;
        }
    return scale_for(peak);
}
// Retain each historical row's original scale, then requantize the joined window for the dot.
template<int W>
static int32_t join(const int32_t* raw, int rows, int stride, int offset,
                    int8_t past[max_left][W], int32_t* scales, int8_t joined[80][W]) {
    int left=4*rows;
    int32_t fresh=slice_scale(raw,rows,stride,offset,W);
    const auto fresh_divisor=ratio(fresh);
    int64_t peak=0;
    int32_t row_scales[80];
    for (int r=0;r<left+rows;++r) {
        int32_t s = r<left ? scales[r] : fresh;
        row_scales[r]=s;
        for (int d=0;d<W;++d) {
#pragma HLS PIPELINE II=1
            int8_t q = r<left ? past[r][d] : int8_t(scale_code(raw[(r-left)*stride+offset+d],fresh_divisor));
            joined[r][d]=q;
            int64_t a=mag(q*s);
            if (a>peak) peak=a;
            if (r>=rows) past[r-rows][d]=q;
        }
        if (r>=rows) scales[r-rows]=s;
    }
    int32_t s=scale_for(peak);
    const auto joined_divisor=ratio(s);
    for (int r=0;r<left+rows;++r)
        for (int d=0;d<W;++d) {
            FZ_ELEMENT_PIPELINE
            joined[r][d]=int8_t(scale_code(int32_t(joined[r][d])*row_scales[r],joined_divisor));
        }
    return s;
}
static int exp_weight(int8_t code) {
    static constexpr int table[8]={16383,15023,13776,12633,11585,10623,9741,8933};
    int m=-int(code);
#ifdef FZ_AREA_REVISION
    if(m==128) return 0;
    unsigned shift=m/8, d=1U<<shift, value=table[m%8];
    return int(nearest(value>>shift,value&(d-1),d));
#else
    return m==128 ? 0 : int(divide(table[m%8],1<<(m/8)));
#endif
}
#ifdef FZ_AREA_REVISION
#include "mixed_attention.inc"
#else
static void scores(int b,int rows,const Params& p,History& h,Work& w) {
#pragma HLS INLINE off
    int8_t keys[80][192];
    const int32_t* raw=w.raw[b];
    const int32_t sk=join(raw,rows,496,192,h.key,h.scale[0],keys);
    const int32_t sq=slice_scale(raw,rows,496,0,192), sp=slice_scale(raw,rows,496,480,16);
    for (int head=0;head<4;++head) {
        for (int r=0;r<rows;++r) {
            int8_t query[48], pos[4];
            for (int d=0;d<48;++d) query[d]=int8_t(divide(raw[r*496+head*48+d],sq));
            for (int d=0;d<4;++d) pos[d]=int8_t(divide(raw[r*496+480+head*4+d],sp));
            int32_t row[80], peak=INT32_MIN;
            for (int t=0;t<5*rows;++t) {
                int32_t qk=0, rel=0;
                for (int d=0;d<48;++d) qk+=query[d]*keys[t][head*48+d];
                for (int d=0;d<4;++d) rel+=pos[d]*p.position[rows-1-r+t][head*4+d];
                row[t]=int32_t(divide(int64_t(qk)*sq*sk,65536)
                              +divide(int64_t(rel)*sp*p.position_scale,65536));
                if (row[t]>peak) peak=row[t];
            }
            for (int t=0;t<5*rows;++t) {
                int64_t code=divide((int64_t(row[t])-peak)*756388, int64_t(1)<<32);
                w.logp[b][head][r][t]=int8_t(code < -128 ? -128 : code);
            }
        }
    }
}
void attention(int op,int b,int rows,const Params& p,History& h,Work& w) {
#pragma HLS INLINE off
    bool first=op==a1_in;
    if (first) scores(b,rows,p,h,w);
    int8_t values[80][96];
    int32_t s=first ? join(w.raw[b],rows,496,384,h.val,h.scale[1],values)
                    : join(w.raw[b],rows,96,0,h.val2,h.scale[2],values);
    for (int head=0;head<4;++head)
        for (int r=0;r<rows;++r) {
            int32_t denom=0, hi[24]={}, lo[24]={};
            for (int t=0;t<5*rows;++t) {
                int e=exp_weight(w.logp[b][head][r][t]);
                denom+=e;
                for (int d=0;d<24;++d) {
                    int8_t v=values[t][head*24+d];
                    hi[d]+=int8_t(e>>7)*v;
                    lo[d]+=int8_t(e&127)*v;
                }
            }
            for (int d=0;d<24;++d)
                w.feed[b][r*96+head*24+d]=int32_t(divide((int64_t(hi[d])*128+lo[d])*s,denom));
        }
}
#endif
}

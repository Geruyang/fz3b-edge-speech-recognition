#include "layer.hpp"

using namespace fzint;
static void unpack_bytes(const uint32_t* src, int8_t* dst, unsigned n) {
    for (unsigned i=0;i<n;i+=4) {
#pragma HLS PIPELINE II=1
        uint32_t t=src[i/4];
        for (int lane=0;lane<4;++lane) dst[i+lane]=int8_t(t>>(lane*8));
    }
}
static void pack_bytes(uint32_t* dst, const int8_t* src, unsigned n) {
    for (unsigned i=0;i<n;i+=4) {
#pragma HLS PIPELINE II=1
        uint32_t t=0;
        for (int lane=0;lane<4;++lane) t|=uint32_t(uint8_t(src[i+lane]))<<(lane*8);
        dst[i/4]=t;
    }
}
static int64_t load64(const uint32_t* src) {
    return int64_t((uint64_t(src[1])<<32)|src[0]);
}
static void store64(uint32_t* dst,int64_t x) {
    dst[0]=uint32_t(x);
    dst[1]=uint32_t(uint64_t(x)>>32);
}
static void read_history(const uint32_t* src, History& h) {
    h.count=load64(src);
    for (int c=0;c<256;++c) h.sum[c]=load64(src+2+2*c);
    unsigned i=514;
    for (int k=0;k<5;++k) for (int t=0;t<64;++t) h.scale[k][t]=src[i++];
    for (int t=0;t<64;++t,i+=48) unpack_bytes(src+i,h.key[t],192);
    for (int t=0;t<64;++t,i+=24) unpack_bytes(src+i,h.val[t],96);
    for (int t=0;t<64;++t,i+=24) unpack_bytes(src+i,h.val2[t],96);
    for (int k=0;k<2;++k) for (int t=0;t<30;++t,i+=64) unpack_bytes(src+i,h.conv[k][t],256);
}
static void write_history(uint32_t* dst,const History& h) {
    store64(dst,h.count);
    for (int c=0;c<256;++c) store64(dst+2+2*c,h.sum[c]);
    unsigned i=514;
    for (int k=0;k<5;++k) for (int t=0;t<64;++t) dst[i++]=uint32_t(h.scale[k][t]);
    for (int t=0;t<64;++t,i+=48) pack_bytes(dst+i,h.key[t],192);
    for (int t=0;t<64;++t,i+=24) pack_bytes(dst+i,h.val[t],96);
    for (int t=0;t<64;++t,i+=24) pack_bytes(dst+i,h.val2[t],96);
    for (int k=0;k<2;++k) for (int t=0;t<30;++t,i+=64) pack_bytes(dst+i,h.conv[k][t],256);
}
static bool read_params(const uint32_t* src, Params& p, unsigned rows) {
    unsigned i=0, offset=0;
    for (int op=0;op<op_count;++op) {
        if (src[i++]!=offset) return false;
        p.offsets[op]=offset;
        offset+=2*((cols_out(op)+31)/32)*cols_in(op);
    }
    for (int op=0;op<op_count;++op)
        for (int j=0;j<max_width;++j) p.scales[op][j]=int32_t(src[i++]);
    for (int op=0;op<op_count;++op)
        for (int j=0;j<max_width;++j) p.bias[op][j]=int32_t(src[i++]);
    p.epsilon=int32_t(src[i++]);
    p.beta=int32_t(src[i++]);
    p.position_scale=int32_t(src[i++]);
    for (unsigned r=0;r<6*rows-1;++r,i+=4) unpack_bytes(src+i,p.position[r],16);
    return p.epsilon>0 && p.position_scale>0;
}
extern "C" int fz_int8_layer(WeightPort weights, const uint32_t* params, const uint32_t* input,
        const uint32_t* old_state, uint32_t* output, uint32_t* working_state,
        unsigned rows, unsigned batch, volatile unsigned& stage,
        volatile uint8_t* mac_event, volatile uint8_t* vector_event) {
#pragma HLS INTERFACE m_axi port=weights bundle=weight offset=slave max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=params bundle=coeff offset=slave
#pragma HLS INTERFACE m_axi port=input bundle=data offset=slave
#pragma HLS INTERFACE m_axi port=old_state bundle=data offset=slave
#pragma HLS INTERFACE m_axi port=output bundle=data offset=slave
#pragma HLS INTERFACE m_axi port=working_state bundle=data offset=slave
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=old_state bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=working_state bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=batch bundle=control
#pragma HLS INTERFACE ap_none port=stage
#pragma HLS INTERFACE ap_fifo port=mac_event
#pragma HLS INTERFACE ap_fifo port=vector_event
    static Work w;
    static History h;
    static Boundary in, out;
    static Params p;
    stage=0;
    if ((rows!=2 && rows!=4 && rows!=8 && rows!=16) || (batch!=1 && batch!=2)) return 1;
    if (!read_params(params,p,rows)) return 2;
    read_history(old_state,h);
    if (h.count<0 || h.count>(int64_t(1)<<30)) return 2;
    for (unsigned b=0;b<batch;++b) {
#ifdef FZ_CACHED_SCALE
        w.feed_scale[b]=0;
#endif
        in.scale[b]=int32_t(input[b]);
        if (in.scale[b]<=0 || in.scale[b]>INT32_MAX/127) return 3;
        for (unsigned r=0;r<rows;++r) unpack_bytes(input+2+b*1024+r*64,in.codes[b][r],256);
        for (unsigned j=0;j<rows*channels;++j) {
            FZ_ELEMENT_PIPELINE
            w.feed[b][j]=w.x[b][j]=in.codes[b][j/256][j%256]*in.scale[b];
        }
    }
    for (int op=0;op<op_count;++op) {
        stage=1+op;
        if (op==pool)
            for (unsigned b=0;b<batch;++b) prefix(b,rows,h,w);
        weighted(op,rows,batch,weights,p,w,mac_event,vector_event);
        for (unsigned b=0;b<batch;++b) {
            int size=int(rows)*cols_out(op);
#ifdef FZ_FUSED_COPY
            if(op==a1_in || op==a2_in) {
                w.feed_scale[b]=0;
                attention(op,b,rows,p,h,w);
            } else if(op==c1_in || op==c2_in) {
                convolution_input(op,b,rows,h,w);
            } else {
                const bool activation=op==f1_in || op==f2_in || op==f3_in || op==c1_dw || op==c2_dw;
                finish_projection(activation,b,size,w);
            }
#else
            if (op==f1_in || op==f2_in || op==f3_in || op==c1_dw || op==c2_dw) {
                for (int i=0;i<size;++i) {
                    FZ_ELEMENT_PIPELINE
                    w.raw[b][i]=swish(w.raw[b][i]);
                }
#ifdef FZ_CACHED_SCALE
                w.feed_scale[b]=
#endif
                quantized_copy(w.feed[b],w.raw[b],size);
            } else if (op==a1_in || op==a2_in) {
#ifdef FZ_CACHED_SCALE
                w.feed_scale[b]=0;
#endif
                attention(op,b,rows,p,h,w);
            } else if (op==c1_in || op==c2_in) {
#ifdef FZ_CACHED_SCALE
                w.feed_scale[b]=0;
#endif
                convolution_input(op,b,rows,h,w);
            } else {
                for (int i=0;i<size;++i) {
                    FZ_ELEMENT_PIPELINE
                    w.x[b][i]+=w.raw[b][i];
                }
#ifdef FZ_CACHED_SCALE
                w.feed_scale[b]=
#endif
                quantized_copy(w.x[b],w.x[b],size);
                for (int i=0;i<size;++i) {
                    FZ_ELEMENT_PIPELINE
                    w.feed[b][i]=w.x[b][i];
                }
            }
#endif
        }
    }
    stage=1+op_count;
    for (unsigned b=0;b<batch;++b) normalize(b,rows,p,in,out,w);
    stage=2+op_count;
    write_history(working_state,h);
    for (unsigned b=0;b<batch;++b) {
        output[b]=uint32_t(out.scale[b]);
        for (unsigned r=0;r<rows;++r)
            pack_bytes(output+2+b*1024+r*64,out.codes[b][r],256);
    }
    stage=0;
    return 0;
}

// Q.K, the relative-position dot, and both byte planes of weighted V share the
// same eight signed multipliers. Buffers are banked in the consumed dimension.
#ifdef FZ_BURST_FLOW
#include "attention_strip.inc"
#else
static int32_t attention_dot(int task,int index,int head,int rows,int r,
        const int8_t query[48],const int8_t pos[4],const Params& p,
        const int8_t keys[80][192],const int8_t values[80][96],
        const int8_t high[80],const int8_t low[80]) {
#pragma HLS INLINE off
    int length=task==0 ? 48 : task==1 ? 4 : 5*rows;
    int32_t sum=0;
    for (int base=0;base<length;base+=8) {
#pragma HLS PIPELINE II=1
        int32_t terms[8];
#pragma HLS ARRAY_PARTITION variable=terms complete
        for(int i=0;i<8;++i) {
#pragma HLS UNROLL
            const int at=base+i;
            int8_t a=0,b=0;
            if(at<length) {
                if(task==0){ a=query[at]; b=keys[index][head*48+at]; }
                else if(task==1){ a=pos[at]; b=p.position[rows-1-r+index][head*4+at]; }
                else { a=task==2 ? high[at] : low[at]; b=values[at][head*24+index]; }
            }
            terms[i]=int32_t(a)*b;
        }
        sum+=((terms[0]+terms[1])+(terms[2]+terms[3]))+
             ((terms[4]+terms[5])+(terms[6]+terms[7]));
    }
    return sum;
}
#endif

void attention(int op,int b,int rows,const Params& p,History& h,Work& w) {
#pragma HLS INLINE off
#ifdef FZ_BURST_FLOW
#pragma HLS ALLOCATION function instances=attention_strip limit=1
#else
#pragma HLS ALLOCATION function instances=attention_dot limit=1
#endif
    const bool first=op==a1_in;
    int8_t keys[80][192], values[80][96];
#pragma HLS ARRAY_PARTITION variable=keys cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=values cyclic factor=8 dim=1
    const int32_t* raw=w.raw[b];
    int32_t sk=1,sq=1,sp=1;
    if(first){
        sk=join(raw,rows,496,192,h.key,h.scale[0],keys);
        sq=slice_scale(raw,rows,496,0,192);
        sp=slice_scale(raw,rows,496,480,16);
    }
    const auto qr=ratio(sq), pr=ratio(sp);
    const int32_t sv=first ? join(raw,rows,496,384,h.val,h.scale[1],values)
                          : join(raw,rows,96,0,h.val2,h.scale[2],values);
    const int n=5*rows;
    for(int head=0;head<4;++head){
#ifdef FZ_BURST_FLOW
        uint32_t positions[95];
        if(first) for(int i=0;i<6*rows-1;++i) {
            uint32_t word=0;
            for(int j=0;j<4;++j) word|=uint32_t(uint8_t(p.position[i][4*head+j]))<<(8*j);
            positions[i]=word;
        }
#endif
        for(int r=0;r<rows;++r){
            int8_t query[48], pos[4], high[80], low[80];
#pragma HLS ARRAY_PARTITION variable=query cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=pos complete
#pragma HLS ARRAY_PARTITION variable=high cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=low cyclic factor=8
            int32_t score[80],peak=INT32_MIN;
#ifdef FZ_BURST_FLOW
            int32_t dots[80],relative[80];
#endif
            if(first){
                for(int i=0;i<48;++i) query[i]=int8_t(scale_code(raw[r*496+head*48+i],qr));
                for(int i=0;i<4;++i) pos[i]=int8_t(scale_code(raw[r*496+480+head*4+i],pr));
#ifdef FZ_BURST_FLOW
                attention_strip(0,head,rows,r,query,pos,positions,keys,values,high,low,dots);
                attention_strip(1,head,rows,r,query,pos,positions,keys,values,high,low,relative);
#endif
                for(int t=0;t<n;++t){
#ifdef FZ_BURST_FLOW
                    FZ_ELEMENT_PIPELINE
                    const int32_t qk=dots[t],rel=relative[t];
#else
                    int32_t qk=attention_dot(0,t,head,rows,r,query,pos,p,keys,values,high,low);
                    int32_t rel=attention_dot(1,t,head,rows,r,query,pos,p,keys,values,high,low);
#endif
                    score[t]=int32_t(divide(int64_t(qk)*sq*sk,65536)
                                  +divide(int64_t(rel)*sp*p.position_scale,65536));
                    if(score[t]>peak) peak=score[t];
                }
            }
            int32_t denom=0;
            for(int t=0;t<n;++t){
#ifdef FZ_BURST_FLOW
                FZ_ELEMENT_PIPELINE
#endif
                if(first){
                    int64_t code=divide((int64_t(score[t])-peak)*756388,int64_t(1)<<32);
                    w.logp[b][head][r][t]=int8_t(code < -128 ? -128 : code);
                }
                int e=exp_weight(w.logp[b][head][r][t]);
                high[t]=int8_t(e>>7); low[t]=int8_t(e&127); denom+=e;
            }
            const auto weight_ratio=row_ratio(denom);
#ifdef FZ_BURST_FLOW
            attention_strip(2,head,rows,r,query,pos,positions,keys,values,high,low,dots);
            attention_strip(3,head,rows,r,query,pos,positions,keys,values,high,low,relative);
#endif
            for(int d=0;d<24;++d){
#ifdef FZ_BURST_FLOW
                const int32_t hi=dots[d],lo=relative[d];
#else
                int32_t hi=attention_dot(2,d,head,rows,r,query,pos,p,keys,values,high,low);
                int32_t lo=attention_dot(3,d,head,rows,r,query,pos,p,keys,values,high,low);
#endif
                w.feed[b][r*96+head*24+d]=int32_t(apply_row_ratio((int64_t(hi)*128+lo)*sv,weight_ratio));
            }
        }
    }
}

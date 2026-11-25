static void multiply(bool dw,int rows,int batch,int cols,int depth,
        const Weights32 inputs[3][288],hls::stream<Weights32>& weights,
        hls::stream<Sums32>& results,volatile uint8_t* mac_event) {
#ifdef FZ_RING_ACC
    // 768 signed-byte products have magnitude at most 12,582,912.
#ifdef __SYNTHESIS__
    using Partial=ap_int<25>;
#else
    using Partial=int32_t;
#endif
    static Partial recent[3][32][12];
#pragma HLS ARRAY_PARTITION variable=recent complete dim=0
#else
#if defined(FZ_SDP_ACC) && defined(__SYNTHESIS__)
    // At most 768 signed-byte products per output; one independent read and write per lane.
    ap_int<25> acc[3][12][32];
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram latency=1
#else
    int32_t acc[3][12][32];
#endif
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#endif
    const unsigned groups=(rows+2)/3, live_visits=groups*batch, stride=dw?8:depth/32;
#ifdef FZ_SPACED_ACC
    // Every access uses visit as the address. Padding only unused slots makes a
    // four-iteration true dependency distance valid even for a one-block tail.
    const unsigned visits=live_visits<4 ? 4 : live_visits;
#else
    const unsigned visits=live_visits;
#endif
    for(int tile=0;tile<(cols+31)/32;++tile) {
        Weights32 word{};
        unsigned char visit=0,group=0,b=0,rotation=0;
        uint16_t k=0,third=0;
        const unsigned iterations=depth*visits;
        FLAT_ISSUE: for(unsigned step=0;step<iterations;++step) {
#pragma HLS PIPELINE II=1
#ifdef FZ_SPACED_ACC
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=4 dependent=true
#endif
            if(visit==0) word=weights.read();
            const unsigned row=3*group;
            const unsigned origin=b*144+group*stride+(dw ? 8*third+tile : k/32);
            Weights32 in[3];
#pragma HLS ARRAY_PARTITION variable=in complete
            for(int bank=0;bank<3;++bank) {
#pragma HLS UNROLL
                in[bank]=inputs[bank][origin+(dw && bank<rotation ? 8 : 0)];
            }
            for(int lane=0;lane<3;++lane) {
#pragma HLS UNROLL
                int bank=lane+(dw ? rotation : 0);
                if(bank>=3) bank-=3;
                const Weights32 selected=in[bank];
                const int8_t dense=in[lane].half[(k%32)/16].byte[k%16];
                for(int col=0;col<32;++col) {
#pragma HLS UNROLL
                    const int8_t a=dw ? selected.half[col/16].byte[col%16] : dense;
                    const int8_t w=word.half[col/16].byte[col%16];
                    // Tail rows occupy private slots and never enter the output stream.
#ifdef FZ_RING_ACC
                    const int32_t product=int32_t(a)*w;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
                    const int32_t sum=(k==0 ? 0 : int32_t(recent[lane][col][visits-1]))+product;
                    for(int t=11;t>0;--t) {
#pragma HLS UNROLL
                        recent[lane][col][t]=recent[lane][col][t-1];
                    }
                    recent[lane][col][0]=Partial(sum);
#else
                    acc[lane][visit][col]=(k==0 ? 0 : int32_t(acc[lane][visit][col]))+int32_t(a)*w;
#endif
                }
            }
            const unsigned live_rows=rows-row<3 ? rows-row : 3;
            const unsigned live_cols=cols-32*tile<32 ? cols-32*tile : 32;
#ifdef FZ_SPACED_ACC
            *mac_event=uint8_t(visit<live_visits ? live_rows*live_cols : 0);
#else
            *mac_event=uint8_t(live_rows*live_cols);
#endif
            if(visit+1==visits) {
                visit=group=b=0; ++k;
                if(rotation==2){rotation=0; ++third;} else ++rotation;
            } else {
                ++visit;
#ifdef FZ_SPACED_ACC
                if(visit>=live_visits){group=b=0;} else
#endif
                if(group+1==groups){group=0;b=1;} else ++group;
            }
        }
        for(int i=0;i<rows*batch;++i) {
#pragma HLS PIPELINE II=1
            const int row=i>=rows ? i-rows : i;
            const int v=(i>=rows ? groups : 0)+row/3;
            Sums32 s;
            for(int col=0;col<32;++col) {
#pragma HLS UNROLL
#ifdef FZ_RING_ACC
                s.value[col]=int32_t(recent[row%3][col][visits-1-v]);
#else
                s.value[col]=acc[row%3][v][col];
#endif
            }
            results.write(s);
        }
    }
}

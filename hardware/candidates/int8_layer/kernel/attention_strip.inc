// One launch covers all scores or all output channels for a row and head.
// With aligned eight-element groups, each RAM bank contributes at most one byte.
static void attention_strip(int mode, int head, int rows, int r, const int8_t query[48],
        const int8_t pos[4], const uint32_t positions[95], const int8_t keys[80][192],
        const int8_t values[80][96], const int8_t high[80], const int8_t low[80], int32_t result[80]) {
#pragma HLS INLINE off
    const int n=5*rows;
    const int length=mode==0 ? 48 : mode==1 ? 4 : n;
    const int outputs=mode<2 ? n : 24;
    const int groups=(length+7)/8;
    unsigned block=0, index=0;
    int32_t partial=0;
    DOT_GROUPS: for(int step=0;step<groups*outputs;++step) {
#pragma HLS PIPELINE II=1
        const unsigned base=block*8;
        const uint32_t pos_word=mode==1 ? positions[rows-1-r+index] : 0;
        int32_t terms[8];
#pragma HLS ARRAY_PARTITION variable=terms complete
        for(unsigned lane=0;lane<8;++lane) {
#pragma HLS UNROLL
            const unsigned at=base+lane;
            int8_t a=0,b=0;
            if(at<unsigned(length)) {
                if(mode==0) {a=query[at]; b=keys[index][head*48+at];}
                else if(mode==1) {a=pos[lane%4]; b=int8_t(pos_word>>(8*(lane%4)));}
                else {a=mode==2?high[at]:low[at]; b=values[at][head*24+index];}
            }
            terms[lane]=int32_t(a)*b;
        }
        const int32_t x=((terms[0]+terms[1])+(terms[2]+terms[3]))+
                        ((terms[4]+terms[5])+(terms[6]+terms[7]));
        partial=(block?partial:0)+x;
        if(block+1==unsigned(groups)){result[index++]=partial;block=0;} else ++block;
    }
}

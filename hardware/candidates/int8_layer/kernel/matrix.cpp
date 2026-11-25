#include "scaling.hpp"
#ifdef FZ_WEIGHT_BURSTS
#include <cstring>
#endif
#ifdef __SYNTHESIS__
#include <hls_stream.h>
#if defined(FZ_RING_ACC) || defined(FZ_SDP_ACC)
#include <ap_int.h>
#endif
#else
#include <deque>
namespace hls {
template<class T> class stream {
    std::deque<T> data;
public:
    void write(const T& x) { data.push_back(x); }
    T read() { T x = data.front(); data.pop_front(); return x; }
};
}
#endif

namespace fzint {
struct Weights32 { Word half[2]; };
struct Sums32 { int32_t value[32]; };

static void fetch(WeightPort bank,
#ifdef FZ_REQUEST_WEIGHTS
                  unsigned word_offset,
#endif
                  int tiles, int k, hls::stream<Weights32>& weights) {
#ifdef FZ_REQUEST_WEIGHTS
    const unsigned beats=2*unsigned(tiles)*unsigned(k);
    REQUEST_WEIGHTS: for(unsigned start=0;start<beats;start+=64){
        const unsigned count=beats-start<64 ? beats-start : 64;
#ifdef __SYNTHESIS__
        bank.read_request(word_offset+start,count);
#endif
        REQUEST_PAIRS: for(unsigned i=0;i<count;i+=2){
#pragma HLS PIPELINE II=2
#ifdef __SYNTHESIS__
            Word first=bank.read();
            Word second=bank.read();
#else
            Word first=bank[word_offset+start+i];
            Word second=bank[word_offset+start+i+1];
#endif
            weights.write(Weights32{{first,second}});
        }
    }
#elif defined(FZ_WEIGHT_BURSTS)
    constexpr unsigned beat_count=16;
    Word local[beat_count];
#pragma HLS ARRAY_PARTITION variable=local cyclic factor=2
    const unsigned beats=2*unsigned(tiles)*unsigned(k);
    const unsigned full=beats/beat_count;
    WEIGHT_BURSTS: for(unsigned block=0;block<full;++block){
        // The fixed contiguous copy has no dependency on output FIFO readiness.
        // The matching address range is exactly 256 bytes at the weight port.
        std::memcpy(local,bank+block*beat_count,sizeof(local));
        WEIGHT_PAIRS: for(unsigned p=0;p<beat_count;p+=2){
#pragma HLS PIPELINE II=1
            weights.write(Weights32{{local[p],local[p+1]}});
        }
    }
    for(unsigned p=full*beat_count;p<beats;p+=2){
        weights.write(Weights32{{bank[p],bank[p+1]}});
    }
#elif defined(FZ_BURST_FLOW)
    Word first;
    WEIGHT_BEAT: for(int beat=0;beat<2*tiles*k;++beat) {
#pragma HLS PIPELINE II=1
        Word w=bank[beat];
        if(beat%2) weights.write(Weights32{{first,w}});
        else first=w;
    }
#else
    for (int i = 0; i < tiles * k; ++i) {
#pragma HLS PIPELINE II=2
        Weights32 w{{bank[2*i], bank[2*i+1]}};
        weights.write(w);
    }
#endif
}

#ifdef FZ_FAST_FLOW
#include "fast_matrix.inc"
#else
static void multiply(bool dw, int rows, int batch, int cols, int depth,
                     const Weights32 inputs[3][288], hls::stream<Weights32>& weights,
                     hls::stream<Sums32>& results, volatile uint8_t* mac_event) {
    const int count = rows * batch;
    int32_t acc[3][12][32];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
    for (int tile = 0; tile < (cols+31)/32; ++tile) {
        for (int k = 0; k < depth; ++k) {
            const Weights32 w = weights.read();
            const int groups = (rows+2)/3, stride = dw ? 8 : depth/32;
            MAC_ISSUE: for (int visit=0; visit<groups*batch; ++visit) {
#pragma HLS PIPELINE II=1
                const int b=visit>=groups, base=3*(visit-b*groups);
                const int shifted=base+(dw?k:0), rot=dw?k%3:0;
                Weights32 banked[3];
#pragma HLS ARRAY_PARTITION variable=banked complete
                for (int bank=0;bank<3;++bank) {
#pragma HLS UNROLL
                    int delta=bank-rot;
                    if (delta<0) delta+=3;
                    int address=b*144+(shifted+delta)/3*stride+(dw?tile:k/32);
                    banked[bank]=inputs[bank][address];
                }
                for (int row_lane = 0; row_lane < 3; ++row_lane) {
#pragma HLS UNROLL
                    const int row = base + row_lane;
                    if (row < rows) {
                        int bank=rot+row_lane;
                        if (bank>=3) bank-=3;
                        const Weights32 selected=banked[bank];
                        for (int col = 0; col < 32; ++col) {
#pragma HLS UNROLL
                            int byte=dw?col:k%32;
                            int8_t a=selected.half[byte/16].byte[byte%16];
                            int8_t weight = w.half[col/16].byte[col%16];
                            acc[row_lane][visit][col] = (k == 0 ? 0 : acc[row_lane][visit][col]) + int32_t(a)*weight;
                        }
                    }
                }
                int live_rows=rows-base < 3 ? rows-base : 3;
                int live_cols=cols-tile*32 < 32 ? cols-tile*32 : 32;
                *mac_event=uint8_t(live_rows*live_cols);
            }
        }
        for (int r = 0; r < count; ++r) {
#pragma HLS PIPELINE II=1
            Sums32 s;
            const int b=r>=rows, row=r-b*rows;
            const int visit=b*((rows+2)/3)+row/3;
            for (int col = 0; col < 32; ++col) {
#pragma HLS UNROLL
                s.value[col] = acc[row%3][visit][col];
            }
            results.write(s);
        }
    }
}

#endif
static void convert(int rows, int batch, int cols, const int32_t sx[2], const int32_t* scale,
                    const int32_t* bias, hls::stream<Sums32>& results, int32_t raw[2][elements],
                    volatile uint8_t* vector_event) {
    for (int tile = 0; tile < (cols+31)/32; ++tile) {
        int32_t sw[32], bias_tile[32];
#pragma HLS ARRAY_PARTITION variable=sw cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=bias_tile cyclic factor=4
        for (int c = 0; c < 32; ++c) {
#pragma HLS PIPELINE II=1
            const int col = tile*32 + c;
            sw[c] = col < cols ? scale[col] : 0;
            bias_tile[c] = col < cols ? bias[col] : 0;
        }
        for (int r = 0; r < rows * batch; ++r) {
            const Sums32 v = results.read();
            const int b = r >= rows, y = r - b*rows;
            FOUR_RESULTS: for (int c = 0; c < 32; c += 4) {
#pragma HLS PIPELINE II=1
                if (tile*32+c < cols) *vector_event=4;
                for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                    const int col = tile*32+c+lane;
                    if (col < cols) {
                        int64_t sum = int64_t(v.value[c+lane]) * sx[b] * sw[c+lane];
                        raw[b][y*cols+col] = int32_t(divide(sum, 1<<24)) + bias_tile[c+lane];
                    }
                }
            }
        }
    }
}

static void engine(WeightPort bank,
#ifdef FZ_REQUEST_WEIGHTS
                   unsigned word_offset,
#endif
                   bool dw, int rows, int batch, int cols, int tiles, int k,
                   const Weights32 inputs[3][288], const int32_t sx[2],
                   const int32_t* scale, const int32_t* bias, int32_t raw[2][elements],
                   volatile uint8_t* mac_event, volatile uint8_t* vector_event) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<Weights32> weights;
    hls::stream<Sums32> results;
#pragma HLS STREAM variable=weights depth=64
#pragma HLS STREAM variable=results depth=8
    fetch(bank,
#ifdef FZ_REQUEST_WEIGHTS
          word_offset,
#endif
          tiles, k, weights);
    multiply(dw, rows, batch, cols, k, inputs, weights, results, mac_event);
    convert(rows, batch, cols, sx, scale, bias, results, raw, vector_event);
}

void weighted(int op, int rows, int batch, WeightPort bank, const Params& p, Work& w,
              volatile uint8_t* mac_event, volatile uint8_t* vector_event) {
#pragma HLS INLINE off
    static Weights32 inputs[3][288];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=1
#pragma HLS AGGREGATE variable=inputs compact=bit
    int32_t sx[2] = {1,1};
    const bool dw = op == c1_dw || op == c2_dw;
    const int k = cols_in(op), cols = cols_out(op);
    const int n = dw ? (30+rows)*channels : rows*k;
    for (int b=0; b<batch; ++b) {
        sx[b] =
#ifdef FZ_CACHED_SCALE
            w.feed_scale[b]>0 ? w.feed_scale[b] :
#endif
            tensor_scale(w.feed[b], n);
        const auto divisor = ratio(sx[b]);
        const int stride = dw ? 8 : k/32;
#ifdef FZ_CACHED_SCALE
        Weights32 packed{};
        unsigned word=0, bank_id=0, base=0;
        PACK_INPUT: for(int j=0;j<n;++j) {
#pragma HLS PIPELINE II=1
            unsigned lane=unsigned(j)%32;
            packed.half[lane/16].byte[lane%16]=int8_t(scale_code(w.feed[b][j],divisor));
            if(lane==31) {
                inputs[bank_id][b*144+base+word]=packed;
                if(++word==unsigned(stride)){
                    word=0;
                    if(++bank_id==3){bank_id=0;base+=stride;}
                }
            }
        }
#else
        for (int r=0;r<(dw?30+rows:rows);++r) {
            for (int word=0;word<stride;++word) {
                Weights32 packed;
                for (int lane=0;lane<32;++lane) {
#pragma HLS PIPELINE II=1
                    int j=(r*stride+word)*32+lane;
                    packed.half[lane/16].byte[lane%16]=int8_t(scale_code(w.feed[b][j],divisor));
                }
                inputs[r%3][b*144+r/3*stride+word]=packed;
            }
        }
#endif
    }
    engine(
#ifdef FZ_REQUEST_WEIGHTS
           bank,p.offsets[op],
#else
           bank+p.offsets[op],
#endif
           dw, rows, batch, cols, (cols+31)/32, k, inputs, sx,
           p.scales[op], p.bias[op], w.raw, mac_event, vector_event);
}
}

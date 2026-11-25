#pragma once
#include <cstdint>
#if defined(FZ_REQUEST_WEIGHTS) && defined(__SYNTHESIS__)
#include <hls_burst_maxi.h>
#endif
#if defined(FZ_STREAM_FLOW) && defined(__SYNTHESIS__)
#define FZ_ELEMENT_PIPELINE _Pragma("HLS PIPELINE II=1")
#else
#define FZ_ELEMENT_PIPELINE
#endif

namespace fzint {
constexpr int channels = 256, max_rows = 16, max_left = 64, max_width = 768;
constexpr int elements = max_rows * max_width;
enum Op { f1_in, f1_out, pool, a1_in, a1_out, c1_in, c1_dw, c1_out, f2_in, f2_out,
          a2_in, a2_out, c2_in, c2_dw, c2_out, f3_in, f3_out, op_count };
struct Word { int8_t byte[16]; };
#if defined(FZ_REQUEST_WEIGHTS) && defined(__SYNTHESIS__)
using WeightPort=hls::burst_maxi<Word>;
#else
using WeightPort=const Word*;
#endif
struct Params {
    int32_t offsets[op_count];                 // 128-bit word offsets in the weight bank
    int32_t scales[op_count][max_width];       // Q24, one per output channel
    int32_t bias[op_count][max_width];         // Q16
    int32_t epsilon, beta, position_scale;     // Q16
    int8_t position[95][16];                  // once-projected, INT8 coded table
};
struct History {
    int64_t count, sum[channels];              // Q16 sum has extended range
    int32_t scale[5][max_left];
    int8_t key[max_left][192], val[max_left][96], val2[max_left][96];
    int8_t conv[2][30][channels];
};
struct Boundary {
    int32_t scale[2];
    int8_t codes[2][max_rows][channels];
};
struct Work {
#ifdef FZ_CACHED_SCALE
    int32_t feed_scale[2];  // exact scale of the expanded tensor, zero if a producer did not compute it
#endif
    int32_t x[2][max_rows * channels];         // residual stream
    int32_t feed[2][elements];                 // working operands, Q16
    int32_t raw[2][elements];                  // projection outputs until consumed
    int8_t logp[2][4][max_rows][max_left + max_rows];
};
inline int cols_in(int op) {
    if (op == f1_out || op == f2_out || op == f3_out) return 768;
    if (op == a1_out || op == a2_out) return 96;
    if (op == c1_dw || op == c2_dw) return 31;
    return 256;
}
inline int cols_out(int op) {
    if (op == f1_in || op == f2_in || op == f3_in) return 768;
    if (op == a1_in) return 496;
    if (op == a2_in) return 96;
    if (op == c1_in || op == c2_in) return 512;
    return 256;
}
// Floor quotient adjusted to round-nearest-even. All callers have positive divisors.
inline int64_t divide(int64_t n, int64_t d) {
    const int64_t q = n / d, r = n % d;
    const int64_t magnitude = r < 0 ? -r : r;
    return q + ((magnitude > d - magnitude || (magnitude == d - magnitude && (q & 1)))
                ? (n < 0 ? -1 : 1) : 0);
}
inline int32_t scale_for(int64_t peak) { return peak ? int32_t((peak + 126) / 127) : 1; }
inline int64_t mag(int32_t x) { return x < 0 ? -int64_t(x) : int64_t(x); }
int32_t tensor_scale(const int32_t* src, int size);
int32_t quantized_copy(int32_t* dst, const int32_t* src, int size);
void weighted(int op, int rows, int batch, WeightPort bank, const Params& p, Work& work,
              volatile uint8_t* mac_event, volatile uint8_t* vector_event);
void attention(int op, int b, int rows, const Params& p, History& h, Work& work);
void convolution_input(int op, int b, int rows, History& h, Work& work);
void prefix(int b, int rows, History& h, Work& work);
void normalize(int b, int rows, const Params& p, const Boundary& in, Boundary& out, Work& work);
#ifdef FZ_FUSED_COPY
void finish_projection(bool activation, int b, int size, Work& work);
#endif
int32_t swish(int32_t x);
}
extern "C" int fz_int8_layer(fzint::WeightPort weights, const uint32_t* params,
    const uint32_t* input, const uint32_t* old_state, uint32_t* output,
    uint32_t* working_state, unsigned rows, unsigned batch, volatile unsigned& stage,
    volatile uint8_t* mac_event, volatile uint8_t* vector_event);

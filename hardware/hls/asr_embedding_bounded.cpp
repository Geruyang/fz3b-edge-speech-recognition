#include "asr_embedding_bounded.hpp"
#include "mac_array.hpp"
#include "vector_ops.hpp"

#include <cstring>

namespace {
using namespace edge_speech::pl;
struct Stage { unsigned m, n, k, height, width, channels, output_width, sh, sw, pw; };
constexpr Stage stages[] = {
    {37 * 80, 8, 9, 39, 80, 1, 80, 1, 1, 1},
    {18 * 39, 32, 72, 37, 80, 8, 39, 2, 2, 0},
    {16 * 19, 128, 288, 18, 39, 32, 19, 1, 2, 0},
    {16, 256, 2432, 16, 19, 128, 0, 1, 1, 0},
};
unsigned tiles(unsigned x, unsigned tile) { return (x + tile - 1) / tile; }
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
bool finite(float x) { return (bits(x) & 0x7f800000u) != 0x7f800000u; }

int8_t read_code(const uint32_t* words, unsigned index) {
#pragma HLS INLINE
    const unsigned byte = (words[index / 4] >> (8 * (index % 4))) & 255u;
    return int8_t(int(byte ^ 128u) - 128);
}
void write_code(uint32_t* words, unsigned index, int8_t code) {
#pragma HLS INLINE
    const unsigned shift = 8 * (index % 4);
    const uint32_t previous = index % 4 ? words[index / 4] : 0;
    words[index / 4] = (previous & ~(255u << shift)) | (uint32_t(uint8_t(code)) << shift);
}
int8_t code_at(const uint32_t* codes, Stage s, unsigned row, unsigned k, uint64_t& reads) {
#pragma HLS INLINE
    if (s.output_width == 0) {
        ++reads;
        return read_code(codes, (row * s.width + k % s.width) * s.channels + k / s.width);
    }
    const unsigned y = (row / s.output_width) * s.sh + (k % 9) / 3;
    const int x = int((row % s.output_width) * s.sw + k % 3) - int(s.pw);
    if (y >= s.height || x < 0 || x >= int(s.width)) return 0;
    ++reads;
    return read_code(codes, (y * s.width + unsigned(x)) * s.channels + k / 9);
}

// pass=0 scans the whole activated tensor; pass=1 regenerates it and emits
// INT8 at that tensor's amax. pass=2 writes the final FP32 projection.
AsrStatus run_pass(Stage s, const uint64_t* weights, const float* scales, const float* bias,
                   const uint32_t* source, float sx, unsigned pass, float& amax,
                   uint32_t* destination, uint32_t* final_output, bool debug, int32_t* raw,
                   uint64_t accounting[embedding_accounting_words],
                   float values[embedding_bounded_tile_elements], int8_t encoded[embedding_bounded_tile_elements]) {
#pragma HLS INLINE off
    bool nonfinite = false;
    float next_scale = 1;
    if (pass == 1 && asr_scale_from_amax(amax, next_scale) != VectorStatus::ok)
        return AsrStatus::invalid_quantization;
    for (unsigned nt = 0; nt < tiles(s.n, mac_n_tile); ++nt) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2
        const unsigned n0 = nt * mac_n_tile;
        const uint16_t n_live = s.n - n0 < mac_n_tile ? s.n - n0 : mac_n_tile;
        float tile_scales[mac_n_tile], tile_bias[mac_n_tile];
        for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE off
            tile_scales[n] = scales[n0 + n];
            tile_bias[n] = bias[n0 + n];
            accounting[1] += 8;
        }
        bool invalid_parameters = false;
        for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
            if (!valid_asr_weight_scale(tile_scales[n]) || !finite(tile_bias[n]))
                invalid_parameters = true;
        }
        if (invalid_parameters) return AsrStatus::invalid_parameter;
        for (unsigned m0 = 0; m0 < s.m; m0 += mac_m_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=185
            const uint16_t m_live = s.m - m0 < mac_m_tile ? s.m - m0 : mac_m_tile;
            int32_t collected[mac_m_tile][mac_n_tile];
            for (unsigned m = 0; m < m_live; ++m)
                for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                    collected[m][n] = 0;
                }
            for (unsigned kt = 0; kt < tiles(s.k, mac_k_tile); ++kt) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=38
                const unsigned k0 = kt * mac_k_tile;
                const uint16_t k_live = s.k - k0 < mac_k_tile ? s.k - k0 : mac_k_tile;
                const unsigned base = (nt * tiles(s.k, mac_k_tile) + kt) * mac_k_tile * mac_weight_banks;
                uint64_t tile[mac_k_tile][mac_weight_banks];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=2
                for (unsigned pair = 0; pair < mac_weight_banks / 2; ++pair)
                    for (unsigned k = 0; k < mac_k_tile; ++k) {
#pragma HLS PIPELINE II=1
                        const unsigned at = base + 2 * (pair * mac_k_tile + k);
                        tile[k][2 * pair] = weights[at];
                        tile[k][2 * pair + 1] = weights[at + 1];
                        accounting[0] += 16;
                    }
                for (unsigned m = 0; m < m_live; m += 2) {
                    int8_t activation[2][mac_k_tile];
#pragma HLS ARRAY_PARTITION variable=activation complete dim=1
                    const uint16_t m_pair = m + 1 < m_live ? 2 : 1;
                    for (unsigned k = 0; k < k_live; ++k) {
#pragma HLS PIPELINE II=1
                        activation[0][k] = code_at(source, s, m0 + m, k0 + k, accounting[7]);
                        activation[1][k] = m_pair == 2 ? code_at(source, s, m0 + m + 1, k0 + k, accounting[7]) : 0;
                    }
                    int32_t partial[2][mac_n_tile];
                    MacAccounting cost;
                    if (mac_pair(m_pair == 1 ? MacMode::w8_gemv : MacMode::w8_gemm,
                                 activation, tile, {m_pair, n_live, k_live}, partial, cost) != MacStatus::ok)
                        return AsrStatus::invalid_code;
                    accounting[5] += cost.issue_steps;
                    accounting[6] += cost.useful_macs;
                    for (unsigned row = 0; row < m_pair; ++row)
                        for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                            collected[m + row][n] += partial[row][n];
                        }
                }
            }
            for (unsigned m = 0; m < m_live; ++m) {
                for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                    const unsigned at = (m0 + m) * s.n + n0 + n;
                    const int32_t dot = collected[m][n];
                    const float x = asr_dequantize(dot, sx, tile_scales[n], tile_bias[n]);
                    nonfinite |= !finite(x);
                    const float y = pass == 2 ? x : double_swish(x);
                    nonfinite |= !finite(y);
                    values[n] = y;
                    if (debug && pass != 1) { raw[at] = dot; accounting[4] += 4; }
                }
                if (nonfinite) return AsrStatus::nonfinite_output;
                if (pass == 0) {
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                        const uint32_t magnitude = bits(values[n]) & 0x7fffffffu;
                        if (magnitude > bits(amax)) std::memcpy(&amax, &magnitude, 4);
                    }
                } else if (pass == 1) {
                    if (quantize_at_fixed_scale(values, encoded, n_live, next_scale) != VectorStatus::ok)
                        return AsrStatus::invalid_quantization;
                    ++accounting[9];
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=2
                        const unsigned at = (m0 + m) * s.n + n0 + n;
                        write_code(destination, at, encoded[n]);
                        ++accounting[8];
                    }
                } else {
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                        final_output[(m0 + m) * s.n + n0 + n] = bits(values[n]);
                    }
                }
            }
        }
    }
    return nonfinite ? AsrStatus::nonfinite_output : AsrStatus::ok;
}
} // namespace

uint8_t asr_embedding_bounded(const uint64_t* weights, const float* weight_scale,
    const float* bias, const float* input, float* output, bool debug_raw,
    int32_t* raw, float* stage_scales, uint64_t accounting[embedding_accounting_words]) {
#pragma HLS INTERFACE m_axi port=weights bundle=memory depth=86016 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=weight_scale bundle=memory depth=1024 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=bias bundle=memory depth=1024 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=input bundle=memory depth=3120 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=output bundle=memory depth=4096 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=raw bundle=memory depth=89152 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=stage_scales bundle=memory depth=4 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE ap_memory port=accounting
#pragma HLS INTERFACE s_axilite port=debug_raw bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ALLOCATION function instances=run_pass limit=1
    uint32_t codes_a[embedding_bounded_codes_a / 4];
    uint32_t buffer_b[embedding_bounded_codes_b / 4];
    float values[embedding_bounded_tile_elements];
    int8_t encoded[embedding_bounded_tile_elements];
#pragma HLS BIND_STORAGE variable=codes_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_b type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=values type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=encoded type=ram_2p impl=bram
    uint64_t cost[embedding_accounting_words] = {};
    uint32_t input_amax_bits = 0;
    bool invalid_input = false;
    for (unsigned i = 0; i < 39 * 80; ++i) {
#pragma HLS PIPELINE II=1
        const float value = input[i];
        invalid_input |= !finite(value);
        const uint32_t magnitude = bits(value) & 0x7fffffffu;
        if (magnitude > input_amax_bits) input_amax_bits = magnitude;
        cost[2] += 4;
    }
    if (invalid_input) return uint8_t(AsrStatus::invalid_quantization);
    float input_amax;
    std::memcpy(&input_amax, &input_amax_bits, 4);
    float sx;
    if (asr_scale_from_amax(input_amax, sx) != VectorStatus::ok)
        return uint8_t(AsrStatus::invalid_quantization);
    for (unsigned base = 0; base < 39 * 80; base += embedding_bounded_tile_elements) {
        const unsigned count = 39 * 80 - base < embedding_bounded_tile_elements ?
            39 * 80 - base : embedding_bounded_tile_elements;
        for (unsigned i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
            values[i] = input[base + i]; cost[2] += 4;
        }
        if (quantize_at_fixed_scale(values, encoded, count, sx) != VectorStatus::ok)
            return uint8_t(AsrStatus::invalid_quantization);
        ++cost[9];
        for (unsigned i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=2
            write_code(buffer_b, base + i, encoded[i]); ++cost[8];
        }
    }
    unsigned weight_at = 0, parameter_at = 0, raw_at = 0;
    for (unsigned stage_index = 0; stage_index < 4; ++stage_index) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        const Stage stage = stages[stage_index];
        const uint32_t* source = stage_index % 2 == 0 ? buffer_b : codes_a;
        uint32_t* destination = stage_index % 2 == 0 ? codes_a : buffer_b;
        if (debug_raw) { stage_scales[stage_index] = sx; cost[4] += 4; }
        float amax = 0;
        auto result = run_pass(stage, weights + weight_at, weight_scale + parameter_at, bias + parameter_at,
            source, sx, stage_index == 3 ? 2 : 0, amax, destination, buffer_b,
            debug_raw, raw + raw_at, cost, values, encoded);
        if (result != AsrStatus::ok) return uint8_t(result);
        if (stage_index != 3) {
            result = run_pass(stage, weights + weight_at, weight_scale + parameter_at, bias + parameter_at,
                source, sx, 1, amax, destination, buffer_b, false, raw + raw_at, cost, values, encoded);
            if (result != AsrStatus::ok) return uint8_t(result);
            if (asr_scale_from_amax(amax, sx) != VectorStatus::ok)
                return uint8_t(AsrStatus::invalid_quantization);
        }
        weight_at += tiles(stage.n, mac_n_tile) * tiles(stage.k, mac_k_tile) * mac_k_tile * mac_weight_banks;
        parameter_at += stage.n;
        raw_at += stage.m * stage.n;
    }
    for (unsigned i = 0; i < 16 * 256; ++i) {
#pragma HLS PIPELINE II=1
        float value;
        std::memcpy(&value, &buffer_b[i], 4);
        output[i] = value; cost[3] += 4;
    }
    for (unsigned i = 0; i < embedding_accounting_words; ++i) accounting[i] = cost[i];
    return uint8_t(AsrStatus::ok);
}

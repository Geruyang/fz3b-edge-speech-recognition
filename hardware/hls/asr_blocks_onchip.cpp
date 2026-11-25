// Independent T015 candidate. This file replaces asr_blocks.cpp in the
// candidate build and implements the same external interface.
#include "asr_embedding_bounded.hpp"
#include "mac_array.hpp"
#include "vector_ops.hpp"

#include <cstring>

namespace edge_speech::pl {
namespace {

constexpr unsigned word_count = mac_k_tile * mac_weight_banks;

enum class Layout : uint8_t { rows, image, flatten };

struct Stage {
    uint16_t m, n, k;
    Layout layout = Layout::rows;
    uint8_t height = 0, width = 0, input_channels = 0, output_width = 0;
    uint8_t stride_h = 1, stride_w = 1, pad_w = 0;
    bool activate = false;
};

constexpr Stage embedding_stages[] = {
    {37 * 80, 8, 9, Layout::image, 39, 80, 1, 80, 1, 1, 1, true},
    {18 * 39, 32, 72, Layout::image, 37, 80, 8, 39, 2, 2, 0, true},
    {16 * 19, 128, 288, Layout::image, 18, 39, 32, 19, 1, 2, 0, true},
    {16, 256, 2432, Layout::flatten, 16, 19, 128},
};

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool finite(float value) {
    return (float_bits(value) & 0x7f800000u) != 0x7f800000u;
}

bool compatible(const float* scales, const float* bias, unsigned n) {
    bool valid = true;
    for (unsigned channel = 0; channel < n; ++channel) {
#pragma HLS PIPELINE II=1
        const float scale = scales[channel];
        valid &= valid_asr_weight_scale(scale) && finite(bias[channel]);
    }
    return valid;
}

unsigned tile_count(unsigned elements, unsigned tile) {
    return (elements + tile - 1) / tile;
}

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

int8_t embedding_code(const uint32_t* codes, Stage stage, uint16_t m, uint16_t k) {
#pragma HLS INLINE
    if (stage.layout == Layout::flatten) {
        // The last Conv has spatial-major storage. Projection K follows C then F.
        return read_code(codes, (m * 19u + k % 19u) * 128u + k / 19u);
    }
    // Only these frozen Conv2d widths reach this path. Constant divisors and
    // bounded indices avoid inferring two general-purpose 32-bit dividers;
    // the dense/MAC engine remains shared across all stages.
    const uint16_t row = stage.output_width == 80 ? m / 80u :
                         stage.output_width == 39 ? m / 39u : m / 19u;
    const uint16_t column = stage.output_width == 80 ? m % 80u :
                            stage.output_width == 39 ? m % 39u : m % 19u;
    const uint16_t y = row * stage.stride_h + (k % 9) / 3;
    const int x = int(column * stage.stride_w + k % 3) - stage.pad_w;
    if (y >= stage.height || x < 0 || x >= stage.width) return 0;
    const uint16_t position = y * stage.width + unsigned(x);
    const unsigned base = stage.input_channels == 1 ? position :
                          stage.input_channels == 8 ? unsigned(position) << 3 : unsigned(position) << 5;
    return read_code(codes, base + k / 9);
}

enum class DensePass : uint8_t { external, scan, encode, projection };

AsrStatus dense(Stage stage, const uint64_t* weights, const float* scales, const float* bias,
                const int8_t* external_codes, float sx, float* next, bool debug_raw, int32_t* raw,
                DensePass pass, bool read_a, float& amax, uint32_t* codes_a, uint32_t* buffer_b,
                float values[embedding_bounded_tile_elements], int8_t encoded[embedding_bounded_tile_elements]) {
#pragma HLS INLINE off
    const bool onchip = pass != DensePass::external;
    const uint32_t* source = read_a ? codes_a : buffer_b;
    uint32_t* destination = read_a ? buffer_b : codes_a;
    if (!onchip && !compatible(scales, bias, stage.n)) return AsrStatus::invalid_parameter;
    float next_scale = 1;
    if (pass == DensePass::encode && asr_scale_from_amax(amax, next_scale) != VectorStatus::ok)
        return AsrStatus::invalid_quantization;

    bool invalid = false;
    for (unsigned nt = 0; nt < tile_count(stage.n, mac_n_tile); ++nt) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=6
        const unsigned n0 = nt * mac_n_tile;
        const uint16_t n_live = stage.n - n0 < mac_n_tile ? stage.n - n0 : mac_n_tile;
        float tile_scales[mac_n_tile], tile_bias[mac_n_tile];
        if (onchip) {
            for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE off
                tile_scales[n] = scales[n0 + n];
                tile_bias[n] = bias[n0 + n];
            }
            if (!compatible(tile_scales, tile_bias, n_live)) return AsrStatus::invalid_parameter;
        }
        for (unsigned m0 = 0; m0 < stage.m; m0 += mac_m_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=185
            const uint16_t m_live = stage.m - m0 < mac_m_tile ? stage.m - m0 : mac_m_tile;
            int32_t collected[mac_m_tile][mac_n_tile];
            for (unsigned m = 0; m < m_live; ++m) {
                for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                    collected[m][n] = 0;
                }
            }
            for (unsigned kt = 0; kt < tile_count(stage.k, mac_k_tile); ++kt) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=38
                const unsigned k0 = kt * mac_k_tile;
                const uint16_t k_live = stage.k - k0 < mac_k_tile ? stage.k - k0 : mac_k_tile;
                const unsigned weight_base = (nt * tile_count(stage.k, mac_k_tile) + kt) * word_count;
                uint64_t weight_tile[mac_k_tile][mac_weight_banks];
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
                for (unsigned pair = 0; pair < mac_weight_banks / 2; ++pair) {
                    for (unsigned k = 0; k < mac_k_tile; ++k) {
#pragma HLS PIPELINE II=1
                        const unsigned at = weight_base + 2 * (pair * mac_k_tile + k);
                        weight_tile[k][2 * pair] = weights[at];
                        weight_tile[k][2 * pair + 1] = weights[at + 1];
                    }
                }
                for (unsigned m = 0; m < m_live; m += 2) {
                    int8_t activation[2][mac_k_tile];
#pragma HLS ARRAY_PARTITION variable=activation complete dim=1
                    const uint16_t m_pair = m + 1 < m_live ? 2 : 1;
                    if (onchip) {
                        for (unsigned k = 0; k < k_live; ++k) {
#pragma HLS PIPELINE II=1
                            activation[0][k] = embedding_code(source, stage, m0 + m, k0 + k);
                            activation[1][k] = m_pair == 2 ? embedding_code(source, stage, m0 + m + 1, k0 + k) : 0;
                        }
                    } else {
                        for (unsigned k = 0; k < k_live; ++k) {
#pragma HLS PIPELINE II=1
                            const unsigned at = (m0 + m) * stage.k + k0 + k;
                            activation[0][k] = external_codes[at];
                            activation[1][k] = m_pair == 2 ? external_codes[at + stage.k] : 0;
                        }
                    }
                    int32_t part[2][mac_n_tile];
                    MacAccounting counters;
                    const auto mode = m_pair == 1 ? MacMode::w8_gemv : MacMode::w8_gemm;
                    const auto status = mac_pair(mode, activation, weight_tile, {m_pair, n_live, k_live}, part, counters);
                    if (status != MacStatus::ok) return AsrStatus::invalid_code;
                    for (unsigned row = 0; row < m_pair; ++row) {
                        for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                            collected[m + row][n] += part[row][n];
                        }
                    }
                }
            }
            for (unsigned m = 0; m < m_live; ++m) {
                for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                    const int32_t dot = collected[m][n];
                    const unsigned at = (m0 + m) * stage.n + n0 + n;
                    const float weight_scale = onchip ? tile_scales[n] : scales[n0 + n];
                    const float weight_bias = onchip ? tile_bias[n] : bias[n0 + n];
                    const float value = asr_dequantize(dot, sx, weight_scale, weight_bias);
                    // Check before the LUT, which can clamp the gate of a bad stage.
                    invalid |= !finite(value);
                    const float result = stage.activate ? double_swish(value) : value;
                    if (onchip) {
                        invalid |= !finite(result);
                        values[n] = result;
                    } else {
                        next[at] = result;
                    }
                    if (debug_raw) raw[at] = dot;
                }
                if (onchip && invalid) return AsrStatus::nonfinite_output;
                if (pass == DensePass::scan) {
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                        const uint32_t magnitude = float_bits(values[n]) & 0x7fffffffu;
                        if (magnitude > float_bits(amax)) std::memcpy(&amax, &magnitude, sizeof(amax));
                    }
                } else if (pass == DensePass::encode) {
                    if (quantize_at_fixed_scale(values, encoded, n_live, next_scale) != VectorStatus::ok)
                        return AsrStatus::invalid_quantization;
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=2
                        write_code(destination, (m0 + m) * stage.n + n0 + n, encoded[n]);
                    }
                } else if (pass == DensePass::projection) {
                    for (unsigned n = 0; n < n_live; ++n) {
#pragma HLS PIPELINE II=1
                        buffer_b[(m0 + m) * stage.n + n0 + n] = float_bits(values[n]);
                    }
                }
            }
        }
    }
    return invalid ? AsrStatus::nonfinite_output : AsrStatus::ok;
}

AsrStatus depthwise(unsigned rows, const uint64_t* weights, const float* scales, const float* bias,
                    const int8_t* codes, float sx, float* next, bool debug_raw, int32_t* raw) {
#pragma HLS INLINE off
    if (!compatible(scales, bias, asr_channels)) return AsrStatus::invalid_parameter;
    bool invalid_code = false;
    bool invalid_output = false;
    for (unsigned channel = 0; channel < asr_channels; ++channel) {
        int8_t taps[asr_history + 1];
        for (unsigned k = 0; k <= asr_history; ++k) {
#pragma HLS PIPELINE II=1
            const unsigned byte = unsigned(weights[4 * channel + k / 8] >> (8 * (k % 8))) & 255u;
            invalid_code |= byte == 128;
            taps[k] = int8_t(int(byte ^ 128u) - 128);
        }
        for (unsigned row = 0; row < rows; ++row) {
            int32_t dot = 0;
            for (unsigned k = 0; k <= asr_history; ++k) {
#pragma HLS PIPELINE II=1
                dot += int16_t(codes[(row + k) * asr_channels + channel] * taps[k]);
            }
            const unsigned at = row * asr_channels + channel;
            const float value = asr_dequantize(dot, sx, scales[channel], bias[channel]);
            invalid_output |= !finite(value);
            next[at] = double_swish(value);
            if (debug_raw) raw[at] = dot;
        }
    }
    if (invalid_code) return AsrStatus::invalid_code;
    return invalid_output ? AsrStatus::nonfinite_output : AsrStatus::ok;
}

Stage get_stage(AsrBlock op, uint16_t rows, unsigned index) {
    if (op == AsrBlock::ffn) {
        Stage stage{rows, asr_channels, 768};
        if (index == 0) stage = {rows, 768, asr_channels, Layout::rows, 0, 0, 0, 0, 1, 1, 0, true};
        return stage;
    }
    if (index == 0) return {rows, 2 * asr_channels, asr_channels};
    if (index == 1) return {rows, asr_channels, asr_history + 1};
    return {rows, asr_channels, asr_channels};
}

unsigned input_count(Stage stage, bool depth) {
    if (depth) return (stage.m + asr_history) * asr_channels;
    return stage.m * stage.k;
}

AsrStatus staged_quantize(const float* source, int8_t* codes, unsigned count, float& scale,
                         float block[embedding_bounded_tile_elements],
                         int8_t encoded[embedding_bounded_tile_elements]) {
#pragma HLS INLINE off
    // The complete tensor is scanned before any encoding. Small local blocks
    // then separate AXI reads, shared vector arithmetic, and AXI writes, so the
    // adapter does not force all FP operations into one memory dependency state.
    uint32_t maximum_bits = 0;
    bool nonfinite = false;
    for (unsigned at = 0; at < count; ++at) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=512 max=38912
        const float value = source[at];
        uint32_t magnitude;
        std::memcpy(&magnitude, &value, sizeof(magnitude));
        magnitude &= 0x7fffffffu;
        nonfinite |= magnitude >= 0x7f800000u;
        // Positive finite FP32 bit patterns have the same order as their
        // values, including subnormals. Integer max avoids an FP feedback path.
        if (magnitude > maximum_bits) maximum_bits = magnitude;
    }
    float maximum;
    std::memcpy(&maximum, &maximum_bits, sizeof(maximum));
    if (nonfinite || asr_scale_from_amax(maximum, scale) != VectorStatus::ok)
        return AsrStatus::invalid_quantization;
    for (unsigned base = 0; base < count; base += 128) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=304
        const unsigned live = count - base < 128 ? count - base : 128;
        for (unsigned at = 0; at < live; ++at) {
#pragma HLS PIPELINE II=1
            block[at] = source[base + at];
        }
        if (quantize_at_fixed_scale(block, encoded, live, scale) != VectorStatus::ok)
            return AsrStatus::invalid_quantization;
        for (unsigned at = 0; at < live; ++at) {
#pragma HLS PIPELINE II=1
            codes[base + at] = encoded[at];
        }
    }
    return AsrStatus::ok;
}

void staged_glu(float* tensor, unsigned rows) {
#pragma HLS INLINE off
    float values[2 * asr_channels];
    float result[asr_channels];
    for (unsigned row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=16
        for (unsigned at = 0; at < 2 * asr_channels; ++at) {
#pragma HLS PIPELINE II=1
            values[at] = tensor[2 * row * asr_channels + at];
        }
        for (unsigned at = 0; at < asr_channels; ++at) {
#pragma HLS PIPELINE II=1
            result[at] = glu(values[at], values[at + asr_channels]);
        }
        for (unsigned at = 0; at < asr_channels; ++at) {
#pragma HLS PIPELINE II=1
            tensor[row * asr_channels + at] = result[at];
        }
    }
}

AsrStatus quantize_embedding_input(const float* input, float& scale, uint32_t* buffer_b,
                                   float values[embedding_bounded_tile_elements],
                                   int8_t encoded[embedding_bounded_tile_elements]) {
#pragma HLS INLINE off
    constexpr unsigned input_elements = 39 * 80;
    uint32_t input_amax_bits = 0;
    bool invalid_input = false;
    for (unsigned i = 0; i < input_elements; ++i) {
#pragma HLS PIPELINE II=1
        const float value = input[i];
        invalid_input |= !finite(value);
        const uint32_t magnitude = float_bits(value) & 0x7fffffffu;
        if (magnitude > input_amax_bits) input_amax_bits = magnitude;
    }
    if (invalid_input) return AsrStatus::invalid_quantization;
    float input_amax;
    std::memcpy(&input_amax, &input_amax_bits, sizeof(input_amax));
    if (asr_scale_from_amax(input_amax, scale) != VectorStatus::ok)
        return AsrStatus::invalid_quantization;
    for (unsigned base = 0; base < input_elements; base += embedding_bounded_tile_elements) {
        const unsigned count = input_elements - base < embedding_bounded_tile_elements ?
            input_elements - base : embedding_bounded_tile_elements;
        for (unsigned i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
            values[i] = input[base + i];
        }
        if (quantize_at_fixed_scale(values, encoded, count, scale) != VectorStatus::ok)
            return AsrStatus::invalid_quantization;
        for (unsigned i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=2
            write_code(buffer_b, base + i, encoded[i]);
        }
    }
    return AsrStatus::ok;
}

} // namespace
} // namespace edge_speech::pl

uint8_t asr_blocks(uint8_t operation, uint16_t rows, const uint64_t* weights, const float* weight_scale,
    const float* bias, const float* input, const float* old_cache, float* next_cache, float* output,
    float* scratch, int8_t* codes, bool debug_raw, int32_t* raw, float* stage_scales) {
#pragma HLS INTERFACE m_axi port=weights bundle=memory depth=86016 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=weight_scale bundle=memory depth=1024 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=bias bundle=memory depth=1024 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=input bundle=memory depth=4096 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=old_cache bundle=memory depth=7680 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=next_cache bundle=memory depth=7680 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=output bundle=memory depth=4096 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=scratch bundle=memory depth=38912 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=codes bundle=memory depth=38912 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=raw bundle=memory depth=89152 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=stage_scales bundle=memory depth=4 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=operation bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=debug_raw bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace edge_speech::pl;
#pragma HLS ALLOCATION function instances=dense limit=1
    const auto block = static_cast<AsrBlock>(operation);
    const bool conv = block == AsrBlock::convolution;
    const bool embedding = block == AsrBlock::embedding;
    if (operation > uint8_t(AsrBlock::embedding) ||
        (rows != 2 && rows != 4 && rows != 8 && rows != 16) ||
        (embedding && rows != 16) || (conv && old_cache == next_cache)) {
        return uint8_t(AsrStatus::invalid_shape);
    }
    uint32_t codes_a[embedding_bounded_codes_a / 4];
    uint32_t buffer_b[embedding_bounded_codes_b / 4];
    float values[embedding_bounded_tile_elements];
    int8_t encoded[embedding_bounded_tile_elements];
#pragma HLS BIND_STORAGE variable=codes_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_b type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=values type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=encoded type=ram_2p impl=bram
    float activation_scale = 1;
    if (embedding) {
        const auto status = quantize_embedding_input(input, activation_scale, buffer_b, values, encoded);
        if (status != AsrStatus::ok) return uint8_t(status);
    }
    // Each of Embedding's three convolution stages uses a complete amax scan
    // and a recomputation/encode pass. Projection is the seventh pass.
    const unsigned passes = embedding ? 7 : (conv ? 3 : 2);
    const unsigned conv_work_base = (asr_history + rows) * asr_channels;
    unsigned weight_offset = 0;
    unsigned parameter_offset = 0;
    unsigned raw_offset = 0;
    float* final_tensor = scratch;
    float amax = 0;
    for (unsigned iteration = 0; iteration < passes; ++iteration) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=7
        const unsigned i = embedding ? iteration / 2 : iteration;
        const Stage stage = embedding ? embedding_stages[i] : get_stage(block, rows, i);
        const bool dw = conv && i == 1;
        DensePass pass = DensePass::external;
        if (embedding) {
            if (i == 3) pass = DensePass::projection;
            else if (iteration % 2 != 0) pass = DensePass::encode;
            else {
                pass = DensePass::scan;
                amax = 0;
            }
        } else {
            const unsigned count = input_count(stage, dw);
            const float* source = i == 0 ? input : final_tensor;
            const auto quant_status = staged_quantize(source, codes, count, activation_scale, values, encoded);
            if (quant_status != AsrStatus::ok) return uint8_t(quant_status);
        }
        const bool report_raw = debug_raw && pass != DensePass::encode;
        if (report_raw) stage_scales[i] = activation_scale;

        // PW1 yields row-major channel halves. The first half of each row
        // can be compacted forward in place after reading its gate half.
        final_tensor = conv ? scratch + (i == 0 ? asr_cache_elements : conv_work_base) : scratch;
        const uint64_t* stage_weights = weights + weight_offset;
        const float* scales = weight_scale + parameter_offset;
        const float* stage_bias = bias + parameter_offset;
        int32_t* stage_raw = report_raw ? raw + raw_offset : raw;
        // One dense call site keeps HLS from specializing separate MAC
        // instances for Embedding and DDR-backed FFN/Conv.
        const auto status = dw ? depthwise(rows, stage_weights, scales, stage_bias, codes,
                                         activation_scale, final_tensor, debug_raw, stage_raw)
                               : dense(stage, stage_weights, scales, stage_bias, codes,
                                       activation_scale, final_tensor, report_raw, stage_raw,
                                       pass, i % 2 != 0, amax, codes_a, buffer_b, values, encoded);
        if (status != AsrStatus::ok) return uint8_t(status);
        if (pass == DensePass::scan) continue;
        if (pass == DensePass::encode && asr_scale_from_amax(amax, activation_scale) != VectorStatus::ok)
            return uint8_t(AsrStatus::invalid_quantization);
        if (conv && i == 0) {
            for (unsigned at = 0; at < asr_cache_elements; ++at) {
#pragma HLS PIPELINE II=1
                scratch[at] = old_cache[at];
            }
            staged_glu(final_tensor, rows);
            final_tensor = scratch;
        }
        weight_offset += dw ? 4 * asr_channels : tile_count(stage.n, mac_n_tile) * tile_count(stage.k, mac_k_tile) * word_count;
        parameter_offset += stage.n;
        raw_offset += stage.m * stage.n;
    }
    if (embedding) {
        for (unsigned at = 0; at < 16 * asr_channels; ++at) {
#pragma HLS PIPELINE II=1
            float value;
            std::memcpy(&value, &buffer_b[at], sizeof(value));
            output[at] = value;
        }
        return uint8_t(AsrStatus::ok);
    }
    for (unsigned at = 0; at < rows * asr_channels; ++at) {
#pragma HLS PIPELINE II=1
        output[at] = final_tensor[at];
    }
    if (conv) {
        for (unsigned at = 0; at < asr_cache_elements; ++at) {
#pragma HLS PIPELINE II=1
            next_cache[at] = scratch[rows * asr_channels + at];
        }
    }
    return uint8_t(AsrStatus::ok);
}

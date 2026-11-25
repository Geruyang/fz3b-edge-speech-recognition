#pragma once

#include <cstdint>

namespace edge_speech::pl {

inline constexpr unsigned asr_channels = 256;
inline constexpr unsigned asr_history = 30;
inline constexpr unsigned asr_cache_elements = asr_channels * asr_history;
inline constexpr unsigned asr_conv_slots = 20;
inline constexpr unsigned asr_input_elements = 16 * asr_channels;
inline constexpr unsigned asr_scratch_elements = 16 * 19 * 128;
// Conv1/2/3 occupy 1/2/5 padded tiles, and projection uses 2 x 38.
inline constexpr unsigned asr_weight_words = (1 + 2 + 5 + 2 * 38) * 64 * 16;
inline constexpr unsigned asr_parameter_elements = 1024;
inline constexpr unsigned asr_debug_elements = 37 * 80 * 8 + 18 * 39 * 32 + 16 * 19 * 128 + 16 * 256;
inline constexpr unsigned asr_stage_limit = 4;

enum class AsrBlock : uint8_t { ffn, convolution, embedding };
enum class AsrStatus : uint8_t { ok, invalid_shape, invalid_quantization, invalid_code, invalid_parameter, nonfinite_output };

// A dense stage consumes padded FZBANK16-v1 W8 words in Ntile, Ktile, pair,
// k, lane order, with separate FP32 per-output scales and bias. The dedicated
// depthwise stage uses 4 words per channel (31 values and one unused byte).
// Stages are concatenated in execution order without inter-stage padding.
// Input/output/state use time-major order. Embedding input is 39 x 80;
// its fixed output is 16 x 256. All other rows are from the frozen stage set.
// Cache ports must refer to distinct old/working bank ranges. The owner
// publishes the work bank after the whole chunk succeeds and DMA is coherent.
// This offline candidate maps caller-preallocated scratch/codes to one shared
// AXI memory master: max 155648 FP32 bytes + 38912 INT8 bytes. This is external
// capacity, not a change to the platform's original 64KiB fusion scratch budget.
// Small local quantization/GLU staging is separate from that external capacity.
// Scratch is device-owned for the call. In debug mode each stage appends
// [row][output_channel] RAW_I32 and reports its activation scale. On failure
// scratch/diagnostics may be partial and output/next_cache remain unpublished.

} // namespace edge_speech::pl

uint8_t asr_blocks(uint8_t operation, uint16_t rows,
    const uint64_t weights[edge_speech::pl::asr_weight_words],
    const float weight_scale[edge_speech::pl::asr_parameter_elements],
    const float bias[edge_speech::pl::asr_parameter_elements],
    const float input[edge_speech::pl::asr_input_elements],
    const float old_cache[edge_speech::pl::asr_cache_elements],
    float next_cache[edge_speech::pl::asr_cache_elements],
    float output[edge_speech::pl::asr_input_elements],
    float scratch[edge_speech::pl::asr_scratch_elements],
    int8_t codes[edge_speech::pl::asr_scratch_elements],
    bool debug_raw, int32_t raw[edge_speech::pl::asr_debug_elements],
    float stage_scales[edge_speech::pl::asr_stage_limit]);

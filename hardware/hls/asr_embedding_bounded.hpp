#pragma once

#include "asr_blocks.hpp"
#include <cstdint>

namespace edge_speech::pl {
inline constexpr unsigned embedding_bounded_codes_a = 16 * 19 * 128;
inline constexpr unsigned embedding_bounded_codes_b = 18 * 39 * 32;
inline constexpr unsigned embedding_bounded_tensor_bytes = embedding_bounded_codes_a + embedding_bounded_codes_b;
inline constexpr unsigned embedding_bounded_tile_elements = 128;
inline constexpr unsigned embedding_bounded_tile_bytes = embedding_bounded_tile_elements * (sizeof(float) + sizeof(int8_t));
inline constexpr unsigned embedding_bounded_scratch_bytes = embedding_bounded_tensor_bytes + embedding_bounded_tile_bytes;
static_assert(embedding_bounded_tensor_bytes == 61376 && embedding_bounded_scratch_bytes == 62016 &&
              embedding_bounded_scratch_bytes <= 65536);

// Logical byte/operation counts, not AXI bus transaction counts or RTL cycles.
// 0 weight reads; 1 scale/bias reads; 2 input reads; 3 output writes;
// 4 debug writes; 5 MAC issue steps; 6 useful MACs;
// 7 local INT8 reads; 8 local INT8 writes; 9 fixed-scale quantization calls.
inline constexpr unsigned embedding_accounting_words = 10;
}

// Offline alternative to external FP32 scratch. The 61,376-byte INT8 pingpong
// plus one 640-byte value/code tile fit the original 64KiB scratch budget.
// Shared weight/accumulator tiles are separate. B's storage is reused for
// final-output staging after Conv3. Conv1/2/3 are recomputed to preserve one
// full-tensor scale.
uint8_t asr_embedding_bounded(
    const uint64_t weights[edge_speech::pl::asr_weight_words],
    const float weight_scale[edge_speech::pl::asr_parameter_elements],
    const float bias[edge_speech::pl::asr_parameter_elements],
    const float input[39 * 80], float output[16 * 256], bool debug_raw,
    int32_t raw[edge_speech::pl::asr_debug_elements], float stage_scales[4],
    uint64_t accounting[edge_speech::pl::embedding_accounting_words]);

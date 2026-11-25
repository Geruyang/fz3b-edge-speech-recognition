#pragma once

#include <cstdint>

namespace edge_speech::pl {

// An external FP32 scratch view can hold one M16 x 4864 activation intermediate.
inline constexpr unsigned vector_scratch_capacity = 16 * 4864;

enum class ScalePolicy : uint8_t { asr_tensor, qwen_row };
enum class VectorStatus : uint8_t { ok, invalid_extent, nonfinite_input, underflow_scale };

struct QuantizationStats {
    float scale;
    unsigned scan_elements;
    unsigned code_elements;
    unsigned scratch_read_bytes;
    unsigned code_write_bytes;
};

bool valid_weight_scale(uint16_t serialized);
float half_scale_to_float(uint16_t serialized);
float qwen_group_accumulate(float accumulator, int32_t dot, uint16_t serialized_scale);
float qwen_finalize(float accumulator, float row_scale, float bias);
float asr_dequantize(int32_t dot, float activation_scale, float weight_scale, float bias);
float add_residual(float left, float right);

float sigmoid_approx(float value);
float double_swish(float value);
float glu(float left, float gate);
float silu(float value);

// One call covers the entire ASR tensor or one complete Qwen row.
VectorStatus requantize(const float* scratch, int8_t* codes, unsigned elements,
                        ScalePolicy policy, QuantizationStats& stats);

// For staged ASR encoders: amax is from the complete finite tensor, never a
// tile maximum. Zero maps to scale 1; ASR subnormal underflow maps to min FP32.
// Both helpers preserve destination values on rejected arguments/input.
VectorStatus asr_scale_from_amax(float amax, float& scale);
bool valid_asr_weight_scale(float scale);
VectorStatus quantize_at_fixed_scale(const float* values, int8_t* codes,
                                     unsigned elements, float scale);

}  // namespace edge_speech::pl

// Standalone HLS qualification of activation and two-pass requantization.
// Opcode 0/1 quantizes a complete ASR tensor/Qwen row; 2 is DoubleSwish,
// 3 is GLU, and 4 is SiLU. Completion words report VectorStatus, scale
// bits, scan/code elements and scratch/code byte counts.
void vector_ops(const float* input, const float* gate, float* activated, int8_t* codes,
                unsigned count, uint8_t opcode, uint32_t completion[6]);

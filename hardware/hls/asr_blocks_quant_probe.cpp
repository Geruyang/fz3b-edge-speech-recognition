#include "vector_ops.hpp"
#include <cstring>

// Small offline qualification top for the shared known-scale API. It is not
// a second deployed vector engine and is never integrated into the platform.
uint8_t asr_blocks_quant_probe(uint8_t mode, float parameter, const float input[4], int8_t codes[4], uint32_t result[2]) {
#pragma HLS INTERFACE ap_memory port=input
#pragma HLS INTERFACE ap_memory port=codes
#pragma HLS INTERFACE ap_memory port=result
    using namespace edge_speech::pl;
    int8_t local_codes[4];
    for (unsigned i = 0; i < 4; ++i) local_codes[i] = codes[i];
    float scale = mode == 0 ? 17 : parameter;
    auto status = mode == 0 ? asr_scale_from_amax(parameter, scale) : VectorStatus::ok;
    if (mode == 2) status = valid_asr_weight_scale(parameter) ? VectorStatus::ok : VectorStatus::nonfinite_input;
    else if (status == VectorStatus::ok) status = quantize_at_fixed_scale(input, local_codes, 4, scale);
    for (unsigned i = 0; i < 4; ++i) codes[i] = local_codes[i];
    result[0] = uint32_t(status);
    std::memcpy(&result[1], &scale, 4);
    return uint8_t(status);
}

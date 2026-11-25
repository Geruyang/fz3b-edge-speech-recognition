#include "vector_ops.hpp"

#include <cstring>

namespace edge_speech::pl {
namespace {

constexpr uint32_t sign_mask = 0x80000000u;
constexpr uint32_t exponent_mask = 0x7f800000u;
constexpr uint32_t fraction_mask = 0x007fffffu;

// Samples of sigmoid(-12 + i / 8), rounded to binary32.
constexpr float sigmoid_lut[] = {
    0.000006144174602f, 0.000006962256250f, 0.000007889262586f, 0.000008939696305f, 0.00001012999098f, 0.00001147876813f,
    0.00001300712847f, 0.00001473898197f, 0.00001670142185f, 0.00001892514825f, 0.00002144494842f, 0.00002430024074f,
    0.00002753569111f, 0.00003120191139f, 0.00003535625074f, 0.00004006369223f, 0.00004539786870f, 0.00005144221374f,
    0.00005829126566f, 0.00006605214485f, 0.00007484622751f, 0.00008481104172f, 0.00009610241550f, 0.0001088969100f,
    0.0001233945760f, 0.0001398220756f, 0.0001584362191f, 0.0001795279693f, 0.0002034269781f, 0.0002305067220f,
    0.0002611903191f, 0.0002959571135f, 0.0003353501305f, 0.0003799845148f, 0.0004305570813f, 0.0004878571228f,
    0.0005527786369f, 0.0006263341581f, 0.0007096703991f, 0.0008040859356f, 0.0009110511944f, 0.001032231037f,
    0.001169510265f, 0.001325022417f, 0.001501182257f, 0.001700722411f, 0.001926734663f, 0.002182716445f,
    0.002472623157f, 0.002800926967f, 0.003172682842f, 0.003593602581f, 0.004070137716f, 0.004609572179f,
    0.005220125694f, 0.005911068856f, 0.006692850924f, 0.007577241268f, 0.008577485414f, 0.009708476481f,
    0.01098694263f, 0.01243165085f, 0.01406362704f, 0.01590639171f, 0.01798620996f, 0.02033235334f,
    0.02297736991f, 0.02595735720f, 0.02931223075f, 0.03308597839f, 0.03732688734f, 0.04208772792f,
    0.04742587318f, 0.05340332980f, 0.06008665017f, 0.06754669114f, 0.07585818002f, 0.08509904501f,
    0.09534946490f, 0.1066905939f, 0.1192029220f, 0.1329642402f, 0.1480471980f, 0.1645164629f,
    0.1824255238f, 0.2018132223f, 0.2227001388f, 0.2450850131f, 0.2689414214f, 0.2942149722f,
    0.3208213008f, 0.3486451353f, 0.3775406688f, 0.4073334000f, 0.4378234991f, 0.4687906266f,
    0.5000000000f, 0.5312093734f, 0.5621765009f, 0.5926666000f, 0.6224593312f, 0.6513548647f,
    0.6791786992f, 0.7057850278f, 0.7310585786f, 0.7549149869f, 0.7772998612f, 0.7981867777f,
    0.8175744762f, 0.8354835371f, 0.8519528020f, 0.8670357598f, 0.8807970780f, 0.8933094061f,
    0.9046505351f, 0.9149009550f, 0.9241418200f, 0.9324533089f, 0.9399133498f, 0.9465966702f,
    0.9525741268f, 0.9579122721f, 0.9626731127f, 0.9669140216f, 0.9706877692f, 0.9740426428f,
    0.9770226301f, 0.9796676467f, 0.9820137900f, 0.9840936083f, 0.9859363730f, 0.9875683491f,
    0.9890130574f, 0.9902915235f, 0.9914225146f, 0.9924227587f, 0.9933071491f, 0.9940889311f,
    0.9947798743f, 0.9953904278f, 0.9959298623f, 0.9964063974f, 0.9968273172f, 0.9971990730f,
    0.9975273768f, 0.9978172836f, 0.9980732653f, 0.9982992776f, 0.9984988177f, 0.9986749776f,
    0.9988304897f, 0.9989677690f, 0.9990889488f, 0.9991959141f, 0.9992903296f, 0.9993736658f,
    0.9994472214f, 0.9995121429f, 0.9995694429f, 0.9996200155f, 0.9996646499f, 0.9997040429f,
    0.9997388097f, 0.9997694933f, 0.9997965730f, 0.9998204720f, 0.9998415638f, 0.9998601779f,
    0.9998766054f, 0.9998911031f, 0.9999038976f, 0.9999151890f, 0.9999251538f, 0.9999339479f,
    0.9999417087f, 0.9999485578f, 0.9999546021f, 0.9999599363f, 0.9999646437f, 0.9999687981f,
    0.9999724643f, 0.9999756998f, 0.9999785551f, 0.9999810749f, 0.9999832986f, 0.9999852610f,
    0.9999869929f, 0.9999885212f, 0.9999898700f, 0.9999910603f, 0.9999921107f, 0.9999930377f,
    0.9999938558f,
};

uint32_t float_bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}

float from_bits(uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}

// For magnitudes up to exponent 7, round amax / 127 near underflow using
// exact integer units of 2^-149. The divisor is odd, so no halfway tie exists.
float activation_scale(uint32_t maximum) {
    if (maximum == 0) return 1.0f;
    const unsigned exponent = maximum >> 23;
    if (exponent <= 7) {
        const uint32_t mantissa = maximum & fraction_mask;
        const uint32_t units = exponent ? (mantissa | (1u << 23)) << (exponent - 1) : mantissa;
        const uint32_t rounded = (units + 63) / 127;
        if (rounded <= fraction_mask) return from_bits(rounded);
    }
    return from_bits(maximum) / 127.0f;
}

// Called for finite, very small values. Scale both operands by 2^126 before
// division when a subnormal could affect the rounded INT8 result.
float normalize_tiny(uint32_t magnitude) {
    if (magnitude <= fraction_mask) return float(magnitude) * 0x1p-23f;
    return from_bits(magnitude + (126u << 23));
}

int8_t quantize_one(float value, float scale) {
    const uint32_t bits = float_bits(value);
    const uint32_t magnitude = bits & ~sign_mask;
    const uint32_t scale_bits = float_bits(scale);
    float quotient;
    if (scale_bits <= fraction_mask || (magnitude <= fraction_mask && (scale_bits >> 23) <= 2)) {
        if ((magnitude >> 23) > 8) return (bits & sign_mask) ? -127 : 127;
        quotient = normalize_tiny(magnitude) / normalize_tiny(scale_bits);
    } else {
        quotient = from_bits(magnitude) / scale;
    }
    if (quotient >= 127.0f) return (bits & sign_mask) ? -127 : 127;
    const int integral = static_cast<int>(quotient);
    const float remainder = quotient - float(integral);
    const int rounded = integral + (remainder > 0.5f || (remainder == 0.5f && (integral & 1)));
    return static_cast<int8_t>((bits & sign_mask) ? -rounded : rounded);
}

}  // namespace

bool valid_weight_scale(uint16_t serialized) {
    return serialized > 0 && serialized < 0x7c00;
}

float half_scale_to_float(uint16_t serialized) {
    const uint32_t exponent = (serialized >> 10) & 0x1f;
    const uint32_t fraction = serialized & 0x3ff;
    if (exponent == 0) {
        // Binary16 subnormals are normal and exact in FP32.
        if (fraction == 0) return 0.0f;
        unsigned leading = 0;
        for (unsigned bit = 1; bit < 10; ++bit) {
#pragma HLS UNROLL
            if (fraction & (1u << bit)) leading = bit;
        }
        return from_bits(((leading + 103) << 23) |
                         ((fraction << (23 - leading)) & fraction_mask));
    }
    return from_bits(((exponent + 112) << 23) | (fraction << 13));
}

float qwen_group_accumulate(float accumulator, int32_t dot, uint16_t serialized_scale) {
#pragma HLS INLINE
    const float scaled = float(dot) * half_scale_to_float(serialized_scale);
#pragma HLS BIND_OP variable=scaled op=fmul
    const float result = accumulator + scaled;
#pragma HLS BIND_OP variable=result op=fadd
    return result;
}

float qwen_finalize(float accumulator, float row_scale, float bias) {
#pragma HLS INLINE
    const float scaled = row_scale * accumulator;
#pragma HLS BIND_OP variable=scaled op=fmul
    const float result = scaled + bias;
#pragma HLS BIND_OP variable=result op=fadd
    return result;
}

float asr_dequantize(int32_t dot, float activation_scale, float weight_scale, float bias) {
    const float scale_product = activation_scale * weight_scale;
#pragma HLS BIND_OP variable=scale_product op=fmul
    const float scaled = float(dot) * scale_product;
#pragma HLS BIND_OP variable=scaled op=fmul
    const float result = scaled + bias;
#pragma HLS BIND_OP variable=result op=fadd
    return result;
}

float add_residual(float left, float right) {
    return left + right;
}

float sigmoid_approx(float value) {
    // Clamp outside the table and interpolate linearly between its samples.
    if (!(value == value)) return value;
    if (value < -12.0f) return 0.0f;
    if (value > 12.0f) return 1.0f;
    const float position = (value + 12.0f) * 8.0f;
    const unsigned low = static_cast<unsigned>(position);
    if (low >= 192) return sigmoid_lut[192];
    const float fraction = position - float(low);
    const float step = sigmoid_lut[low + 1] - sigmoid_lut[low];
    return sigmoid_lut[low] + step * fraction;
}

float double_swish(float value) {
    return value * sigmoid_approx(value - 1.0f);
}

float glu(float left, float gate) {
    return left * sigmoid_approx(gate);
}

float silu(float value) {
    return value * sigmoid_approx(value);
}

VectorStatus requantize(const float* scratch, int8_t* codes, unsigned elements,
                        ScalePolicy policy, QuantizationStats& stats) {
    stats = {};
    if (elements == 0 || elements > vector_scratch_capacity || scratch == nullptr || codes == nullptr ||
        (policy != ScalePolicy::asr_tensor && policy != ScalePolicy::qwen_row)) {
        return VectorStatus::invalid_extent;
    }
    uint32_t maximum = 0;
    bool nonfinite = false;
    scan_amax:
    for (unsigned at = 0; at < elements; ++at) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=77824
        const uint32_t magnitude = float_bits(scratch[at]) & ~sign_mask;
        nonfinite |= magnitude >= exponent_mask;
        if (magnitude > maximum) maximum = magnitude;
    }
    stats.scan_elements = elements;
    stats.scratch_read_bytes = elements * sizeof(float);
    if (nonfinite) return VectorStatus::nonfinite_input;

    float scale = activation_scale(maximum);
    if (float_bits(scale) == 0) {
        if (policy == ScalePolicy::qwen_row) return VectorStatus::underflow_scale;
        scale = from_bits(1);
    }
    emit_codes:
    for (unsigned at = 0; at < elements; ++at) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=77824
        codes[at] = quantize_one(scratch[at], scale);
    }
    stats.scale = scale;
    stats.code_elements = elements;
    stats.scratch_read_bytes += elements * sizeof(float);
    stats.code_write_bytes = elements * sizeof(int8_t);
    return VectorStatus::ok;
}

VectorStatus asr_scale_from_amax(float amax, float& scale) {
    const uint32_t bits = float_bits(amax);
    const uint32_t magnitude = bits & ~sign_mask;
    if (magnitude >= exponent_mask || ((bits & sign_mask) != 0 && magnitude != 0))
        return VectorStatus::nonfinite_input;
    float result = activation_scale(magnitude);
    if (float_bits(result) == 0) result = from_bits(1);
    scale = result;
    return VectorStatus::ok;
}

bool valid_asr_weight_scale(float scale) {
    const uint32_t bits = float_bits(scale);
    return bits > 0 && bits < exponent_mask;
}

VectorStatus quantize_at_fixed_scale(const float* values, int8_t* codes, unsigned elements, float scale) {
    if (values == nullptr || codes == nullptr || elements == 0 || elements > vector_scratch_capacity)
        return VectorStatus::invalid_extent;
    const uint32_t bits = float_bits(scale);
    const uint32_t magnitude = bits & ~sign_mask;
    if (magnitude >= exponent_mask) return VectorStatus::nonfinite_input;
    // FPO comparisons treat subnormals as zero. Validate encoded sign and
    // magnitude so the positive ASR minimum scale reaches quantize_one's
    // existing, explicitly normalized subnormal path.
    if ((bits & sign_mask) != 0 || magnitude == 0) return VectorStatus::underflow_scale;
    bool nonfinite = false;
    for (unsigned i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        nonfinite |= (float_bits(values[i]) & ~sign_mask) >= exponent_mask;
    }
    if (nonfinite) return VectorStatus::nonfinite_input;
    for (unsigned i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        codes[i] = quantize_one(values[i], scale);
    }
    return VectorStatus::ok;
}

}  // namespace edge_speech::pl

void vector_ops(const float* input, const float* gate, float* activated, int8_t* codes,
                unsigned count, uint8_t opcode, uint32_t completion[6]) {
#pragma HLS INTERFACE m_axi port=input bundle=input depth=77824
#pragma HLS INTERFACE m_axi port=gate bundle=gate depth=77824
#pragma HLS INTERFACE m_axi port=activated bundle=activated depth=77824
#pragma HLS INTERFACE m_axi port=codes bundle=codes depth=77824
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=opcode bundle=control
#pragma HLS INTERFACE ap_memory port=completion
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace edge_speech::pl;
    QuantizationStats stats{};
    VectorStatus status = VectorStatus::invalid_extent;
    if (count != 0 && count <= vector_scratch_capacity && opcode <= 4) {
        if (opcode <= 1) {
            const ScalePolicy policy = opcode == 0 ? ScalePolicy::asr_tensor : ScalePolicy::qwen_row;
            status = requantize(input, codes, count, policy, stats);
        } else {
            for (unsigned index = 0; index < count; ++index) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=77824
                const float x = input[index];
                float sigmoid_input = x;
                if (opcode == 2) sigmoid_input = x - 1.0f;
                if (opcode == 3) sigmoid_input = gate[index];
                activated[index] = x * sigmoid_approx(sigmoid_input);
            }
            status = VectorStatus::ok;
        }
    }
    completion[0] = static_cast<uint32_t>(status);
    std::memcpy(&completion[1], &stats.scale, sizeof stats.scale);
    completion[2] = stats.scan_elements;
    completion[3] = stats.code_elements;
    completion[4] = stats.scratch_read_bytes;
    completion[5] = stats.code_write_bytes;
}


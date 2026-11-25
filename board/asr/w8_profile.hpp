#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace edge_speech::asr {

enum class W8Part : unsigned {
    ffn_input_quant, ffn_linear1, ffn_double_swish, ffn_hidden_quant, ffn_linear2,
    conv_input_quant, conv_pw1, conv_glu, conv_old_history, conv_depth_quant, conv_dw31,
    conv_tail_quant, conv_pw2, conv_next_history,
    embedding_input_quant, embedding_conv1_scan, embedding_conv1_encode, embedding_conv2_scan,
    embedding_conv2_encode, embedding_conv3_scan, embedding_conv3_encode, embedding_projection,
    write_output, count
};

inline constexpr std::array<const char*, static_cast<unsigned>(W8Part::count)> w8_part_names{
    "ffn_input_quant_ns", "ffn_linear1_ns", "ffn_double_swish_ns", "ffn_hidden_quant_ns", "ffn_linear2_ns",
    "conv_input_quant_ns", "conv_pw1_ns", "conv_glu_ns", "conv_old_history_ns", "conv_depth_quant_ns", "conv_dw31_ns",
    "conv_tail_quant_ns", "conv_pw2_ns", "conv_next_history_ns",
    "embedding_input_quant_ns", "embedding_conv1_scan_ns", "embedding_conv1_encode_ns", "embedding_conv2_scan_ns",
    "embedding_conv2_encode_ns", "embedding_conv3_scan_ns", "embedding_conv3_encode_ns", "embedding_projection_ns",
    "write_output_ns"
};

struct W8Profile {
    std::array<std::uint64_t, w8_part_names.size()> ns{};
    std::uint64_t kernel_ns = 0;
    std::uint32_t double_swish_elements = 0;
};

// The alternate CPU build inserts scopes in the same code that performs the arithmetic.
// Each invocation has one owner and ORT calls the custom node on that thread.
inline thread_local W8Profile* current_w8_profile = nullptr;
using W8Clock = std::chrono::steady_clock;

inline W8Clock::time_point w8_clock() {
    return current_w8_profile ? W8Clock::now() : W8Clock::time_point{};
}

inline void w8_record(W8Part phase, W8Clock::time_point start) {
    if (!current_w8_profile) return;
    current_w8_profile->ns[static_cast<unsigned>(phase)] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(W8Clock::now() - start).count();
    if (phase == W8Part::ffn_double_swish) ++current_w8_profile->double_swish_elements;
}

class W8ProfileScope {
public:
    explicit W8ProfileScope(W8Profile* profile) : saved_(current_w8_profile) { current_w8_profile = profile; }
    ~W8ProfileScope() { current_w8_profile = saved_; }
    W8ProfileScope(const W8ProfileScope&) = delete;
    W8ProfileScope& operator=(const W8ProfileScope&) = delete;

private:
    W8Profile* saved_;
};

}  // namespace edge_speech::asr

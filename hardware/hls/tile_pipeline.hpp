#pragma once

#include <ap_int.h>
#include <cstdint>

namespace edge_speech {
struct PortProfileConfig;

namespace pl {

using TileBeat = ap_uint<128>;
inline constexpr unsigned tile_limit_m = 16;
inline constexpr unsigned tile_limit_n = 128;
inline constexpr unsigned tile_limit_k = 4864;

enum class TileStatus : uint8_t {
    ok, invalid_descriptor, invalid_profile, invalid_mac_input, invalid_scale, nonfinite_output
};

// Each HP pointer maps the same FZBANK16-v1 image. Weight and scale bases
// are independent and beat-indexed. Activation groups use [group][m][64]
// with zero padding in the last group and sixteen signed bytes per beat.
struct TileDescriptor {
    const TileBeat* hp0;
    const TileBeat* hp1;
    const TileBeat* hp2;
    const TileBeat* hp3;
    const TileBeat* activation;
    const float* row_scales;
    const float* bias;
    float* output;
    uint16_t m, n, k;
    uint32_t weight_base, scale_base;
    uint32_t tile_id;
    bool zero_bias;
};

struct TileCompletion {
    TileStatus status;
    uint32_t tile_id;
    uint64_t profile_epoch;
    uint32_t read_bytes[4];
    uint32_t group_results;
    uint32_t mac_issue_steps;
};

// Host interface validates the full snapshot before calling the hardware ABI.
TileCompletion run_tile_pipeline(const TileDescriptor& descriptor, const PortProfileConfig& profile);

}  // namespace pl
}  // namespace edge_speech

// Completion words are status, tile ID, profile epoch, per-HP read bytes
// (four words), useful group results and MAC logical issue steps.
void tile_pipeline(const edge_speech::pl::TileBeat* hp0, const edge_speech::pl::TileBeat* hp1,
                   const edge_speech::pl::TileBeat* hp2, const edge_speech::pl::TileBeat* hp3,
                   const edge_speech::pl::TileBeat* activation, const float* row_scales,
                   const float* bias, float* output, uint16_t m, uint16_t n, uint16_t k,
                   uint32_t weight_base, uint32_t scale_base, uint32_t tile_id, bool zero_bias,
                   uint8_t active_hp_mask, uint64_t profile_epoch, uint64_t completion[9]);

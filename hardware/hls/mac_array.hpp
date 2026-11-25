#pragma once

#include <cstdint>

namespace edge_speech::pl {

inline constexpr unsigned mac_m_tile = 16;
inline constexpr unsigned mac_n_tile = 128;
inline constexpr unsigned mac_k_tile = 64;
inline constexpr unsigned mac_weight_banks = 16;
inline constexpr unsigned mac_physical_lanes = 256;
inline constexpr unsigned mac_w8_gemv_lanes = 128;

enum class MacMode : uint8_t { w8_gemm, w8_gemv, w4_gemm, w4_gemv };
enum class MacStatus : uint8_t { ok, invalid_mode, invalid_extent, invalid_code };

struct MacExtent {
    uint16_t m;
    uint16_t n;
    uint16_t k;
};

struct MacAccounting {
    uint32_t useful_macs;
    uint32_t issued_slots;
    uint32_t tail_slots;
    uint32_t issue_steps;
    uint32_t mode_lanes;
};

// Weight words are [word_index][bank]. Within a W8 word, channel c occupies
// byte c. Within W4, that channel's byte holds even K in its low nibble and
// odd K in its high nibble. One W4 call returns one raw group partial.
// On success, out-of-extent results are zero. Rejected tiles preserve raw and
// clear accounting. Accounting describes the logical schedule, without stalls.
MacStatus mac_tile(MacMode mode, const int8_t activation[mac_m_tile][mac_k_tile],
                   const uint64_t weight[mac_k_tile][mac_weight_banks], MacExtent extent,
                   int32_t raw[mac_m_tile][mac_n_tile], MacAccounting& accounting);

// Same compute engine, with a two-row buffer for a downstream tile pipeline.
MacStatus mac_pair(MacMode mode, const int8_t activation[2][mac_k_tile],
                   const uint64_t weight[mac_k_tile][mac_weight_banks], MacExtent extent,
                   int32_t raw[2][mac_n_tile], MacAccounting& accounting);

} // namespace edge_speech::pl

// Standalone HLS qualification top, backed by bank-local on-chip word ports.
uint8_t mac_array(uint8_t mode, uint16_t m, uint16_t n, uint16_t k,
                  const int8_t activation[16][64], const uint64_t weight[64][16],
                  int32_t raw[16][128], uint32_t accounting[5]);

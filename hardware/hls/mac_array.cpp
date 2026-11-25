#include "mac_array.hpp"

namespace edge_speech::pl {
namespace {

bool gemv(MacMode mode) {
    return mode == MacMode::w8_gemv || mode == MacMode::w4_gemv;
}

MacStatus validate(MacMode mode, MacExtent extent, unsigned max_rows) {
    if (static_cast<uint8_t>(mode) > static_cast<uint8_t>(MacMode::w4_gemv)) {
        return MacStatus::invalid_mode;
    }
    if (extent.m == 0 || extent.m > max_rows || (gemv(mode) && extent.m != 1) ||
        extent.n == 0 || extent.n > mac_n_tile || extent.k == 0 || extent.k > mac_k_tile) {
        return MacStatus::invalid_extent;
    }
    return MacStatus::ok;
}

MacAccounting account(MacMode mode, MacExtent extent) {
    const unsigned mode_lanes = mode == MacMode::w8_gemv ?
        mac_w8_gemv_lanes : mac_physical_lanes;
    const unsigned row_pairs = (extent.m + 1u) / 2u;
    const unsigned k_steps = mode == MacMode::w4_gemv ? (extent.k + 1u) / 2u : extent.k;
    const unsigned steps = row_pairs * k_steps;
    const unsigned useful = static_cast<unsigned>(extent.m) * extent.n * extent.k;
    const unsigned slots = steps * mode_lanes;
    return {useful, slots, slots - useful, steps, mode_lanes};
}

int8_t signed_code(unsigned code, unsigned offset) {
#pragma HLS INLINE
    return static_cast<int8_t>(static_cast<int>(code ^ offset) - static_cast<int>(offset));
}

// The mode muxes feed these same two lane groups in every mode. W4 GEMV uses
// both groups for one row and folds the groups only when emitting the result.
MacStatus compute_pair(MacMode mode, const int8_t activation[2][mac_k_tile],
                       const uint64_t weight[mac_k_tile][mac_weight_banks], MacExtent extent,
                       int32_t raw[2][mac_n_tile]) {
#pragma HLS INLINE off
    const bool four_bit = mode == MacMode::w4_gemm || mode == MacMode::w4_gemv;
    const bool dual_k = mode == MacMode::w4_gemv;
    const bool second_row = extent.m == 2;
    const unsigned steps = dual_k ? (extent.k + 1u) / 2u : extent.k;
    // Seed each lane on the first K step, avoiding a separate clearing pass.
    int32_t sum_a[mac_n_tile];
    int32_t sum_b[mac_n_tile];
    bool bad_weight[mac_n_tile];
    uint64_t saved_word[mac_weight_banks];
    bool bad_activation = false;
#pragma HLS ARRAY_PARTITION variable=sum_a complete
#pragma HLS ARRAY_PARTITION variable=sum_b complete
#pragma HLS ARRAY_PARTITION variable=bad_weight complete
#pragma HLS ARRAY_PARTITION variable=saved_word complete
#pragma HLS ARRAY_PARTITION variable=activation complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=raw cyclic factor=4 dim=2

    issue_k: for (unsigned step = 0; step < steps; ++step) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        const unsigned k = dual_k ? 2 * step : step;
        const bool odd = (k & 1u) != 0;
        const bool use_next_k = dual_k && k + 1u < extent.k;
        const bool use_b = second_row || use_next_k;
        const unsigned word_index = four_bit ? k / 2 : k;
        const int8_t a = activation[0][k];
        int8_t b = 0;
        if (second_row) {
            b = activation[1][k];
        } else if (use_next_k) {
            b = activation[0][k + 1];
        }
        bad_activation |= (a == -128 || b == -128);
        for (unsigned bank = 0; bank < mac_weight_banks; ++bank) {
#pragma HLS UNROLL
            uint64_t word;
            if (four_bit && odd) {
                word = saved_word[bank];
            } else {
                word = weight[word_index][bank];
                saved_word[bank] = word;
            }
            for (unsigned c = 0; c < 8; ++c) {
#pragma HLS UNROLL
                const unsigned n = bank * 8 + c;
                const unsigned byte = static_cast<unsigned>(word >> (8 * c)) & 255u;
                const uint8_t low = byte & 15u;
                const uint8_t high = byte >> 4;
                const uint8_t nibble = odd ? high : low;
                const int8_t wa = four_bit ? signed_code(nibble, 8u) : signed_code(byte, 128u);
                const int8_t wb = dual_k ? signed_code(high, 8u) : wa;
                const bool bad_a = four_bit ? nibble == 8 : byte == 128;
                const bool bad_code = n < extent.n && (bad_a || (use_next_k && high == 8));
                bad_weight[n] = bad_code || (step != 0 && bad_weight[n]);
                const int16_t product_a = a * wa;
                const int16_t product_b = b * wb;
#pragma HLS BIND_OP variable=product_a op=mul impl=dsp
#pragma HLS BIND_OP variable=product_b op=mul impl=dsp
                const int16_t gated_b = use_b ? product_b : 0;
                sum_a[n] = step == 0 ? product_a : sum_a[n] + product_a;
                sum_b[n] = step == 0 ? gated_b : sum_b[n] + gated_b;
            }
        }
    }

    bool invalid = bad_activation;
    for (unsigned n = 0; n < mac_n_tile; ++n) {
#pragma HLS UNROLL
        invalid |= bad_weight[n];
    }
    if (invalid) {
        return MacStatus::invalid_code;
    }
    for (unsigned block = 0; block < mac_n_tile / 4; ++block) {
#pragma HLS PIPELINE II=1
        for (unsigned lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
            const unsigned n = 4 * block + lane;
            raw[0][n] = n < extent.n ? sum_a[n] + (dual_k ? sum_b[n] : 0) : 0;
            raw[1][n] = second_row && n < extent.n ? sum_b[n] : 0;
        }
    }
    return MacStatus::ok;
}

} // namespace

MacStatus mac_pair(MacMode mode, const int8_t activation[2][mac_k_tile],
                   const uint64_t weight[mac_k_tile][mac_weight_banks], MacExtent extent,
                   int32_t raw[2][mac_n_tile], MacAccounting& accounting) {
#pragma HLS INLINE
    accounting = {};
    const auto validation = validate(mode, extent, 2);
    if (validation != MacStatus::ok) {
        return validation;
    }
    const auto status = compute_pair(mode, activation, weight, extent, raw);
    if (status == MacStatus::ok) {
        accounting = account(mode, extent);
    }
    return status;
}

MacStatus mac_tile(MacMode mode, const int8_t activation[mac_m_tile][mac_k_tile],
                   const uint64_t weight[mac_k_tile][mac_weight_banks], MacExtent extent,
                   int32_t raw[mac_m_tile][mac_n_tile], MacAccounting& accounting) {
#pragma HLS INLINE
    accounting = {};
    const auto validation = validate(mode, extent, mac_m_tile);
    if (validation != MacStatus::ok) {
        return validation;
    }

    int32_t shadow[mac_m_tile / 2][2][mac_n_tile];
#pragma HLS ARRAY_PARTITION variable=shadow cyclic factor=4 dim=3
    row_pairs: for (unsigned pair = 0; pair < (extent.m + 1u) / 2u; ++pair) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
        const unsigned row = 2 * pair;
        const uint16_t rows = row + 1u < extent.m ? 2 : 1;
        int8_t inputs[2][mac_k_tile];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=1
#pragma HLS ARRAY_PARTITION variable=inputs cyclic factor=2 dim=2
        for (unsigned k = 0; k < mac_k_tile; ++k) {
#pragma HLS PIPELINE II=1
            inputs[0][k] = activation[row][k];
            inputs[1][k] = rows == 2 ? activation[row + 1][k] : 0;
        }
        int32_t partials[2][mac_n_tile];
#pragma HLS ARRAY_PARTITION variable=partials cyclic factor=4 dim=2
        const auto status = compute_pair(mode, inputs, weight, {rows, extent.n, extent.k}, partials);
        if (status != MacStatus::ok) {
            return status;
        }
        for (unsigned n = 0; n < mac_n_tile; ++n) {
#pragma HLS PIPELINE II=1
            shadow[pair][0][n] = partials[0][n];
            shadow[pair][1][n] = partials[1][n];
        }
    }
    for (unsigned row = 0; row < mac_m_tile; ++row) {
        for (unsigned n = 0; n < mac_n_tile; ++n) {
#pragma HLS PIPELINE II=1
            raw[row][n] = row < extent.m ? shadow[row / 2][row % 2][n] : 0;
        }
    }
    accounting = account(mode, extent);
    return MacStatus::ok;
}

} // namespace edge_speech::pl

uint8_t mac_array(uint8_t mode, uint16_t m, uint16_t n, uint16_t k,
                  const int8_t activation[16][64], const uint64_t weight[64][16],
                  int32_t raw[16][128], uint32_t counters[5]) {
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
#pragma HLS INTERFACE mode=ap_memory port=activation
#pragma HLS INTERFACE mode=ap_memory port=weight
#pragma HLS INTERFACE mode=ap_memory port=raw
#pragma HLS INTERFACE mode=ap_memory port=counters
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
    using namespace edge_speech::pl;
    MacAccounting accounting;
    const auto status = mac_tile(static_cast<MacMode>(mode), activation, weight,
                                 {m, n, k}, raw, accounting);
    counters[0] = accounting.useful_macs;
    counters[1] = accounting.issued_slots;
    counters[2] = accounting.tail_slots;
    counters[3] = accounting.issue_steps;
    counters[4] = accounting.mode_lanes;
    return static_cast<uint8_t>(status);
}

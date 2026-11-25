#include "mac_pingpong.hpp"

namespace edge_speech::pl {
namespace {
int8_t signed_code(unsigned code, unsigned offset) {
#pragma HLS INLINE
    return static_cast<int8_t>(static_cast<int>(code ^ offset) - static_cast<int>(offset));
}

// One physical 2x128 MAC datapath. Current accumulators are local, while
// a second bounded snapshot emits four lanes per cycle. A complete snapshot
// replaces the old one only after all prior writes, including stalls, finish.
MacStatus advance(MacMode mode, const int8_t activation[2][64],
                   const uint64_t weight[][16], MacExtent extent,
                   bool compute, unsigned previous_rows,
                   bool previous_valid, uint32_t previous_steps,
                   ap_int<21> pending[256], hls::stream<MacPingPongFour>& partials) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=activation complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=pending complete dim=0
    const bool four_bit = mode == MacMode::w4_gemm || mode == MacMode::w4_gemv;
    const bool dual_k = mode == MacMode::w4_gemv;
    const bool second_row = extent.m == 2;
    const unsigned issue_steps = compute ? (dual_k ? (extent.k + 1u) / 2u : extent.k) : 0;
    const unsigned emit_steps = previous_rows * 32;
    const unsigned steps = issue_steps > emit_steps ? issue_steps : emit_steps;
    // Signed 21 bits cover every legal partial: 64*127*127=1,032,256.
    // W4 dual-K folding is smaller; packets sign-extend to original INT32.
    ap_int<21> sum[2][128];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=0
    bool bad_activation = false;
    bool bad_weight[128];
    uint64_t saved_word[16];
#pragma HLS ARRAY_PARTITION variable=bad_weight complete
#pragma HLS ARRAY_PARTITION variable=saved_word complete
    issue_and_emit: for (unsigned step = 0; step < steps; ++step) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        if (step < issue_steps) {
            const unsigned k = dual_k ? 2 * step : step;
            const bool odd = (k & 1u) != 0;
            const bool use_next_k = dual_k && k + 1u < extent.k;
            const bool use_b = second_row || use_next_k;
            const unsigned word_index = four_bit ? k / 2 : k;
            const int8_t a = activation[0][k];
            const int8_t b = second_row ? activation[1][k] :
                (use_next_k ? activation[0][k + 1] : int8_t(0));
            bad_activation |= a == -128 || b == -128;
            for (unsigned bank = 0; bank < 16; ++bank) {
#pragma HLS UNROLL
                uint64_t word;
                if (four_bit && odd) word = saved_word[bank];
                else {
                    word = weight[word_index][bank];
                    saved_word[bank] = word;
                }
                for (unsigned c = 0; c < 8; ++c) {
#pragma HLS UNROLL
                    const unsigned n = bank * 8 + c;
                    const unsigned byte = static_cast<unsigned>(word >> (8 * c)) & 255u;
                    const unsigned low = byte & 15u;
                    const unsigned high = byte >> 4;
                    const unsigned nibble = odd ? high : low;
                    const int8_t wa = four_bit ? signed_code(nibble, 8u) : signed_code(byte, 128u);
                    const int8_t wb = dual_k ? signed_code(high, 8u) : wa;
                    const bool bad_a = four_bit ? nibble == 8 : byte == 128;
                    const bool bad = n < extent.n && (bad_a || (use_next_k && high == 8));
                    bad_weight[n] = bad || (step != 0 && bad_weight[n]);
                    const int16_t product_a = a * wa;
                    const int16_t product_b = b * wb;
#pragma HLS BIND_OP variable=product_a op=mul impl=dsp
#pragma HLS BIND_OP variable=product_b op=mul impl=dsp
                    const int16_t gated_b = use_b ? product_b : int16_t(0);
                    sum[0][n] = step == 0 ? ap_int<21>(product_a) : ap_int<21>(sum[0][n] + product_a);
                    sum[1][n] = step == 0 ? ap_int<21>(gated_b) : ap_int<21>(sum[1][n] + gated_b);
                }
            }
        }
        if (step < emit_steps) {
            const unsigned row = step / 32;
            const unsigned low_col = (step % 32) * 4;
            MacPingPongFour packet{};
            packet.valid = previous_valid;
            packet.issue_steps = step == 0 && previous_valid ? previous_steps : 0;
            for (unsigned lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                const unsigned n = low_col + lane;
                const int32_t first = pending[lane];
                const int32_t second = pending[128 + lane];
                packet.dot[lane] = previous_valid && n < extent.n ?
                    first + (dual_k ? second : 0) : 0;
            }
            partials.write(packet);
            // Fixed taps avoid a variable 256-to-four mux. Increasing indices
            // preserve the old higher elements in C and simultaneous RTL.
            shift_pending: for (unsigned at = 0; at < 256; ++at) {
#pragma HLS UNROLL
                pending[at] = at < 252 ? pending[at + 4] : ap_int<21>(0);
            }
        }
    }
    bool invalid = bad_activation;
    for (unsigned n = 0; n < 128; ++n) {
#pragma HLS UNROLL
        invalid |= compute && bad_weight[n];
    }
    if (compute) {
        snapshot: for (unsigned n = 0; n < 128; ++n) {
#pragma HLS UNROLL
            pending[n] = sum[0][n];
            pending[128 + n] = sum[1][n];
        }
    }
    return invalid ? MacStatus::invalid_code : MacStatus::ok;
}
}

MacStatus mac_pingpong_group(MacMode mode, const TileBeat* activation,
                             const uint64_t weight[][16], MacExtent extent,
                             hls::stream<MacPingPongFour>& partials,
                             MacAccounting& accounting) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
    accounting = {};
    const bool gemv = mode == MacMode::w8_gemv || mode == MacMode::w4_gemv;
    if (static_cast<uint8_t>(mode) > static_cast<uint8_t>(MacMode::w4_gemv))
        return MacStatus::invalid_mode;
    if (extent.m == 0 || extent.m > 16 || (gemv && extent.m != 1) ||
        extent.n == 0 || extent.n > 128 || extent.k == 0 || extent.k > 64)
        return MacStatus::invalid_extent;
    ap_int<21> pending[256];
#pragma HLS ARRAY_PARTITION variable=pending complete dim=0
    const unsigned pairs = (extent.m + 1u) / 2u;
    const unsigned issue_steps = mode == MacMode::w4_gemv ? (extent.k + 1u) / 2u : extent.k;
    unsigned previous_rows = 0;
    bool previous_valid = false;
    MacStatus final_status = MacStatus::ok;
    pair_pingpong: for (unsigned pair = 0; pair <= pairs; ++pair) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=9
        const bool compute = pair < pairs;
        const unsigned row = pair * 2;
        const uint16_t rows = row + 1u < extent.m ? 2 : 1;
        int8_t inputs[2][64];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=1
#pragma HLS ARRAY_PARTITION variable=inputs cyclic factor=16 dim=2
        if (compute) {
            load_activations: for (unsigned index = 0; index < 8; ++index) {
#pragma HLS PIPELINE II=1
                const TileBeat beat = index / 4 < rows ? activation[row * 4 + index] : TileBeat(0);
                for (unsigned lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
                    const uint8_t byte = beat.range(lane * 8 + 7, lane * 8);
                    inputs[index / 4][(index % 4) * 16 + lane] = static_cast<int8_t>(byte);
                }
            }
        }
        const MacExtent pair_extent = {rows, extent.n, extent.k};
        const MacStatus status = advance(mode, inputs, weight, pair_extent, compute,
                                         previous_rows, previous_valid, issue_steps, pending, partials);
        if (status != MacStatus::ok) final_status = status;
        previous_rows = rows;
        previous_valid = status == MacStatus::ok;
    }
    if (final_status == MacStatus::ok) {
        const unsigned lanes = mode == MacMode::w8_gemv ? 128 : 256;
        const unsigned useful = unsigned(extent.m) * extent.n * extent.k;
        const unsigned steps = pairs * issue_steps;
        accounting = {useful, steps * lanes, steps * lanes - useful, steps, lanes};
    }
    return final_status;
}
}

uint8_t mac_pingpong(uint8_t mode, uint16_t m, uint16_t n, uint16_t k,
                     const edge_speech::pl::TileBeat activation[64],
                     const uint64_t weight[64][16],
                     hls::stream<edge_speech::pl::MacPingPongFour>& partials,
                     uint32_t completion[6]) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=activation
#pragma HLS INTERFACE ap_memory port=weight
#pragma HLS INTERFACE ap_fifo port=partials
#pragma HLS INTERFACE ap_memory port=completion
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
    using namespace edge_speech::pl;
    MacAccounting accounting{};
    const auto status = mac_pingpong_group(static_cast<MacMode>(mode), activation, weight,
                                           {m, n, k}, partials, accounting);
    completion[0] = static_cast<uint32_t>(status);
    completion[1] = accounting.useful_macs;
    completion[2] = accounting.issued_slots;
    completion[3] = accounting.tail_slots;
    completion[4] = accounting.issue_steps;
    completion[5] = accounting.mode_lanes;
    return static_cast<uint8_t>(status);
}

#include "tile_pipeline_pingpong.hpp"

#include "bank_router.hpp"
#include "mac_pingpong.hpp"
#include "vector_ops.hpp"

#ifndef __SYNTHESIS__
#include "../../shared/types.hpp"
#endif

#include <hls_stream.h>
#include <hls_streamofblocks.h>

#include <cstring>

namespace edge_speech::pl {
namespace {

constexpr unsigned pair_count = 8;
constexpr unsigned value_beats = 32;
constexpr unsigned scale_beats = 2;
constexpr unsigned beats_per_pair = value_beats + scale_beats;
constexpr unsigned group_stride = pair_count * value_beats;
constexpr unsigned scale_stride = pair_count * scale_beats;
constexpr unsigned max_beats = 76 * (group_stride + scale_stride);
using WeightBlock = uint64_t[value_beats][16];
using Weights = hls::stream_of_blocks<WeightBlock, 2>;

using RawFour = MacPingPongFour;

bool finite_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

bool positive_finite(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits != 0 && bits < 0x7f800000u;
}

unsigned owned_pairs(unsigned hp, unsigned mask) {
#pragma HLS INLINE
    unsigned count = 0;
    for (unsigned pair = 0; pair < pair_count; ++pair) {
#pragma HLS UNROLL
        count += bank_owner(2 * pair, mask) == hp;
    }
    return count;
}

unsigned first_pair(unsigned hp, unsigned mask) {
#pragma HLS INLINE
    unsigned count = 0;
    for (unsigned pair = 0; pair < pair_count; ++pair) {
#pragma HLS UNROLL
        count += bank_owner(2 * pair, mask) < hp;
    }
    return count;
}

template <unsigned Hp>
void read_port(const TileBeat* image, unsigned groups, unsigned mask, unsigned weight_base,
               unsigned scale_base, hls::stream<TileBeat>& stream, uint32_t& read_bytes) {
#pragma HLS INLINE off
    const unsigned pairs = owned_pairs(Hp, mask);
    const unsigned first = first_pair(Hp, mask);
    for (unsigned group = 0; group < groups; ++group) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=76
        for (unsigned slot = 0; slot < pairs; ++slot) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=4
            const unsigned pair = first + slot;
            const unsigned weight_at = weight_base + group * group_stride + pair * value_beats;
            load_values:
            for (unsigned word = 0; word < value_beats; ++word) {
#pragma HLS PIPELINE II=1
                stream.write(image[weight_at + word]);
            }
            const unsigned scale_at = scale_base + group * scale_stride + pair * scale_beats;
            load_scales:
            for (unsigned word = 0; word < scale_beats; ++word) {
#pragma HLS PIPELINE II=1
                stream.write(image[scale_at + word]);
            }
        }
    }
    read_bytes = groups * pairs * beats_per_pair * 16;
}

void load_weights(unsigned groups, unsigned mask, hls::stream<TileBeat>& hp0,
                  hls::stream<TileBeat>& hp1, hls::stream<TileBeat>& hp2,
                  hls::stream<TileBeat>& hp3, Weights& weights, hls::stream<TileBeat>& scales) {
#pragma HLS INLINE off
    unsigned count[4] = {owned_pairs(0, mask), owned_pairs(1, mask),
                         owned_pairs(2, mask), owned_pairs(3, mask)};
    unsigned first[4] = {first_pair(0, mask), first_pair(1, mask),
                         first_pair(2, mask), first_pair(3, mask)};
#pragma HLS ARRAY_PARTITION variable=count complete
#pragma HLS ARRAY_PARTITION variable=first complete
    for (unsigned group = 0; group < groups; ++group) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=76
        hls::write_lock<WeightBlock> lock(weights);
        WeightBlock& words = lock;
        TileBeat latched_scales[pair_count][scale_beats];
#pragma HLS ARRAY_PARTITION variable=latched_scales complete dim=1
        for (unsigned slot = 0; slot < count[0]; ++slot) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=4
            collect_beats:
            for (unsigned word = 0; word < beats_per_pair; ++word) {
#pragma HLS PIPELINE II=1
                TileBeat received[4];
#pragma HLS ARRAY_PARTITION variable=received complete
                received[0] = hp0.read();
                received[1] = slot < count[1] ? hp1.read() : TileBeat(0);
                received[2] = slot < count[2] ? hp2.read() : TileBeat(0);
                received[3] = hp3.read();
                for (unsigned pair = 0; pair < pair_count; ++pair) {
#pragma HLS UNROLL
                    const unsigned source = bank_owner(2 * pair, mask);
                    if (slot < count[source] && pair == first[source] + slot) {
                        const TileBeat beat = received[source];
                        if (word < value_beats) {
                            const BankPairWrite write = split_beat(beat.range(63, 0),
                                                                   beat.range(127, 64), pair);
                            words[word][write.first_bank] = write.first_word;
                            words[word][write.second_bank] = write.second_word;
                        } else {
                            latched_scales[pair][word - value_beats] = beat;
                        }
                    }
                }
            }
        }
        send_scales:
        for (unsigned word = 0; word < scale_stride; ++word) {
#pragma HLS PIPELINE II=1
            scales.write(latched_scales[word / scale_beats][word % scale_beats]);
        }
    }
}

// Isolated comparison: hold one weight-group lock while the sole shared
// MAC datapath alternates its two raw-result banks and emits row-major packets.
void run_mac(const TileBeat* activation, unsigned m, unsigned n, unsigned k,
             unsigned groups, Weights& weights, hls::stream<RawFour>& partials) {
#pragma HLS INLINE off
    const MacMode mode = m == 1 ? MacMode::w4_gemv : MacMode::w4_gemm;
    for (unsigned group = 0; group < groups; ++group) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=76
        hls::read_lock<WeightBlock> weight_lock(weights);
        WeightBlock& words = weight_lock;
        const unsigned valid_k = group + 1 == groups ? k - group * 64 : 64;
        const MacExtent extent = {static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                  static_cast<uint16_t>(valid_k)};
        MacAccounting accounting{};
        const TileBeat* group_activation = activation + group * m * 4;
        const MacStatus status = mac_pingpong_group(mode, group_activation, words,
                                                    extent, partials, accounting);
        (void)status;  // Pair validity and issue counts are carried by packets.
    }
}

void run_vector(unsigned m, unsigned n, unsigned groups, bool zero_bias,
                const float* row_scales, const float* bias, float* output,
                hls::stream<TileBeat>& scales, hls::stream<RawFour>& partials,
                TileStatus& final_status, uint32_t& group_results, uint32_t& mac_issue_steps) {
#pragma HLS INLINE off
    float accumulator[16][128];
#pragma HLS ARRAY_PARTITION variable=accumulator cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE variable=accumulator type=ram_t2p impl=bram
    TileStatus status = TileStatus::ok;
    uint32_t steps = 0;
    for (unsigned group = 0; group < groups; ++group) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=76
        uint16_t group_scales[128];
#pragma HLS ARRAY_PARTITION variable=group_scales cyclic factor=8
        bool valid_scales = true;
        read_scales:
        for (unsigned beat_at = 0; beat_at < scale_stride; ++beat_at) {
#pragma HLS PIPELINE II=1
            const TileBeat beat = scales.read();
            for (unsigned lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                const uint16_t value = beat.range(lane * 16 + 15, lane * 16);
                const unsigned col = beat_at * 8 + lane;
                group_scales[col] = value;
                if (col < n && !valid_weight_scale(value)) valid_scales = false;
            }
        }
        if (!valid_scales && status == TileStatus::ok) status = TileStatus::invalid_scale;
        bool valid_mac = true;
        update_group:
        for (unsigned lane_group = 0; lane_group < m * 32; ++lane_group) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
#pragma HLS DEPENDENCE variable=accumulator inter false
            const unsigned row = lane_group / 32;
            const unsigned low_col = (lane_group % 32) * 4;
            const RawFour packet = partials.read();
            valid_mac &= packet.valid;
            steps += packet.issue_steps;
            for (unsigned lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                const unsigned col = low_col + lane;
                if (status == TileStatus::ok && col < n) {
                    const float previous = group == 0 ? 0.0f : accumulator[row][col];
                    accumulator[row][col] =
                        qwen_group_accumulate(previous, packet.dot[lane], group_scales[col]);
                }
            }
        }
        if (!valid_mac && status == TileStatus::ok) status = TileStatus::invalid_mac_input;
    }

    float bias_cache[128];
    if (status == TileStatus::ok) {
        // Keep the AXI read loop unconditional within its branch. A per-item
        // ternary can infer a zero-length read request for nonzero bias.
        if (zero_bias) {
            clear_bias:
            for (unsigned col = 0; col < n; ++col) {
#pragma HLS PIPELINE II=1
                bias_cache[col] = 0.0f;
            }
        } else {
            load_bias:
            for (unsigned col = 0; col < n; ++col) {
#pragma HLS PIPELINE II=1
                const float value = bias[col];
                if (!finite_bits(value)) status = TileStatus::invalid_scale;
                bias_cache[col] = value;
            }
        }
        for (unsigned row = 0; row < m; ++row) {
            const float sx = row_scales[row];
            if (!positive_finite(sx)) status = TileStatus::invalid_scale;
            if (status != TileStatus::ok) break;
            finalize:
            for (unsigned col = 0; col < n; ++col) {
#pragma HLS PIPELINE II=1
                const float value = qwen_finalize(accumulator[row][col], sx, bias_cache[col]);
                if (!finite_bits(value)) status = TileStatus::nonfinite_output;
                accumulator[row][col] = value;
            }
        }
    }
    if (status == TileStatus::ok) {
        for (unsigned row = 0; row < m; ++row) {
            emit_results:
            for (unsigned col = 0; col < n; ++col) {
#pragma HLS PIPELINE II=1
                output[row * n + col] = accumulator[row][col];
            }
        }
    }
    final_status = status;
    group_results = status == TileStatus::ok ? groups * m * n : 0;
    mac_issue_steps = steps;
}

void run_streams(const TileBeat* hp0, const TileBeat* hp1, const TileBeat* hp2, const TileBeat* hp3,
                 const TileBeat* activation, const float* row_scales, const float* bias, float* output,
                 unsigned m, unsigned n, unsigned k, unsigned groups, unsigned weight_base,
                 unsigned scale_base, unsigned mask, unsigned tile_id, bool zero_bias,
                 uint32_t& bytes0, uint32_t& bytes1, uint32_t& bytes2, uint32_t& bytes3,
                 TileStatus& status, uint32_t& results, uint32_t& steps) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<TileBeat> reader0("reader0"), reader1("reader1"), reader2("reader2"), reader3("reader3");
#pragma HLS STREAM variable=reader0 depth=16
#pragma HLS STREAM variable=reader1 depth=16
#pragma HLS STREAM variable=reader2 depth=16
#pragma HLS STREAM variable=reader3 depth=16
    hls::stream<TileBeat> scales("scales");
#pragma HLS STREAM variable=scales depth=32
    hls::stream<RawFour> partials("partials");
#pragma HLS STREAM variable=partials depth=4
    Weights weights;
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS BIND_STORAGE variable=weights type=ram_s2p impl=bram
    read_port<0>(hp0, groups, mask, weight_base, scale_base, reader0, bytes0);
    read_port<1>(hp1, groups, mask, weight_base, scale_base, reader1, bytes1);
    read_port<2>(hp2, groups, mask, weight_base, scale_base, reader2, bytes2);
    read_port<3>(hp3, groups, mask, weight_base, scale_base, reader3, bytes3);
    load_weights(groups, mask, reader0, reader1, reader2, reader3, weights, scales);
    run_mac(activation, m, n, k, groups, weights, partials);
    run_vector(m, n, groups, zero_bias, row_scales, bias, output,
               scales, partials, status, results, steps);
}

}  // namespace



}  // namespace edge_speech::pl

void tile_pp(const edge_speech::pl::TileBeat* hp0, const edge_speech::pl::TileBeat* hp1,
                   const edge_speech::pl::TileBeat* hp2, const edge_speech::pl::TileBeat* hp3,
                   const edge_speech::pl::TileBeat* activation, const float* row_scales,
                   const float* bias, float* output, uint16_t m, uint16_t n, uint16_t k,
                   uint32_t weight_base, uint32_t scale_base, uint32_t tile_id, bool zero_bias,
                   uint8_t active_hp_mask, uint64_t profile_epoch, uint64_t completion[9]) {
#pragma HLS INTERFACE m_axi port=hp0 bundle=hp0 depth=20672 max_read_burst_length=16 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=hp1 bundle=hp1 depth=20672 max_read_burst_length=16 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=hp2 bundle=hp2 depth=20672 max_read_burst_length=16 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=hp3 bundle=hp3 depth=20672 max_read_burst_length=16 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=activation bundle=x depth=4864 max_read_burst_length=8 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=row_scales bundle=state depth=16 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=bias bundle=state depth=128 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=output bundle=state depth=2048 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=weight_base bundle=control
#pragma HLS INTERFACE s_axilite port=scale_base bundle=control
#pragma HLS INTERFACE s_axilite port=tile_id bundle=control
#pragma HLS INTERFACE s_axilite port=zero_bias bundle=control
#pragma HLS INTERFACE s_axilite port=active_hp_mask bundle=control
#pragma HLS INTERFACE s_axilite port=profile_epoch bundle=control
#pragma HLS INTERFACE ap_memory port=completion
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace edge_speech::pl;
    const unsigned groups = (k + 63) / 64;
    uint32_t bytes0 = 0, bytes1 = 0, bytes2 = 0, bytes3 = 0;
    uint32_t results = 0, steps = 0;
    TileStatus status;
    if (bank_owner(0, active_hp_mask) == kInvalidHp) {
        status = TileStatus::invalid_profile;
    } else if (m == 0 || m > tile_limit_m || n == 0 || n > tile_limit_n || k == 0 ||
               k > tile_limit_k || weight_base > max_beats - groups * group_stride ||
               scale_base > max_beats - groups * scale_stride ||
               !(weight_base + groups * group_stride <= scale_base ||
                 scale_base + groups * scale_stride <= weight_base)) {
        status = TileStatus::invalid_descriptor;
    } else {
        run_streams(hp0, hp1, hp2, hp3, activation, row_scales, bias, output, m, n, k, groups,
                    weight_base, scale_base, active_hp_mask, tile_id, zero_bias, bytes0, bytes1,
                    bytes2, bytes3, status, results, steps);
    }
    completion[0] = static_cast<uint64_t>(status);
    completion[1] = tile_id;
    completion[2] = profile_epoch;
    completion[3] = bytes0;
    completion[4] = bytes1;
    completion[5] = bytes2;
    completion[6] = bytes3;
    completion[7] = results;
    completion[8] = steps;
}


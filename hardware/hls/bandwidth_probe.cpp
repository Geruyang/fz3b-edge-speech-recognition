#include "bandwidth_top.hpp"
#include "bank_router.hpp"

#include <hls_stream.h>

namespace {
constexpr unsigned kPairs = 8;
constexpr unsigned kValueBeats = 32;
constexpr unsigned kScaleBeats = 2;
constexpr unsigned kMaxTiles = ((151936 + 127) / 128) * ((4864 + 63) / 64);

struct ReadBeat {
    BandwidthBeat word;
    std::uint32_t index;
    ap_uint<3> pair;
};

template <unsigned Hp>
void read_port(const BandwidthBeat* memory, unsigned mask, std::uint32_t tiles,
               std::uint32_t scale_base, hls::stream<ReadBeat>& fifo) {
#pragma HLS INLINE off
    if ((mask & (1u << Hp)) == 0 || !tiles) return;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        for (unsigned pair = 0; pair < kPairs; ++pair) {
            if (edge_speech::pl::bank_owner(2 * pair, mask) != Hp) continue;
            const auto value_start = (tile * kPairs + pair) * kValueBeats;
            const auto scale_start = scale_base + (tile * kPairs + pair) * kScaleBeats;
            for (unsigned beat = 0; beat < kValueBeats; ++beat) {
#pragma HLS PIPELINE II=1
                const auto index = value_start + beat;
                fifo.write({memory[index], index, pair});
            }
            for (unsigned beat = 0; beat < kScaleBeats; ++beat) {
#pragma HLS PIPELINE II=1
                const auto index = scale_start + beat;
                fifo.write({memory[index], index, pair});
            }
        }
    }
}

template <unsigned Hp>
void consume_port(BandwidthBeat* memory, unsigned mask, std::uint32_t tiles,
                  std::uint32_t write_base, unsigned write_enable, hls::stream<ReadBeat>& fifo,
                  BandwidthBeat& result) {
#pragma HLS INLINE off
    unsigned pairs = 0;
    for (unsigned pair = 0; pair < kPairs; ++pair) {
#pragma HLS UNROLL
        pairs += edge_speech::pl::bank_owner(2 * pair, mask) == Hp;
    }
    const std::uint32_t beats = tiles * pairs * (kValueBeats + kScaleBeats);
    std::uint64_t low_signature = 0;
    std::uint64_t high_signature = 0;
    for (std::uint32_t beat = 0; beat < beats; ++beat) {
#pragma HLS PIPELINE II=1
        const auto input = fifo.read();
        if (write_enable) memory[write_base + input.index] = input.word;
        const auto words = edge_speech::pl::split_beat(input.word.range(63, 0),
            input.word.range(127, 64), input.pair);
        const auto tag = std::uint64_t{input.index} << 4;
        low_signature += words.first_word ^ (tag + words.first_bank);
        high_signature += words.second_word ^ ~(tag + words.second_bank);
    }
    result.range(63, 0) = low_signature + high_signature;
    result.range(127, 64) = std::uint64_t{beats} * 16;
}

void scan(BandwidthBeat* hp0, BandwidthBeat* hp1,
          BandwidthBeat* hp2, BandwidthBeat* hp3,
          unsigned mask, std::uint32_t tiles, std::uint32_t scale_base, std::uint32_t write_base, unsigned write_enable,
          BandwidthBeat& result0, BandwidthBeat& result1,
          BandwidthBeat& result2, BandwidthBeat& result3) {
#pragma HLS DATAFLOW
    hls::stream<ReadBeat> fifo0, fifo1, fifo2, fifo3;
#pragma HLS STREAM variable=fifo0 depth=32
#pragma HLS STREAM variable=fifo1 depth=32
#pragma HLS STREAM variable=fifo2 depth=32
#pragma HLS STREAM variable=fifo3 depth=32
    read_port<0>(hp0, mask, tiles, scale_base, fifo0);
    read_port<1>(hp1, mask, tiles, scale_base, fifo1);
    read_port<2>(hp2, mask, tiles, scale_base, fifo2);
    read_port<3>(hp3, mask, tiles, scale_base, fifo3);
    consume_port<0>(hp0, mask, tiles, write_base, write_enable, fifo0, result0);
    consume_port<1>(hp1, mask, tiles, write_base, write_enable, fifo1, result1);
    consume_port<2>(hp2, mask, tiles, write_base, write_enable, fifo2, result2);
    consume_port<3>(hp3, mask, tiles, write_base, write_enable, fifo3, result3);
}
}  // namespace

void bandwidth_top(BandwidthBeat* hp0, BandwidthBeat* hp1,
                   BandwidthBeat* hp2, BandwidthBeat* hp3,
                   unsigned active_hp_mask, std::uint32_t tiles, std::uint32_t scale_base, std::uint32_t write_base, unsigned write_enable,
                   BandwidthBeat& result0, BandwidthBeat& result1,
                   BandwidthBeat& result2, BandwidthBeat& result3) {
#pragma HLS INTERFACE m_axi port=hp0 bundle=hp0 offset=slave depth=2176 max_read_burst_length=16 num_read_outstanding=16 max_write_burst_length=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=hp1 bundle=hp1 offset=slave depth=2176 max_read_burst_length=16 num_read_outstanding=16 max_write_burst_length=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=hp2 bundle=hp2 offset=slave depth=2176 max_read_burst_length=16 num_read_outstanding=16 max_write_burst_length=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=hp3 bundle=hp3 offset=slave depth=2176 max_read_burst_length=16 num_read_outstanding=16 max_write_burst_length=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=hp0 bundle=control
#pragma HLS INTERFACE s_axilite port=hp1 bundle=control
#pragma HLS INTERFACE s_axilite port=hp2 bundle=control
#pragma HLS INTERFACE s_axilite port=hp3 bundle=control
#pragma HLS INTERFACE ap_none port=active_hp_mask
#pragma HLS INTERFACE s_axilite port=tiles bundle=control
#pragma HLS INTERFACE s_axilite port=scale_base bundle=control
#pragma HLS INTERFACE s_axilite port=write_base bundle=control
#pragma HLS INTERFACE s_axilite port=write_enable bundle=control
#pragma HLS INTERFACE s_axilite port=result0 bundle=control
#pragma HLS INTERFACE s_axilite port=result1 bundle=control
#pragma HLS INTERFACE s_axilite port=result2 bundle=control
#pragma HLS INTERFACE s_axilite port=result3 bundle=control
#pragma HLS INTERFACE ap_ctrl_hs port=return
    const auto value_end = std::uint64_t{tiles} * kPairs * kValueBeats;
    const auto scale_end = std::uint64_t{scale_base} + std::uint64_t{tiles} * kPairs * kScaleBeats;
    const bool valid = edge_speech::pl::bank_owner(0, active_hp_mask) != edge_speech::pl::kInvalidHp
        && tiles <= kMaxTiles && scale_base >= value_end && write_enable <= 1 &&
        (!write_enable || (scale_end <= write_base &&
         std::uint64_t{write_base} + scale_end <= 0x100000000ull));
    scan(hp0, hp1, hp2, hp3, active_hp_mask, valid ? tiles : 0, scale_base, write_base, write_enable,
         result0, result1, result2, result3);
}

#pragma once

#include "asr_blocks.hpp"
#include "board/asr/stream_engine.hpp"

#include <array>
#include <bitset>
#include <limits>
#include <optional>

namespace edge_speech::pl {

// The serialized PS owner binds the HLS cache ports to these two banks.
// Completion must include the device's drain and DMA synchronization proof.
class AsrConvBanks {
public:
    using History = std::array<float, asr_cache_elements>;

    bool reset(const std::string& session, uint64_t generation) {
        if (pending_ || session.empty() || generation == 0 ||
            (session == session_ && generation <= generation_)) return false;
        session_ = session;
        generation_ = generation;
        epoch_ = 0;
        committed_ = 0;
        banks_[0] = {};
        return true;
    }

    bool begin(const asr::ChunkStamp& stamp) {
        if (pending_ || generation_ == 0 || stamp.session_id != session_ ||
            stamp.stream_generation != generation_ || stamp.state_epoch != epoch_ ||
            epoch_ == std::numeric_limits<uint64_t>::max() || stamp.chunk_id != epoch_ + 1) return false;
        pending_ = stamp;
        received_.reset();
        return true;
    }

    const History* committed(unsigned slot) const {
        if (generation_ == 0 || slot >= asr_conv_slots) return nullptr;
        return &banks_[committed_][slot];
    }

    History* work(const asr::ChunkStamp& stamp, unsigned slot) {
        if (!matches(stamp) || slot >= asr_conv_slots || received_[slot]) return nullptr;
        return &banks_[committed_ ^ 1u][slot];
    }

    bool complete_slot(const asr::ChunkStamp& stamp, unsigned slot, AsrStatus status) {
        if (status != AsrStatus::ok || work(stamp, slot) == nullptr) return false;
        received_.set(slot);
        return true;
    }

    bool commit(const asr::ChunkStamp& stamp, bool outputs_ready_and_dma_drained) {
        if (!matches(stamp) || !received_.all() || !outputs_ready_and_dma_drained) return false;
        committed_ ^= 1u;
        ++epoch_;
        pending_.reset();
        return true;
    }

    bool abort(const asr::ChunkStamp& stamp, bool dma_drained) {
        if (!matches(stamp) || !dma_drained) return false;
        pending_.reset();
        return true;
    }

    uint64_t epoch() const { return epoch_; }

private:
    bool matches(const asr::ChunkStamp& stamp) const {
        return pending_ && *pending_ == stamp;
    }

    std::array<std::array<History, asr_conv_slots>, 2> banks_{};
    std::bitset<asr_conv_slots> received_;
    std::optional<asr::ChunkStamp> pending_;
    std::string session_;
    uint64_t generation_ = 0, epoch_ = 0;
    unsigned committed_ = 0;
};

} // namespace edge_speech::pl

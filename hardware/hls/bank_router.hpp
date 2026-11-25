#pragma once

#include <cstdint>

namespace edge_speech {
struct PortProfileConfig;

namespace pl {
constexpr unsigned kInvalidHp = 4;

constexpr unsigned bank_owner(unsigned bank, unsigned active_hp_mask) {
    if (bank >= 16) return kInvalidHp;
    switch (active_hp_mask) {
    case 0x9: return bank < 8 ? 0 : 3;
    case 0xb: return bank < 6 ? 0 : (bank < 10 ? 1 : 3);
    case 0xf: return bank / 4;
    default: return kInvalidHp;
    }
}

struct BankPairWrite {
    unsigned first_bank;
    unsigned second_bank;
    std::uint64_t first_word;
    std::uint64_t second_word;
};

// Precondition: pair is an FZBANK16 pair ID in [0, 8).
constexpr BankPairWrite split_beat(std::uint64_t low, std::uint64_t high, unsigned pair) {
    return {2 * pair, 2 * pair + 1, low, high};
}

unsigned route_bank(unsigned bank, const PortProfileConfig& profile);
}  // namespace pl
}  // namespace edge_speech

#include "bank_router.hpp"
#include "../../shared/types.hpp"

namespace edge_speech::pl {
unsigned route_bank(unsigned bank, const PortProfileConfig& profile) {
    validate_port_profile(profile);
    const auto owner = bank_owner(bank, profile.active_hp_mask);
    if (owner == kInvalidHp || owner != profile.bank_to_hp[bank]) {
        throw std::invalid_argument("invalid bank ID or owner");
    }
    return owner;
}
}  // namespace edge_speech::pl

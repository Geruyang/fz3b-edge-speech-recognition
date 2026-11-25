#include "board/pl/device.hpp"
#include "driver/fzspeech_job.h"
#include "driver/fzspeech_transport_regs.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

Bytes read(const std::string& path, std::size_t expected_bytes) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "fixture file unavailable");
    Bytes data(std::istreambuf_iterator<char>(file), {});
    require(data.size() == expected_bytes, "fixture length differs from FZBANK16 probe layout");
    return data;
}

std::uint64_t le64(const Bytes& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(bytes.at(offset + i)) << (8 * i);
    }
    return value;
}

struct Expected {
    std::array<std::uint64_t, 4> signature{}, bytes{};
};

Expected reference(const Bytes& payload, unsigned mask) {
    Expected result;
    for (unsigned tile = 0; tile < 4; ++tile) {
        for (unsigned pair = 0; pair < 8; ++pair) {
            const auto port = fz_pair_port(mask, pair);
            for (unsigned kind = 0; kind < 2; ++kind) {
                for (unsigned beat = 0; beat < (kind ? 2U : 32U); ++beat) {
                    const auto index = kind ? 1024 + (tile * 8 + pair) * 2 + beat :
                                              (tile * 8 + pair) * 32 + beat;
                    const auto tag = std::uint64_t(index) << 4;
                    result.signature[port] += (le64(payload, 16 * index) ^ (tag + 2 * pair)) +
                        (le64(payload, 16 * index + 8) ^ ~(tag + 2 * pair + 1));
                    result.bytes[port] += 16;
                }
            }
        }
    }
    return result;
}

void identity(const fz_transport_stats& stats, std::uint64_t generation, const fz_profile& profile) {
    require(stats.generation == generation && stats.chain_id == 0,
            "idle transport generation/held-chain identity differs");
    require(stats.profile.epoch == profile.epoch && stats.profile.id == profile.id &&
            stats.profile.mask == profile.mask, "actual transport profile readback differs");
    require(stats.read_limit == 8 && stats.write_limit == 8, "actual credit limits differ");
    require(stats.status != FZ_STATUS_UNKNOWN && stats.fifo_empty_mask == 15 &&
            (stats.flags & FZ_TR_FLAG_ENGINE_IDLE), "transport is not provably idle");
    for (const auto& port : stats.ports) {
        require(!port.read_outstanding && !port.write_outstanding && !port.pending_flags,
                "physical transactions, pending fragments or faults remain");
        require(port.read_peak <= 16 && port.write_peak <= 16, "invalid cumulative peak readback");
    }
}

std::uint64_t delta(std::uint64_t before, std::uint64_t after) {
    require(after >= before, "physical counter wrapped or regressed");
    return after - before;
}

void counters(const fz_transport_stats& before, const fz_transport_stats& after,
              const fz_probe_receipt& receipt, const Expected& expected, bool write_enable) {
    require(after.counter_epoch == before.counter_epoch, "physical counter epoch changed");
    require(after.snapshot_sequence > before.snapshot_sequence,
            "snapshot sequence repeated or wrapped during this single-job check");
    require(receipt.span_count >= (write_enable ? 128U : 64U) && receipt.span_count <= FZ_JOB_MAX_SPANS &&
            receipt.packet_bytes == FZ_JOB_HEADER_BYTES + FZ_JOB_ENTRY_BYTES * receipt.span_count,
            "provider packet length/span provenance is invalid");
    for (unsigned port = 0; port < 4; ++port) {
        const auto& a = before.ports[port];
        const auto& b = after.ports[port];
        const auto metadata = port == 0 ? receipt.packet_bytes : 0;
        require(delta(a.read_bytes, b.read_bytes) == expected.bytes[port] + metadata,
                "physical read bytes differ from payload plus actual HP0 descriptor fetch");
        require(delta(a.write_bytes, b.write_bytes) == (write_enable ? expected.bytes[port] : 0),
                "physical write bytes differ from the submitted mode");
        (void)delta(a.ar_stall, b.ar_stall);
        (void)delta(a.r_stall, b.r_stall);
        (void)delta(a.aw_stall, b.aw_stall);
        (void)delta(a.w_stall, b.w_stall);
        (void)delta(a.b_stall, b.b_stall);
        if (!(after.profile.mask & (1U << port))) {
            require(b.read_bytes == a.read_bytes && b.write_bytes == a.write_bytes,
                    "disabled physical port emitted traffic");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2,
                "fixture-directory [--execute] [--read-only] [--release-provider] [--profile=2|3|4] required");
        const std::string directory = argv[1];
        bool execute = false, write_enable = true, release_provider = false;
        unsigned profile_id = 2;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--execute") {
                execute = true;
            } else if (option == "--read-only") {
                write_enable = false;
            } else if (option == "--release-provider") {
                release_provider = true;
            } else if (option == "--profile=2" || option == "--profile=3" || option == "--profile=4") {
                profile_id = static_cast<unsigned>(option.back() - '0');
            } else {
                throw std::runtime_error("unknown or malformed probe option");
            }
        }
        const auto mask = fz_profile_mask(profile_id);
        auto payload = read(directory + "/values.bin", 16384);
        const auto scales = read(directory + "/scales.bin", 1024);
        payload.insert(payload.end(), scales.begin(), scales.end());
        const auto expected = reference(payload, mask);
        if (!execute) {
            std::cout << "{\"mode\":\"offline_plan_only\",\"payload_bytes\":17408,\"profile\":" << profile_id
                      << ",\"write_enable\":" << write_enable << "}\n";
            return 0;
        }
#ifdef __linux__
        using namespace edge_speech::pl;
        LinuxDevice device;
        const auto initial = device.info();
        require((initial.capabilities & (FZ_CAP_BANK16_PROBE | FZ_CAP_JOB_RECEIPT)) ==
                (FZ_CAP_BANK16_PROBE | FZ_CAP_JOB_RECEIPT), "provider lacks probe/receipt capability");
        const auto info = device.prepare();
        auto profile = device.set_port_profile({info.profile.epoch, profile_id, mask});
        profile = device.set_transport_limits({info.generation, profile, 8, 8}).profile;
        const auto input = device.register_buffer(info.generation, 17440, Access::read);
        fz_buffer output{};
        if (write_enable) {
            output = device.register_buffer(info.generation, 17472, Access::read_write);
        }
        Bytes source(17440, 0xa5), destination(17472, 0xd3);
        std::copy(payload.begin(), payload.end(), source.begin() + 16);
        device.write_buffer(input, 0, source.data(), source.size());
        if (write_enable) {
            device.write_buffer(output, 0, destination.data(), destination.size());
        }
        const auto before = device.transport_statistics();
        identity(before, info.generation, profile);
        const fz_chain chain{info.generation, 1, profile, FZ_DESCRIPTOR_VERSION, 0, 0, 0};
        device.submit_probe(chain, input, 16, output, 32, write_enable, 1);
        const auto complete = device.wait(info.generation, 1, 5000);
        require(complete.status == FZ_STATUS_SUCCESS, "completion failed; driver owns isolation/retained mappings");
        require(complete.generation == info.generation && complete.chain_id == 1 &&
                complete.profile.epoch == profile.epoch && complete.profile.id == profile.id &&
                complete.profile.mask == profile.mask, "wait receipt identity differs");
        const auto receipt = device.job_receipt(info.generation, 1);
        require(receipt.generation == info.generation && receipt.chain_id == 1 &&
                receipt.profile.epoch == profile.epoch && receipt.profile.id == profile.id &&
                receipt.profile.mask == profile.mask, "HLS receipt identity differs");
        require(receipt.status == FZ_STATUS_SUCCESS && !receipt.error && !receipt.failed_job_id &&
                receipt.failed_span == FZ_NO_DEPENDENCY && receipt.failed_transfer == FZ_NO_DEPENDENCY &&
                receipt.failed_dma_span == FZ_NO_DEPENDENCY, "HLS receipt carries failure provenance");
        for (unsigned port = 0; port < 4; ++port) {
            require(receipt.signature[port] == expected.signature[port], "HLS signature differs from CPU reference");
            require(receipt.read_bytes[port] == expected.bytes[port], "HLS payload-byte count differs");
        }
        Bytes actual_source(source.size());
        device.read_buffer(input, 0, actual_source.data(), actual_source.size());
        require(actual_source == source, "source or source guards changed");
        if (write_enable) {
            Bytes actual_destination(destination.size());
            device.read_buffer(output, 0, actual_destination.data(), actual_destination.size());
            std::copy(payload.begin(), payload.end(), destination.begin() + 32);
            require(actual_destination == destination, "destination payload or guards differ");
        }
        const auto after = device.transport_statistics();
        identity(after, info.generation, profile);
        counters(before, after, receipt, expected, write_enable);
        const auto idle = device.quiesce(info.generation, 5000);
        require(idle.status == FZ_STATUS_IDLE && idle.generation == info.generation &&
                idle.profile.epoch == profile.epoch && idle.profile.id == profile.id && idle.profile.mask == profile.mask,
                "quiesce receipt does not prove the submitted session idle");
        const auto stopped = device.transport_statistics();
        identity(stopped, info.generation, profile);
        require((stopped.flags & 3U) == 3U, "final issuance is not stopped");
        device.unregister(input);
        if (write_enable) {
            device.unregister(output);
        }
        if (release_provider) {
            device.release_provider(info.generation, 5000);
        }
        std::cout << "{\"status\":\"PASS\",\"generation\":" << info.generation << ",\"chain_id\":1,\"epoch\":"
                  << profile.epoch << ",\"payload_read_bytes\":17408,\"packet_fetch_read_bytes\":" << receipt.packet_bytes
                  << ",\"span_count\":" << receipt.span_count << ",\"write_enable\":" << write_enable
                  << ",\"scope\":\"single_correctness_job_not_bandwidth_qualification\"}\n";
#else
        (void)expected;
        (void)release_provider;
        (void)identity;
        (void)counters;
        throw std::runtime_error("--execute requires the Linux device interface");
#endif
    } catch (const std::exception& error) {
        std::cerr << "probe failed: " << error.what() << '\n';
        return 1;
    }
}

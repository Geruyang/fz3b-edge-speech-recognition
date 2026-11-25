#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace edge_speech::pl {

enum class AsrOperation : std::uint32_t {
    ffn = 0, conv = 1, embedding = 2,
    ffn_hidden_left = 3, ffn_hidden_right = 4, ffn_output_left = 5, ffn_output_right = 6
};
inline constexpr std::uint32_t asr_job_receipt_all = 31;

struct AsrJobRequest {
    AsrOperation operation{};
    std::uint32_t rows = 0;
    std::uint16_t call_id = 0;
    const void* input = nullptr;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    const void* old_history = nullptr;  // Null selects the driver's committed Conv slot.
    std::size_t history_bytes = 0;
};

struct AsrJobReceipt {
    std::uint64_t generation = 0, epoch = 0, chain_id = 0;
    std::uint32_t sequence = 0;
    AsrOperation operation{};
    std::uint16_t call_id = 0;
    std::uint32_t rows = 0, irq_events = 0, flags = 0;
    std::uint64_t device_ns = 0;
    std::uint64_t prepare_ns = 0, retire_ns = 0;
    std::uint32_t packet_bytes = 0, span_count = 0, packet_crc32 = 0;
    std::uint64_t ioctl_ns = 0, audit_ns = 0;
    std::uint64_t queue_ns = 0;
};

struct AsrJobResult {
    std::vector<std::uint8_t> output;
    AsrJobReceipt receipt;
    std::vector<std::uint8_t> working_history;
};

struct AsrPairedFfnReceipt {
    std::array<AsrJobReceipt, 2> first, second;
    std::uint64_t first_join_ns = 0, second_join_ns = 0;
    std::uint64_t first_wall_ns = 0, second_wall_ns = 0;
    std::uint64_t audit_ns = 0;
};

// One hardware lease. Job output and history are provisional until commit.
// A failed chunk requires abort. Only cancel can overlap a transaction method.
class AsrJobTransport {
public:
    virtual ~AsrJobTransport() = default;
    virtual void begin_chunk(bool reset) = 0;
    virtual AsrJobResult submit(const AsrJobRequest& request) = 0;
    virtual AsrJobReceipt submit_into(const AsrJobRequest& request, void* output, void* history) {
        if (!output || (request.history_bytes && !history))
            throw std::invalid_argument("missing ASR result destination");
        auto result = submit(request);
        if (result.output.size() != request.output_bytes || result.working_history.size() != request.history_bytes)
            throw std::runtime_error("ASR result size differs from its request");
        std::memcpy(output, result.output.data(), request.output_bytes);
        if (request.history_bytes)
            std::memcpy(history, result.working_history.data(), request.history_bytes);
        return result.receipt;
    }
    virtual bool supports_paired_ffn(std::uint16_t) const noexcept { return false; }
    virtual AsrPairedFfnReceipt submit_paired_ffn_into(const AsrJobRequest&, void*) {
        throw std::logic_error("ASR transport does not support paired FFN");
    }
    virtual void commit_chunk() = 0;
    virtual void abort_chunk() noexcept = 0;
    virtual void cancel() noexcept = 0;
};

std::unique_ptr<AsrJobTransport> open_linux_asr_jobs(
    const std::filesystem::path& device = "/dev/fzspeech-asr-jobs");
struct AsrLaneIdentity {
    std::uint64_t platform_generation;
    std::uint32_t lane;
};
std::unique_ptr<AsrJobTransport> open_linux_asr_lane(const std::filesystem::path& device,
                                                   AsrLaneIdentity* identity = nullptr);
std::unique_ptr<AsrJobTransport> make_dual_asr_jobs(
    std::unique_ptr<AsrJobTransport> lane0, std::unique_ptr<AsrJobTransport> lane1);
inline std::unique_ptr<AsrJobTransport> open_linux_dual_asr_jobs(
    const std::filesystem::path& lane0_path = "/dev/fzspeech-asr-lane0",
    const std::filesystem::path& lane1_path = "/dev/fzspeech-asr-lane1")
{
    AsrLaneIdentity first{}, second{};
    auto lane0 = open_linux_asr_lane(lane0_path, &first);
    auto lane1 = open_linux_asr_lane(lane1_path, &second);
    if (first.lane != 0 || second.lane != 1 || first.platform_generation != second.platform_generation)
        throw std::runtime_error("ASR lanes do not share the same platform lease");
    return make_dual_asr_jobs(std::move(lane0), std::move(lane1));
}

}  // namespace edge_speech::pl

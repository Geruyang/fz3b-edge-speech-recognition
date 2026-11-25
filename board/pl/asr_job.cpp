#include "asr_job.hpp"

#include <cerrno>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef __linux__
#include "driver/uapi/fzspeech_asr_jobs.h"
#include "shared/asr_job.h"

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <limits>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace edge_speech::pl {
namespace {
#ifdef __linux__
static_assert(sizeof(fz_asr_chunk) == 32 && sizeof(fz_asr_request) == 88 &&
              sizeof(fz_asr_receipt) == 96 && sizeof(fz_asr_submit) == 184);
static_assert(asr_job_receipt_all == FZ_ASR_RECEIPT_ALL);

void check_ioctl(int fd, unsigned long command, void* data)
{
    if (ioctl(fd, command, data) < 0)
        throw std::system_error(errno, std::generic_category(), "ASR job ioctl");
}

class LinuxAsrJobs final : public AsrJobTransport {
public:
    explicit LinuxAsrJobs(const std::filesystem::path& device, bool dual = false) : dual_(dual)
    {
        fd_ = open(device.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0)
            throw std::system_error(errno, std::generic_category(), device.string());
        try {
            fz_asr_info info{};
            check_ioctl(fd_, FZ_ASR_GET_INFO, &info);
            const auto& digest = dual_ ? fz_asr_dual_layout_sha256 : fz_asr_layout_sha256;
            if (info.abi != (dual_ ? FZ_ASR_DUAL_USERS_ABI : FZ_ASR_USERS_ABI) ||
                info.wire_version != FZ_ASR_JOB_VERSION ||
                info.dma_bits != 30 || info.reserved ||
                !std::equal(std::begin(info.capability_sha256), std::end(info.capability_sha256),
                            std::begin(digest)))
                throw std::runtime_error("ASR hardware capability differs from the selected model");
            if (dual_) {
                check_ioctl(fd_, FZ_ASR_GET_LANE, &lane_);
                if (!lane_.platform_generation || lane_.lane_count != 2 || lane_.lane >= lane_.lane_count)
                    throw std::runtime_error("ASR hardware lane identity is invalid");
            }
        } catch (...) {
            close(fd_);
            throw;
        }
    }

    const fz_asr_lane_info& lane_info() const noexcept { return lane_; }

    ~LinuxAsrJobs() override
    {
        abort_chunk();
        close(fd_);
    }

    void begin_chunk(bool reset) override
    {
        if (chunk_.chunk_id)
            throw std::logic_error("ASR chunk is already open");
        fz_asr_chunk next{};
        next.flags = reset ? FZ_ASR_CHUNK_RESET : 0;
        check_ioctl(fd_, FZ_ASR_BEGIN_CHUNK, &next);
        if (!next.generation || !next.epoch || next.chunk_id <= last_chunk_ || next.reserved ||
            next.flags != (reset ? FZ_ASR_CHUNK_RESET : 0) ||
            (generation_ && (next.generation != generation_ || next.epoch != epoch_))) {
            abort_chunk();
            throw std::runtime_error("ASR chunk identity differs");
        }
        chunk_ = next;
        generation_ = next.generation;
        epoch_ = next.epoch;
        last_chunk_ = next.chunk_id;
        failed_ = false;
    }

    AsrJobResult submit(const AsrJobRequest& request) override
    {
        start_submit(request);
        AsrJobResult result{};
        result.output.resize(request.output_bytes);
        result.working_history.resize(request.history_bytes);
        result.receipt = perform_submit(request, result.output.data(), result.working_history.data());
        return result;
    }

    AsrJobReceipt submit_into(const AsrJobRequest& request, void* output, void* history) override
    {
        start_submit(request);
        if (!output || (request.history_bytes && !history))
            throw std::invalid_argument("missing ASR result destination");
        return perform_submit(request, output, history);
    }

    void commit_chunk() override
    {
        if (!chunk_.chunk_id || failed_)
            throw std::logic_error("ASR chunk cannot be committed");
        check_ioctl(fd_, FZ_ASR_COMMIT_CHUNK, &chunk_);
        chunk_ = {};
    }

    void abort_chunk() noexcept override
    {
        (void)ioctl(fd_, FZ_ASR_ABORT_CHUNK, nullptr);
        chunk_ = {};
        failed_ = false;
    }

    void cancel() noexcept override { (void)ioctl(fd_, FZ_ASR_CANCEL_JOB, nullptr); }

private:
    void start_submit(const AsrJobRequest& request)
    {
        if (!chunk_.chunk_id || failed_)
            throw std::logic_error("ASR chunk is unavailable");
        failed_ = true;
        check_request(request);
    }

    AsrJobReceipt perform_submit(const AsrJobRequest& request, void* output, void* history)
    {
        fz_asr_submit job{};
        auto& req = job.request;
        req.generation = chunk_.generation;
        req.epoch = chunk_.epoch;
        req.chunk_id = chunk_.chunk_id;
        req.input_ptr = reinterpret_cast<std::uintptr_t>(request.input);
        req.output_ptr = reinterpret_cast<std::uintptr_t>(output);
        req.old_history_ptr = reinterpret_cast<std::uintptr_t>(request.old_history);
        req.working_history_ptr = request.history_bytes
            ? reinterpret_cast<std::uintptr_t>(history) : 0;
        req.operation = static_cast<std::uint32_t>(request.operation);
        req.rows = request.rows;
        req.call_id = request.call_id;
        req.input_bytes = static_cast<std::uint32_t>(request.input_bytes);
        req.output_bytes = static_cast<std::uint32_t>(request.output_bytes);
        req.history_bytes = static_cast<std::uint32_t>(request.history_bytes);
        const auto started = std::chrono::steady_clock::now();
        if (ioctl(fd_, FZ_ASR_SUBMIT_JOB, static_cast<void*>(&job)) < 0) {
            const int error = errno;
            constexpr const char* operations[] = {"ffn", "conv", "embedding", "ffn-hidden-left",
                "ffn-hidden-right", "ffn-output-left", "ffn-output-right"};
            const auto operation = operations[static_cast<std::uint32_t>(request.operation)];
            throw std::system_error(error, std::generic_category(),
                std::string("ASR job ioctl=FZ_ASR_SUBMIT_JOB operation=") + operation +
                " call_id=" + std::to_string(request.call_id) + " chunk_id=" + std::to_string(chunk_.chunk_id) +
                " device_generation=" + std::to_string(chunk_.generation) + " device_epoch=" + std::to_string(chunk_.epoch) +
                " rows=" + std::to_string(request.rows) + " errno=" + std::to_string(error));
        }
        const auto ioctl_elapsed = std::chrono::steady_clock::now() - started;
        const auto& r = job.receipt;
        if (r.generation != chunk_.generation || r.epoch != chunk_.epoch ||
            r.chunk_id != chunk_.chunk_id || r.chain_id <= last_chain_ ||
            !r.sequence || r.sequence <= last_sequence_ || r.hw_done_sequence != r.sequence ||
            r.operation != req.operation || r.rows != req.rows || r.call_id != req.call_id ||
            r.flags != FZ_ASR_RECEIPT_ALL || !r.irq_events || !r.span_count ||
            r.span_count > FZ_JOB_MAX_SPANS ||
            r.packet_bytes != FZ_JOB_HEADER_BYTES + r.span_count * FZ_JOB_ENTRY_BYTES)
            throw std::runtime_error("ASR completion receipt differs");
        last_chain_ = r.chain_id;
        last_sequence_ = r.sequence;
        AsrJobReceipt receipt{};
        receipt.generation = r.generation;
        receipt.epoch = r.epoch;
        receipt.chain_id = r.chain_id;
        receipt.sequence = r.sequence;
        receipt.operation = request.operation;
        receipt.call_id = request.call_id;
        receipt.rows = r.rows;
        receipt.irq_events = r.irq_events;
        receipt.flags = r.flags;
        receipt.prepare_ns = r.prepare_ns;
        receipt.device_ns = r.device_ns;
        receipt.retire_ns = r.retire_ns;
        receipt.packet_bytes = r.packet_bytes;
        receipt.span_count = r.span_count;
        receipt.packet_crc32 = r.packet_crc32;
        receipt.ioctl_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ioctl_elapsed).count();
        failed_ = false;
        return receipt;
    }

    void check_request(const AsrJobRequest& request) const
    {
        if (request.call_id >= FZ_ASR_CALL_COUNT)
            throw std::invalid_argument("ASR call ID is outside the selected model");
        const auto op = static_cast<std::uint32_t>(request.operation);
        const auto& call = fz_asr_calls[request.call_id];
        if (!fz_asr_operation_allowed(request.call_id, op, dual_, lane_.lane) ||
            request.rows != call.rows || !request.input ||
            request.input_bytes != fz_asr_operation_region_bytes(request.call_id, op, FZ_ASR_INPUT) ||
            request.output_bytes != fz_asr_operation_region_bytes(request.call_id, op, FZ_ASR_OUTPUT) ||
            request.history_bytes != fz_asr_operation_region_bytes(request.call_id, op, FZ_ASR_OLD) ||
            (!request.history_bytes && request.old_history))
            throw std::invalid_argument("ASR request shape differs from the selected call");
    }

    int fd_ = -1;
    bool dual_;
    fz_asr_lane_info lane_{};
    fz_asr_chunk chunk_{};
    std::uint64_t generation_ = 0, epoch_ = 0, last_chunk_ = 0, last_chain_ = 0;
    std::uint32_t last_sequence_ = 0;
    bool failed_ = false;
};
#endif
}  // namespace

std::unique_ptr<AsrJobTransport> open_linux_asr_jobs(const std::filesystem::path& device)
{
#ifdef __linux__
    return std::make_unique<LinuxAsrJobs>(device);
#else
    (void)device;
    throw std::system_error(std::make_error_code(std::errc::not_supported), "ASR job device");
#endif
}

std::unique_ptr<AsrJobTransport> open_linux_asr_lane(const std::filesystem::path& device, AsrLaneIdentity* identity)
{
#ifdef __linux__
    auto lane = std::make_unique<LinuxAsrJobs>(device, true);
    if (identity)
        *identity = {lane->lane_info().platform_generation, lane->lane_info().lane};
    return lane;
#else
    (void)device;
    (void)identity;
    throw std::system_error(std::make_error_code(std::errc::not_supported), "ASR lane device");
#endif
}

}  // namespace edge_speech::pl

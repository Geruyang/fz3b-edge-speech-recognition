#include "board/pl/asr_job.hpp"
#include "shared/asr_job.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

namespace edge_speech::pl {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point started)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
}

class DualAsrJobs final : public AsrJobTransport {
public:
    DualAsrJobs(std::unique_ptr<AsrJobTransport> lane0, std::unique_ptr<AsrJobTransport> lane1)
        : lanes_{std::move(lane0), std::move(lane1)}
    {
        if (!lanes_[0] || !lanes_[1])
            throw std::invalid_argument("paired FFN requires both ASR lanes");
        worker_ = std::thread([this] { work(); });
    }

    ~DualAsrJobs() override
    {
        abort_chunk();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        worker_.join();
    }

    void begin_chunk(bool reset) override
    {
        if (active_)
            throw std::logic_error("ASR chunk is already open");
        cancelled_.store(false);
        failed_ = true;
        previous_ = {};
        try {
            lanes_[0]->begin_chunk(reset);
            lanes_[1]->begin_chunk(reset);
            if (cancelled_.load()) throw std::runtime_error("paired ASR chunk cancelled during open");
        } catch (...) {
            abort_chunk();
            throw;
        }
        active_ = true;
        failed_ = false;
    }

    AsrJobResult submit(const AsrJobRequest& request) override
    {
        check_full(request);
        auto result = lanes_[0]->submit(request);
        verify(0, request, result.receipt);
        finish_submit();
        return result;
    }

    AsrJobReceipt submit_into(const AsrJobRequest& request, void* output, void* history) override
    {
        check_full(request);
        auto receipt = lanes_[0]->submit_into(request, output, history);
        verify(0, request, receipt);
        finish_submit();
        return receipt;
    }

    bool supports_paired_ffn(std::uint16_t call_id) const noexcept override
    {
        return call_id < FZ_ASR_CALL_COUNT && fz_asr_calls[call_id].operation == FZ_ASR_FFN;
    }

    AsrPairedFfnReceipt submit_paired_ffn_into(const AsrJobRequest& request, void* output) override
    {
        start_submit();
        if (request.operation != AsrOperation::ffn || !supports_paired_ffn(request.call_id) ||
            request.rows != fz_asr_calls[request.call_id].rows || !request.input || !output ||
            request.input_bytes != request.rows * 1024 || request.output_bytes != request.rows * 1024 ||
            request.history_bytes || request.old_history)
            throw std::invalid_argument("paired FFN request differs from the selected model");

        AsrPairedFfnReceipt result{};
        const auto first = Clock::now();
        result.first = submit_phase(request, AsrOperation::ffn_hidden_left, AsrOperation::ffn_hidden_right,
                                    request.input, request.input_bytes, request.rows * 1536);
        const auto first_join = Clock::now();
        join_rows(request.rows, 384, hidden_.data());
        result.first_join_ns = elapsed(first_join);
        result.first_wall_ns = elapsed(first);
        if (cancelled_.load()) throw std::runtime_error("paired FFN cancelled between stages");

        const auto second = Clock::now();
        result.second = submit_phase(request, AsrOperation::ffn_output_left, AsrOperation::ffn_output_right,
                                     hidden_.data(), request.rows * 3072, request.rows * 512);
        const auto second_join = Clock::now();
        // Stage two is assembled in private storage until both validated halves are present.
        join_rows(request.rows, 128, hidden_.data());
        finish_submit();
        std::memcpy(output, hidden_.data(), request.output_bytes);
        result.second_join_ns = elapsed(second_join);
        result.second_wall_ns = elapsed(second);
        return result;
    }

    void commit_chunk() override
    {
        if (!active_ || failed_ || cancelled_.load())
            throw std::logic_error("paired ASR chunk cannot be committed");
        failed_ = true;
        // Lane 1 accepts only stateless shards in this logical session.
        lanes_[1]->commit_chunk();
        lanes_[0]->commit_chunk();
        active_ = false;
        failed_ = false;
    }

    void abort_chunk() noexcept override
    {
        for (const auto& lane : lanes_) lane->abort_chunk();
        active_ = false;
        failed_ = false;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true);
        for (const auto& lane : lanes_) lane->cancel();
    }

private:
    struct PeerTask {
        AsrJobRequest request;
        Clock::time_point queued;
        AsrJobReceipt receipt;
        std::exception_ptr error;
    };

    void start_submit()
    {
        if (!active_ || failed_ || cancelled_.load())
            throw std::logic_error("paired ASR chunk is unavailable");
        failed_ = true;
    }

    void check_full(const AsrJobRequest& request)
    {
        start_submit();
        if (static_cast<std::uint32_t>(request.operation) > FZ_ASR_EMBEDDING)
            throw std::invalid_argument("paired ASR transport accepts logical calls");
    }

    void finish_submit()
    {
        if (cancelled_.load()) throw std::runtime_error("paired ASR chunk cancelled");
        failed_ = false;
    }

    void verify(unsigned lane, const AsrJobRequest& request, const AsrJobReceipt& receipt)
    {
        const auto& prior = previous_[lane];
        if (receipt.call_id != request.call_id || receipt.operation != request.operation ||
            receipt.rows != request.rows || receipt.flags != asr_job_receipt_all || receipt.irq_events != 1 ||
            !receipt.generation || !receipt.epoch || !receipt.chain_id || !receipt.sequence ||
            (prior && (receipt.generation != prior->generation || receipt.epoch != prior->epoch ||
                       receipt.chain_id != prior->chain_id + 1 || receipt.sequence != prior->sequence + 1)))
            throw std::runtime_error("ASR lane receipt does not match the request or sequence");
        previous_[lane] = receipt;
    }

    AsrJobReceipt run(unsigned lane, const AsrJobRequest& request, Clock::time_point queued)
    {
        const auto queue_ns = elapsed(queued);
        auto receipt = lanes_[lane]->submit_into(request, halves_[lane].data(), nullptr);
        verify(lane, request, receipt);
        receipt.queue_ns = queue_ns;
        return receipt;
    }

    void work() noexcept
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [this] { return stopping_ || pending_; });
            if (stopping_) return;
            auto& task = *pending_;
            lock.unlock();
            try {
                task.receipt = run(1, task.request, task.queued);
            } catch (...) {
                task.error = std::current_exception();
                lanes_[0]->cancel();
            }
            lock.lock();
            pending_ = nullptr;
            completed_.notify_one();
        }
    }

    std::array<AsrJobReceipt, 2> submit_phase(const AsrJobRequest& job, AsrOperation left, AsrOperation right,
                                               const void* input, std::size_t input_bytes, std::size_t output_bytes)
    {
        const auto queued = Clock::now();
        PeerTask peer{{right, job.rows, job.call_id, input, input_bytes, output_bytes}, queued, {}, {}};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_ = &peer;
        }
        ready_.notify_one();
        AsrJobReceipt receipt{};
        std::exception_ptr error;
        try {
            receipt = run(0, {left, job.rows, job.call_id, input, input_bytes, output_bytes}, queued);
        } catch (...) {
            error = std::current_exception();
            lanes_[1]->cancel();
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            completed_.wait(lock, [this] { return !pending_; });
        }
        if (error) std::rethrow_exception(error);
        if (peer.error) std::rethrow_exception(peer.error);
        if (cancelled_.load()) throw std::runtime_error("paired FFN phase cancelled");
        return {receipt, peer.receipt};
    }

    void join_rows(std::size_t rows, std::size_t columns, float* result) const
    {
        for (const auto& half : halves_) {
            if (!std::all_of(half.begin(), half.begin() + rows * columns,
                             [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("nonfinite paired FFN stage");
        }
        for (std::size_t row = 0; row < rows; ++row) {
            std::copy_n(halves_[0].data() + row * columns, columns, result + row * 2 * columns);
            std::copy_n(halves_[1].data() + row * columns, columns, result + (row * 2 + 1) * columns);
        }
    }

    std::array<std::unique_ptr<AsrJobTransport>, 2> lanes_;
    std::array<std::optional<AsrJobReceipt>, 2> previous_;
    std::array<std::array<float, 16 * 384>, 2> halves_;
    std::array<float, 16 * 768> hidden_;
    std::atomic<bool> cancelled_{false};
    bool active_ = false, failed_ = false, stopping_ = false;
    std::mutex mutex_;
    std::condition_variable ready_, completed_;
    PeerTask* pending_ = nullptr;
    std::thread worker_;
};

} // namespace

std::unique_ptr<AsrJobTransport> make_dual_asr_jobs(
    std::unique_ptr<AsrJobTransport> lane0, std::unique_ptr<AsrJobTransport> lane1)
{
    return std::make_unique<DualAsrJobs>(std::move(lane0), std::move(lane1));
}

} // namespace edge_speech::pl

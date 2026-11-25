#pragma once

#include "board/asr/ort_backend.hpp"
#include "shared/asr_job_table.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace edge_speech::asr {

using OrtJobClock = std::chrono::steady_clock;

inline std::uint64_t job_elapsed_ns(OrtJobClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(OrtJobClock::now() - start).count();
}

// The deployment encoder uses the sequential CPU provider with one intra-op
// worker. A custom op running outside its caller's transaction fails closed.
class OrtAsrJobScope {
public:
    OrtAsrJobScope(const OrtAsrJobs& jobs, const ChunkStamp& stamp)
        : jobs_(jobs), stamp_(stamp), trace_{stamp, 0, 0, 0} {
        if (current_) throw std::logic_error("nested ORT ASR transaction");
        if (jobs_.transport) {
            try {
                jobs_.transport->begin_chunk(stamp_.state_epoch == 0);
            } catch (...) {
                jobs_.transport->abort_chunk();
                throw;
            }
        }
        current_ = this;
    }

    ~OrtAsrJobScope() {
        current_ = nullptr;
        if (!committed_ && jobs_.transport) jobs_.transport->abort_chunk();
    }

    OrtAsrJobScope(const OrtAsrJobScope&) = delete;
    OrtAsrJobScope& operator=(const OrtAsrJobScope&) = delete;

    static OrtAsrJobScope& current() {
        if (!current_) throw std::logic_error("fused node outside its ORT ASR transaction");
        return *current_;
    }

    bool prepare(std::uint16_t call_id) {
        if (seen_.test(call_id)) throw std::runtime_error("duplicate fused ASR node in one encoder run");
        seen_.set(call_id);
        return jobs_.enabled.test(call_id);
    }

    bool paired(std::uint16_t call_id) const {
        return jobs_.paired_ffn.test(call_id);
    }

    bool trace_jobs() const { return bool(jobs_.on_job); }

    pl::AsrJobResult submit(const pl::AsrJobRequest& request) {
        auto result = jobs_.transport->submit(request);
        if (result.output.size() != request.output_bytes || result.working_history.size() != request.history_bytes)
            throw std::runtime_error("ASR job output size does not match its request");
        verify_receipt(request, request.operation, result.receipt, 0);
        return result;
    }

    pl::AsrJobReceipt submit_into(const pl::AsrJobRequest& request, void* output, void* history) {
        auto receipt = jobs_.transport->submit_into(request, output, history);
        verify_receipt(request, request.operation, receipt, 0);
        return receipt;
    }

    pl::AsrPairedFfnReceipt submit_paired_into(const pl::AsrJobRequest& request, void* output) {
        if (!paired(request.call_id) || request.operation != pl::AsrOperation::ffn)
            throw std::logic_error("ASR call is not a paired FFN");
        auto result = jobs_.transport->submit_paired_ffn_into(request, output);
        constexpr std::array<pl::AsrOperation, 2> first{
            pl::AsrOperation::ffn_hidden_left, pl::AsrOperation::ffn_hidden_right};
        constexpr std::array<pl::AsrOperation, 2> second{
            pl::AsrOperation::ffn_output_left, pl::AsrOperation::ffn_output_right};
        for (unsigned lane = 0; lane < 2; ++lane) {
            verify_receipt(request, first[lane], result.first[lane], lane);
            verify_receipt(request, second[lane], result.second[lane], lane);
            if (result.first[lane].irq_events != 1 || result.second[lane].irq_events != 1)
                throw std::runtime_error("paired FFN shard did not complete with exactly one IRQ");
        }
        return result;
    }

    void record(std::uint16_t call_id, std::uint64_t elapsed_ns, std::optional<pl::AsrJobReceipt> receipt,
                std::uint64_t marshal_ns = 0, std::uint64_t submit_ns = 0, std::uint64_t output_ns = 0,
                std::optional<pl::AsrPairedFfnReceipt> pair = {}, const W8Profile* w8_profile = nullptr) {
        const auto& call = fz_asr_calls[call_id];
        ++(receipt || pair ? trace_.pl_calls : trace_.ps_calls)[call.operation];
        if (pair) {
            ++trace_.paired_ffn_calls;
            trace_.physical_pl_calls[0] += 2;
            trace_.physical_pl_calls[1] += 2;
        } else if (receipt) {
            ++trace_.physical_pl_calls[0];
        }
        if (jobs_.on_job) {
            const auto start = OrtJobClock::now();
            jobs_.on_job({stamp_, call_id, static_cast<pl::AsrOperation>(call.operation), call.rows, elapsed_ns,
                          std::move(receipt), marshal_ns, submit_ns, output_ns, std::move(pair), w8_profile});
            trace_.job_trace_ns += job_elapsed_ns(start);
        }
    }

    void end_encoder(std::uint64_t elapsed_ns) {
        if (!seen_.all()) throw std::runtime_error("encoder did not execute all 51 fused ASR nodes");
        trace_.encoder_ns = elapsed_ns;
    }

    void end_decoder(std::uint64_t elapsed_ns) { trace_.decoder_joiner_ns = elapsed_ns; }

    void commit() {
        const auto start = OrtJobClock::now();
        if (jobs_.transport) jobs_.transport->commit_chunk();
        committed_ = true;
        trace_.commit_ns = job_elapsed_ns(start);
    }

    void report() const noexcept {
        if (!jobs_.on_chunk) return;
        try {
            jobs_.on_chunk(trace_);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "ASR committed chunk trace failed: %s\n", error.what());
        } catch (...) {
            std::fputs("ASR committed chunk trace failed\n", stderr);
        }
    }

private:
    void verify_receipt(const pl::AsrJobRequest& request, pl::AsrOperation operation,
                        const pl::AsrJobReceipt& receipt, unsigned lane) {
        auto& previous = previous_[lane];
        if (receipt.call_id != request.call_id || receipt.operation != operation || receipt.rows != request.rows ||
            !receipt.generation || !receipt.epoch || !receipt.chain_id || !receipt.sequence ||
            receipt.flags != pl::asr_job_receipt_all || !receipt.irq_events ||
            (previous && (receipt.generation != previous->generation || receipt.epoch != previous->epoch ||
                          receipt.chain_id != previous->chain_id + 1 || receipt.sequence != previous->sequence + 1))) {
            throw std::runtime_error("ASR job completion does not match its request or lease");
        }
        previous = receipt;
    }

    inline static thread_local OrtAsrJobScope* current_ = nullptr;
    const OrtAsrJobs& jobs_;
    const ChunkStamp& stamp_;
    OrtAsrChunkTrace trace_;
    std::bitset<ort_asr_job_calls> seen_;
    std::array<std::optional<pl::AsrJobReceipt>, 2> previous_;
    bool committed_ = false;
};

}  // namespace edge_speech::asr

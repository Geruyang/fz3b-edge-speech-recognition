#pragma once

#include "board/asr/ort_asr_jobs.hpp"

#include <stdexcept>
#include <string_view>

namespace edge_speech::asr {

struct AuditedAsrJob {
    pl::AsrJobReceipt receipt;
    std::string input_sha256, output_sha256, old_sha256, working_sha256;
    std::size_t input_bytes, output_bytes, history_bytes;
    std::vector<std::uint8_t> embedding_input, embedding_output;
    std::optional<pl::AsrPairedFfnReceipt> paired_ffn;
};

// Hash only selected live payloads, before submit and after the driver has
// delivered its validated receipt. The recorder publishes at chunk commit.
class OrtAsrJobAudit final : public pl::AsrJobTransport {
public:
    using Digest = std::function<std::string(std::string_view)>;

    OrtAsrJobAudit(std::unique_ptr<pl::AsrJobTransport> transport, std::bitset<ort_asr_job_calls> selected,
                   bool save_embedding, Digest digest)
        : transport_(std::move(transport)), selected_(selected), save_embedding_(save_embedding), digest_(std::move(digest)) {
        if (!transport_ || (selected_.any() && !digest_) || (save_embedding_ && !selected_.test(0)))
            throw std::invalid_argument("invalid ASR job evidence selection");
    }

    void begin_chunk(bool reset) override {
        clear();
        transport_->begin_chunk(reset);
    }

    pl::AsrJobResult submit(const pl::AsrJobRequest& request) override {
        if (!selected_.test(request.call_id)) return transport_->submit(request);
        const auto start = OrtJobClock::now();
        auto evidence = capture_input(request);
        const auto input_ns = job_elapsed_ns(start);
        auto result = transport_->submit(request);
        if (result.output.size() != request.output_bytes || result.working_history.size() != request.history_bytes)
            throw std::runtime_error("invalid ASR job evidence size");
        capture_output(std::move(evidence), request, result.receipt, result.output.data(),
                       result.working_history.data(), input_ns);
        return result;
    }

    pl::AsrJobReceipt submit_into(const pl::AsrJobRequest& request, void* output, void* history) override {
        if (!selected_.test(request.call_id)) return transport_->submit_into(request, output, history);
        const auto start = OrtJobClock::now();
        auto evidence = capture_input(request);
        const auto input_ns = job_elapsed_ns(start);
        auto receipt = transport_->submit_into(request, output, history);
        capture_output(std::move(evidence), request, receipt, output, history, input_ns);
        return receipt;
    }

    bool supports_paired_ffn(std::uint16_t call_id) const noexcept override {
        return transport_->supports_paired_ffn(call_id);
    }

    pl::AsrPairedFfnReceipt submit_paired_ffn_into(const pl::AsrJobRequest& request, void* output) override {
        if (!selected_.test(request.call_id)) return transport_->submit_paired_ffn_into(request, output);
        const auto start = OrtJobClock::now();
        auto evidence = capture_input(request);
        const auto input_ns = job_elapsed_ns(start);
        auto pair = transport_->submit_paired_ffn_into(request, output);
        const auto after = OrtJobClock::now();
        capture_payload(evidence, request, output, nullptr);
        pair.audit_ns = input_ns + job_elapsed_ns(after);
        evidence.paired_ffn = pair;
        jobs_.push_back(std::move(evidence));
        return pair;
    }

    void observe(const OrtAsrJobTrace& trace) {
        if (!selected_.test(trace.call_id)) return;
        if (observed_.test(trace.call_id) || jobs_.size() != observed_.count() + 1 ||
            (stamp_ && !(*stamp_ == trace.stamp))) {
            throw std::runtime_error("ASR job evidence has no matching ORT call");
        }
        const auto& staged = jobs_.back();
        bool matched = false;
        if (trace.paired_ffn && staged.paired_ffn && !trace.receipt && trace.operation == pl::AsrOperation::ffn) {
            matched = true;
            for (unsigned lane = 0; lane < 2; ++lane) {
                const auto& first = trace.paired_ffn->first[lane];
                const auto& second = trace.paired_ffn->second[lane];
                matched = matched && first.call_id == trace.call_id && second.call_id == trace.call_id &&
                    first.rows == trace.rows && second.rows == trace.rows &&
                    same_receipt(staged.paired_ffn->first[lane], first) &&
                    same_receipt(staged.paired_ffn->second[lane], second);
            }
        } else if (trace.receipt && !trace.paired_ffn && !staged.paired_ffn) {
            matched = trace.receipt->call_id == trace.call_id && trace.receipt->operation == trace.operation &&
                trace.receipt->rows == trace.rows && same_receipt(staged.receipt, *trace.receipt);
        }
        if (!matched) {
            throw std::runtime_error("ASR job evidence receipt does not match the ORT call");
        }
        stamp_ = trace.stamp;
        observed_.set(trace.call_id);
    }

    void commit_chunk() override {
        if (observed_ != selected_) throw std::runtime_error("selected ASR job evidence is incomplete");
        transport_->commit_chunk();
        committed_ = true;
    }

    std::vector<AuditedAsrJob> take_committed(const ChunkStamp& stamp) {
        if (!committed_ || (selected_.any() && (!stamp_ || !(*stamp_ == stamp))))
            throw std::runtime_error("ASR job evidence does not belong to the committed chunk");
        auto result = std::move(jobs_);
        clear();
        return result;
    }

    void abort_chunk() noexcept override {
        transport_->abort_chunk();
        clear();
    }

    void cancel() noexcept override { transport_->cancel(); }

private:
    static bool same_receipt(const pl::AsrJobReceipt& staged, const pl::AsrJobReceipt& actual) {
        return staged.call_id == actual.call_id && staged.operation == actual.operation && staged.rows == actual.rows &&
            staged.generation == actual.generation && staged.epoch == actual.epoch &&
            staged.chain_id == actual.chain_id && staged.sequence == actual.sequence;
    }

    AuditedAsrJob capture_input(const pl::AsrJobRequest& request) const {
        if (observed_.test(request.call_id) || jobs_.size() != observed_.count())
            throw std::logic_error("unpaired ASR job evidence submission");
        AuditedAsrJob evidence{};
        evidence.input_bytes = request.input_bytes;
        evidence.output_bytes = request.output_bytes;
        evidence.history_bytes = request.history_bytes;
        evidence.input_sha256 = digest(request.input, request.input_bytes);
        evidence.old_sha256 = digest(request.old_history, request.history_bytes);
        if (save_embedding_ && request.call_id == 0) {
            const auto* bytes = static_cast<const std::uint8_t*>(request.input);
            evidence.embedding_input.assign(bytes, bytes + request.input_bytes);
        }
        return evidence;
    }

    void capture_output(AuditedAsrJob evidence, const pl::AsrJobRequest& request, pl::AsrJobReceipt& receipt,
                        const void* output, const void* history, std::uint64_t input_ns) {
        const auto start = OrtJobClock::now();
        if (receipt.call_id != request.call_id || receipt.operation != request.operation ||
            receipt.rows != request.rows || receipt.flags != pl::asr_job_receipt_all)
            throw std::runtime_error("invalid ASR job evidence result");
        capture_payload(evidence, request, output, history);
        receipt.audit_ns = input_ns + job_elapsed_ns(start);
        evidence.receipt = receipt;
        jobs_.push_back(std::move(evidence));
    }

    void capture_payload(AuditedAsrJob& evidence, const pl::AsrJobRequest& request,
                         const void* output, const void* history) {
        evidence.output_sha256 = digest(output, request.output_bytes);
        evidence.working_sha256 = digest(history, request.history_bytes);
        if (save_embedding_ && request.call_id == 0) {
            const auto* bytes = static_cast<const std::uint8_t*>(output);
            evidence.embedding_output.assign(bytes, bytes + request.output_bytes);
        }
    }

    std::string digest(const void* bytes, std::size_t size) const {
        if (!size) return {};
        if (!bytes) throw std::invalid_argument("missing ASR job payload for audit");
        return digest_(std::string_view(static_cast<const char*>(bytes), size));
    }

    void clear() noexcept {
        jobs_.clear();
        stamp_.reset();
        observed_.reset();
        committed_ = false;
    }

    std::unique_ptr<pl::AsrJobTransport> transport_;
    std::bitset<ort_asr_job_calls> selected_, observed_;
    bool save_embedding_, committed_ = false;
    Digest digest_;
    std::optional<ChunkStamp> stamp_;
    std::vector<AuditedAsrJob> jobs_;
};

}  // namespace edge_speech::asr

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(__MINGW32__) && !defined(_stdcall)
#define _stdcall __stdcall
#endif
#endif

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include "hardware/hls/asr_blocks.hpp"
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
#include "board/asr/ort_asr_jobs.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace edge_speech::asr {
namespace {

using namespace edge_speech::pl;
constexpr char domain_name[] = "com.edge_speech.t015";
constexpr std::int64_t ffn_weight_words = 49152;
constexpr std::int64_t conv_weight_words = 25600;
constexpr std::int64_t embedding_parameters = 8 + 32 + 128 + 256;

struct Workspace {
    std::array<float, asr_scratch_elements> scratch;
    std::array<int8_t, asr_scratch_elements> codes;
    std::array<float, asr_input_elements> input, output;
    std::array<float, asr_cache_elements> history, new_history;
};

void to_time_major(const float* source, float* result, unsigned time) {
    for (unsigned row = 0; row < time; ++row)
        for (unsigned channel = 0; channel < asr_channels; ++channel)
            result[row * asr_channels + channel] = source[channel * time + row];
}

void to_channel_major(const float* source, float* result, unsigned time) {
    for (unsigned channel = 0; channel < asr_channels; ++channel)
        for (unsigned row = 0; row < time; ++row)
            result[channel * time + row] = source[row * asr_channels + channel];
}

struct Kernel {
    const OrtApi& api;
    AsrBlock block;
    unsigned rows;
    std::int64_t id;
    bool unified;

    OrtStatusPtr ComputeV2(OrtKernelContext* location) noexcept {
        try {
            compute(location);
            return nullptr;
        } catch (const Ort::Exception& error) {
            return api.CreateStatus(error.GetOrtErrorCode(), error.what());
        } catch (const std::exception& error) {
            return api.CreateStatus(ORT_FAIL, error.what());
        } catch (...) {
            return api.CreateStatus(ORT_FAIL, "unexpected fused ASR failure");
        }
    }

    void compute(OrtKernelContext* location) const {
        Ort::KernelContext context(location);
        const auto input = context.GetInput(0);
        const auto cache = context.GetInput(1);
        const auto weights = context.GetInput(2);
        const auto scales = context.GetInput(3);
        const auto bias = context.GetInput(4);
        const bool conv = block == AsrBlock::convolution;
        const bool embedding = block == AsrBlock::embedding;
        const std::vector<std::int64_t> shape = conv ? std::vector<std::int64_t>{1, asr_channels, rows}
            : embedding ? std::vector<std::int64_t>{1, 39, 80} : std::vector<std::int64_t>{rows, 1, asr_channels};
        const auto out_shape = embedding ? std::vector<std::int64_t>{1, rows, asr_channels} : shape;
        const std::vector<std::int64_t> cache_shape = conv ? std::vector<std::int64_t>{1, asr_channels, asr_history}
                                                           : std::vector<std::int64_t>{1};
        const std::vector<std::int64_t> weight_shape{conv ? conv_weight_words : embedding ? asr_weight_words : ffn_weight_words};
        const std::vector<std::int64_t> parameters{embedding ? embedding_parameters : asr_parameter_elements};
        if (input.GetTensorTypeAndShapeInfo().GetShape() != shape ||
            cache.GetTensorTypeAndShapeInfo().GetShape() != cache_shape ||
            weights.GetTensorTypeAndShapeInfo().GetShape() != weight_shape ||
            scales.GetTensorTypeAndShapeInfo().GetShape() != parameters ||
            bias.GetTensorTypeAndShapeInfo().GetShape() != parameters) {
            throw std::invalid_argument("invalid fused ASR tensor ABI for block " + std::to_string(id));
        }

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        auto* chunk = unified ? &OrtAsrJobScope::current() : nullptr;
        const auto started = OrtJobClock::now();
        auto output_started = started;
        std::uint64_t marshal_ns = 0, submit_ns = 0;
        std::optional<AsrJobReceipt> receipt;
        std::optional<AsrPairedFfnReceipt> pair;
        const W8Profile* cpu_detail = nullptr;
#if defined(EDGE_SPEECH_W8_PROFILE)
        W8Profile w8_measurements{};
#endif
#endif
        thread_local Workspace work;
        const float* x = input.GetTensorData<float>();
        if (conv) {
            to_time_major(x, work.input.data(), rows);
            to_time_major(cache.GetTensorData<float>(), work.history.data(), asr_history);
            x = work.input.data();
        }
        const auto output_values = rows * asr_channels;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (chunk && chunk->prepare(static_cast<std::uint16_t>(id))) {
            const auto input_values = embedding ? 39 * 80 : output_values;
            const auto history_bytes = conv ? asr_cache_elements * sizeof(float) : 0;
            const auto submit_started = OrtJobClock::now();
            const AsrJobRequest request{static_cast<AsrOperation>(block), rows, static_cast<std::uint16_t>(id),
                x, input_values * sizeof(float), output_values * sizeof(float),
                conv ? work.history.data() : nullptr, history_bytes};
            if (chunk->paired(request.call_id))
                pair = chunk->submit_paired_into(request, work.output.data());
            else
                receipt = chunk->submit_into(request, work.output.data(), conv ? work.new_history.data() : nullptr);
            output_started = OrtJobClock::now();
            using Ns = std::chrono::nanoseconds;
            marshal_ns = std::chrono::duration_cast<Ns>(submit_started - started).count();
            submit_ns = std::chrono::duration_cast<Ns>(output_started - submit_started).count();
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!std::all_of(work.output.begin(), work.output.begin() + output_values, finite) ||
                (conv && !std::all_of(work.new_history.begin(), work.new_history.end(), finite))) {
                throw std::runtime_error("nonfinite PL ASR job output");
            }
        } else
#endif
        {
#if defined(EDGE_SPEECH_ENABLE_ASR_JOBS) && defined(EDGE_SPEECH_W8_PROFILE)
            auto* active = chunk && chunk->trace_jobs() ? &w8_measurements : nullptr;
            const W8ProfileScope timing(active);
            const auto w8_start = w8_clock();
#endif
            const auto status = static_cast<AsrStatus>(asr_blocks(static_cast<uint8_t>(block), rows,
                weights.GetTensorData<uint64_t>(), scales.GetTensorData<float>(), bias.GetTensorData<float>(), x,
                conv ? work.history.data() : nullptr, conv ? work.new_history.data() : nullptr,
                work.output.data(), work.scratch.data(), work.codes.data(), false, nullptr, nullptr));
#if defined(EDGE_SPEECH_ENABLE_ASR_JOBS) && defined(EDGE_SPEECH_W8_PROFILE)
            if (active) {
                active->kernel_ns = job_elapsed_ns(w8_start);
                cpu_detail = active;
            }
#endif
            if (status != AsrStatus::ok) {
                throw std::runtime_error("PS ASR block " + std::to_string(id) +
                                         " failed with status " + std::to_string(static_cast<unsigned>(status)));
            }
        }

        auto* y = context.GetOutput(0, out_shape).GetTensorMutableData<float>();
        auto* next_cache = context.GetOutput(1, cache_shape).GetTensorMutableData<float>();
        if (conv) {
            to_channel_major(work.output.data(), y, rows);
            to_channel_major(work.new_history.data(), next_cache, asr_history);
        } else {
            std::copy_n(work.output.data(), output_values, y);
            next_cache[0] = 0.0F;
        }
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (chunk) {
            const auto output_ns = receipt || pair ? job_elapsed_ns(output_started) : 0;
            chunk->record(static_cast<std::uint16_t>(id), job_elapsed_ns(started), std::move(receipt),
                          marshal_ns, submit_ns, output_ns, std::move(pair), cpu_detail);
        }
#endif
    }
};

struct FusedBlock : Ort::CustomOpBase<FusedBlock, Kernel, true> {
    explicit FusedBlock(bool unified) : unified(unified) {}

    const char* GetName() const { return "FusedBlock"; }
    std::size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(std::size_t index) const {
        return index == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
        try {
            const Ort::ConstKernelInfo attrs(info);
            const auto kind = attrs.GetAttribute<std::int64_t>("kind");
            const auto rows = attrs.GetAttribute<std::int64_t>("rows");
            const auto id = attrs.GetAttribute<std::int64_t>("block_id");
            bool valid = (kind == static_cast<std::int64_t>(AsrBlock::ffn) ||
                          kind == static_cast<std::int64_t>(AsrBlock::convolution)) &&
                         (rows == 2 || rows == 4 || rows == 8 || rows == 16) && id >= 1 && id <= 50;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
            if (unified) valid = id >= 0 && id < static_cast<std::int64_t>(ort_asr_job_calls) &&
                fz_asr_calls[id].operation == kind && fz_asr_calls[id].rows == rows;
#endif
            if (!valid) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT, "invalid fused ASR attributes");
            }
            *kernel = new Kernel{api, static_cast<AsrBlock>(kind), static_cast<unsigned>(rows), id, unified};
            return nullptr;
        } catch (const Ort::Exception& error) {
            return api.CreateStatus(error.GetOrtErrorCode(), error.what());
        } catch (const std::exception& error) {
            return api.CreateStatus(ORT_FAIL, error.what());
        } catch (...) {
            return api.CreateStatus(ORT_FAIL, "unexpected fused ASR creation failure");
        }
    }

    bool unified;
};

}  // namespace

void add_ort_pl_custom_ops(Ort::SessionOptions& options, bool unified) {
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    if (unified) {
        static const FusedBlock unified_op(true);
        static const auto unified_domain = [] {
            Ort::CustomOpDomain value(domain_name);
            value.Add(&unified_op);
            return value;
        }();
        options.Add(unified_domain);
        return;
    }
#else
    (void)unified;
#endif
    static const FusedBlock op(false);
    static const auto domain = [] {
        Ort::CustomOpDomain value(domain_name);
        value.Add(&op);
        return value;
    }();
    options.Add(domain);
}

}  // namespace edge_speech::asr

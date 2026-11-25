#pragma once

#include "board/asr/stream_engine.hpp"

#include <filesystem>
#include <functional>
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
#include "board/asr/w8_profile.hpp"
#include "board/pl/asr_job.hpp"

#include <bitset>
#include <optional>
#endif

namespace edge_speech::asr {

struct OrtModelPaths {
    std::string runtime_library;
    std::string encoder;
    std::string decoder;
    std::string joiner;
    std::string tokens;
};

struct OrtCpuChunkTrace {
    const ChunkStamp& stamp;
    std::uint64_t encoder_ns = 0, decoder_joiner_ns = 0;
    std::uint64_t marshal_ns = 0, cache_ns = 0, text_ns = 0;
    std::uint64_t decoder_ns = 0, joiner_ns = 0, greedy_ns = 0;
    std::uint32_t decoder_calls = 0, joiner_calls = 0;
};

struct OrtCpuDiagnostics {
    std::filesystem::path profile_prefix;
    std::function<void(const OrtCpuChunkTrace&)> on_chunk;
    std::array<unsigned, 3> intra_threads{1, 1, 1};
    bool require_int8_encoder = false;
    bool require_int8_decoder = false;
    bool require_int8_joiner = false;
    bool detailed_timing = false;
};

// Paths refer to the hash-checked T002 FP32 release assets and private runtime.
std::shared_ptr<ChunkBackend> make_ort_cpu_backend(const OrtModelPaths& paths, OrtCpuDiagnostics diagnostics = {});
std::array<std::string, 3> finish_ort_profile(ChunkBackend& backend);

inline constexpr std::size_t embedding_frames = 16;
inline constexpr std::size_t embedding_channels = 256;
using OrtEmbeddingFeatures = std::array<float, embedding_frames * embedding_channels>;

struct OrtEmbeddingResult {
    ChunkStamp stamp;
    OrtEmbeddingFeatures features;
};

// The submitter owns the device identity and returns only after the matching
// job has completed and drained, with a CPU-readable result for this stamp.
using OrtEmbeddingSubmit = std::function<OrtEmbeddingResult(const ChunkStamp&, const FeatureWindow&)>;

// paths.encoder is the exported downstream graph with a pl_embedding input.
std::shared_ptr<ChunkBackend> make_ort_pl_backend(const OrtModelPaths& paths, OrtEmbeddingSubmit submit);

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
inline constexpr std::size_t ort_asr_job_calls = 51;

struct OrtAsrJobTrace {
    const ChunkStamp& stamp;
    std::uint16_t call_id;
    pl::AsrOperation operation;
    std::uint32_t rows;
    std::uint64_t elapsed_ns;
    std::optional<pl::AsrJobReceipt> receipt;
    std::uint64_t marshal_ns = 0, submit_ns = 0, output_ns = 0;
    std::optional<pl::AsrPairedFfnReceipt> paired_ffn;
    const W8Profile* w8_profile = nullptr;
};

struct OrtAsrChunkTrace {
    const ChunkStamp& stamp;
    std::uint64_t encoder_ns, decoder_joiner_ns, commit_ns;
    std::array<std::uint32_t, 3> ps_calls{}, pl_calls{};
    std::uint64_t job_trace_ns = 0;
    std::array<std::uint32_t, 2> physical_pl_calls{};
    std::uint32_t paired_ffn_calls = 0;
};

struct OrtAsrJobs {
    std::shared_ptr<pl::AsrJobTransport> transport;
    std::bitset<ort_asr_job_calls> enabled;
    std::function<void(const OrtAsrJobTrace&)> on_job;
    std::function<void(const OrtAsrChunkTrace&)> on_chunk;
    std::bitset<ort_asr_job_calls> paired_ffn;
};

// A fixed selector per backend preserves device and model state across chunks.
// Empty selection runs the identical fused graph with all kernels on the CPU.
std::shared_ptr<ChunkBackend> make_ort_asr_jobs_backend(const OrtModelPaths& paths, OrtAsrJobs jobs,
                                                       OrtCpuDiagnostics diagnostics = {});
#endif

}  // namespace edge_speech::asr

#pragma once

#include "shared/types.hpp"

#include <chrono>
#include <functional>
#include <memory>

namespace edge_speech::asr {

inline constexpr std::size_t window_frames = 39;
inline constexpr std::size_t consume_frames = 32;
inline constexpr std::size_t feature_dim = 80;
using FeatureWindow = std::array<float, window_frames * feature_dim>;

struct ChunkStamp {
    std::string session_id;
    std::uint64_t stream_generation, chunk_id, state_epoch;

    bool operator==(const ChunkStamp& other) const {
        return session_id == other.session_id && stream_generation == other.stream_generation &&
            chunk_id == other.chunk_id && state_epoch == other.state_epoch;
    }
};

struct ChunkState {
    virtual ~ChunkState() = default;
};

struct ChunkRequest {
    ChunkStamp stamp;
    FeatureWindow features;
    std::shared_ptr<const ChunkState> input_state;
};

struct ChunkResult {
    ChunkStamp stamp;
    std::shared_ptr<const ChunkState> state;
    std::string text;
    std::uint64_t trailing_blank_frames;
    Backend backend;
    std::vector<std::int64_t> token_ids{};
    std::uint64_t decoder_steps = 0;
};

class ChunkBackend {
public:
    virtual ~ChunkBackend() = default;
    virtual std::shared_ptr<const ChunkState> initial_state() = 0;
    // Build the next state independently. A failed call leaves input_state reusable.
    // Backends with asynchronous devices must make the job safe to replay before
    // reporting a recoverable failure.
    virtual ChunkResult run(const ChunkRequest&) = 0;
    // Preserve model context and start the next display segment with no tokens.
    // This operation must leave the input state usable if it fails.
    virtual std::shared_ptr<const ChunkState> continue_segment(
        const std::shared_ptr<const ChunkState>&) {
        throw std::logic_error("ASR backend does not support continuous endpoint context");
    }
};

enum class EndpointPolicy { reset_model, preserve_context };

struct StreamConfig {
    std::string version = "zipformer-stream-v1";
    std::uint32_t trailing_silence_ms = 800;
    std::uint32_t max_segment_ms = 20000;
    EndpointPolicy endpoint_policy = EndpointPolicy::preserve_context;
    bool frozen_fbank = false;
};

using FeatureClock = std::chrono::steady_clock;

struct PreparedFeatureWindow {
    FeatureWindow features;
    FeatureClock::time_point started, ready;
};

struct FeaturePreparationTrace {
    ChunkStamp stamp;
    bool prepared_ahead;
    std::uint64_t prepare_ns, queue_ns, overlap_ns, backend_ns, service_ns;
    unsigned queue_peak;
};

// The capture producer lends only PCM to the feature window. The stream owner
// resolves state after consuming it and computes the padded tail at STOP.
class FeatureStaging {
public:
    explicit FeatureStaging(bool frozen);
    ~FeatureStaging();
    void offer(const AudioBlock& block);
    void prepare_ready();
    void discontinuity();
    void reset(const WireUuid& stream, std::uint64_t generation, std::uint64_t origin);
    std::optional<PreparedFeatureWindow> take(std::uint64_t generation, std::uint64_t origin, std::uint64_t epoch);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct StreamPreparation {
    std::shared_ptr<FeatureStaging> staging;
    std::function<void(const FeaturePreparationTrace&)> on_chunk;
};

struct StreamGap {
    std::uint64_t sample_start;
    std::optional<std::uint64_t> sample_end;
    std::string reason;
};

struct FinalizedStream {
    std::uint64_t stop_sample;
    Integrity integrity;
    std::vector<TranscriptEvent> events;
    std::vector<TranscriptEvent> segments;
    std::vector<StreamGap> gaps;
    StreamConfig config;
    // Decoder token IDs in the same order as segments. A backend may omit them.
    std::vector<std::vector<std::int64_t>> segment_tokens{};
    std::uint64_t decoder_steps = 0;
};

// The session's ASR owner serializes all calls, including capture discontinuities.
class StreamEngine {
public:
    StreamEngine(std::string session_id, std::string model_release,
                 std::shared_ptr<ChunkBackend> backend, StreamConfig config = {}, StreamPreparation preparation = {});
    ~StreamEngine();

    // A decoding exception can occur after PCM was accepted. Retry replays the
    // preserved chunk without accepting those samples a second time. enqueue()
    // is available when capture already queues work separately from decoding.
    std::vector<TranscriptEvent> accept(const AudioBlock& block);
    void enqueue(const AudioBlock& block);
    std::vector<TranscriptEvent> retry();
    std::vector<TranscriptEvent> gap(std::uint64_t start, std::optional<std::uint64_t> end,
                                     std::string reason);
    FinalizedStream finish(std::uint64_t stop_sample);

    std::uint64_t accepted_sample() const;
    std::uint64_t committed_sample() const;
    std::uint64_t stream_generation() const;
    std::uint64_t state_epoch() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_speech::asr

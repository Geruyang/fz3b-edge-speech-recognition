#include "board/asr/stream_engine.hpp"

#include "board/asr/fbank.hpp"

#include <cstdio>
#include <mutex>

namespace edge_speech::asr {
namespace {

constexpr std::uint64_t samples_per_ms = 16;
constexpr std::uint64_t max_pending_samples = 800 * samples_per_ms;
constexpr std::uint64_t chunk_samples = consume_frames * samples_per_frame;
constexpr std::uint64_t encoder_frame_ms = 40;

bool valid_pcm(const AudioBlock& block) {
    return block.sample_count && block.sample_count <= 320 && block.audio_flags == 0 &&
        block.pcm_s16le.size() == 2 * block.sample_count;
}

void append_pcm(std::deque<std::int16_t>& target, const AudioBlock& block) {
    std::array<std::int16_t, 320> samples{};
    for (std::size_t i = 0; i < block.sample_count; ++i) {
        const auto word = edge_speech::detail::read_le<std::uint16_t>(&block.pcm_s16le[2 * i]);
        const auto decoded = static_cast<std::int32_t>(word) - (word >= 32768 ? 65536 : 0);
        samples[i] = static_cast<std::int16_t>(decoded);
    }
    target.insert(target.end(), samples.begin(), samples.begin() + block.sample_count);
}

std::uint64_t time_ns(FeatureClock::time_point start, FeatureClock::time_point end) {
    return end > start ? std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() : 0;
}

WireUuid wire_uuid(std::string_view id) {
    if (!is_uuid(id)) {
        throw std::invalid_argument("invalid ASR session UUID");
    }
    const auto nibble = [](char digit) { return digit <= '9' ? digit - '0' : digit - 'a' + 10; };
    WireUuid result{};
    std::size_t index = 0;
    for (std::size_t pos = 0; pos < id.size();) {
        if (id[pos] == '-') {
            ++pos;
        }
        result[index++] = static_cast<std::uint8_t>(16 * nibble(id[pos]) + nibble(id[pos + 1]));
        pos += 2;
    }
    return result;
}

}  // namespace

struct FeatureStaging::Impl {
    explicit Impl(bool frozen) : fbank(frozen) {}

    void retain_from(std::uint64_t sample) {
        while (first < sample && !pcm.empty()) {
            pcm.pop_front();
            ++first;
        }
    }

    void prepare_ready() {
        if (!stream || prepared) return;
        const auto begin = origin + epoch * chunk_samples;
        const auto required = begin + window_frames * samples_per_frame + frame_end;
        const auto left = epoch ? begin - frame_left_context : origin;
        if (first > left || first + pcm.size() < required) return;
        retain_from(left);
        const auto started = FeatureClock::now();
        auto features = fbank.window(pcm, first - origin, epoch * consume_frames, required - origin, false);
        prepared.emplace(PreparedFeatureWindow{std::move(features), started, FeatureClock::now()});
    }

    Fbank fbank;
    std::mutex mutex;
    std::optional<WireUuid> stream;
    std::uint64_t generation = 0, origin = 0, epoch = 0, first = 0;
    std::deque<std::int16_t> pcm;
    std::optional<PreparedFeatureWindow> prepared;
};

FeatureStaging::FeatureStaging(bool frozen) : impl_(std::make_unique<Impl>(frozen)) {}
FeatureStaging::~FeatureStaging() = default;

void FeatureStaging::offer(const AudioBlock& block) {
    auto& s = *impl_;
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.stream || block.session_id != *s.stream || !valid_pcm(block)) return;
    if (block.sample_start != s.first + s.pcm.size()) {
        s.pcm.clear();
        s.prepared.reset();
        s.first = block.sample_start;
    }
    append_pcm(s.pcm, block);
    if (s.pcm.size() > max_pending_samples)
        s.retain_from(s.first + s.pcm.size() - max_pending_samples);
    s.prepare_ready();
}

void FeatureStaging::prepare_ready() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->prepare_ready();
}

void FeatureStaging::discontinuity() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->pcm.clear();
    impl_->prepared.reset();
}

void FeatureStaging::reset(const WireUuid& stream, std::uint64_t generation, std::uint64_t origin) {
    auto& s = *impl_;
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.stream || s.stream != stream) s.pcm.clear();
    s.stream = stream;
    s.generation = generation;
    s.origin = origin;
    s.epoch = 0;
    s.prepared.reset();
    s.retain_from(origin);
}

std::optional<PreparedFeatureWindow> FeatureStaging::take(std::uint64_t generation, std::uint64_t origin,
                                                         std::uint64_t epoch) {
    auto& s = *impl_;
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.generation != generation || s.origin != origin || s.epoch > epoch) return {};
    auto result = s.epoch == epoch ? std::move(s.prepared) : std::nullopt;
    s.prepared.reset();
    s.epoch = epoch + 1;
    return result;
}

struct StreamEngine::Impl {
    Impl(std::string id, std::string release, std::shared_ptr<ChunkBackend> backend, StreamConfig config,
         StreamPreparation preparation)
        : id(std::move(id)), wire_id(wire_uuid(this->id)), release(std::move(release)),
          backend(std::move(backend)), config(std::move(config)), preparation(std::move(preparation)),
          fbank(this->config.frozen_fbank) {
        if (!this->backend || this->release.empty() || this->config.version != "zipformer-stream-v1" ||
            !this->config.trailing_silence_ms || !this->config.max_segment_ms ||
            (this->config.endpoint_policy != EndpointPolicy::reset_model &&
             this->config.endpoint_policy != EndpointPolicy::preserve_context)) {
            throw std::invalid_argument("invalid ASR backend, release, or stream configuration");
        }
        state = new_state();
        if (this->preparation.staging) this->preparation.staging->reset(wire_id, generation, run_origin);
    }

    std::shared_ptr<const ChunkState> new_state() {
        auto initial = backend->initial_state();
        if (!initial) {
            throw std::runtime_error("ASR backend returned an empty initial state");
        }
        return initial;
    }

    void publish(const std::string& text, std::uint64_t end, bool final, Backend source) {
        TranscriptEvent event{next_event, id, segment_id, revision + 1, final,
                              segment_start, end, text, release, source};
        validate_transcript_revision(event, nullptr, next_event - 1, accepted);
        if (final) {
            finals.push_back(event);
            try {
                final_tokens.push_back(tokens);
            } catch (...) {
                finals.pop_back();
                throw;
            }
            try {
                events.push_back(std::move(event));
            } catch (...) {
                final_tokens.pop_back();
                finals.pop_back();
                throw;
            }
        } else {
            events.push_back(std::move(event));
        }
        ++next_event;
        ++revision;
    }

    bool is_endpoint(const std::string& result_text, std::uint64_t blank_frames) const {
        const auto blanks = (std::uint64_t{config.trailing_silence_ms} + encoder_frame_ms - 1) /
            encoder_frame_ms;
        return !result_text.empty() && blank_frames >= blanks;
    }

    void discard_before(std::uint64_t sample) {
        while (pcm_start < sample && !pcm.empty()) {
            pcm.pop_front();
            ++pcm_start;
        }
        pcm_start = sample;
    }

    void finish_display_segment(std::uint64_t end) {
        if (!text.empty()) {
            publish(text, end, true, *segment_backend);
        }
        segment_start = end;
        text.clear();
        tokens.clear();
        segment_backend.reset();
        endpoint_due = false;
        revision = 0;
        ++segment_id;
    }

    void finish_segment(std::uint64_t end, std::uint64_t next_start,
                        const StreamGap* discontinuity = nullptr) {
        auto initial = new_state();
        if (discontinuity) {
            gaps.push_back(*discontinuity);
        }
        try {
            finish_display_segment(end);
        } catch (...) {
            if (discontinuity) {
                gaps.pop_back();
            }
            throw;
        }
        discard_before(next_start);
        run_origin = segment_start = committed = next_start;
        state = std::move(initial);
        pending.reset();
        epoch = 0;
        ++generation;
        if (preparation.staging) preparation.staging->reset(wire_id, generation, run_origin);
    }

    void finish_endpoint() {
        if (config.endpoint_policy == EndpointPolicy::reset_model) {
            finish_segment(committed, committed);
            return;
        }
        auto continuation = backend->continue_segment(state);
        if (!continuation) {
            throw std::runtime_error("ASR backend returned an empty continuation state");
        }
        finish_display_segment(committed);
        state = std::move(continuation);
    }

    void decode(std::uint64_t limit, bool draining, bool publish_partial) {
        const auto first_frame = epoch * consume_frames;
        if (!pending) {
            const auto started = FeatureClock::now();
            auto prepared = preparation.staging && !draining
                ? preparation.staging->take(generation, run_origin, epoch) : std::nullopt;
            auto features = prepared ? std::move(prepared->features)
                : fbank.window(pcm, pcm_start - run_origin, first_frame, limit - run_origin, draining);
            const auto ready = FeatureClock::now();
            pending.emplace(ChunkRequest{{id, generation, epoch + 1, epoch}, std::move(features), state});
            preparation_begin = prepared ? prepared->started : started;
            preparation_end = prepared ? prepared->ready : ready;
            prepared_ahead = prepared.has_value();
            if (prepared_ahead) queue_peak = 1;
        }
        const auto execution_begin = FeatureClock::now();
        auto result = backend->run(*pending);
        const auto execution_end = FeatureClock::now();
        if (!(result.stamp == pending->stamp) || !result.state ||
            (result.backend != Backend::cpu && result.backend != Backend::pl && result.backend != Backend::mixed)) {
            throw std::runtime_error("invalid or stale ASR chunk completion");
        }
        const auto end = run_origin + std::min((epoch + 1) * chunk_samples, limit - run_origin);
        const auto source = segment_backend && *segment_backend != result.backend ? Backend::mixed : result.backend;
        const bool endpoint = !draining && is_endpoint(result.text, result.trailing_blank_frames);
        if (publish_partial && result.text != text && !endpoint) {
            publish(result.text, end, false, source);
        }
        // All backend stages and the tentative event succeeded. Move the new
        // cache into the committed slot before releasing the replay snapshot.
        state = std::move(result.state);
        text = std::move(result.text);
        tokens = std::move(result.token_ids);
        decoder_steps += result.decoder_steps;
        endpoint_due = endpoint;
        segment_backend = source;
        committed = end;
        ++epoch;
        pending.reset();
        const auto next_frame_first = epoch * chunk_samples;
        discard_before(run_origin + std::min(next_frame_first - frame_left_context, limit - run_origin));
        if (preparation.on_chunk) {
            const auto overlap_ns = prepared_ahead
                ? time_ns(std::max(preparation_begin, previous_execution_begin),
                          std::min(preparation_end, previous_execution_end)) : 0;
            try {
                preparation.on_chunk({result.stamp, prepared_ahead, time_ns(preparation_begin, preparation_end),
                    prepared_ahead ? time_ns(preparation_end, execution_begin) : 0,
                    overlap_ns, time_ns(execution_begin, execution_end),
                    time_ns(preparation_begin, FeatureClock::now()), queue_peak});
            } catch (...) {
                std::fputs("ASR feature preparation trace failed\n", stderr);
            }
        }
        previous_execution_begin = execution_begin;
        previous_execution_end = execution_end;
    }

    bool live_ready(std::uint64_t limit) const {
        // One extra frame preserves the locked sherpa IsReady boundary.
        return epoch * chunk_samples + window_frames * samples_per_frame + frame_end <= limit - run_origin;
    }

    void flush(std::uint64_t limit) {
        if (limit == segment_start) {
            return;
        }
        const auto padded_frames = (limit - run_origin + tail_silence_samples +
                                    samples_per_frame / 2) / samples_per_frame;
        while (epoch * consume_frames + window_frames < padded_frames) {
            decode(limit, true, false);
        }
    }

    void pump() {
        while (true) {
            const auto remaining = accepted - segment_start;
            const auto max_samples = std::uint64_t{config.max_segment_ms} * samples_per_ms;
            const auto limit = segment_start + std::min(remaining, max_samples);
            if (endpoint_due && committed - segment_start < max_samples) {
                finish_endpoint();
            } else if (live_ready(limit)) {
                decode(limit, false, !stop_requested && !requested_gap);
            } else if (remaining >= max_samples) {
                flush(limit);
                finish_segment(limit, limit);
            } else {
                return;
            }
        }
    }

    const std::string id;
    const WireUuid wire_id;
    const std::string release;
    const std::shared_ptr<ChunkBackend> backend;
    const StreamConfig config;
    const StreamPreparation preparation;
    Fbank fbank;
    std::shared_ptr<const ChunkState> state;
    std::optional<ChunkRequest> pending;
    FeatureClock::time_point preparation_begin{}, preparation_end{}, previous_execution_begin{}, previous_execution_end{};
    unsigned queue_peak = 0;
    std::deque<std::int16_t> pcm;
    // The frame grid and replay epoch only reset at a hard model boundary.
    std::uint64_t run_origin = 0, pcm_start = 0, accepted = 0, committed = 0, segment_start = 0;
    std::uint64_t generation = 1, epoch = 0, segment_id = 1, revision = 0, next_event = 1;
    std::uint64_t decoder_steps = 0;
    std::string text;
    std::vector<std::int64_t> tokens;
    std::optional<Backend> segment_backend;
    std::vector<TranscriptEvent> events, finals;
    std::vector<std::vector<std::int64_t>> final_tokens;
    std::vector<StreamGap> gaps;
    std::optional<StreamGap> requested_gap;
    bool endpoint_due = false, stop_requested = false, stopped = false, prepared_ahead = false;
};

StreamEngine::StreamEngine(std::string session_id, std::string model_release,
                           std::shared_ptr<ChunkBackend> backend, StreamConfig config, StreamPreparation preparation)
    : impl_(std::make_unique<Impl>(std::move(session_id), std::move(model_release),
                                  std::move(backend), std::move(config), std::move(preparation))) {}

StreamEngine::~StreamEngine() = default;

void StreamEngine::enqueue(const AudioBlock& block) {
    auto& s = *impl_;
    if (s.stop_requested || s.requested_gap) {
        throw std::logic_error("ASR stream is closing or recovering a gap");
    }
    if (block.session_id != s.wire_id || block.sample_start != s.accepted ||
        !valid_pcm(block)) {
        throw std::invalid_argument("invalid or noncontiguous ASR audio block");
    }
    const auto end = checked_extent(block.sample_start, block.sample_count);
    if (end - s.committed > max_pending_samples) {
        const StreamGap overrun{s.committed, end, "asr_overrun"};
        s.finish_segment(s.committed, end, &overrun);
    } else {
        append_pcm(s.pcm, block);
    }
    s.accepted = end;
}

std::vector<TranscriptEvent> StreamEngine::accept(const AudioBlock& block) {
    enqueue(block);
    return retry();
}

std::vector<TranscriptEvent> StreamEngine::retry() {
    auto& s = *impl_;
    if (s.stop_requested) {
        throw std::logic_error("retry a pending stop through finish()");
    }
    s.pump();
    if (s.requested_gap) {
        const auto& gap = *s.requested_gap;
        const auto end = gap.sample_end.value_or(gap.sample_start);
        s.flush(gap.sample_start);
        s.finish_segment(gap.sample_start, end, &gap);
        s.accepted = end;
        s.requested_gap.reset();
    }
    return std::exchange(s.events, {});
}

std::vector<TranscriptEvent> StreamEngine::gap(std::uint64_t start, std::optional<std::uint64_t> end,
                                                std::string reason) {
    auto& s = *impl_;
    if (s.stop_requested) {
        throw std::logic_error("ASR stream is closing");
    }
    if (start != s.accepted || (end && *end < start) || reason.empty()) {
        throw std::invalid_argument("invalid ASR gap");
    }
    if (s.requested_gap) {
        if (s.requested_gap->sample_end != end || s.requested_gap->reason != reason) {
            throw std::logic_error("finish recovering the pending ASR gap first");
        }
    } else {
        s.requested_gap.emplace(StreamGap{start, end, std::move(reason)});
    }
    return retry();
}

FinalizedStream StreamEngine::finish(std::uint64_t stop_sample) {
    auto& s = *impl_;
    if (stop_sample != s.accepted || s.requested_gap) {
        throw std::invalid_argument("stop boundary does not match the accepted ASR input");
    }
    s.stop_requested = true;
    if (!s.stopped) {
        s.pump();
        s.flush(stop_sample);
        if (!s.text.empty()) {
            s.publish(s.text, stop_sample, true, *s.segment_backend);
        }
        s.stopped = true;
    }
    FinalizedStream result{stop_sample, s.gaps.empty() ? Integrity::complete : Integrity::with_gaps,
                           {}, s.finals, s.gaps, s.config, s.final_tokens, s.decoder_steps};
    result.events = std::exchange(s.events, {});
    return result;
}

std::uint64_t StreamEngine::accepted_sample() const {
    return impl_->accepted;
}

std::uint64_t StreamEngine::committed_sample() const {
    return impl_->committed;
}

std::uint64_t StreamEngine::stream_generation() const {
    return impl_->generation;
}

std::uint64_t StreamEngine::state_epoch() const {
    return impl_->epoch;
}

}  // namespace edge_speech::asr

#include "board/session/coordinator.hpp"
#include "board/audio/capture_spool.hpp"

#include <ctime>
#include <istream>
#include <map>
#include <mutex>
#include <random>
#include <tuple>

namespace edge_speech::session {
namespace {

constexpr auto session_limit = std::chrono::minutes(60);
constexpr std::uint64_t sample_limit = 16000 * 60 * 60;
constexpr char hex[] = "0123456789abcdef";

std::string uuid() {
    std::random_device entropy;
    std::uniform_int_distribution<unsigned> byte(0, 255);
    WireUuid data;
    for (auto& octet : data) octet = static_cast<std::uint8_t>(byte(entropy));
    data[6] = (data[6] & 0x0f) | 0x40;
    data[8] = (data[8] & 0x3f) | 0x80;

    std::string result;
    result.reserve(36);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result += '-';
        result += hex[data[i] >> 4];
        result += hex[data[i] & 15];
    }
    return result;
}

std::string utc_now() {
    const auto now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    const auto valid = gmtime_s(&utc, &now) == 0;
#else
    const auto valid = gmtime_r(&now, &utc) != nullptr;
#endif
    char result[32];
    if (!valid || !std::strftime(result, sizeof(result), "%Y-%m-%dT%H:%M:%SZ", &utc)) {
        throw std::runtime_error("could not format snapshot creation time");
    }
    return result;
}

void append_u64(Bytes& bytes, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_text(Bytes& bytes, std::string_view value) {
    append_u64(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t rotate_right(std::uint32_t word, unsigned count) {
    return (word >> count) | (word << (32 - count));
}

// FIPS 180-4. Keeping the digest local avoids a crypto runtime dependency on the board.
class Sha256 {
public:
    void update(const std::uint8_t* bytes, std::size_t size) {
        if (size > (UINT64_MAX - bit_length_) / 8) {
            throw std::invalid_argument("SHA-256 input exceeds the 64-bit bit length");
        }
        bit_length_ += std::uint64_t{size} * 8;
        while (size != 0) {
            const auto count = std::min(size, block_.size() - filled_);
            std::copy_n(bytes, count, block_.data() + filled_);
            filled_ += count;
            if (filled_ == block_.size()) {
                compress();
                filled_ = 0;
            }
            bytes += count;
            size -= count;
        }
    }

    std::string finish() {
        block_[filled_++] = 0x80;
        std::fill(block_.begin() + filled_, block_.end(), 0);
        if (filled_ > 56) {
            compress();
            block_.fill(0);
        }
        for (unsigned i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<std::uint8_t>(bit_length_ >> (i * 8));
        }
        compress();

        std::string result;
        result.reserve(64);
        for (const auto word : digest_) {
            for (int shift = 28; shift >= 0; shift -= 4) result += hex[(word >> shift) & 15];
        }
        return result;
    }

private:
    void compress() {
        constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            for (std::size_t octet = 0; octet < 4; ++octet) {
                words[i] = (words[i] << 8) | block_[i * 4 + octet];
            }
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const auto u = words[i - 15], v = words[i - 2];
            const auto lower = rotate_right(u, 7) ^ rotate_right(u, 18) ^ (u >> 3);
            const auto upper = rotate_right(v, 17) ^ rotate_right(v, 19) ^ (v >> 10);
            words[i] = words[i - 16] + lower + words[i - 7] + upper;
        }
        auto [a, b, c, d, e, f, g, h] = digest_;
        for (std::size_t i = 0; i < 64; ++i) {
            const auto upper = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto t1 = h + upper + choose + constants[i] + words[i];
            const auto lower = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = lower + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        const std::array<std::uint32_t, 8> state{a, b, c, d, e, f, g, h};
        for (std::size_t i = 0; i < state.size(); ++i) digest_[i] += state[i];
    }

    std::array<std::uint32_t, 8> digest_{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block_{};
    std::size_t filled_ = 0;
    std::uint64_t bit_length_ = 0;
};

SessionSnapshot frozen_snapshot(const Status& status, asr::FinalizedStream& stream) {
    std::sort(stream.segments.begin(), stream.segments.end(), [](const auto& left, const auto& right) {
        return std::tie(left.sample_start, left.segment_id) < std::tie(right.sample_start, right.segment_id);
    });
    SessionSnapshot snapshot{uuid(), status.session_id, 1, {}, stream.stop_sample, status.high_watermark,
                             {}, utc_now(), true, stream.integrity};

    // v1 canonical source: big-endian u64 integers, length-prefixed UTF-8 text,
    // and segments in (sample_start, segment_id) order. UUID and UTC time of
    // snapshot creation are envelope fields outside this source digest.
    Bytes content;
    append_text(content, "edge-speech/raw-snapshot/v1");
    append_text(content, snapshot.session_id);
    append_u64(content, stream.stop_sample);
    append_u64(content, snapshot.high_watermark);
    append_u64(content, static_cast<unsigned>(stream.integrity));
    append_text(content, stream.config.version);
    append_u64(content, stream.config.trailing_silence_ms);
    append_u64(content, stream.config.max_segment_ms);
    append_u64(content, stream.gaps.size());
    for (const auto& gap : stream.gaps) {
        append_u64(content, gap.sample_start);
        append_u64(content, gap.sample_end.value_or(UINT64_MAX));
        append_text(content, gap.reason);
    }
    append_u64(content, stream.segments.size());
    for (const auto& segment : stream.segments) {
        if (!segment.is_final || segment.session_id != status.session_id ||
            segment.event_id > snapshot.high_watermark || segment.sample_end > stream.stop_sample) {
            throw std::logic_error("uncommitted transcript segment in frozen source");
        }
        snapshot.ordered_segment_revisions.push_back({segment.segment_id, segment.revision});
        append_u64(content, segment.segment_id);
        append_u64(content, segment.revision);
        append_u64(content, segment.event_id);
        append_u64(content, segment.sample_start);
        append_u64(content, segment.sample_end);
        append_text(content, segment.model_release);
        append_u64(content, static_cast<unsigned>(segment.backend));
        append_text(content, segment.text);
    }
    Sha256 digest;
    digest.update(content.data(), content.size());
    snapshot.content_sha256 = digest.finish();
    return snapshot;
}

const char* gap_reason(audio::CaptureGapReason reason) {
    switch (reason) {
    case audio::CaptureGapReason::device_xrun: return "device_xrun";
    case audio::CaptureGapReason::disconnect: return "disconnect";
    case audio::CaptureGapReason::queue_overrun: return "queue_overrun";
    case audio::CaptureGapReason::timeout: return "other";
    }
    return "other";
}

}  // namespace

std::string sha256(std::string_view bytes) {
    Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    return digest.finish();
}

std::string sha256(std::istream& input) {
    Sha256 digest;
    std::array<char, 65536> bytes;
    do {
        input.read(bytes.data(), bytes.size());
        digest.update(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                      static_cast<std::size_t>(input.gcount()));
    } while (input);
    if (input.bad() || !input.eof()) throw std::runtime_error("cannot read SHA-256 input stream");
    return digest.finish();
}

std::string_view state_name(State state) {
    switch (state) {
    case State::ready: return "ready";
    case State::recording: return "recording";
    case State::draining: return "draining";
    case State::finalized: return "finalized";
    case State::refining: return "refining";
    case State::completed: return "completed";
    case State::faulted: return "faulted";
    }
    throw std::invalid_argument("invalid session state");
}

WireUuid wire_uuid(std::string_view id) {
    if (!is_uuid(id)) throw std::invalid_argument("invalid session UUID");
    WireUuid result;
    std::size_t cursor = 0;
    const auto digit = [](char ch) { return ch <= '9' ? ch - '0' : ch - 'a' + 10; };
    for (auto& octet : result) {
        if (id[cursor] == '-') ++cursor;
        octet = static_cast<std::uint8_t>(digit(id[cursor]) * 16 + digit(id[cursor + 1]));
        cursor += 2;
    }
    return result;
}

struct SessionCoordinator::Impl {
    explicit Impl(Dependencies dependencies) : deps(std::move(dependencies)) {
        if (!deps.make_source || !deps.make_stream || !deps.freeze) {
            throw std::invalid_argument("session source, ASR factory and freeze journal are required");
        }
        if (static_cast<bool>(deps.device.quiesce) != static_cast<bool>(deps.device.switch_mode)) {
            throw std::invalid_argument("device quiesce and mode-switch hooks must be paired");
        }
        if (!deps.now) deps.now = Clock::now;
    }

    StartReceipt receipt(bool ok, std::string code) const {
        return {ok, std::move(code), current.state, current.session_id};
    }

    void publish_status() {
        std::lock_guard<std::mutex> lock(observer);
        observed = current;
        observed_source = source;
        if (current.state != State::recording) stop_pending = false;
    }

    Status capture_status() const {
        auto status = observed;
        if (stop_pending && status.state == State::recording) status.state = State::draining;
        if (observed_source) {
            if (const auto progress = observed_source->capture_progress()) {
                status.capture_end_sample = progress->sample_end;
                if (progress->sealed && status.state != State::recording) status.stop_sample = progress->sample_end;
                if (progress->storage_failed && (status.state == State::recording || status.state == State::draining)) {
                    status.state = State::draining;
                    status.integrity = Integrity::failed;
                    status.error_code = "capture_storage_error";
                    status.stop_sample = progress->sample_end;
                }
            }
        }
        return status;
    }

    bool stop_is_pending() const {
        std::lock_guard<std::mutex> lock(observer);
        return stop_pending;
    }

    void fail(State state, std::string code) {
        if (source) {
            if (const auto progress = source->capture_progress(); progress && progress->sealed) {
                current.stop_sample = progress->sample_end;
            }
        }
        current.state = state;
        current.error_code = std::move(code);
        current.integrity = Integrity::failed;
        publish_status();
    }

    void publish(const std::vector<TranscriptEvent>& events) {
        for (const auto& event : events) {
            if (deps.on_event) deps.on_event(event);
            current.high_watermark = event.event_id;
        }
    }

    // Audits and bounds source events before handing them to audio/ASR sinks.
    // A finite gap advances the capture timeline, an unknown gap keeps its index.
    bool accept_boundary(audio::CaptureEvent& event) {
        auto& end = current.capture_end_sample;
        if (end == sample_limit) return false;
        if (auto* block = std::get_if<AudioBlock>(&event)) {
            if (block->session_id != wire_id || block->sample_start != end ||
                block->sample_count == 0 || block->sample_count > audio::samples_per_block ||
                block->pcm_s16le.size() != 2 * block->sample_count) {
                throw std::invalid_argument("invalid normalized PCM boundary");
            }
            if (block->sample_count > sample_limit - end) {
                block->sample_count = static_cast<std::uint32_t>(sample_limit - end);
                block->pcm_s16le.resize(2 * block->sample_count);
            }
            end += block->sample_count;
        } else {
            auto& gap = std::get<audio::CaptureGap>(event);
            if (gap.sample_start != end || (gap.sample_end && *gap.sample_end <= end)) {
                throw std::invalid_argument("invalid capture gap boundary");
            }
            if (gap.sample_end) {
                gap.sample_end = std::min(*gap.sample_end, sample_limit);
                end = *gap.sample_end;
            }
            current.gaps.push_back({gap.sample_start, gap.sample_end, gap_reason(gap.reason)});
            current.integrity = Integrity::with_gaps;
        }
        return true;
    }

    void record_capture(const audio::CaptureEvent& event) {
        if (const auto* block = std::get_if<AudioBlock>(&event)) {
            if (deps.on_audio) deps.on_audio(*block);
        } else {
            if (deps.on_gap) deps.on_gap(std::get<audio::CaptureGap>(event));
        }
    }

    std::vector<TranscriptEvent> decode(const audio::CaptureEvent& event) {
        std::vector<TranscriptEvent> result;
        if (const auto* block = std::get_if<AudioBlock>(&event)) {
            result = stream->accept(*block);
        } else {
            const auto& gap = std::get<audio::CaptureGap>(event);
            result = stream->gap(gap.sample_start, gap.sample_end, gap_reason(gap.reason));
        }
        current.processed_sample = stream->committed_sample();
        return result;
    }

    void drain() {
        current.state = State::draining;
        publish_status();
        std::string error;
        bool capture_sink_available = true;
        const auto failure_state = source->capture_progress() ? State::faulted : State::draining;
        try {
            source->stop();
            if (const auto progress = source->capture_progress()) {
                current.stop_sample = progress->sample_end;
                if (progress->storage_failed) error = "capture_storage_error";
                publish_status();
            }
            while (auto event = source->read()) {
                if (!accept_boundary(*event)) continue;
                if (capture_sink_available) {
                    try {
                        record_capture(*event);
                    } catch (...) {
                        capture_sink_available = false;
                        error = "storage_error";
                    }
                }
                // Keep advancing the sealed capture watermark and delivering raw
                // audio even if the decoder or transcript journal has failed.
                if (!error.empty()) continue;
                std::vector<TranscriptEvent> events;
                try {
                    events = decode(*event);
                } catch (...) {
                    error = "drain_failed";
                    continue;
                }
                try {
                    publish(events);
                } catch (...) {
                    error = "storage_error";
                }
                publish_status();
            }
            if (current.stop_sample && *current.stop_sample != current.capture_end_sample) {
                throw std::runtime_error("sealed capture watermark does not match the drained source");
            }
            current.stop_sample = current.capture_end_sample;
        } catch (const audio::CaptureStorageError&) {
            fail(failure_state, "capture_storage_error");
            return;
        } catch (...) {
            fail(failure_state, "capture_error");
            return;
        }
        if (!error.empty()) {
            fail(failure_state, std::move(error));
            return;
        }
        publish_status();

        std::optional<asr::FinalizedStream> finalized;
        try {
            finalized = stream->finish(*current.stop_sample);
            current.processed_sample = *current.stop_sample;
        } catch (...) {
            fail(failure_state, "drain_failed");
            return;
        }
        try {
            publish(finalized->events);
            auto snapshot = frozen_snapshot(current, *finalized);
            deps.freeze(snapshot, *finalized);
            current.integrity = finalized->integrity;
            current.gaps = std::move(finalized->gaps);
            current.snapshot = std::move(snapshot);
            current.state = State::finalized;
            publish_status();
        } catch (...) {
            fail(failure_state, "storage_error");
        }
    }

    std::string can_release_device() const {
        if (!deps.device.quiesce) return {};
        try {
            switch (deps.device.quiesce()) {
            case DeviceStatus::idle: return {};
            case DeviceStatus::busy: return "accelerator_busy";
            case DeviceStatus::unknown: break;
            }
        } catch (...) {
            return "unknown_dma";
        }
        return "unknown_dma";
    }

    std::string prepare(DeviceMode mode) {
        if (auto reason = can_release_device(); !reason.empty()) return reason;
        if (deps.device.switch_mode && (!device_mode || *device_mode != mode)) {
            device_mode.reset();
            try {
                deps.device.switch_mode(mode);
            } catch (...) {
                return "accelerator_unavailable";
            }
        }
        device_mode = mode;
        return {};
    }

    Dependencies deps;
    std::mutex owner;
    std::mutex requests;
    mutable std::mutex observer;
    Status current, observed;
    WireUuid wire_id{};
    std::shared_ptr<audio::AudioSource> source, observed_source;
    std::unique_ptr<asr::StreamEngine> stream;
    Clock::time_point deadline;
    std::optional<DeviceMode> device_mode = DeviceMode::asr;
    bool cancel_requested = false;
    bool stop_pending = false;
    std::map<std::string, std::pair<std::string, StartReceipt>> start_receipts;
    std::map<std::string, std::pair<std::string, StopReceipt>> stop_receipts;
    std::map<std::string, std::pair<std::string, StopReceipt>> async_stop_receipts;
    std::map<std::string, CancelReceipt> cancel_receipts;
};

SessionCoordinator::SessionCoordinator(Dependencies deps) : impl_(std::make_unique<Impl>(std::move(deps))) {}
SessionCoordinator::~SessionCoordinator() = default;

StartReceipt SessionCoordinator::start(const StartConfig& config) {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    std::lock_guard<std::mutex> requests(session.requests);
    if (!is_uuid(config.request_id) || config.model_release.empty()) {
        return session.receipt(false, "invalid_request");
    }
    const auto previous = session.start_receipts.find(config.request_id);
    if (previous != session.start_receipts.end()) {
        return previous->second.first == config.model_release ? previous->second.second
            : session.receipt(false, "request_conflict");
    }
    if (session.stop_receipts.count(config.request_id) || session.async_stop_receipts.count(config.request_id)) {
        return session.receipt(false, "request_conflict");
    }

    StartReceipt result;
    if (session.current.state != State::ready) {
        result = session.receipt(false, "busy");
    } else if (auto reason = session.prepare(DeviceMode::asr); !reason.empty()) {
        result = session.receipt(false, std::move(reason));
    } else {
        std::string error = "accelerator_unavailable";
        try {
            auto id = uuid();
            const auto wire = wire_uuid(id);
            auto stream = session.deps.make_stream(id, config);
            if (!stream) throw std::runtime_error("ASR factory returned no stream");
            // Model initialization can outlast the ALSA queue. Start capture once the decoder is ready.
            error = "capture_unavailable";
            auto source = session.deps.make_source(id);
            if (!source) throw std::runtime_error("audio factory returned no source");
            session.deadline = session.deps.now() + session_limit;
            session.wire_id = wire;
            session.source = std::move(source);
            session.stream = std::move(stream);
            session.current.session_id = std::move(id);
            session.current.state = State::recording;
            session.publish_status();
            result = session.receipt(true, "ok");
        } catch (const audio::CaptureStorageError&) {
            result = session.receipt(false, "capture_storage_error");
        } catch (...) {
            result = session.receipt(false, std::move(error));
        }
    }
    session.start_receipts.emplace(config.request_id, std::make_pair(config.model_release, result));
    return result;
}

StopReceipt SessionCoordinator::stop(std::string request_id, std::string session_id) {
    auto& session = *impl_;
    bool background_capture = false;
    {
        std::lock_guard<std::mutex> observer(session.observer);
        background_capture = session.observed_source && session.observed_source->capture_progress().has_value();
    }
    if (background_capture) {
        const auto scheduled = request_stop(request_id, session_id);
        if (scheduled.code == "request_conflict" || scheduled.code == "invalid_request") return scheduled;
    }
    std::lock_guard<std::mutex> owner(session.owner);
    if (!is_uuid(request_id) || (!session_id.empty() && !is_uuid(session_id))) {
        return {session.receipt(false, "invalid_request"), std::nullopt};
    }
    {
        std::lock_guard<std::mutex> requests(session.requests);
        const auto previous = session.stop_receipts.find(request_id);
        if (previous != session.stop_receipts.end()) {
            if (previous->second.first == session_id) return previous->second.second;
            return {session.receipt(false, "request_conflict"), std::nullopt};
        }
        const auto pending = session.async_stop_receipts.find(request_id);
        if (session.start_receipts.count(request_id) ||
            (pending != session.async_stop_receipts.end() && pending->second.first != session_id)) {
            return {session.receipt(false, "request_conflict"), std::nullopt};
        }
    }

    auto result = StopReceipt{session.receipt(false, "invalid_state"), std::nullopt};
    if (!session.current.session_id.empty() &&
        (session_id.empty() || session_id == session.current.session_id)) {
        if (session.current.state == State::recording) session.drain();
        if (session.current.snapshot) {
            result = {session.receipt(true, "ok"), session.current.stop_sample};
        } else if (!session.current.error_code.empty()) {
            result = {session.receipt(false, session.current.error_code), session.current.stop_sample};
        }
    }
    std::lock_guard<std::mutex> requests(session.requests);
    session.stop_receipts.emplace(std::move(request_id), std::make_pair(std::move(session_id), result));
    return result;
}

StopReceipt SessionCoordinator::request_stop(std::string request_id, std::string session_id) {
    auto& session = *impl_;
    std::lock_guard<std::mutex> requests(session.requests);
    std::lock_guard<std::mutex> observer(session.observer);
    const auto status = session.capture_status();
    const auto response = [&](bool ok, std::string code) -> StopReceipt {
        return {{ok, std::move(code), status.state, status.session_id}, status.stop_sample};
    };
    if (!is_uuid(request_id) || (!session_id.empty() && !is_uuid(session_id))) {
        return response(false, "invalid_request");
    }
    const auto completed = session.stop_receipts.find(request_id);
    if (completed != session.stop_receipts.end()) {
        return completed->second.first == session_id ? completed->second.second : response(false, "request_conflict");
    }
    const auto previous = session.async_stop_receipts.find(request_id);
    if (previous != session.async_stop_receipts.end()) {
        if (previous->second.first != session_id) return response(false, "request_conflict");
        const auto& original = previous->second.second;
        if (original.ok && original.session_id == status.session_id) {
            if (!status.error_code.empty()) return response(false, status.error_code);
            return response(true, "ok");
        }
        return original;
    }
    if (session.start_receipts.count(request_id)) return response(false, "request_conflict");

    auto result = response(false, "invalid_state");
    if (!status.session_id.empty() && (session_id.empty() || session_id == status.session_id)) {
        if (status.snapshot) {
            result = response(true, "ok");
        } else if (!status.error_code.empty()) {
            result = response(false, status.error_code);
        } else if (status.state == State::recording || status.state == State::draining) {
            if (status.state == State::recording) {
                session.observed_source->request_stop();
                session.stop_pending = true;
            }
            result = response(true, "ok");
            result.state = State::draining;
        }
    }
    session.async_stop_receipts.emplace(std::move(request_id), std::make_pair(std::move(session_id), result));
    return result;
}

Status SessionCoordinator::status() const {
    const auto& session = *impl_;
    std::lock_guard<std::mutex> lock(session.observer);
    return session.capture_status();
}

bool SessionCoordinator::pump() {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    if (session.current.state != State::recording) return false;
    const auto progress = session.source->capture_progress();
    if ((progress && progress->storage_failed) || session.stop_is_pending() || session.deps.now() >= session.deadline) {
        session.drain();
        return true;
    }

    std::optional<audio::CaptureEvent> event;
    try {
        event = session.source->read();
        if (!event) {
            if (const auto progress = session.source->capture_progress(); progress && progress->sealed) {
                session.drain();
                return true;
            }
            return false;
        }
        session.accept_boundary(*event);
    } catch (const audio::CaptureStorageError&) {
        session.source->stop();
        session.fail(State::faulted, "capture_storage_error");
        return true;
    } catch (...) {
        session.fail(State::faulted, "capture_error");
        return true;
    }
    try {
        session.record_capture(*event);
        session.publish(session.decode(*event));
        session.publish_status();
    } catch (...) {
        if (session.source->capture_progress()) session.source->stop();
        session.fail(State::faulted, "recording_failed");
        return true;
    }
    if (session.current.capture_end_sample == sample_limit) session.drain();
    return true;
}

StartReceipt SessionCoordinator::begin_refine(std::string job_id) {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    if (!is_uuid(job_id)) return session.receipt(false, "invalid_request");
    if (session.current.state == State::refining && session.current.refinement_job_id == job_id &&
        !session.cancel_requested) return session.receipt(true, "ok");
    if (session.current.state != State::finalized || session.cancel_receipts.count(job_id)) {
        return session.receipt(false, "invalid_state");
    }
    if (auto reason = session.prepare(DeviceMode::refine); !reason.empty()) {
        return session.receipt(false, std::move(reason));
    }
    session.current.refinement_job_id = std::move(job_id);
    session.current.state = State::refining;
    session.publish_status();
    return session.receipt(true, "ok");
}

CancelReceipt SessionCoordinator::cancel_refine(std::string job_id) {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    const auto previous = session.cancel_receipts.find(job_id);
    if (previous != session.cancel_receipts.end()) return previous->second;
    if (!is_uuid(job_id)) return {session.receipt(false, "invalid_request"), std::move(job_id)};
    if (session.current.state != State::refining || session.current.refinement_job_id != job_id) {
        return {session.receipt(false, "invalid_state"), std::move(job_id)};
    }
    if (!session.cancel_requested) {
        try {
            if (session.deps.device.request_cancel) session.deps.device.request_cancel(job_id);
            session.cancel_requested = true;
        } catch (...) {
            return {session.receipt(false, "cancel_failed"), std::move(job_id)};
        }
    }
    if (auto reason = session.prepare(DeviceMode::asr); !reason.empty()) {
        return {session.receipt(false, std::move(reason)), std::move(job_id)};
    }
    session.cancel_requested = false;
    session.current.refinement_job_id.clear();
    session.current.state = State::finalized;
    session.publish_status();
    CancelReceipt result{session.receipt(true, "ok"), std::move(job_id)};
    session.cancel_receipts.emplace(result.refinement_job_id, result);
    return result;
}

StartReceipt SessionCoordinator::complete_refine(std::string job_id) {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    if (session.current.state != State::refining || session.current.refinement_job_id != job_id ||
        session.cancel_requested) return session.receipt(false, "invalid_state");
    if (auto reason = session.prepare(DeviceMode::asr); !reason.empty()) {
        return session.receipt(false, std::move(reason));
    }
    session.current.refinement_job_id.clear();
    session.current.state = State::completed;
    session.publish_status();
    return session.receipt(true, "ok");
}

StartReceipt SessionCoordinator::complete() {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    if (session.current.state != State::finalized) return session.receipt(false, "invalid_state");
    session.current.state = State::completed;
    session.publish_status();
    return session.receipt(true, "ok");
}

StartReceipt SessionCoordinator::reset() {
    auto& session = *impl_;
    std::lock_guard<std::mutex> owner(session.owner);
    if (session.current.state != State::completed && session.current.state != State::faulted) {
        return session.receipt(false, "invalid_state");
    }
    if (auto reason = session.prepare(DeviceMode::asr); !reason.empty()) {
        return session.receipt(false, std::move(reason));
    }
    if (session.current.state == State::faulted) {
        try {
            session.source->stop();
        } catch (...) {
            return session.receipt(false, "capture_error");
        }
    }
    session.source.reset();
    session.stream.reset();
    session.current = {};
    session.publish_status();
    return session.receipt(true, "ok");
}

}  // namespace edge_speech::session

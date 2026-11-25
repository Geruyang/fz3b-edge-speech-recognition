#pragma once

#include "board/asr/stream_engine.hpp"
#include "board/audio/capture.hpp"

#include <chrono>
#include <functional>
#include <iosfwd>

namespace edge_speech::session {

using Clock = std::chrono::steady_clock;
enum class State { ready, recording, draining, finalized, refining, completed, faulted };
enum class DeviceStatus { idle, busy, unknown };
enum class DeviceMode { asr, refine };

std::string_view state_name(State state);
WireUuid wire_uuid(std::string_view id);
std::string sha256(std::string_view bytes);
std::string sha256(std::istream& input);

struct StartConfig {
    std::string request_id;
    std::string model_release;
};

struct StartReceipt {
    bool ok = false;
    std::string code;
    State state = State::ready;
    std::string session_id;
};

struct StopReceipt : StartReceipt {
    std::optional<std::uint64_t> stop_sample;
};

struct CancelReceipt : StartReceipt {
    std::string refinement_job_id;
};

struct Status {
    State state = State::ready;
    std::string session_id;
    std::uint64_t capture_end_sample = 0;
    std::optional<std::uint64_t> stop_sample;
    std::uint64_t processed_sample = 0;
    std::uint64_t high_watermark = 0;
    Integrity integrity = Integrity::pending;
    std::vector<asr::StreamGap> gaps;
    std::optional<SessionSnapshot> snapshot;
    std::string error_code;
    std::string refinement_job_id;
};

struct DeviceHooks {
    // PL deployments supply both callbacks. Quiesce must positively confirm idle
    // (including DMA) before switch_mode may release or repurpose device buffers.
    // Pinned mappings must remain owned by the device supervisor during teardown.
    std::function<DeviceStatus()> quiesce;
    std::function<void(DeviceMode)> switch_mode;
    std::function<void(std::string_view)> request_cancel;
};

struct Dependencies {
    std::function<std::unique_ptr<audio::AudioSource>(const std::string&)> make_source;
    std::function<std::unique_ptr<asr::StreamEngine>(const std::string&, const StartConfig&)> make_stream;
    std::function<void(const AudioBlock&)> on_audio;
    std::function<void(const TranscriptEvent&)> on_event;
    std::function<void(const SessionSnapshot&, const asr::FinalizedStream&)> freeze;
    std::function<Clock::time_point()> now;
    std::function<void(const audio::CaptureGap&)> on_gap;
    DeviceHooks device;
};

// Calls serialize source/ASR ownership across capture and control threads.
// Callbacks run synchronously and may inspect status(). Event and freeze callbacks
// return only after commit. Audio callbacks should use a nonblocking sink.
class SessionCoordinator {
public:
    explicit SessionCoordinator(Dependencies deps);
    ~SessionCoordinator();

    StartReceipt start(const StartConfig& config);
    StopReceipt stop(std::string request_id, std::string session_id = {});
    // Acknowledge STOP and signal background capture without waiting for ASR.
    // pump() then uses the same drain and snapshot commit path as stop().
    StopReceipt request_stop(std::string request_id, std::string session_id = {});
    Status status() const;

    // The capture loop calls pump regularly. Each call reads at most one live
    // capture event or seals the capture when its 60 minute deadline is reached.
    bool pump();

    StartReceipt begin_refine(std::string refinement_job_id);
    CancelReceipt cancel_refine(std::string refinement_job_id);
    StartReceipt complete_refine(std::string refinement_job_id);
    StartReceipt complete();
    // The owner has resolved archive disposition before calling reset.
    StartReceipt reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_speech::session

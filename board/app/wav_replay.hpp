#pragma once

#include "board/audio/capture.hpp"

#include <chrono>
#include <functional>
#include <istream>
#include <mutex>

namespace edge_speech::app {

// Real-time replay of normalized PCM fixtures through the same AudioSource
// interface as the board microphone. Only due real samples are eligible at stop.
class WavReplaySource final : public audio::AudioSource {
public:
    using Clock = std::chrono::steady_clock;
    using Now = std::function<Clock::time_point()>;

    WavReplaySource(WireUuid session, std::unique_ptr<std::istream> wav, Now now = Clock::now);
    std::optional<audio::CaptureEvent> read() override;
    void stop() override;
    void request_stop() override;
    bool exhausted() const { return position_ == stop_frame_.value_or(frames_); }

private:
    std::uint64_t due_frame() const;

    WireUuid session_;
    std::unique_ptr<std::istream> wav_;
    Now now_;
    Clock::time_point start_;
    std::uint64_t frames_ = 0;
    std::uint64_t position_ = 0;
    std::optional<std::uint64_t> stop_frame_;
    std::mutex request_mutex_;
    std::optional<std::uint64_t> requested_frame_;
};

}  // namespace edge_speech::app

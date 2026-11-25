#pragma once

#include "board/audio/capture.hpp"

#include <filesystem>
#include <functional>
#include <stdexcept>

namespace edge_speech::audio {

class CaptureStorageError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

// The worker owns the source and writes PCM before the ASR consumer can read it.
// The consumer owns read()/stop(). Progress and stop requests are thread-safe.
class SpoolCaptureSource final : public AudioSource {
public:
    using OnStored = std::function<void(const CaptureEvent&)>;

    SpoolCaptureSource(std::unique_ptr<AudioSource> source, WireUuid session,
                       const std::filesystem::path& directory, OnStored on_stored = {});
    ~SpoolCaptureSource() override;
    std::optional<CaptureEvent> read() override;
    void stop() override;
    void request_stop() override;
    std::optional<CaptureProgress> capture_progress() const override;
    CaptureProgress progress() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<SpoolCaptureSource> make_spooled_capture_source(
    std::unique_ptr<AudioSource> source, WireUuid session, const std::filesystem::path& directory,
    SpoolCaptureSource::OnStored on_stored = {});

}  // namespace edge_speech::audio

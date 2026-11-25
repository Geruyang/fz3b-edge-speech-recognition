#pragma once

#include "board/audio/resampler.hpp"
#include "shared/types.hpp"

#include <memory>
#include <variant>

namespace edge_speech::audio {

inline constexpr char alsa_device[] = "hw:CARD=U2K,DEV=0";

// Persist timeout as model reason "other" and retain its explanatory detail.
enum class CaptureGapReason { device_xrun, disconnect, timeout, queue_overrun };

struct CaptureGap {
    std::uint64_t sample_start;
    std::optional<std::uint64_t> sample_end;
    CaptureGapReason reason;
    std::uint64_t detected_at_ns;  // steady_clock, local to the capture process
    std::string detail;
};

using CaptureEvent = std::variant<AudioBlock, CaptureGap>;

struct CaptureProgress {
    std::uint64_t sample_end = 0;
    bool sealed = false;
    bool storage_failed = false;
};

class AudioSource {
public:
    virtual ~AudioSource() = default;
    virtual std::optional<CaptureEvent> read() = 0;
    virtual void stop() = 0;
    // Background sources can honor this while the decoder owns read()/stop().
    virtual void request_stop() {}
    virtual std::optional<CaptureProgress> capture_progress() const { return std::nullopt; }
};

enum class NativeStatus { no_data, samples, xrun, disconnected, timeout };

struct NativeRead {
    NativeStatus status = NativeStatus::no_data;
    unsigned sample_rate = 0;
    std::vector<std::int16_t> interleaved_stereo;
    std::string detail;
    // A sealed read may recover a positive prefix before an unrecoverable tail.
    std::optional<NativeStatus> tail_gap{};
};

class NativeCapture {
public:
    virtual ~NativeCapture() = default;
    virtual NativeRead read() = 0;
    virtual void stop() = 0;
    // A device may preserve frames already available at the instant of sealing.
    virtual NativeRead stop_tail() { stop(); return {}; }
};

enum class ChannelSelection { left, right, average };

struct CaptureFormat {
    unsigned sample_rate;
    ChannelSelection channel_selection;
    std::optional<double> channel_correlation;
};

// A single owner calls read() and stop(). When read() has no current event it
// returns nullopt. After stop() seals the source, drain through read() to nullopt.
// A gap is ordered after the valid pre-gap tail. Unknown gap lengths do not
// advance the sample index, so consumers must handle the gap even at equal indices.
class CaptureSource final : public AudioSource {
public:
    CaptureSource(WireUuid session_id, std::unique_ptr<NativeCapture> native, bool preserve_device_tail = false);
    std::optional<CaptureEvent> read() override;
    void stop() override;
    // Most recent native rate and channel decision, available once PCM is normalized.
    std::optional<CaptureFormat> format() const { return format_; }

private:
    void normalize(const NativeRead& result);
    void start_segment();
    void mix(const std::vector<std::int16_t>& stereo);
    void emit_blocks(bool drain);
    void flush_segment();
    void record_gap(CaptureGapReason reason, const std::string& detail);

    WireUuid session_id_;
    std::unique_ptr<NativeCapture> native_;
    bool preserve_device_tail_;
    std::optional<CaptureFormat> format_;
    std::optional<Resampler> resampler_;
    std::vector<std::int16_t> probing_stereo_;
    unsigned probing_rate_ = 0;
    std::vector<std::int16_t> normalized_;
    std::deque<CaptureEvent> events_;
    std::uint64_t next_sample_ = 0;
    std::optional<CaptureGapReason> outage_;
    bool stopped_ = false;
};

// The Linux ALSA backend selects a native S16_LE stereo rate. A caller can
// explicitly request one measured rate, or let the backend select in order
// 16000, 48000, 44100 Hz. The host-reference library has no ALSA dependency.
std::unique_ptr<NativeCapture> make_alsa_capture(unsigned preferred_rate = 0);

// One caller reads and stops this facade. The worker owns the underlying source.
// Pending PCM is bounded to max_blocks. Overrun keeps newest PCM and records
// discarded sample ranges. After successful stop(), read() drains the queue.
std::unique_ptr<AudioSource> make_buffered_capture_source(std::unique_ptr<AudioSource> source,
                                                         std::size_t max_blocks = 40);

}  // namespace edge_speech::audio

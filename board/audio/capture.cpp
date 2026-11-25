#include "board/audio/capture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(EDGE_SPEECH_HAS_ALSA) && EDGE_SPEECH_HAS_ALSA
#include <alsa/asoundlib.h>
#include <cerrno>
#endif

namespace edge_speech::audio {
namespace {

using Clock = std::chrono::steady_clock;

bool supported_rate(unsigned rate) {
    return rate == 16000 || rate == 44100 || rate == 48000;
}

CaptureFormat inspect_channels(unsigned rate, const std::vector<std::int16_t>& stereo) {
    double left = 0;
    double right = 0;
    double cross = 0;
    const auto probe_samples = std::min(stereo.size(), std::size_t(2 * (rate / 50)));
    for (std::size_t i = 0; i < probe_samples; i += 2) {
        const double l = stereo[i];
        const double r = stereo[i + 1];
        left += l * l;
        right += r * r;
        cross += l * r;
    }
    CaptureFormat format{rate, left >= right ? ChannelSelection::left : ChannelSelection::right,
                         std::nullopt};
    if (left > 0 && right > 0) {
        format.channel_correlation = std::clamp(cross / std::sqrt(left * right), -1.0, 1.0);
        if (*format.channel_correlation >= 0.7 && std::min(left, right) >= 0.25 * std::max(left, right)) {
            format.channel_selection = ChannelSelection::average;
        }
    }
    return format;
}

}  // namespace

CaptureSource::CaptureSource(WireUuid session_id, std::unique_ptr<NativeCapture> native, bool preserve_device_tail)
    : session_id_(session_id), native_(std::move(native)), preserve_device_tail_(preserve_device_tail) {
    if (!native_) throw std::invalid_argument("native audio capture is required");
}

std::optional<CaptureEvent> CaptureSource::read() {
    if (events_.empty() && !stopped_) {
        auto result = native_->read();
        switch (result.status) {
        case NativeStatus::no_data:
            break;
        case NativeStatus::samples:
            normalize(result);
            outage_.reset();
            emit_blocks(false);
            break;
        case NativeStatus::xrun:
            record_gap(CaptureGapReason::device_xrun, result.detail);
            break;
        case NativeStatus::disconnected:
            record_gap(CaptureGapReason::disconnect, result.detail);
            break;
        case NativeStatus::timeout:
            record_gap(CaptureGapReason::timeout, result.detail);
            break;
        }
    }
    if (events_.empty()) return std::nullopt;
    auto event = std::move(events_.front());
    events_.pop_front();
    return event;
}

void CaptureSource::stop() {
    if (stopped_) return;
    NativeRead tail;
    if (preserve_device_tail_) tail = native_->stop_tail();
    else native_->stop();
    stopped_ = true;
    if (tail.status == NativeStatus::samples) normalize(tail);
    switch (tail.tail_gap.value_or(tail.status)) {
    case NativeStatus::xrun: record_gap(CaptureGapReason::device_xrun, tail.detail); break;
    case NativeStatus::disconnected: record_gap(CaptureGapReason::disconnect, tail.detail); break;
    case NativeStatus::timeout: record_gap(CaptureGapReason::timeout, tail.detail); break;
    case NativeStatus::samples:
    case NativeStatus::no_data: break;
    }
    flush_segment();
}

void CaptureSource::normalize(const NativeRead& result) {
    if (!supported_rate(result.sample_rate) || result.interleaved_stereo.empty() ||
        result.interleaved_stereo.size() % 2 != 0) {
        throw std::invalid_argument("invalid native S16_LE stereo capture");
    }
    if (probing_rate_ != 0 && probing_rate_ != result.sample_rate) {
        throw std::runtime_error("native audio rate changed without a capture gap");
    }
    if (resampler_) {
        mix(result.interleaved_stereo);
        return;
    }
    probing_rate_ = result.sample_rate;
    probing_stereo_.insert(probing_stereo_.end(), result.interleaved_stereo.begin(),
                           result.interleaved_stereo.end());
    if (probing_stereo_.size() / 2 >= probing_rate_ / 50) start_segment();
}

void CaptureSource::start_segment() {
    format_ = inspect_channels(probing_rate_, probing_stereo_);
    resampler_.emplace(probing_rate_);
    mix(probing_stereo_);
    probing_stereo_.clear();
}

void CaptureSource::mix(const std::vector<std::int16_t>& stereo) {
    std::vector<std::int16_t> mono(stereo.size() / 2);
    for (std::size_t i = 0; i < mono.size(); ++i) {
        const auto l = stereo[2 * i];
        const auto r = stereo[2 * i + 1];
        switch (format_->channel_selection) {
        case ChannelSelection::left:
            mono[i] = l;
            break;
        case ChannelSelection::right:
            mono[i] = r;
            break;
        case ChannelSelection::average:
            mono[i] = static_cast<std::int16_t>((static_cast<int>(l) + r) / 2);
            break;
        }
    }
    resampler_->push(mono.data(), mono.size(), normalized_);
}

void CaptureSource::emit_blocks(bool drain) {
    std::size_t offset = 0;
    while (normalized_.size() - offset >= samples_per_block || (drain && offset < normalized_.size())) {
        const auto count = std::min(samples_per_block, normalized_.size() - offset);
        AudioBlock block{session_id_, next_sample_, static_cast<std::uint32_t>(count), 0, {}};
        block.pcm_s16le.reserve(count * 2);
        for (std::size_t i = 0; i < count; ++i) {
            const auto sample = static_cast<std::uint16_t>(normalized_[offset + i]);
            block.pcm_s16le.push_back(static_cast<std::uint8_t>(sample & 0xff));
            block.pcm_s16le.push_back(static_cast<std::uint8_t>(sample >> 8));
        }
        events_.emplace_back(std::move(block));
        next_sample_ += count;
        offset += count;
    }
    normalized_.erase(normalized_.begin(), normalized_.begin() + offset);
}

void CaptureSource::flush_segment() {
    if (!probing_stereo_.empty()) start_segment();
    if (resampler_) resampler_->finish(normalized_);
    emit_blocks(true);
    resampler_.reset();
    probing_rate_ = 0;
}

void CaptureSource::record_gap(CaptureGapReason reason, const std::string& detail) {
    if (outage_ == reason) return;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch());
    flush_segment();
    events_.emplace_back(CaptureGap{next_sample_, std::nullopt, reason,
                                    static_cast<std::uint64_t>(now.count()), detail});
    outage_ = reason;
}

namespace {

class BufferedCaptureSource final : public AudioSource {
public:
    BufferedCaptureSource(std::unique_ptr<AudioSource> source, std::size_t max_blocks)
        : source_(std::move(source)), max_blocks_(max_blocks), worker_(&BufferedCaptureSource::run, this) {}

    ~BufferedCaptureSource() override { join(); }

    std::optional<CaptureEvent> read() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.empty()) {
            if (failure_) std::rethrow_exception(failure_);
            return std::nullopt;
        }
        auto event = std::move(events_.front());
        events_.pop_front();
        if (std::holds_alternative<AudioBlock>(event)) --queued_blocks_;
        return event;
    }

    void stop() override {
        join();
        if (failure_) std::rethrow_exception(failure_);
    }

    void request_stop() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stop_requested_ && source_) source_->request_stop();
            stop_requested_ = true;
        }
        wake_.notify_one();
    }

private:
    void join() {
        request_stop();
        if (worker_.joinable()) worker_.join();
    }

    void run() {
        std::exception_ptr failure;
        try {
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stop_requested_) break;
                }
                if (auto event = source_->read()) {
                    enqueue(std::move(*event));
                } else {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_.wait_for(lock, std::chrono::milliseconds(2), [this] { return stop_requested_; });
                }
            }
        } catch (...) {
            failure = std::current_exception();
        }

        try {
            source_->stop();
            if (!failure) {
                while (auto event = source_->read()) enqueue(std::move(*event));
            }
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        source_.reset();
        failure_ = std::move(failure);
    }

    void enqueue(CaptureEvent event) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto* block = std::get_if<AudioBlock>(&event);
        const bool has_pcm = block != nullptr;
        if (block) {
            if (block->sample_count == 0 || block->sample_count > samples_per_block ||
                block->pcm_s16le.size() != 2 * block->sample_count ||
                block->sample_start > std::numeric_limits<std::uint64_t>::max() - block->sample_count) {
                throw std::runtime_error("buffered capture received an invalid PCM block extent");
            }
            if (queued_blocks_ == max_blocks_) discard_oldest_pcm();
        }
        // Sustained distinct hardware faults cannot be coalesced without losing
        // evidence. Fail visibly if an unresponsive consumer outlasts this bound.
        if (events_.size() >= 4 * max_blocks_ + 16) {
            throw std::runtime_error("buffered capture event limit exceeded");
        }
        events_.push_back(std::move(event));
        if (has_pcm) ++queued_blocks_;
    }

    void discard_oldest_pcm() {
        const auto first_pcm = std::find_if(events_.begin(), events_.end(), [](const CaptureEvent& event) {
            return std::holds_alternative<AudioBlock>(event);
        });
        const auto& block = std::get<AudioBlock>(*first_pcm);
        const auto end = block.sample_start + block.sample_count;
        --queued_blocks_;
        if (first_pcm != events_.begin()) {
            auto* previous = std::get_if<CaptureGap>(&*std::prev(first_pcm));
            if (previous && previous->reason == CaptureGapReason::queue_overrun &&
                previous->sample_end == block.sample_start) {
                previous->sample_end = end;
                events_.erase(first_pcm);
                return;
            }
        }
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch());
        *first_pcm = CaptureGap{block.sample_start, end, CaptureGapReason::queue_overrun,
                                static_cast<std::uint64_t>(now.count()), "capture PCM queue overflow"};
    }

    std::unique_ptr<AudioSource> source_;
    const std::size_t max_blocks_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::list<CaptureEvent> events_;
    std::size_t queued_blocks_ = 0;
    std::exception_ptr failure_;
    bool stop_requested_ = false;
    std::thread worker_;
};

}  // namespace

std::unique_ptr<AudioSource> make_buffered_capture_source(std::unique_ptr<AudioSource> source,
                                                         std::size_t max_blocks) {
    if (!source || max_blocks == 0 || max_blocks > 40) {
        throw std::invalid_argument("buffered capture requires a source and 1 to 40 PCM blocks");
    }
    return std::make_unique<BufferedCaptureSource>(std::move(source), max_blocks);
}

#if defined(EDGE_SPEECH_HAS_ALSA) && EDGE_SPEECH_HAS_ALSA
namespace {

constexpr auto retry_interval = std::chrono::milliseconds(250);
constexpr auto no_progress_limit = std::chrono::milliseconds(250);

bool disconnected(int error) {
    return error == -ENOENT || error == -ENODEV || error == -ENXIO || error == -EIO;
}

std::string alsa_error(const char* operation, int error) {
    return std::string(alsa_device) + " " + operation + ": " + snd_strerror(error);
}

int configure(snd_pcm_t* pcm, unsigned rate) {
    snd_pcm_hw_params_t* config;
    snd_pcm_hw_params_alloca(&config);
    int result = snd_pcm_hw_params_any(pcm, config);
    if (result >= 0) result = snd_pcm_hw_params_set_access(pcm, config, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (result >= 0) result = snd_pcm_hw_params_set_format(pcm, config, SND_PCM_FORMAT_S16_LE);
    if (result >= 0) result = snd_pcm_hw_params_set_channels(pcm, config, 2);
    if (result >= 0) result = snd_pcm_hw_params_set_rate(pcm, config, rate, 0);
    unsigned period_us = 20000;
    unsigned buffer_us = 80000;
    if (result >= 0) result = snd_pcm_hw_params_set_period_time_near(pcm, config, &period_us, nullptr);
    if (result >= 0) result = snd_pcm_hw_params_set_buffer_time_near(pcm, config, &buffer_us, nullptr);
    if (result >= 0) result = snd_pcm_hw_params(pcm, config);
    return result;
}

class AlsaCapture final : public NativeCapture {
public:
    explicit AlsaCapture(unsigned preferred_rate) : preferred_rate_(preferred_rate) {}
    ~AlsaCapture() override { stop(); }

    NativeRead read() override {
        if (stopped_) return {};
        if (!pcm_) {
            if (Clock::now() < reopen_at_) return {};
            int result = open();
            if (result < 0) {
                if (disconnected(result)) return lost("open/configure", result);
                throw std::runtime_error(alsa_error("open/configure", result));
            }
            last_progress_ = Clock::now();
        }

        std::vector<std::int16_t> samples(2 * (rate_ / 50));
        const auto count = snd_pcm_readi(pcm_.get(), samples.data(), samples.size() / 2);
        if (count > 0) {
            last_progress_ = Clock::now();
            samples.resize(2 * static_cast<std::size_t>(count));
            return {NativeStatus::samples, rate_, std::move(samples), {}};
        }
        if (snd_pcm_state(pcm_.get()) == SND_PCM_STATE_DISCONNECTED) {
            return lost("read", -ENODEV);
        }
        if (count == 0 || count == -EAGAIN || count == -EINTR) {
            if (Clock::now() - last_progress_ < no_progress_limit) return {};
            pcm_.reset();
            return {NativeStatus::timeout, 0, {}, std::string(alsa_device) + " no native frames within " +
                    std::to_string(no_progress_limit.count()) + " ms"};
        }
        const auto error = static_cast<int>(count);
        if (error == -EPIPE || error == -ESTRPIPE) {
            const auto prepared = snd_pcm_prepare(pcm_.get());
            if (prepared < 0) {
                if (disconnected(prepared)) return lost("recover", prepared);
                throw std::runtime_error(alsa_error("recover", prepared));
            }
            last_progress_ = Clock::now();
            return {NativeStatus::xrun, 0, {}, alsa_error("read", error)};
        }
        if (disconnected(error)) return lost("read", error);
        throw std::runtime_error(alsa_error("read", error));
    }

    void stop() override {
        stopped_ = true;
        pcm_.reset();
    }

    NativeRead stop_tail() override {
        if (!pcm_) {
            stop();
            return {};
        }
        // Take a bounded snapshot of unread frames. The device may keep
        // advancing while readi copies them, so read only the snapshot extent.
        const auto available = snd_pcm_avail_update(pcm_.get());
        if (available > 0 && static_cast<snd_pcm_uframes_t>(available) > buffer_frames_) {
            stop();
            return {NativeStatus::xrun, 0, {}, std::string(alsa_device) + " capture buffer overrun at seal"};
        }
        auto error = static_cast<int>(available);
        std::vector<std::int16_t> tail;
        if (available > 0) {
            tail.resize(2 * static_cast<std::size_t>(available));
            snd_pcm_sframes_t copied = 0;
            unsigned interruptions = 0;
            while (copied < available) {
                const auto count = snd_pcm_readi(pcm_.get(), tail.data() + 2 * copied, available - copied);
                if (count > 0) {
                    copied += count;
                } else if (count != -EINTR || ++interruptions == 4) {
                    error = static_cast<int>(count);
                    break;
                }
            }
            tail.resize(2 * static_cast<std::size_t>(copied));
        }
        stop();
        if (available > 0 && tail.size() == 2 * static_cast<std::size_t>(available)) {
            return {NativeStatus::samples, rate_, std::move(tail), {}};
        }
        if (available == 0 || available == -EAGAIN) return {};
        const auto status = error == -EPIPE || error == -ESTRPIPE ? NativeStatus::xrun :
            disconnected(error) ? NativeStatus::disconnected : NativeStatus::timeout;
        const auto detail = error ? alsa_error("seal", error) : std::string(alsa_device) + " unread frames at seal";
        if (!tail.empty()) {
            return {NativeStatus::samples, rate_, std::move(tail), detail, status};
        }
        return {status, 0, {}, detail};
    }

private:
    int open() {
        snd_pcm_t* handle = nullptr;
        auto result = snd_pcm_open(&handle, alsa_device, SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
        if (result < 0) return result;
        decltype(pcm_) candidate(handle, snd_pcm_close);
        const std::array<unsigned, 3> default_rates{16000, 48000, 44100};
        for (const auto default_rate : default_rates) {
            const auto rate = preferred_rate_ == 0 ? default_rate : preferred_rate_;
            result = configure(candidate.get(), rate);
            snd_pcm_uframes_t period_frames = 0;
            if (result >= 0) result = snd_pcm_get_params(candidate.get(), &buffer_frames_, &period_frames);
            if (result >= 0) {
                rate_ = rate;
                pcm_ = std::move(candidate);
                return 0;
            }
            if (preferred_rate_ != 0 || disconnected(result)) break;
        }
        return result;
    }

    NativeRead lost(const char* operation, int error) {
        pcm_.reset();
        reopen_at_ = Clock::now() + retry_interval;
        return {NativeStatus::disconnected, 0, {}, alsa_error(operation, error)};
    }

    std::unique_ptr<snd_pcm_t, decltype(&snd_pcm_close)> pcm_{nullptr, snd_pcm_close};
    unsigned preferred_rate_;
    unsigned rate_ = 0;
    snd_pcm_uframes_t buffer_frames_ = 0;
    Clock::time_point reopen_at_ = Clock::time_point::min();
    Clock::time_point last_progress_;
    bool stopped_ = false;
};

}  // namespace
#endif

std::unique_ptr<NativeCapture> make_alsa_capture(unsigned preferred_rate) {
    if (preferred_rate != 0 && !supported_rate(preferred_rate)) {
        throw std::invalid_argument("unsupported ALSA capture rate");
    }
#if defined(EDGE_SPEECH_HAS_ALSA) && EDGE_SPEECH_HAS_ALSA
    return std::make_unique<AlsaCapture>(preferred_rate);
#else
    throw std::runtime_error("this build has no ALSA capture backend");
#endif
}

}  // namespace edge_speech::audio

#include "board/app/wav_replay.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace edge_speech::app {
namespace {

constexpr std::uint64_t sample_rate = 16000;
constexpr std::uint64_t block_frames = 320;
constexpr std::uint64_t max_frames = 3600 * sample_rate;

void read_exact(std::istream& input, char* output, std::streamsize size) {
    if (!input.read(output, size)) throw std::invalid_argument("truncated replay WAV");
}

std::uint32_t le(const char* bytes, unsigned count) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
        value |= std::uint32_t{static_cast<unsigned char>(bytes[i])} << (8 * i);
    }
    return value;
}

std::uint64_t seek_capture_payload(std::istream& input) {
    input.seekg(0, std::ios::end);
    const auto file_end = input.tellg();
    input.seekg(0);
    std::array<char, 12> riff{};
    read_exact(input, riff.data(), riff.size());
    const auto end = std::uint64_t{le(riff.data() + 4, 4)} + 8;
    if (std::string_view(riff.data(), 4) != "RIFF" || std::string_view(riff.data() + 8, 4) != "WAVE" ||
        file_end < 0 || end < 12 || end > static_cast<std::uint64_t>(file_end)) {
        throw std::invalid_argument("invalid replay RIFF bounds");
    }

    bool format_seen = false;
    for (std::uint64_t offset = 12; end - offset >= 8;) {
        std::array<char, 8> header{};
        read_exact(input, header.data(), header.size());
        const auto chunk_size = std::uint64_t{le(header.data() + 4, 4)};
        const auto next = offset + 8 + chunk_size + (chunk_size & 1);
        if (next > end) throw std::invalid_argument("replay WAV chunk exceeds RIFF");
        const auto id = std::string_view(header.data(), 4);
        if (id == "fmt ") {
            if (format_seen || chunk_size < 16) throw std::invalid_argument("invalid replay WAV format chunk");
            std::array<char, 16> format{};
            read_exact(input, format.data(), format.size());
            if (le(format.data(), 2) != 1 || le(format.data() + 2, 2) != 1 ||
                le(format.data() + 4, 4) != sample_rate || le(format.data() + 8, 4) != 2 * sample_rate ||
                le(format.data() + 12, 2) != 2 || le(format.data() + 14, 2) != 16) {
                throw std::invalid_argument("replay requires mono 16 kHz PCM_S16LE WAV");
            }
            format_seen = true;
        } else if (id == "data") {
            if (!format_seen || (chunk_size & 1) || chunk_size / 2 > max_frames) {
                throw std::invalid_argument("invalid WAV PCM payload size or missing format");
            }
            return chunk_size / 2;
        }
        input.seekg(static_cast<std::streamoff>(next));
        offset = next;
    }
    throw std::invalid_argument("replay WAV does not contain a data chunk");
}

}  // namespace

WavReplaySource::WavReplaySource(WireUuid session, std::unique_ptr<std::istream> wav, Now now)
    : session_(session), wav_(std::move(wav)), now_(std::move(now)) {
    if (!wav_ || !*wav_ || !now_) throw std::invalid_argument("missing replay stream or clock");
    frames_ = seek_capture_payload(*wav_);
    start_ = now_();
}

std::uint64_t WavReplaySource::due_frame() const {
    const auto age = now_() - start_;
    if (age <= Clock::duration::zero()) return 0;
    if (age >= std::chrono::hours(1)) return frames_;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(age).count();
    return std::min(frames_, static_cast<std::uint64_t>(elapsed) * sample_rate / 1000000);
}

std::optional<audio::CaptureEvent> WavReplaySource::read() {
    const auto limit = stop_frame_.value_or(frames_);
    auto due = limit;
    if (!stop_frame_) {
        std::lock_guard<std::mutex> lock(request_mutex_);
        due = requested_frame_ ? *requested_frame_ : due_frame();
    }
    if (due <= position_) return std::nullopt;
    const auto available = due - position_;
    if (available < block_frames && due != limit) return std::nullopt;
    const auto count = static_cast<std::uint32_t>(std::min(block_frames, available));
    Bytes pcm(std::size_t{count} * 2);
    read_exact(*wav_, reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
    AudioBlock block{session_, position_, count, 0, std::move(pcm)};
    position_ += count;
    return audio::CaptureEvent{std::move(block)};
}

void WavReplaySource::stop() {
    if (stop_frame_) return;
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (!requested_frame_) requested_frame_ = due_frame();
    stop_frame_ = std::max(position_, *requested_frame_);
}

void WavReplaySource::request_stop() {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (!requested_frame_) requested_frame_ = due_frame();
}

}  // namespace edge_speech::app

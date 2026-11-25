#pragma once

#include "board/asr/stream_engine.hpp"

#include <complex>
#include <deque>

namespace edge_speech::asr {

inline constexpr std::uint64_t samples_per_frame = 160;
inline constexpr std::uint64_t frame_left_context = 120;
inline constexpr std::uint64_t frame_end = 280;
inline constexpr std::uint64_t tail_silence_samples = 8000;

class Fbank {
public:
    explicit Fbank(bool frozen = false);
    ~Fbank();
    FeatureWindow window(const std::deque<std::int16_t>& pcm, std::uint64_t pcm_offset,
                         std::uint64_t start_frame, std::uint64_t real_samples,
                         bool finishing) const;

private:
    struct Frozen;
    std::unique_ptr<Frozen> frozen_;
    static constexpr std::size_t frame_size = 400;
    static constexpr std::size_t fft_size = 512;
    std::array<float, frame_size> window_;
    std::array<std::size_t, fft_size> bit_reverse_;
    std::array<std::complex<float>, fft_size / 2> roots_;
    std::array<std::vector<std::pair<std::size_t, float>>, feature_dim> filters_;
};

}  // namespace edge_speech::asr

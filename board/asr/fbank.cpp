#include "board/asr/fbank.hpp"

#include <cmath>
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#endif

namespace edge_speech::asr {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr float sample_rate = 16000;

float mel(float hz) {
    return 1127.0F * std::log(1.0F + hz / 700.0F);
}

}  // namespace

struct Fbank::Frozen {
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
    static knf::FbankOptions options() {
        knf::FbankOptions opts;
        opts.frame_opts.samp_freq = 16000;
        opts.frame_opts.frame_shift_ms = 10;
        opts.frame_opts.frame_length_ms = 25;
        opts.frame_opts.dither = 0;
        opts.frame_opts.snip_edges = false;
        opts.frame_opts.remove_dc_offset = true;
        opts.frame_opts.preemph_coeff = 0.97F;
        opts.frame_opts.window_type = "povey";
        opts.mel_opts.num_bins = feature_dim;
        opts.mel_opts.low_freq = 20;
        opts.mel_opts.high_freq = -400;
        opts.mel_opts.is_librosa = false;
        return opts;
    }
    const knf::FbankOptions opts = options();
    knf::FeatureWindowFunction function{opts.frame_opts};
    knf::FbankComputer computer{opts};

    FeatureWindow compute(const std::deque<std::int16_t>& pcm, std::uint64_t offset,
                          std::uint64_t first, std::uint64_t limit, bool finishing) {
        const auto end = std::min(limit + (finishing ? tail_silence_samples : 0),
                                  (first + window_frames - 1) * samples_per_frame + frame_end);
        if (offset >= end || first + window_frames > INT32_MAX)
            throw std::out_of_range("fbank frame outside retained audio");
        std::vector<float> wave(end - offset, 0);
        for (auto i = offset; i < std::min(end, limit); ++i)
            wave[i - offset] = pcm.at(i - offset) / 32768.0F;
        FeatureWindow features{};
        std::vector<float> frame;
        for (std::size_t i = 0; i < window_frames; ++i) {
            std::fill(frame.begin(), frame.end(), 0.0F);
            knf::ExtractWindow(offset, wave, static_cast<std::int32_t>(first + i),
                               opts.frame_opts, function, &frame);
            computer.Compute(0, 1, &frame, features.data() + i * feature_dim);
        }
        return features;
    }
#endif
};

Fbank::~Fbank() = default;

Fbank::Fbank(bool frozen) {
    if (frozen) {
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
        frozen_ = std::make_unique<Frozen>();
        return;
#else
        throw std::invalid_argument("frozen fbank is unavailable in this build");
#endif
    }
    for (std::size_t index = 0; index < window_.size(); ++index) {
        window_[index] = static_cast<float>(
            std::pow(0.5 - 0.5 * std::cos(2 * pi * index / (frame_size - 1)), 0.85));
    }
    for (std::size_t index = 0; index < fft_size; ++index) {
        std::size_t reversed = 0;
        for (std::size_t mask = 1; mask < fft_size; mask <<= 1) {
            reversed = (reversed << 1) | ((index & mask) != 0);
        }
        bit_reverse_[index] = reversed;
    }
    for (std::size_t index = 0; index < roots_.size(); ++index) {
        const auto angle = -2 * pi * index / fft_size;
        roots_[index] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
    }
    const auto low = mel(20);
    const auto step = (mel(sample_rate / 2 - 400) - low) / (feature_dim + 1);
    for (std::size_t bin = 0; bin < feature_dim; ++bin) {
        const auto left = low + bin * step;
        const auto center = low + (bin + 1) * step;
        const auto right = low + (bin + 2) * step;
        for (std::size_t index = 0; index < fft_size / 2; ++index) {
            const auto frequency = mel(index * (sample_rate / fft_size));
            if (left < frequency && frequency < right) {
                const auto weight = frequency <= center ?
                    (frequency - left) / (center - left) : (right - frequency) / (right - center);
                filters_[bin].emplace_back(index, weight);
            }
        }
    }
}

FeatureWindow Fbank::window(const std::deque<std::int16_t>& pcm, std::uint64_t pcm_offset,
                            std::uint64_t start_frame, std::uint64_t real_samples,
                            bool finishing) const {
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
    if (frozen_) return frozen_->compute(pcm, pcm_offset, start_frame, real_samples, finishing);
#endif
    FeatureWindow result{};
    const auto waveform_length = real_samples + (finishing ? tail_silence_samples : 0);
    for (std::size_t frame = 0; frame < window_frames; ++frame) {
        const auto frame_start = static_cast<std::int64_t>((start_frame + frame) * samples_per_frame)
            - static_cast<std::int64_t>(frame_left_context);
        std::array<float, frame_size> input{};
        float sum = 0;
        for (std::size_t index = 0; index < frame_size; ++index) {
            const auto in_wave = frame_start + static_cast<std::int64_t>(index);
            auto reflected = static_cast<std::uint64_t>(in_wave >= 0 ? in_wave : -in_wave - 1);
            if (reflected >= waveform_length) {
                reflected = waveform_length - 1 - (reflected - waveform_length);
            }
            if (reflected < real_samples) {
                input[index] = static_cast<float>(pcm.at(reflected - pcm_offset)) / 32768.0F;
            }
            sum += input[index];
        }
        const auto mean = sum / frame_size;
        for (auto& value : input) {
            value -= mean;
        }
        for (std::size_t index = frame_size - 1; index > 0; --index) {
            input[index] -= 0.97F * input[index - 1];
        }
        input[0] -= 0.97F * input[0];

        std::array<std::complex<float>, fft_size> bins{};
        for (std::size_t index = 0; index < frame_size; ++index) {
            bins[bit_reverse_[index]] = input[index] * window_[index];
        }
        for (std::size_t width = 2; width <= fft_size; width <<= 1) {
            const auto half = width / 2;
            for (std::size_t block = 0; block < fft_size; block += width) {
                for (std::size_t index = 0; index < half; ++index) {
                    const auto even = bins[block + index];
                    const auto odd = bins[block + index + half] * roots_[index * (fft_size / width)];
                    bins[block + index] = even + odd;
                    bins[block + index + half] = even - odd;
                }
            }
        }
        std::array<float, fft_size / 2> energies{};
        for (std::size_t index = 0; index < energies.size(); ++index) {
            energies[index] = std::norm(bins[index]);
        }
        for (std::size_t bin = 0; bin < feature_dim; ++bin) {
            float energy = 0;
            for (const auto& weight : filters_[bin]) {
                energy += weight.second * energies[weight.first];
            }
            result[frame * feature_dim + bin] =
                std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
        }
    }
    return result;
}

}  // namespace edge_speech::asr

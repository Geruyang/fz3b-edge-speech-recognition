#include "board/audio/resampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace edge_speech::audio {

Resampler::Resampler(unsigned source_rate) {
    if (source_rate != 16000 && source_rate != 44100 && source_rate != 48000) {
        throw std::invalid_argument("unsupported native sample rate");
    }
    const auto divisor = std::gcd(source_rate, output_rate);
    step_ = source_rate / divisor;
    phases_ = output_rate / divisor;
    if (source_rate == output_rate) return;

    // 7 kHz cutoff leaves room for the transition to the 8 kHz Nyquist limit.
    constexpr double pi = 3.14159265358979323846;
    const double cutoff = 14000.0 / source_rate;
    kernels_.resize(phases_);
    for (unsigned phase = 0; phase < phases_; ++phase) {
        auto& kernel = kernels_[phase];
        const double fraction = static_cast<double>(phase) / phases_;
        double sum = 0;
        for (int tap = -half_width; tap <= half_width; ++tap) {
            const double distance = tap - fraction;
            const double x = distance / (half_width + 1);
            const double window = 0.42 + 0.5 * std::cos(pi * x) + 0.08 * std::cos(2 * pi * x);
            const double sinc = distance == 0 ? cutoff : std::sin(pi * cutoff * distance) / (pi * distance);
            const double coefficient = window * sinc;
            kernel[tap + half_width] = coefficient;
            sum += coefficient;
        }
        for (double& coefficient : kernel) coefficient /= sum;
    }
}

void Resampler::push(const std::int16_t* samples, std::size_t count,
                     std::vector<std::int16_t>& output) {
    if (finished_) throw std::logic_error("resampler has already been drained");
    if (count == 0) return;
    if (samples == nullptr) throw std::invalid_argument("missing native samples");
    if (kernels_.empty()) {
        output.insert(output.end(), samples, samples + count);
        return;
    }
    history_.insert(history_.end(), samples, samples + count);
    received_ += count;
    produce(output, false);
}

void Resampler::finish(std::vector<std::int16_t>& output) {
    if (finished_) return;
    if (!kernels_.empty()) produce(output, true);
    finished_ = true;
    history_.clear();
}

void Resampler::produce(std::vector<std::int16_t>& output, bool drain) {
    const auto target = (received_ / step_) * phases_ + (received_ % step_) * phases_ / step_;
    while (emitted_ < target && (drain || received_ - center_ > half_width)) {
        double value = 0;
        const auto& coefficients = kernels_[phase_];
        for (int tap = -half_width; tap <= half_width; ++tap) {
            const auto position = static_cast<std::int64_t>(center_) + tap;
            const auto index = std::min(position > 0 ? static_cast<std::uint64_t>(position) : 0,
                                        received_ - 1);
            value += coefficients[tap + half_width] * history_[index - first_stored_];
        }
        output.push_back(static_cast<std::int16_t>(std::clamp(std::lround(value), -32768L, 32767L)));
        ++emitted_;
        advance();
    }
}

void Resampler::advance() {
    phase_ += step_;
    center_ += phase_ / phases_;
    phase_ %= phases_;
    while (first_stored_ + half_width < center_) {
        history_.pop_front();
        ++first_stored_;
    }
}

}  // namespace edge_speech::audio

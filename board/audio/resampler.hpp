#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace edge_speech::audio {

inline constexpr unsigned output_rate = 16000;
inline constexpr std::size_t samples_per_block = output_rate / 50;
inline constexpr char normalizer_release[] = "sinc-blackman-65-v1";

// Accepts mono PCM16. Output sample times stay phase-aligned across pushes.
// finish() emits the remaining valid time positions and may be called repeatedly.
class Resampler {
public:
    explicit Resampler(unsigned source_rate);
    void push(const std::int16_t* samples, std::size_t count, std::vector<std::int16_t>& output);
    void finish(std::vector<std::int16_t>& output);

private:
    static constexpr int half_width = 32;
    using Kernel = std::array<double, 2 * half_width + 1>;

    void produce(std::vector<std::int16_t>& output, bool drain);
    void advance();

    unsigned step_ = 0;
    unsigned phases_ = 1;
    unsigned phase_ = 0;
    std::uint64_t center_ = 0;
    std::uint64_t received_ = 0;
    std::uint64_t emitted_ = 0;
    std::uint64_t first_stored_ = 0;
    std::deque<std::int16_t> history_;
    std::vector<Kernel> kernels_;
    bool finished_ = false;
};

}  // namespace edge_speech::audio

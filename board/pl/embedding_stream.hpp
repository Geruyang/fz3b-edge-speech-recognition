#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>

namespace edge_speech::pl {

inline constexpr std::size_t embedding_input_values = 39 * 80;
inline constexpr std::size_t embedding_output_values = 16 * 256;
using EmbeddingInputBytes = std::array<std::uint8_t, 12480>;
using EmbeddingOutputBytes = std::array<std::uint8_t, 16384>;
using EmbeddingEvidenceBytes = std::array<std::uint8_t, 45392>;

struct EmbeddingIdentity {
    std::uint64_t generation = 0;
    std::uint64_t epoch = 0;
    std::uint64_t chain_id = 0;
    std::uint32_t sequence = 0;
};

struct EmbeddingExchange {
    EmbeddingIdentity identity;
    EmbeddingOutputBytes output{};
    EmbeddingEvidenceBytes evidence{};
};

class EmbeddingStreamError : public std::runtime_error {
public:
    explicit EmbeddingStreamError(const std::string& message, std::error_code code = {});
    std::error_code code() const noexcept { return code_; }

private:
    std::error_code code_;
};

// Internal wire contract of a single lease. Only EmbeddingStream publishes verified FP32 results.
class EmbeddingTransport {
public:
    virtual ~EmbeddingTransport() = default;
    virtual EmbeddingExchange exchange(std::uint64_t chain, const EmbeddingInputBytes& input) = 0;
    virtual void close() noexcept = 0;
};

// The board's maintenance owner must establish the ready window before this open.
std::unique_ptr<EmbeddingTransport> open_linux_embedding_stream(
    const std::filesystem::path& device = "/dev/fzspeech-embedding-stream");
// Takes the directory containing capture.json and the raw physical stream export.
std::unique_ptr<EmbeddingTransport> open_embedding_capture(const std::filesystem::path& directory);

struct EmbeddingResult {
    std::array<float, embedding_output_values> output;
    EmbeddingIdentity identity;
};

// Owns one FZASR015 Embedding lease. submit serializes SUBMIT, GET and verification.
// Errors permanently close this lease. close waits for a concurrent submit to return.
class EmbeddingStream {
public:
    explicit EmbeddingStream(std::unique_ptr<EmbeddingTransport> transport);
    ~EmbeddingStream();

    EmbeddingStream(const EmbeddingStream&) = delete;
    EmbeddingStream& operator=(const EmbeddingStream&) = delete;

    EmbeddingResult submit(const float* features, std::size_t count);
    void close() noexcept;

private:
    struct State;
    void close_locked() noexcept;

    std::mutex mutex_;
    std::unique_ptr<State> state_;
    std::unique_ptr<EmbeddingTransport> transport_;
};

}  // namespace edge_speech::pl

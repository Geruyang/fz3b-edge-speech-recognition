#include "board/pl/embedding_stream.hpp"

#include "board/session/coordinator.hpp"
#include "driver/fzspeech_embedding_job.h"
#include "driver/fzspeech_job_regs.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#ifdef __linux__
#include "driver/uapi/fzspeech_embedding_stream.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace edge_speech::pl {
namespace {

constexpr std::size_t receipt_header_bytes = 128;
constexpr std::size_t registers_bytes = 3 * FZ_SQ_MIN_MMIO_BYTES;
constexpr std::size_t packet_start = receipt_header_bytes + registers_bytes;
constexpr std::size_t packet_capacity = FZ_JOB_MAX_PACKET_BYTES;
constexpr std::size_t spans_count = 176;
constexpr std::size_t packet_bytes = FZ_JOB_HEADER_BYTES + spans_count * FZ_JOB_ENTRY_BYTES;
constexpr std::size_t weight_mapping_bytes = 170 * FZ_JOB_ENTRY_BYTES;
constexpr std::size_t guard_start = packet_start + packet_capacity;
constexpr std::size_t page_bytes = 4096;
constexpr std::size_t guards_bytes = 4 * page_bytes;
constexpr std::uint64_t dma_limit = 1ULL << 30;
constexpr std::uint64_t descriptor_bytes = 3 * page_bytes;
constexpr std::uint32_t profile = 2, mask = 9;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(guard_start + 2 * guards_bytes == EmbeddingEvidenceBytes{}.size());

void require(bool condition, const char* message) {
    if (!condition) throw EmbeddingStreamError(message);
}

std::string sha(const std::uint8_t* bytes, std::size_t size) {
    return session::sha256(std::string_view(reinterpret_cast<const char*>(bytes), size));
}

template <std::size_t N>
std::string sha(const std::array<std::uint8_t, N>& bytes) {
    return sha(bytes.data(), bytes.size());
}

std::string hex(const std::uint8_t* bytes, std::size_t size) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (std::size_t i = 0; i < size; ++i) {
        result[2 * i] = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
}

class RegisterSnapshots {
public:
    explicit RegisterSnapshots(const EmbeddingEvidenceBytes& bytes) : bytes_(bytes) {}

    std::uint32_t word(unsigned stage, unsigned offset) const {
        return fz_job_get32(bytes_.data() + receipt_header_bytes + stage * FZ_SQ_MIN_MMIO_BYTES + offset);
    }
    std::uint64_t wide(unsigned stage, unsigned offset) const {
        return word(stage, offset) | (std::uint64_t(word(stage, offset + 4)) << 32);
    }
    void words(unsigned offset, std::array<std::uint32_t, 3> expected) const {
        require(expected == std::array{word(0, offset), word(1, offset), word(2, offset)},
                "unexpected Embedding lifecycle register");
    }
    void wides(unsigned offset, std::array<std::uint64_t, 3> expected) const {
        require(expected == std::array{wide(0, offset), wide(1, offset), wide(2, offset)},
                "unexpected Embedding lifecycle identity");
    }

private:
    const EmbeddingEvidenceBytes& bytes_;
};

using PortCounts = std::array<std::array<std::uint64_t, 2>, 4>;

struct VerifiedJob {
    EmbeddingIdentity id;
    PortCounts before{}, after{};
    std::array<std::uint8_t, weight_mapping_bytes> weights;
    std::uint64_t counter_epoch;
    std::uint32_t irq, snapshot_first, snapshot_last;
};

void check_registers(const RegisterSnapshots& r, VerifiedJob& job,
                     std::uint64_t descriptor, std::uint32_t crc) {
    const auto [generation, epoch, chain, sequence] = job.id;
    r.words(FZ_TR_ID, {FZ_TR_CORE_ID, FZ_TR_CORE_ID, FZ_TR_CORE_ID});
    r.words(FZ_TR_VERSION, {FZ_TR_ABI, FZ_TR_ABI, FZ_TR_ABI});
    r.words(FZ_SQ_ID, {FZ_JOB_MAGIC, FZ_JOB_MAGIC, FZ_JOB_MAGIC});
    r.words(FZ_SQ_ABI, {2, 2, 2});
    r.words(FZ_SQ_CAP, {7, 7, 7});
    r.words(FZ_SQ_DMA_BITS, {32, 32, 32});
    r.words(FZ_TR_PROFILE_ID, {profile, profile, profile});
    r.words(FZ_TR_PROFILE_MASK, {mask, mask, mask});
    r.wides(FZ_TR_PROFILE_EPOCH_LO, {epoch, epoch, epoch});
    r.wides(FZ_TR_GENERATION_LO, {generation, generation, generation});
    r.words(FZ_TR_RUN, {0, 1, 0});
    r.wides(FZ_TR_CHAIN_LO, {0, chain, 0});
    r.words(FZ_TR_STATUS, {0, FZ_STATUS_SUCCESS, 0});
    constexpr auto stopped = FZ_TR_FLAG_ENGINE_IDLE | FZ_TR_FLAG_ISSUANCE_STOPPED;
    r.words(FZ_TR_FLAGS, {stopped, FZ_TR_FLAG_ENGINE_IDLE, stopped});
    r.words(FZ_TR_FIFO_EMPTY_MASK, {15, 15, 15});
    r.words(FZ_SQ_RESULT, {0, 0, 0});
    r.words(FZ_SQ_FLAGS, {0, FZ_SQ_DONE_VALID, 0});
    r.words(FZ_SQ_STATUS, {0, FZ_STATUS_SUCCESS, 0});
    r.words(FZ_SQ_ADMISSION, {0, 0, 0});
    r.words(FZ_SQ_IRQ_ENABLE, {0, 0, 0});
    r.wides(FZ_SQ_GENERATION_LO, {generation, generation, generation});
    r.wides(FZ_SQ_CHAIN_LO, {0, chain, 0});
    r.words(FZ_SQ_DONE_SEQUENCE, {sequence - 1, sequence, sequence});

    require(r.wide(1, FZ_SQ_GENERATION_REQ_LO) == generation &&
            r.wide(1, FZ_SQ_CHAIN_REQ_LO) == chain && r.wide(1, FZ_SQ_EPOCH_REQ_LO) == epoch &&
            r.word(1, FZ_SQ_PROFILE_ID_REQ) == profile && r.word(1, FZ_SQ_PROFILE_MASK_REQ) == mask,
            "wrong published Embedding identity");
    for (unsigned stage : {1U, 2U}) {
        require(r.wide(stage, FZ_SQ_COMPLETED_GENERATION_LO) == generation &&
                r.wide(stage, FZ_SQ_COMPLETED_CHAIN_LO) == chain &&
                r.wide(stage, FZ_SQ_COMPLETED_EPOCH_LO) == epoch &&
                r.word(stage, FZ_SQ_COMPLETED_PROFILE_ID) == profile &&
                r.word(stage, FZ_SQ_COMPLETED_PROFILE_MASK) == mask &&
                r.word(stage, FZ_SQ_COMPLETED_STATUS) == FZ_STATUS_SUCCESS &&
                r.word(stage, FZ_SQ_COMPLETED_ERROR) == 0 &&
                r.word(stage, FZ_SQ_FAILED_SPAN) == 0xffffffffU, "wrong completed Embedding identity or status");
        require(r.wide(stage, FZ_SQ_PACKET_LO) == descriptor &&
                r.word(stage, FZ_SQ_PACKET_BYTES) == packet_bytes &&
                r.word(stage, FZ_SQ_SPAN_COUNT) == spans_count &&
                r.word(stage, FZ_SQ_PACKET_CRC) == crc, "registers and original DMA packet differ");
    }
    require(r.wide(2, FZ_SQ_ACK_CHAIN_LO) == chain &&
            r.wide(2, FZ_SQ_ACK_GENERATION_LO) == generation, "unmatched Embedding ACK");
    job.snapshot_first = r.word(0, FZ_TR_SNAPSHOT_SEQUENCE);
    job.snapshot_last = r.word(2, FZ_TR_SNAPSHOT_SEQUENCE);
    const auto held = r.word(1, FZ_TR_SNAPSHOT_SEQUENCE);
    require(job.snapshot_first < held && held < job.snapshot_last, "nonmonotonic Embedding snapshots");
    job.counter_epoch = r.wide(0, FZ_TR_COUNTER_EPOCH_LO);
    r.wides(FZ_TR_COUNTER_EPOCH_LO, {job.counter_epoch, job.counter_epoch, job.counter_epoch});

    for (unsigned hp = 0; hp < job.before.size(); ++hp) {
        const auto port = FZ_TR_PORT_BASE(hp);
        r.words(port + FZ_TR_PORT_READ_OUTSTANDING, {0, 0, 0});
        r.words(port + FZ_TR_PORT_WRITE_OUTSTANDING, {0, 0, 0});
        r.words(port + FZ_TR_PORT_PENDING, {0, 0, 0});
        for (unsigned direction = 0; direction < 2; ++direction) {
            const auto reg = port + (direction == 0 ? FZ_TR_PORT_READ_BYTES : FZ_TR_PORT_WRITE_BYTES);
            auto& start = job.before[hp][direction];
            auto& end = job.after[hp][direction];
            start = r.wide(0, reg);
            end = r.wide(2, reg);
            const auto delta = hp != 0 ? 0 : direction == 0 ? 6770688 + packet_bytes : EmbeddingOutputBytes{}.size();
            require(end == r.wide(1, reg) && end >= start && end - start == delta,
                    "unexpected Embedding port traffic");
        }
    }
}

void check_packet(const std::uint8_t* packet, const EmbeddingIdentity& id,
                  std::uint64_t descriptor, std::uint32_t crc) {
    require(descriptor % page_bytes == 0 && descriptor < dma_limit &&
            descriptor_bytes <= dma_limit - descriptor, "invalid DMA descriptor allocation");
    require(fz_job_crc32(packet, packet_bytes) == crc, "Embedding packet CRC mismatch");
    constexpr std::uint64_t extents[] = {688128, 1696, 1696, 12480, 16384};
    std::array<fz_job_span, spans_count> spans;
    std::array<std::pair<std::uint64_t, std::uint64_t>, spans_count + 1> allocations;
    allocations[0] = {descriptor, descriptor + descriptor_bytes};
    std::size_t region = 0;
    std::uint64_t used = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        auto field = packet + FZ_JOB_HEADER_BYTES + i * FZ_JOB_ENTRY_BYTES;
        auto& span = spans[i];
        span = {fz_job_get64(field), fz_job_get64(field + 8), fz_job_get64(field + 16),
                fz_job_get32(field + 24), fz_job_get32(field + 28)};
        require(region < std::size(extents), "extra Embedding span");
        const bool output = region == 4;
        const auto bytes = std::min<std::uint64_t>(extents[region] - used, output ? 2 * page_bytes : page_bytes);
        require(span.bytes == bytes && span.dma_address % page_bytes == 0 && span.dma_address < dma_limit &&
                (!output || span.dma_address >= page_bytes), "invalid Embedding span geometry");
        const std::uint64_t start = span.dma_address - (output ? page_bytes : 0);
        const auto end = span.dma_address + (output ? 3 * page_bytes : page_bytes);
        require(end <= dma_limit, "Embedding allocation outside the 30-bit DMA window");
        allocations[i + 1] = {start, end};
        used += span.bytes;
        if (used == extents[region]) {
            ++region;
            used = 0;
        }
    }
    std::sort(allocations.begin(), allocations.end());
    require(std::adjacent_find(allocations.begin(), allocations.end(), [](auto a, auto b) {
                return a.second > b.first;
            }) == allocations.end(), "Embedding DMA allocations overlap");

    // Re-encode the canonical wire form, including the production layout and weight pack fingerprints.
    const fz_job_meta meta{id.generation, id.chain_id, {id.epoch, profile, mask}, 1};
    std::array<std::uint8_t, packet_bytes> canonical;
    fz_u32 written = 0;
    require(fz_embedding_job_encode(canonical.data(), canonical.size(), &meta, spans.data(), spans.size(),
                                   dma_limit - 1, descriptor, descriptor_bytes, &written) == FZ_JOB_OK &&
            written == packet_bytes && std::equal(canonical.begin(), canonical.end(), packet),
            "invalid Embedding descriptor, layout or model weight pack");
}

VerifiedJob verify(const EmbeddingExchange& reply, std::uint64_t chain) {
    const auto& id = reply.identity;
    require(id.chain_id == chain && id.sequence == chain && id.generation && id.epoch,
            "unexpected Embedding SUBMIT identity");
    const auto& detail = reply.evidence;
    const auto w = [&](std::size_t offset) { return fz_job_get32(detail.data() + offset); };
    const auto q = [&](std::size_t offset) { return fz_job_get64(detail.data() + offset); };
    const auto descriptor = q(40);
    const auto irq = w(64), crc = w(80);
    require(w(0) == 0x56455a46 && w(4) == 1 && w(8) == detail.size() && w(12) == packet_bytes &&
            w(48) == descriptor_bytes && w(76) == spans_count && w(84) == 30 &&
            w(120) == registers_bytes && w(124) == guards_bytes, "wrong FZEV v1 receipt format");
    require(q(16) == id.generation && q(24) == id.epoch && q(32) == chain && w(52) == id.sequence &&
            w(56) == profile && w(60) == mask && irq > 0 && w(68) == 1 && w(72) == 31,
            "wrong FZEV identity, IRQ, ACK, CPU sync, guards or drain");
    require(hex(detail.data() + 88, 32) == sha(reply.output), "FZEV output digest mismatch");
    const auto packet = detail.data() + packet_start;
    require(std::all_of(packet + packet_bytes, detail.data() + guard_start, [](auto b) { return b == 0; }),
            "nonzero Embedding packet padding");
    constexpr std::array<std::uint8_t, 4> guard_values{0xa1, 0xb2, 0xc3, 0xd4};
    for (std::size_t i = guard_start; i < detail.size(); i += page_bytes) {
        auto expect = guard_values[((i - guard_start) / page_bytes) % guard_values.size()];
        require(std::all_of(detail.begin() + i, detail.begin() + i + page_bytes,
                            [expect](auto b) { return b == expect; }), "changed Embedding output guard");
    }
    check_packet(packet, id, descriptor, crc);
    VerifiedJob job{};
    job.id = id;
    job.irq = irq;
    std::copy_n(packet + FZ_JOB_HEADER_BYTES, job.weights.size(), job.weights.begin());
    check_registers(RegisterSnapshots(detail), job, descriptor, crc);
    return job;
}

using Json = nlohmann::json;

std::string bounded_text(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw EmbeddingStreamError("missing capture file " + path.string());
    std::string data(limit + 1, '\0');
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    require(input.eof() && !input.bad(), "oversized or unreadable physical capture text");
    data.resize(static_cast<std::size_t>(input.gcount()));
    return data;
}

template <typename Bytes>
Bytes binary_file(const std::filesystem::path& path) {
    Bytes bytes;
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) ||
        file.peek() != std::char_traits<char>::eof() || !file.eof() || file.bad()) {
        throw EmbeddingStreamError("wrong physical capture extent " + path.string());
    }
    return bytes;
}

bool uuid(const std::string& id) {
    if (id.size() != 36 || id == "00000000-0000-0000-0000-000000000000") return false;
    for (std::size_t i = 0; i < id.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (id[i] != '-') return false;
        } else if ((id[i] < '0' || id[i] > '9') && (id[i] < 'a' || id[i] > 'f')) {
            return false;
        }
    }
    return true;
}

std::uint64_t json_uint(const Json& parent, const char* name) {
    const auto& field = parent.at(name);
    require(field.is_number_unsigned(), "invalid integer in physical capture");
    return field.get<std::uint64_t>();
}

class CaptureTransport final : public EmbeddingTransport {
public:
    explicit CaptureTransport(std::filesystem::path directory) : directory_(std::move(directory)) {
        auto manifest = Json::parse(bounded_text(directory_ / "capture.json", 2 * 1024 * 1024));
        boot_ = manifest.at("boot_id").get<std::string>();
        session_ = manifest.at("session_id").get<std::string>();
        require(json_uint(manifest, "schema_version") == 1 && manifest.at("source") == "board" &&
                uuid(boot_) && uuid(session_), "invalid physical board capture identity or source");
        require(bounded_text(directory_ / "boot.before", 37) == boot_ + '\n' &&
                bounded_text(directory_ / "boot.after", 37) == boot_ + '\n' &&
                bounded_text(directory_ / "session", 37) == session_ + '\n' &&
                bounded_text(directory_ / "closed", 3) == "ok\n",
                "physical capture did not close under the same boot and lease");
        generation_ = json_uint(manifest, "generation");
        epoch_ = json_uint(manifest, "epoch");
        responses_ = std::move(manifest.at("responses"));
        require(generation_ && epoch_ && responses_.is_array() && !responses_.empty(),
                "invalid physical capture lease");
    }

    EmbeddingExchange exchange(std::uint64_t chain, const EmbeddingInputBytes& input) override {
        require(!closed_ && chain && chain <= responses_.size(), "physical capture ended");
        const auto index = chain - 1;
        const auto& row = responses_[static_cast<std::size_t>(index)];
        require(json_uint(row, "chunk") == index && json_uint(row, "chain_id") == chain &&
                json_uint(row, "sequence") == chain && json_uint(row, "generation") == generation_ &&
                json_uint(row, "epoch") == epoch_ && row.at("session_id") == session_ &&
                json_uint(row, "status") == 0 && row.at("completed") == true &&
                row.at("matched_ack") == true && row.at("drained") == true, "wrong physical capture job identity");
        const auto name = (index < 10 ? "chunk-0" : "chunk-") + std::to_string(index);
        std::ostringstream receipt;
        receipt << index << ' ' << generation_ << ' ' << epoch_ << ' ' << chain << ' ' << chain << '\n';
        require(bounded_text(directory_ / (name + ".receipt"), 128) == receipt.str(),
                "physical capture receipt identity mismatch");
        require(binary_file<EmbeddingInputBytes>(directory_ / (name + ".input")) == input &&
                row.at("input_sha256") == sha(input), "runtime input differs from the physical capture");

        EmbeddingExchange reply;
        reply.identity = {generation_, epoch_, chain, static_cast<std::uint32_t>(chain)};
        reply.output = read_job_file<EmbeddingOutputBytes>(row.at("output"), name + ".output");
        reply.evidence = read_job_file<EmbeddingEvidenceBytes>(row.at("detail"), name + ".detail");
        return reply;
    }

    void close() noexcept override { closed_ = true; }

private:
    template <typename Bytes>
    Bytes read_job_file(const Json& metadata, const std::string& name) {
        require(metadata.at("file") == name && json_uint(metadata, "bytes") == Bytes{}.size(),
                "wrong physical capture response metadata");
        auto bytes = binary_file<Bytes>(directory_ / name);
        require(metadata.at("sha256") == sha(bytes), "physical capture response digest mismatch");
        return bytes;
    }

    std::filesystem::path directory_;
    std::string boot_, session_;
    std::uint64_t generation_, epoch_;
    Json responses_;
    bool closed_ = false;
};

#ifdef __linux__
static_assert(sizeof(fz_embedding_stream_request) == 56 && sizeof(fz_embedding_stream_evidence_request) == 48);
static_assert(sizeof(fz_embedding_stream_evidence) == EmbeddingEvidenceBytes{}.size());
static_assert(FZ_EMBED_STREAM_INPUT_BYTES == EmbeddingInputBytes{}.size());
static_assert(FZ_EMBED_STREAM_OUTPUT_BYTES == EmbeddingOutputBytes{}.size());
static_assert(offsetof(fz_embedding_stream_evidence, packet) == packet_start);
static_assert(offsetof(fz_embedding_stream_evidence, guards_before) == guard_start);

[[noreturn]] void system_failure(const char* operation, int error) {
    auto code = std::error_code(error, std::generic_category());
    throw EmbeddingStreamError(std::string(operation) + ": " + code.message(), code);
}

class LinuxTransport final : public EmbeddingTransport {
public:
    explicit LinuxTransport(const std::filesystem::path& device) {
        fd_ = ::open(device.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0) system_failure("open Embedding stream", errno);
        struct stat info{};
        const int rc = ::fstat(fd_, &info);
        if (rc != 0 || !S_ISCHR(info.st_mode)) {
            const int error = rc != 0 ? errno : ENODEV;
            close();
            system_failure("inspect Embedding stream character device", error);
        }
    }
    ~LinuxTransport() override { close(); }

    EmbeddingExchange exchange(std::uint64_t chain, const EmbeddingInputBytes& input) override {
        require(fd_ >= 0, "Linux Embedding lease is closed");
        EmbeddingExchange reply;
        fz_embedding_stream_request submit{};
        submit.chain_id = chain;
        submit.input_ptr = reinterpret_cast<std::uintptr_t>(input.data());
        submit.output_ptr = reinterpret_cast<std::uintptr_t>(reply.output.data());
        submit.input_bytes = input.size();
        submit.output_bytes = reply.output.size();
        if (::ioctl(fd_, FZ_EMBED_STREAM_SUBMIT, &submit) != 0) system_failure("Embedding SUBMIT", errno);
        require(submit.chain_id == chain && submit.sequence == chain && submit.generation && submit.epoch,
                "wrong Embedding SUBMIT reply identity");
        reply.identity = {submit.generation, submit.epoch, submit.chain_id, submit.sequence};

        fz_embedding_stream_evidence_request get{};
        get.generation = submit.generation;
        get.epoch = submit.epoch;
        get.chain_id = submit.chain_id;
        get.sequence = submit.sequence;
        get.evidence_ptr = reinterpret_cast<std::uintptr_t>(reply.evidence.data());
        get.evidence_bytes = reply.evidence.size();
        if (::ioctl(fd_, FZ_EMBED_STREAM_EVIDENCE, &get) != 0) system_failure("Embedding GET evidence", errno);
        return reply;
    }

    void close() noexcept override {
        // The driver's character-device release cancels the single lease.
        if (fd_ >= 0) ::close(std::exchange(fd_, -1));
    }

private:
    int fd_ = -1;
};
#endif
}  // namespace

EmbeddingStreamError::EmbeddingStreamError(const std::string& message, std::error_code code)
    : std::runtime_error(message), code_(code) {}

std::unique_ptr<EmbeddingTransport> open_linux_embedding_stream(const std::filesystem::path& device) {
#ifdef __linux__
    return std::make_unique<LinuxTransport>(device);
#else
    (void)device;
    throw EmbeddingStreamError("the live Embedding device requires Linux");
#endif
}

std::unique_ptr<EmbeddingTransport> open_embedding_capture(const std::filesystem::path& directory) {
    try {
        return std::make_unique<CaptureTransport>(directory);
    } catch (const Json::exception& error) {
        throw EmbeddingStreamError(std::string("invalid physical capture manifest: ") + error.what());
    }
}

struct EmbeddingStream::State {
    std::uint64_t next_chain = 1;
    VerifiedJob previous{};
};

EmbeddingStream::EmbeddingStream(std::unique_ptr<EmbeddingTransport> transport)
    : state_(std::make_unique<State>()), transport_(std::move(transport)) {
    require(static_cast<bool>(transport_), "missing Embedding transport");
}
EmbeddingStream::~EmbeddingStream() { close(); }

EmbeddingResult EmbeddingStream::submit(const float* features, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    require(static_cast<bool>(transport_), "Embedding lease is closed");
    try {
        require(features && count == embedding_input_values, "expected 39 by 80 FP32 Embedding input");
        const auto chain = state_->next_chain;
        require(chain <= std::numeric_limits<std::uint32_t>::max(), "Embedding sequence exhausted");
        EmbeddingInputBytes input;
        for (std::size_t i = 0; i < count; ++i) {
            require(std::isfinite(features[i]), "nonfinite FP32 Embedding input");
            std::uint32_t bits;
            std::memcpy(&bits, features + i, sizeof(bits));
            fz_job_put32(input.data() + sizeof(bits) * i, bits);
        }

        const auto raw = transport_->exchange(chain, input);
        const auto job = verify(raw, chain);
        const auto& last = state_->previous;
        if (chain == 1) {
            require(job.before == PortCounts{}, "Embedding lease started with prior port traffic");
        } else {
            require(job.id.generation == last.id.generation && job.id.epoch == last.id.epoch &&
                    job.irq == last.irq && job.counter_epoch == last.counter_epoch &&
                    job.weights == last.weights && job.snapshot_first > last.snapshot_last &&
                    job.before == last.after, "Embedding lease, weights, IRQ or counters changed between jobs");
        }
        EmbeddingResult result;
        result.identity = job.id;
        for (std::size_t i = 0; i < result.output.size(); ++i) {
            const std::uint32_t bits = fz_job_get32(raw.output.data() + sizeof(std::uint32_t) * i);
            std::memcpy(result.output.data() + i, &bits, sizeof(bits));
            require(std::isfinite(result.output[i]), "nonfinite FP32 Embedding output");
        }
        state_->previous = job;
        ++state_->next_chain;
        return result;
    } catch (const EmbeddingStreamError&) {
        close_locked();
        throw;
    } catch (const std::exception& error) {
        close_locked();
        throw EmbeddingStreamError(error.what());
    } catch (...) {
        close_locked();
        throw;
    }
}

void EmbeddingStream::close_locked() noexcept {
    if (transport_) {
        transport_->close();
        transport_.reset();
    }
}

void EmbeddingStream::close() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    close_locked();
}

}  // namespace edge_speech::pl

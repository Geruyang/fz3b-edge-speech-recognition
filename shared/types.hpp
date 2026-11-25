#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace edge_speech {

inline std::uint64_t checked_extent(
    std::uint64_t start, std::uint64_t count,
    std::uint64_t limit = std::numeric_limits<std::uint64_t>::max()) {
    if (start > limit || count > limit - start) {
        throw std::invalid_argument("extent exceeds limit");
    }
    return start + count;
}

struct TimingEvent {
    std::string name, clock_domain;
    std::uint64_t monotonic_ns;
};

inline std::uint64_t elapsed_ns(const TimingEvent& start, const TimingEvent& end) {
    if (start.clock_domain.empty() || start.clock_domain != end.clock_domain ||
        end.monotonic_ns < start.monotonic_ns) {
        throw std::invalid_argument("incompatible timing events");
    }
    return end.monotonic_ns - start.monotonic_ns;
}

enum class MessageType : std::uint16_t {
    hello = 1, request, response, event, pcm, file_chunk, ack, heartbeat
};

using Bytes = std::vector<std::uint8_t>;
using WireUuid = std::array<std::uint8_t, 16>;

namespace detail {

inline bool is_lower_hex(char digit) {
    return ('0' <= digit && digit <= '9') || ('a' <= digit && digit <= 'f');
}

}  // namespace detail

inline bool is_uuid(std::string_view value) {
    if (value.size() != 36) {
        return false;
    }
    for (std::size_t index = 0; index < value.size(); ++index) {
        const bool separator = index == 8 || index == 13 || index == 18 || index == 23;
        if (separator ? value[index] != '-' : !detail::is_lower_hex(value[index])) {
            return false;
        }
    }
    return true;
}

inline constexpr std::size_t fzas_header_bytes = 32;
inline constexpr std::size_t pcm_header_bytes = 32;
inline constexpr std::uint32_t fzas_json_payload_limit = 64 * 1024;
inline constexpr std::uint32_t fzas_binary_payload_limit = 256 * 1024;
inline constexpr std::array<std::uint8_t, 4> fzas_magic{'F', 'Z', 'A', 'S'};

struct Frame {
    MessageType message_type;
    std::uint64_t transport_seq;
    Bytes payload;
    std::uint32_t flags = 0;
};

struct FrameHeader {
    MessageType message_type;
    std::uint32_t payload_bytes;
    std::uint64_t transport_seq;
    std::uint32_t payload_crc32, flags;
};

namespace detail {

template <typename UInt>
inline UInt read_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
        value |= std::uint64_t{data[byte]} << (8 * byte);
    }
    return static_cast<UInt>(value);
}

template <typename UInt>
inline void write_le(std::uint8_t* data, UInt value) {
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
        data[byte] = static_cast<std::uint8_t>(value >> (8 * byte));
    }
}

inline std::uint32_t payload_limit(MessageType type) {
    switch (type) {
    case MessageType::pcm:
    case MessageType::file_chunk:
        return fzas_binary_payload_limit;
    case MessageType::hello:
    case MessageType::request:
    case MessageType::response:
    case MessageType::event:
    case MessageType::ack:
    case MessageType::heartbeat:
        return fzas_json_payload_limit;
    }
    throw std::invalid_argument("unsupported FZAS message type");
}

inline std::uint32_t crc32(const std::uint8_t* bytes, std::size_t size) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= bytes[index];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
        }
    }
    return ~crc;
}

inline void validate_pcm(const std::uint8_t* bytes, std::size_t size) {
    if (size < pcm_header_bytes) {
        throw std::invalid_argument("truncated PCM header");
    }
    const auto first = read_le<std::uint64_t>(bytes + 16);
    const auto count = read_le<std::uint32_t>(bytes + 24);
    if (size != pcm_header_bytes + 2 * std::uint64_t{count}) {
        throw std::invalid_argument("PCM sample_count does not match payload length");
    }
    checked_extent(first, count);
}

}  // namespace detail

inline FrameHeader decode_frame_header(const std::uint8_t* data, std::size_t size) {
    using detail::read_le;
    if (size != fzas_header_bytes) {
        throw std::invalid_argument("invalid FZAS header length");
    }
    if (!std::equal(fzas_magic.begin(), fzas_magic.end(), data) ||
        read_le<std::uint16_t>(data + 4) != 1 || read_le<std::uint32_t>(data + 28) != 0) {
        throw std::invalid_argument("invalid FZAS magic, version, or reserved field");
    }
    FrameHeader header{
        static_cast<MessageType>(read_le<std::uint16_t>(data + 6)),
        read_le<std::uint32_t>(data + 8), read_le<std::uint64_t>(data + 12),
        read_le<std::uint32_t>(data + 20), read_le<std::uint32_t>(data + 24),
    };
    if (header.payload_bytes > detail::payload_limit(header.message_type)) {
        throw std::invalid_argument("FZAS payload exceeds message limit");
    }
    return header;
}

inline Bytes encode_frame(const Frame& frame) {
    using detail::write_le;
    if (frame.payload.size() > detail::payload_limit(frame.message_type)) {
        throw std::invalid_argument("FZAS payload exceeds message limit");
    }
    if (frame.message_type == MessageType::pcm) {
        detail::validate_pcm(frame.payload.data(), frame.payload.size());
    }
    Bytes wire(fzas_header_bytes + frame.payload.size());
    std::copy(fzas_magic.begin(), fzas_magic.end(), wire.begin());
    write_le(wire.data() + 4, std::uint16_t{1});
    write_le(wire.data() + 6, static_cast<std::uint16_t>(frame.message_type));
    write_le(wire.data() + 8, static_cast<std::uint32_t>(frame.payload.size()));
    write_le(wire.data() + 12, frame.transport_seq);
    write_le(wire.data() + 20, detail::crc32(frame.payload.data(), frame.payload.size()));
    write_le(wire.data() + 24, frame.flags);
    std::copy(frame.payload.begin(), frame.payload.end(), wire.begin() + fzas_header_bytes);
    return wire;
}

inline Frame decode_frame(const Bytes& wire) {
    const auto header = decode_frame_header(wire.data(), std::min(fzas_header_bytes, wire.size()));
    if (wire.size() - fzas_header_bytes != header.payload_bytes) {
        throw std::invalid_argument("FZAS payload length mismatch");
    }
    const auto* payload = wire.data() + fzas_header_bytes;
    if (detail::crc32(payload, header.payload_bytes) != header.payload_crc32) {
        throw std::invalid_argument("FZAS payload CRC32 mismatch");
    }
    if (header.message_type == MessageType::pcm) {
        detail::validate_pcm(payload, header.payload_bytes);
    }
    return {header.message_type, header.transport_seq,
            Bytes(wire.begin() + fzas_header_bytes, wire.end()), header.flags};
}

enum class Backend { cpu, pl, mixed };
enum class Integrity { pending, complete, with_gaps, failed };

struct AudioBlock {
    WireUuid session_id;
    std::uint64_t sample_start;
    std::uint32_t sample_count, audio_flags;
    Bytes pcm_s16le;
};

struct TranscriptEvent {
    std::uint64_t event_id;
    std::string session_id;
    std::uint64_t segment_id, revision;
    bool is_final;
    std::uint64_t sample_start, sample_end;
    std::string text, model_release;
    Backend backend;
};

inline void validate_transcript_revision(
    const TranscriptEvent& event, const TranscriptEvent* previous_segment = nullptr,
    std::optional<std::uint64_t> session_event_watermark = std::nullopt,
    std::uint64_t capture_end_sample = std::numeric_limits<std::uint64_t>::max()) {
    if (!is_uuid(event.session_id)) {
        throw std::invalid_argument("invalid session_id");
    }
    if (event.sample_start > event.sample_end || event.sample_end > capture_end_sample) {
        throw std::invalid_argument("transcript sample range is invalid");
    }
    if (session_event_watermark && event.event_id <= *session_event_watermark) {
        throw std::invalid_argument("session event_id did not advance");
    }
    if (previous_segment) {
        if (previous_segment->session_id != event.session_id ||
            previous_segment->segment_id != event.segment_id) {
            throw std::invalid_argument("previous_segment must describe the same session and segment");
        }
        if (event.event_id <= previous_segment->event_id) {
            throw std::invalid_argument("session event_id did not advance");
        }
        if (previous_segment->is_final || event.revision <= previous_segment->revision) {
            throw std::invalid_argument("segment revision did not advance or segment is final");
        }
    }
}

struct SegmentRevision {
    std::uint64_t segment_id, revision;
};

struct SessionSnapshot {
    std::string snapshot_id, session_id;
    std::uint32_t schema_version;
    std::vector<SegmentRevision> ordered_segment_revisions;
    std::optional<std::uint64_t> stop_sample;
    std::uint64_t high_watermark;
    std::string content_sha256, created_at;
    bool frozen;
    Integrity integrity;
};

struct SourceRange {
    std::string snapshot_id, snapshot_sha256;
    std::uint64_t segment_id, segment_revision;
    std::string source_text_sha256;
    std::uint64_t codepoint_start, codepoint_end;
};

struct RefineRequest {
    std::string refinement_job_id, session_id, source_snapshot_id, source_sha256;
    std::string fragment_plan_id, fragment_plan_sha256, model_release_id;
    std::string prompt_sha256, tokenizer_sha256;
    std::uint64_t context_limit, total_prompt_tokens, output_limit;
};

struct RefineResult {
    std::string refined_segment_id, refinement_job_id, fragment_plan_id, fragment_plan_sha256;
    std::string fragment_id;
    std::uint32_t schema_version;
    std::vector<SourceRange> source_ranges;
    std::string source_text_sha256, draft_text, diff_json, protected_changes_json;
    std::string coverage_state, review_state;
};

enum class PortProfileId { hp2, hp3, hp4 };
using BankOwners = std::array<std::uint8_t, 16>;

struct PortProfileConfig {
    PortProfileId port_profile_id;
    std::uint8_t supported_hp_mask, active_hp_mask;
    std::uint64_t profile_epoch;
    BankOwners bank_to_hp;
    std::uint32_t burst_beats, read_outstanding, write_outstanding;
    std::uint64_t read_queue_depth, write_queue_depth;
};

inline PortProfileConfig validate_port_profile(const PortProfileConfig& config) {
    constexpr BankOwners hp2{0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3};
    constexpr BankOwners hp3{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3};
    constexpr BankOwners hp4{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    const auto legal_outstanding = [](std::uint32_t count) { return count == 8 || count == 16; };
    if (config.supported_hp_mask != 0xf || config.burst_beats != 16 ||
        !legal_outstanding(config.read_outstanding) || !legal_outstanding(config.write_outstanding) ||
        !config.read_queue_depth || !config.write_queue_depth) {
        throw std::invalid_argument("invalid HP capabilities or queue configuration");
    }
    if ((config.port_profile_id == PortProfileId::hp2 && config.active_hp_mask == 0x9 &&
         config.bank_to_hp == hp2) ||
        (config.port_profile_id == PortProfileId::hp3 && config.active_hp_mask == 0xb &&
         config.bank_to_hp == hp3) ||
        (config.port_profile_id == PortProfileId::hp4 && config.active_hp_mask == 0xf &&
         config.bank_to_hp == hp4)) {
        return config;
    }
    throw std::invalid_argument("unsupported HP profile mask or bank map");
}

struct AssetIdentity {
    std::string layout_id, layout_sha256, pack_version, pack_sha256;
};

inline bool is_sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), detail::is_lower_hex);
}

inline AssetIdentity validate_asset_identity(
    const AssetIdentity& actual, const AssetIdentity& expected) {
    if (actual.layout_id != "FZBANK16-v1" || actual.layout_id != expected.layout_id ||
        !is_sha256(actual.layout_sha256) || actual.layout_sha256 != expected.layout_sha256) {
        throw std::invalid_argument("layout identity mismatch");
    }
    if (actual.pack_version.empty() || actual.pack_version != expected.pack_version ||
        !is_sha256(actual.pack_sha256) || actual.pack_sha256 != expected.pack_sha256) {
        throw std::invalid_argument("numeric pack identity mismatch");
    }
    return actual;
}

struct JobHandle : AssetIdentity {
    std::string owner_id;
    std::uint64_t device_generation, chain_id, job_id;
    PortProfileId port_profile_id;
    std::uint8_t active_hp_mask;
    std::uint64_t profile_epoch;
    std::string ps_platform_config_sha256, ps_workload_id;
};

struct BufferExtent {
    std::uint64_t handle, offset, bytes;
};

struct AsrJobContext {
    std::string session_id;
    std::uint64_t stream_generation, chunk_id, state_epoch;
};

struct RefineJobContext {
    std::string refinement_job_id, fragment_plan_sha256, fragment_id;
};

enum class JobMode { asr_w8a8_raw_i32, asr_fused_f32, qwen_w4a8_group64_f32, dynamic_w8a8_f32 };
enum class RhsSource { static_weight, dynamic_buffer };

struct DeviceJobSpec : JobHandle {
    std::uint32_t descriptor_version;
    JobMode mode;
    std::string op_kind;
    RhsSource rhs_source;
    std::string node_id, model_release, quant_contract;
    std::uint32_t descriptor_index;
    std::vector<std::uint32_t> depends_on;
    std::optional<std::uint64_t> weight_handle;
    std::optional<BufferExtent> rhs_handle;
    BufferExtent input_handle, output_handle;
    std::optional<BufferExtent> lhs_scale_handle, rhs_scale_handle, bias_handle;
    std::uint64_t M, K, N;
    std::vector<std::pair<std::string, std::uint64_t>> strides, valid_extents;
    std::variant<AsrJobContext, RefineJobContext> context;
};

struct JobResult : JobHandle {
    std::string status;
    std::optional<std::uint64_t> output_fence;
    std::uint64_t counter_epoch;
    std::vector<TimingEvent> timing_events;
};

struct StallCycles {
    // Each channel counts valid && !ready cycles.
    std::uint64_t AR, R, AW, W, B;
};

struct PortTrafficCounters {
    std::uint64_t read_bytes, write_bytes, useful_weight_bytes;
    StallCycles handshake_stall_cycles;
    std::uint64_t read_outstanding_peak, write_outstanding_peak;
};

struct BandwidthRound {
    std::uint64_t elapsed_ns, ps_time_ns, useful_weight_bytes, counter_epoch;
    std::array<PortTrafficCounters, 4> port_traffic;
};

struct BandwidthSummary {
    double p5, median, min;
};

enum class MeasurementKind { full_scan, integrated_reader_trace, whole_model };
enum class ElapsedTimeDomain { submit_to_scan_complete, reader_window_union, decode_step_wall };

struct FullScanReport {
    static constexpr auto measurement_kind = MeasurementKind::full_scan;
    static constexpr auto elapsed_time_domain = ElapsedTimeDomain::submit_to_scan_complete;
    BandwidthSummary scan_effective_weight_Bps;
};

struct IntegratedReaderReport {
    static constexpr auto measurement_kind = MeasurementKind::integrated_reader_trace;
    static constexpr auto elapsed_time_domain = ElapsedTimeDomain::reader_window_union;
    BandwidthSummary transport_effective_weight_Bps;
};

struct WholeModelReport {
    static constexpr auto measurement_kind = MeasurementKind::whole_model;
    static constexpr auto elapsed_time_domain = ElapsedTimeDomain::decode_step_wall;
    BandwidthSummary model_weight_Bps;
};

struct BandwidthReport : AssetIdentity {
    PortProfileConfig profile;
    std::string bitstream_sha256, driver_sha256, model_release_id;
    std::uint64_t clock_hz;
    std::string ps_workload_id, ps_platform_config_sha256;
    std::vector<BandwidthRound> rounds;
    std::variant<FullScanReport, IntegratedReaderReport, WholeModelReport> measurement;
};

}  // namespace edge_speech

"""Logical contracts shared with the board and fixed FZAS v1 framing."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import IntEnum
import math
import re
from statistics import median
import struct
from typing import Literal, TypedDict
from zlib import crc32


UINT64_MAX = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1
HEADER = struct.Struct("<4sHHIQIII")
PCM_HEADER = struct.Struct("<16sQII")
JSON_PAYLOAD_LIMIT = 64 * 1024
BINARY_PAYLOAD_LIMIT = 256 * 1024
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _uint(value, field, maximum=UINT64_MAX):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an unsigned integer <= {maximum}")
    return value


def _identity(value, field, pattern=None):
    if not isinstance(value, str) or not value.strip() or (pattern and not pattern.fullmatch(value)):
        raise ValueError(f"invalid {field}")
    return value


def checked_extent(start: int, count: int, limit: int = UINT64_MAX) -> int:
    """Compute the right-open end of a range within an unsigned extent."""
    _uint(start, "start")
    _uint(count, "count")
    _uint(limit, "limit")
    if start > limit or count > limit - start:
        raise ValueError("extent exceeds limit")
    return start + count


@dataclass(frozen=True)
class TimingEvent:
    name: str
    clock_domain: str
    monotonic_ns: int


def elapsed_ns(start: TimingEvent, end: TimingEvent) -> int:
    _identity(start.clock_domain, "clock_domain")
    if start.clock_domain != end.clock_domain:
        raise ValueError("timing events use different clock domains")
    first = _uint(start.monotonic_ns, "start.monotonic_ns")
    last = _uint(end.monotonic_ns, "end.monotonic_ns")
    if last < first:
        raise ValueError("timing events are reversed")
    return last - first


class MessageType(IntEnum):
    HELLO = 1
    REQUEST = 2
    RESPONSE = 3
    EVENT = 4
    PCM = 5
    FILE_CHUNK = 6
    ACK = 7
    HEARTBEAT = 8


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    transport_seq: int
    payload: bytes
    flags: int = 0


@dataclass(frozen=True)
class FrameHeader:
    message_type: MessageType
    payload_bytes: int
    transport_seq: int
    payload_crc32: int
    flags: int


def _payload_limit(message_type: MessageType) -> int:
    if message_type in (MessageType.PCM, MessageType.FILE_CHUNK):
        return BINARY_PAYLOAD_LIMIT
    return JSON_PAYLOAD_LIMIT


def decode_frame_header(header: bytes) -> FrameHeader:
    """Validate a fixed header before allocating or reading the payload."""
    if len(header) != HEADER.size:
        raise ValueError("invalid FZAS header length")
    magic, version, tag, size, seq, checksum, flags, reserved = HEADER.unpack(header)
    if magic != b"FZAS" or version != 1 or reserved != 0:
        raise ValueError("invalid FZAS magic, version, or reserved field")
    message_type = MessageType(tag)
    if size > _payload_limit(message_type):
        raise ValueError("FZAS payload exceeds message limit")
    return FrameHeader(message_type, size, seq, checksum, flags)


def _validate_pcm(payload: bytes) -> None:
    if len(payload) < PCM_HEADER.size:
        raise ValueError("truncated PCM header")
    _, start, count, _ = PCM_HEADER.unpack_from(payload)
    if len(payload) != PCM_HEADER.size + 2 * count:
        raise ValueError("PCM sample_count does not match payload length")
    checked_extent(start, count)


def encode_frame(frame: Frame) -> bytes:
    if type(frame.message_type) not in (int, MessageType):
        raise ValueError("invalid FZAS message type")
    message_type = MessageType(frame.message_type)
    seq = _uint(frame.transport_seq, "transport_seq")
    flags = _uint(frame.flags, "flags", UINT32_MAX)
    if not isinstance(frame.payload, bytes) or len(frame.payload) > _payload_limit(message_type):
        raise ValueError("invalid FZAS payload or payload length")
    if message_type == MessageType.PCM:
        _validate_pcm(frame.payload)
    header = HEADER.pack(b"FZAS", 1, message_type, len(frame.payload), seq,
                         crc32(frame.payload), flags, 0)
    return header + frame.payload


def decode_frame(wire: bytes) -> Frame:
    header = decode_frame_header(wire[:HEADER.size])
    if len(wire) - HEADER.size != header.payload_bytes:
        raise ValueError("FZAS payload length mismatch")
    payload = wire[HEADER.size:]
    if crc32(payload) != header.payload_crc32:
        raise ValueError("FZAS payload CRC32 mismatch")
    if header.message_type == MessageType.PCM:
        _validate_pcm(payload)
    return Frame(header.message_type, header.transport_seq, payload, header.flags)


@dataclass(frozen=True)
class AudioBlock:
    session_id: str
    sample_start: int
    sample_count: int
    audio_flags: int
    pcm_s16le: bytes

    def __post_init__(self):
        _identity(self.session_id, "session_id", _UUID)
        checked_extent(self.sample_start, self.sample_count)
        _uint(self.sample_count, "sample_count", UINT32_MAX)
        _uint(self.audio_flags, "audio_flags", UINT32_MAX)
        if not isinstance(self.pcm_s16le, bytes) or len(self.pcm_s16le) != 2 * self.sample_count:
            raise ValueError("PCM bytes do not match sample_count")


Backend = Literal["cpu", "pl", "mixed"]
Integrity = Literal["pending", "complete", "with_gaps", "failed"]


class TranscriptEvent(TypedDict):
    event_id: int
    session_id: str
    type: Literal["transcript"]
    segment_id: int
    revision: int
    is_final: bool
    sample_start: int
    sample_end: int
    text: str
    model_release: str
    backend: Backend


def validate_transcript_event(
        event: TranscriptEvent, *, previous_segment: TranscriptEvent | None = None,
        session_event_watermark: int | None = None,
        capture_end_sample: int = UINT64_MAX) -> TranscriptEvent:
    for field in ("event_id", "segment_id", "revision", "sample_start", "sample_end"):
        _uint(event.get(field), field)
    _identity(event.get("session_id"), "session_id", _UUID)
    _identity(event.get("model_release"), "model_release")
    if event.get("type") != "transcript" or event.get("backend") not in ("cpu", "pl", "mixed"):
        raise ValueError("invalid transcript kind or backend")
    if type(event.get("is_final")) is not bool or not isinstance(event.get("text"), str):
        raise ValueError("invalid transcript final flag or text")
    event["text"].encode("utf-8", errors="strict")
    if not event["sample_start"] <= event["sample_end"] <= _uint(capture_end_sample, "capture_end_sample"):
        raise ValueError("transcript sample range is invalid")
    if session_event_watermark is not None:
        if event["event_id"] <= _uint(session_event_watermark, "session_event_watermark"):
            raise ValueError("session event_id did not advance")
    if previous_segment is not None:
        if (previous_segment["session_id"], previous_segment["segment_id"]) != (
                event["session_id"], event["segment_id"]):
            raise ValueError("previous_segment must describe the same session and segment")
        if event["event_id"] <= previous_segment["event_id"]:
            raise ValueError("session event_id did not advance")
        if previous_segment["is_final"] or event["revision"] <= previous_segment["revision"]:
            raise ValueError("segment revision did not advance or segment is final")
    return event


class SegmentRevision(TypedDict):
    segment_id: int
    revision: int


class SessionSnapshot(TypedDict):
    snapshot_id: str
    session_id: str
    schema_version: int
    ordered_segment_revisions: list[SegmentRevision]
    stop_sample: int | None
    high_watermark: int
    content_sha256: str
    created_at: str
    frozen: bool
    integrity: Integrity


class SourceRange(TypedDict):
    snapshot_id: str
    snapshot_sha256: str
    segment_id: int
    segment_revision: int
    source_text_sha256: str
    codepoint_start: int
    codepoint_end: int


class RefineRequest(TypedDict):
    refinement_job_id: str
    session_id: str
    source_snapshot_id: str
    source_sha256: str
    fragment_plan_id: str
    fragment_plan_sha256: str
    model_release_id: str
    prompt_sha256: str
    tokenizer_sha256: str
    context_limit: int
    total_prompt_tokens: int
    output_limit: int


class RefineResult(TypedDict):
    refined_segment_id: str
    refinement_job_id: str
    fragment_plan_id: str
    fragment_plan_sha256: str
    fragment_id: str
    schema_version: int
    source_ranges: list[SourceRange]
    source_text_sha256: str
    draft_text: str
    diff: list[dict]
    protected_changes: list[dict]
    coverage_state: Literal["valid", "missing", "duplicate", "invalid"]
    review_state: Literal["unreviewed", "accepted", "rejected"]


PortProfileId = Literal["hp2", "hp3", "hp4"]
PROFILE_MAPS = {
    "hp2": (0x9, (0,) * 8 + (3,) * 8),
    "hp3": (0xB, (0,) * 6 + (1,) * 4 + (3,) * 6),
    "hp4": (0xF, (0,) * 4 + (1,) * 4 + (2,) * 4 + (3,) * 4),
}


@dataclass(frozen=True)
class PortProfileConfig:
    port_profile_id: PortProfileId
    supported_hp_mask: int
    active_hp_mask: int
    profile_epoch: int
    bank_to_hp: tuple[int, ...]
    burst_beats: int
    read_outstanding: int
    write_outstanding: int
    read_queue_depth: int
    write_queue_depth: int


def validate_port_profile(config: PortProfileConfig) -> PortProfileConfig:
    profile_id = _identity(config.port_profile_id, "port_profile_id")
    canonical = PROFILE_MAPS.get(profile_id)
    _uint(config.supported_hp_mask, "supported_hp_mask", 0xF)
    _uint(config.active_hp_mask, "active_hp_mask", 0xF)
    _uint(config.profile_epoch, "profile_epoch")
    if not isinstance(config.bank_to_hp, (tuple, list)) or len(config.bank_to_hp) != 16:
        raise ValueError("bank_to_hp must contain 16 owners")
    for owner in config.bank_to_hp:
        _uint(owner, "bank owner", 3)
    if config.supported_hp_mask != 0xF or canonical != (config.active_hp_mask, tuple(config.bank_to_hp)):
        raise ValueError("unsupported HP profile mask or bank map")
    if _uint(config.burst_beats, "burst_beats") != 16:
        raise ValueError("burst_beats must be 16")
    for field in ("read_outstanding", "write_outstanding"):
        if _uint(getattr(config, field), field) not in (8, 16):
            raise ValueError(f"{field} must be 8 or 16")
    for field in ("read_queue_depth", "write_queue_depth"):
        if not _uint(getattr(config, field), field):
            raise ValueError(f"{field} must be positive")
    if isinstance(config.bank_to_hp, tuple):
        return config
    return replace(config, bank_to_hp=tuple(config.bank_to_hp))


@dataclass(frozen=True)
class AssetIdentity:
    layout_id: str
    layout_sha256: str
    pack_version: str
    pack_sha256: str


def _validate_asset_fields(asset: AssetIdentity) -> None:
    if asset.layout_id != "FZBANK16-v1":
        raise ValueError("unsupported layout_id")
    _identity(asset.layout_sha256, "layout_sha256", _SHA256)
    _identity(asset.pack_version, "pack_version")
    _identity(asset.pack_sha256, "pack_sha256", _SHA256)


def validate_asset_identity(actual: AssetIdentity, expected: AssetIdentity) -> AssetIdentity:
    _validate_asset_fields(actual)
    _validate_asset_fields(expected)
    if (actual.layout_id, actual.layout_sha256) != (expected.layout_id, expected.layout_sha256):
        raise ValueError("layout identity mismatch")
    if (actual.pack_version, actual.pack_sha256) != (expected.pack_version, expected.pack_sha256):
        raise ValueError("numeric pack identity mismatch")
    return actual


class JobHandle(TypedDict):
    owner_id: str
    device_generation: int
    chain_id: int
    job_id: int
    port_profile_id: PortProfileId
    active_hp_mask: int
    profile_epoch: int
    layout_id: str
    layout_sha256: str
    pack_version: str
    pack_sha256: str
    ps_platform_config_sha256: str
    ps_workload_id: str


class BufferExtent(TypedDict):
    handle: int
    offset: int
    bytes: int


class AsrJobContext(TypedDict):
    session_id: str
    stream_generation: int
    chunk_id: int
    state_epoch: int


class RefineJobContext(TypedDict):
    refinement_job_id: str
    fragment_plan_sha256: str
    fragment_id: str


class DeviceJobSpec(JobHandle):
    descriptor_version: int
    mode: Literal["ASR_W8A8_RAW_I32", "ASR_FUSED_F32", "QWEN_W4A8_GROUP64_F32",
                  "DYNAMIC_W8A8_F32"]
    op_kind: str
    rhs_source: Literal["static_weight", "dynamic_buffer"]
    node_id: str
    model_release: str
    quant_contract: str
    descriptor_index: int
    depends_on: list[int]
    weight_handle: int | None
    rhs_handle: BufferExtent | None
    input_handle: BufferExtent
    output_handle: BufferExtent
    lhs_scale_handle: BufferExtent | None
    rhs_scale_handle: BufferExtent | None
    bias_handle: BufferExtent | None
    M: int
    K: int
    N: int
    strides: dict[str, int]
    valid_extents: dict[str, int]
    context: AsrJobContext | RefineJobContext


class JobResult(JobHandle):
    status: str
    output_fence: int | None
    counter_epoch: int
    timing_events: list[TimingEvent]


class StallCycles(TypedDict):
    AR: int
    R: int
    AW: int
    W: int
    B: int


class PortTrafficCounters(TypedDict):
    """Counters for one HP. Stalls count cycles with valid true and ready false."""

    read_bytes: int
    write_bytes: int
    useful_weight_bytes: int
    handshake_stall_cycles: StallCycles
    read_outstanding_peak: int
    write_outstanding_peak: int


class BandwidthRound(TypedDict):
    elapsed_ns: int
    ps_time_ns: int
    useful_weight_bytes: int
    counter_epoch: int
    port_traffic: list[PortTrafficCounters]  # Indexed by physical HP ID.


class BandwidthSummary(TypedDict):
    p5: float
    median: float
    min: float


class _BandwidthReportBase(TypedDict):
    profile: dict
    layout_id: str
    layout_sha256: str
    pack_version: str
    pack_sha256: str
    bitstream_sha256: str
    driver_sha256: str
    model_release_id: str
    clock_hz: int
    ps_workload_id: str
    ps_platform_config_sha256: str
    rounds: list[BandwidthRound]


class FullScanReport(_BandwidthReportBase):
    measurement_kind: Literal["full_scan"]
    elapsed_time_domain: Literal["submit_to_scan_complete"]
    scan_effective_weight_Bps: BandwidthSummary


class IntegratedReaderReport(_BandwidthReportBase):
    measurement_kind: Literal["integrated_reader_trace"]
    elapsed_time_domain: Literal["reader_window_union"]
    transport_effective_weight_Bps: BandwidthSummary


class WholeModelReport(_BandwidthReportBase):
    measurement_kind: Literal["whole_model"]
    elapsed_time_domain: Literal["decode_step_wall"]
    model_weight_Bps: BandwidthSummary


BandwidthReport = FullScanReport | IntegratedReaderReport | WholeModelReport
_MEASUREMENTS = {
    "full_scan": ("submit_to_scan_complete", "scan_effective_weight_Bps"),
    "integrated_reader_trace": ("reader_window_union", "transport_effective_weight_Bps"),
    "whole_model": ("decode_step_wall", "model_weight_Bps"),
}


def _bandwidth_round_rate(item, profile):
    if not isinstance(item, Mapping):
        raise ValueError("invalid bandwidth round")
    elapsed = _uint(item.get("elapsed_ns"), "round elapsed_ns")
    unique = _uint(item.get("useful_weight_bytes"), "round useful_weight_bytes")
    _uint(item.get("ps_time_ns"), "round ps_time_ns")
    _uint(item.get("counter_epoch"), "round counter_epoch")
    ports = item.get("port_traffic")
    if not elapsed or not unique or not isinstance(ports, (tuple, list)) or len(ports) != 4:
        raise ValueError("bandwidth round requires a positive duration/payload and four HP counters")
    port_payload = 0
    for index, port in enumerate(ports):
        if not isinstance(port, Mapping):
            raise ValueError("invalid HP counters")
        read = _uint(port.get("read_bytes"), "read_bytes")
        write = _uint(port.get("write_bytes"), "write_bytes")
        useful = _uint(port.get("useful_weight_bytes"), "useful_weight_bytes")
        read_peak = _uint(port.get("read_outstanding_peak"), "read_outstanding_peak",
                          profile.read_outstanding)
        write_peak = _uint(port.get("write_outstanding_peak"), "write_outstanding_peak",
                           profile.write_outstanding)
        if useful > read or (not profile.active_hp_mask & (1 << index)
                             and any((read, write, useful, read_peak, write_peak))):
            raise ValueError("HP counters contradict AXI byte counts or active profile")
        stall = port.get("handshake_stall_cycles")
        if not isinstance(stall, Mapping) or set(stall) != {"AR", "R", "AW", "W", "B"}:
            raise ValueError("expected stalls for all five AXI channels")
        for channel, cycles in stall.items():
            _uint(cycles, f"{channel} stall cycles")
        port_payload += useful
    if port_payload != unique:
        raise ValueError("round useful_weight_bytes must match the per-HP useful payload")
    return unique * 1_000_000_000 / elapsed


def validate_bandwidth_report(report: BandwidthReport) -> BandwidthReport:
    kind = _identity(report.get("measurement_kind"), "measurement_kind")
    if kind not in _MEASUREMENTS:
        raise ValueError("unsupported measurement_kind")
    elapsed_domain, metric = _MEASUREMENTS[kind]
    present_metrics = {name for _, name in _MEASUREMENTS.values() if name in report}
    if report.get("elapsed_time_domain") != elapsed_domain or present_metrics != {metric}:
        raise ValueError("bandwidth metric and elapsed time domain must match measurement_kind")
    for field in ("bitstream_sha256", "driver_sha256", "ps_platform_config_sha256"):
        _identity(report.get(field), field, _SHA256)
    _validate_asset_fields(AssetIdentity(*(report.get(name) for name in (
        "layout_id", "layout_sha256", "pack_version", "pack_sha256"))))
    _identity(report.get("ps_workload_id"), "ps_workload_id")
    _identity(report.get("model_release_id"), "model_release_id")
    if not _uint(report.get("clock_hz"), "clock_hz"):
        raise ValueError("clock_hz must be positive")
    try:
        profile = PortProfileConfig(**report["profile"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid bandwidth profile snapshot") from exc
    profile = validate_port_profile(profile)
    rounds = report.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("bandwidth report requires measurement rounds")
    rates = sorted(_bandwidth_round_rate(item, profile) for item in rounds)
    summary = report.get(metric)
    if not isinstance(summary, Mapping) or set(summary) != {"p5", "median", "min"}:
        raise ValueError("bandwidth summary requires p5, median and min")
    if any(type(value) not in (float, int) or not 0 <= value <= rates[-1]
           for value in summary.values()):
        raise ValueError("bandwidth summary contains an invalid rate")
    p5_index = 0.05 * (len(rates) - 1)
    lower = int(p5_index)
    upper = min(lower + 1, len(rates) - 1)
    p5 = rates[lower] + (p5_index - lower) * (rates[upper] - rates[lower])
    if (not math.isclose(summary["min"], rates[0], rel_tol=1e-9)
            or not math.isclose(summary["median"], median(rates), rel_tol=1e-9)
            or not math.isclose(summary["p5"], p5, rel_tol=1e-9)):
        raise ValueError("bandwidth summary contradicts elapsed time or useful payload")
    return report

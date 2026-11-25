#include "board/app/wav_replay.hpp"
#include "board/asr/ort_backend.hpp"
#include "board/audio/capture_spool.hpp"
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
#include "board/pl/embedding_stream.hpp"
#endif
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
#include "board/asr/asr_job_audit.hpp"
#include "shared/asr_job_table.h"
#endif
#include "board/session/coordinator.hpp"
#include "board/transport/server.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using edge_speech::Integrity;
using edge_speech::SessionSnapshot;
namespace app = edge_speech::app;
namespace asr = edge_speech::asr;
namespace audio = edge_speech::audio;
namespace session = edge_speech::session;
namespace transport = edge_speech::transport;
using Json = nlohmann::json;
using namespace std::chrono_literals;

constexpr char cpu_release[] = "zipformer-bilingual-e2382758-fp32";
constexpr char cpu_int8_release[] = "zipformer-bilingual-e2382758-int8-cpu";
constexpr char pl_release[] = "zipformer-bilingual-e2382758-w8-pl-embedding";
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
constexpr char jobs_release[] = "zipformer-bilingual-e2382758-w8-pl-jobs-v3";
constexpr char jobs_encoder_sha256[] = "826642f4034bad46757ddb29fa612a9584b486a88e929c16b08dc27f5f29370c";
#endif
volatile std::sig_atomic_t running = 1;

void stop_signal(int) { running = 0; }

enum class AsrMode { cpu, cpu_int8, cpu_w8, embedding, jobs };

bool cpu_mode(AsrMode mode) { return mode == AsrMode::cpu || mode == AsrMode::cpu_int8 || mode == AsrMode::cpu_w8; }

struct Options {
    std::string model_dir, ort_library, journal, wav;
    std::string capture_dir, ort_profile_dir, asr_summary, cpu_encoder, cpu_encoder_sha256;
    std::string cpu_decoder, cpu_decoder_sha256, cpu_joiner, cpu_joiner_sha256;
    std::optional<bool> frozen_fbank;
    unsigned ort_intra_threads = 1;
    std::array<unsigned, 3> ort_graph_threads{};
    std::optional<bool> spool_capture;
    std::string pl_encoder, embedding_capture, asr_trace, pl_ops, pl_audit_ops, embedding_audit_dir;
    AsrMode mode = AsrMode::cpu;
    std::string version_lock = "toolchains/versions.lock.json";
    unsigned alsa_rate = 0;
    unsigned endpoint_ms = 800;
    std::optional<bool> prepare_ahead;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    std::optional<unsigned> lane;
    std::string paired_ffn_rows;
#endif
    asr::EndpointPolicy endpoint_policy = asr::StreamConfig{}.endpoint_policy;
    transport::ServerConfig server;
};

unsigned parse_number(const std::string& text, unsigned limit) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected unsigned integer option");
    }
    const auto value = std::stoul(text);
    if (value > limit) throw std::invalid_argument("option exceeds its range");
    return static_cast<unsigned>(value);
}

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
std::bitset<asr::ort_asr_job_calls> paired_selector(std::string_view shapes,
                                                   const std::bitset<asr::ort_asr_job_calls>& enabled) {
    std::bitset<17> rows;
    while (!shapes.empty()) {
        const auto comma = shapes.find(',');
        const auto count = parse_number(std::string(shapes.substr(0, comma)), 16);
        if (count != 2 && count != 4 && count != 8 && count != 16)
            throw std::invalid_argument("--paired-ffn-rows accepts 2,4,8,16");
        rows.set(count);
        if (comma == std::string_view::npos) break;
        shapes.remove_prefix(comma + 1);
        if (shapes.empty()) throw std::invalid_argument("empty item in --paired-ffn-rows");
    }
    std::bitset<asr::ort_asr_job_calls> selected;
    for (std::size_t id = 0; id < selected.size(); ++id)
        if (enabled.test(id) && fz_asr_calls[id].operation == 0 && rows.test(fz_asr_calls[id].rows)) selected.set(id);
    if (rows.any() && selected.none()) throw std::invalid_argument("no enabled FFNs match --paired-ffn-rows");
    return selected;
}

std::bitset<asr::ort_asr_job_calls> job_selector(std::string_view remaining) {
    std::bitset<asr::ort_asr_job_calls> selection;
    while (!remaining.empty()) {
        const auto comma = remaining.find(',');
        const auto item = remaining.substr(0, comma);
        if (item == "all") {
            selection.set();
        } else if (item == "embedding" || item == "ffn" || item == "conv") {
            const unsigned op = item == "embedding" ? 2 : item == "ffn" ? 0 : 1;
            for (std::size_t id = 0; id < selection.size(); ++id) {
                if (fz_asr_calls[id].operation == op) selection.set(id);
            }
        } else {
            selection.set(parse_number(std::string(item), selection.size() - 1));
        }
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1);
        if (remaining.empty()) throw std::invalid_argument("empty item in --pl-ops");
    }
    if (selection.none()) throw std::invalid_argument("jobs requires a nonempty --pl-ops selector");
    return selection;
}

std::bitset<asr::ort_asr_job_calls> audit_selector(const Options& options) {
    if (options.pl_audit_ops == "off") {
        if (!options.embedding_audit_dir.empty()) throw std::invalid_argument("Embedding evidence conflicts with audit off");
        return {};
    }
    auto selection = options.pl_audit_ops.empty() ? std::bitset<asr::ort_asr_job_calls>{} : job_selector(options.pl_audit_ops);
    if (!options.embedding_audit_dir.empty()) selection.set(0);
    if (selection.any() && (options.asr_trace.empty() || (selection & ~job_selector(options.pl_ops)).any())) {
        throw std::invalid_argument("ASR evidence requires --asr-trace and a subset of --pl-ops");
    }
    return selection;
}
#endif

Options options(int argc, char** argv) {
    Options result;
    // Allow up to 800 ms of 20 ms preview blocks when CPU inference returns a burst.
    result.server.archive_queue_depth = 40;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (++i == argc) throw std::invalid_argument("missing option value: " + key);
        const std::string value = argv[i];
        if (key == "--model-dir") result.model_dir = value;
        else if (key == "--ort-library") result.ort_library = value;
        else if (key == "--journal") result.journal = value;
        else if (key == "--version-lock") result.version_lock = value;
        else if (key == "--wav") result.wav = value;
        else if (key == "--capture-dir") result.capture_dir = value;
        else if (key == "--capture-mode") {
            if (value != "spool" && value != "realtime") {
                throw std::invalid_argument("--capture-mode accepts spool or realtime");
            }
            result.spool_capture = value == "spool";
        }
        else if (key == "--asr-trace") result.asr_trace = value;
        else if (key == "--asr-summary") {
            if (value.empty()) throw std::invalid_argument("empty ASR summary path");
            result.asr_summary = value;
        }
        else if (key == "--ort-profile-dir") {
            if (value.empty()) throw std::invalid_argument("empty ORT profile directory");
            result.ort_profile_dir = value;
        }
        else if (key == "--ort-intra-threads" || key == "--ort-encoder-threads" ||
                 key == "--ort-decoder-threads" || key == "--ort-joiner-threads") {
            const auto threads = parse_number(value, 4);
            if (!threads) throw std::invalid_argument("ORT thread counts accept 1 to 4");
            if (key == "--ort-intra-threads") result.ort_intra_threads = threads;
            else result.ort_graph_threads[key == "--ort-encoder-threads" ? 0 : key == "--ort-decoder-threads" ? 1 : 2] = threads;
        }
        else if (key == "--fbank") {
            if (value != "native" && value != "frozen") throw std::invalid_argument("--fbank accepts native or frozen");
            result.frozen_fbank = value == "frozen";
        }
        else if (key == "--cpu-encoder") result.cpu_encoder = value;
        else if (key == "--cpu-encoder-sha256") result.cpu_encoder_sha256 = value;
        else if (key == "--cpu-decoder") result.cpu_decoder = value;
        else if (key == "--cpu-decoder-sha256") result.cpu_decoder_sha256 = value;
        else if (key == "--cpu-joiner") result.cpu_joiner = value;
        else if (key == "--cpu-joiner-sha256") result.cpu_joiner_sha256 = value;
        else if (key == "--asr-backend") {
            if (value == "cpu") result.mode = AsrMode::cpu;
            else if (value == "cpu-int8") result.mode = AsrMode::cpu_int8;
            else if (value == "cpu-w8") result.mode = AsrMode::cpu_w8;
            else if (value == "pl") result.mode = AsrMode::embedding;
            else if (value == "jobs") result.mode = AsrMode::jobs;
            else throw std::invalid_argument("unknown ASR backend");
        }
        else if (key == "--pl-encoder") result.pl_encoder = value;
        else if (key == "--embedding-capture") result.embedding_capture = value;
        else if (key == "--pl-ops") result.pl_ops = value;
        else if (key == "--pl-audit-ops") result.pl_audit_ops = value;
        else if (key == "--embedding-audit-dir") result.embedding_audit_dir = value;
        else if (key == "--prepare-ahead") {
            if (value != "one" && value != "off") throw std::invalid_argument("--prepare-ahead accepts one or off");
            result.prepare_ahead = value == "one";
        }
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        else if (key == "--paired-ffn-rows") {
            if (value.empty()) throw std::invalid_argument("empty --paired-ffn-rows");
            result.paired_ffn_rows = value;
        }
        else if (key == "--pl-lane") result.lane = parse_number(value, 1);
#endif
        else if (key == "--alsa-rate") result.alsa_rate = parse_number(value, 48000);
        else if (key == "--endpoint-ms") result.endpoint_ms = parse_number(value, 20000);
        else if (key == "--endpoint-policy") {
            if (value == "continuous") result.endpoint_policy = asr::EndpointPolicy::preserve_context;
            else if (value != "reset") throw std::invalid_argument("--endpoint-policy must be reset or continuous");
            else result.endpoint_policy = asr::EndpointPolicy::reset_model;
        }
        else if (key == "--control-port") result.server.control_port = parse_number(value, 65535);
        else if (key == "--archive-port") result.server.archive_port = parse_number(value, 65535);
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (result.model_dir.empty() || result.ort_library.empty() || result.journal.empty()) {
        throw std::invalid_argument("--model-dir, --ort-library, and --journal are required");
    }
    if (result.endpoint_ms == 0) throw std::invalid_argument("--endpoint-ms must be positive");
    if (result.alsa_rate && result.alsa_rate != 16000 && result.alsa_rate != 44100 && result.alsa_rate != 48000) {
        throw std::invalid_argument("--alsa-rate must be 16000, 44100, or 48000");
    }
    const bool uses_jobs_encoder = result.mode == AsrMode::cpu_w8 || !cpu_mode(result.mode);
    if (uses_jobs_encoder == result.pl_encoder.empty())
        throw std::invalid_argument("--pl-encoder is required for cpu-w8, pl and jobs only");
    if ((!result.asr_summary.empty() || !result.ort_profile_dir.empty() || result.frozen_fbank.has_value() || result.ort_intra_threads != 1) &&
        !cpu_mode(result.mode)) throw std::invalid_argument("ORT diagnostics options require a CPU backend");
    for (auto& graph_threads : result.ort_graph_threads) {
        if (graph_threads && !cpu_mode(result.mode)) throw std::invalid_argument("ORT graph overrides require a CPU backend");
        if (!graph_threads) graph_threads = result.ort_intra_threads;
    }
    if (result.mode == AsrMode::cpu_w8 && !result.frozen_fbank.value_or(true))
        throw std::invalid_argument("W8 CPU requires frozen fbank");
    const auto validate_cpu_asset = [&](const std::string& path, const std::string& sha, const char* option, bool required) {
        if (result.mode != AsrMode::cpu_int8 && (!path.empty() || !sha.empty()))
            throw std::invalid_argument(std::string(option) + " requires cpu-int8");
        if (required || !path.empty() || !sha.empty()) {
            if (path.empty() || sha.size() != 64 || sha.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::invalid_argument(std::string(option) + " requires a file and its lowercase SHA-256");
        }
    };
    validate_cpu_asset(result.cpu_encoder, result.cpu_encoder_sha256, "--cpu-encoder", result.mode == AsrMode::cpu_int8);
    validate_cpu_asset(result.cpu_decoder, result.cpu_decoder_sha256, "--cpu-decoder", false);
    validate_cpu_asset(result.cpu_joiner, result.cpu_joiner_sha256, "--cpu-joiner", false);
    if (result.mode != AsrMode::embedding && !result.embedding_capture.empty())
        throw std::invalid_argument("--embedding-capture requires --asr-backend pl");
    if (result.mode != AsrMode::jobs && !result.pl_ops.empty())
        throw std::invalid_argument("--pl-ops requires --asr-backend jobs");
    if (result.mode != AsrMode::jobs && (!result.pl_audit_ops.empty() || !result.embedding_audit_dir.empty()))
        throw std::invalid_argument("ASR evidence options require --asr-backend jobs");
#ifndef EDGE_SPEECH_ENABLE_ORT_PL
    if (result.mode == AsrMode::embedding) throw std::invalid_argument("PL backend is unavailable in this build");
    if (result.frozen_fbank.value_or(false)) throw std::invalid_argument("frozen fbank is unavailable in this build");
#endif
#ifndef EDGE_SPEECH_ENABLE_ASR_JOBS
    if (result.mode == AsrMode::cpu_w8) throw std::invalid_argument("W8 CPU backend is unavailable in this build");
#endif
    if (result.mode == AsrMode::jobs) {
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        paired_selector(result.paired_ffn_rows, job_selector(result.pl_ops));
        if (result.lane && !result.paired_ffn_rows.empty())
            throw std::invalid_argument("--pl-lane and --paired-ffn-rows are mutually exclusive");
        if (!result.prepare_ahead) result.prepare_ahead = result.lane.has_value() || !result.paired_ffn_rows.empty();
        if (!result.spool_capture && (result.lane || !result.paired_ffn_rows.empty())) result.spool_capture = true;
        audit_selector(result);
#else
        throw std::invalid_argument("ASR jobs backend is unavailable in this build");
#endif
    }
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    else if (result.lane || !result.paired_ffn_rows.empty()) {
        throw std::invalid_argument("dual-lane options require --asr-backend jobs");
    }
#endif
    if (!result.capture_dir.empty()) {
        if (!result.spool_capture.value_or(true)) throw std::invalid_argument("--capture-dir requires spool capture");
        result.spool_capture = true;
    }
    return result;
}

class PreparingCaptureSource final : public audio::AudioSource {
public:
    PreparingCaptureSource(std::unique_ptr<audio::AudioSource> source, std::shared_ptr<asr::FeatureStaging> staging)
        : source_(std::move(source)), staging_(std::move(staging)) {}

    std::optional<audio::CaptureEvent> read() override {
        auto event = source_->read();
        if (!event) staging_->prepare_ready();
        else if (const auto* block = std::get_if<edge_speech::AudioBlock>(&*event)) staging_->offer(*block);
        else staging_->discontinuity();
        return event;
    }

    void stop() override { source_->stop(); }
    void request_stop() override { source_->request_stop(); }

private:
    std::unique_ptr<audio::AudioSource> source_;
    std::shared_ptr<asr::FeatureStaging> staging_;
};

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
Json paired_receipts(const edge_speech::pl::AsrPairedFfnReceipt& pair) {
    const auto phase = [](const auto& receipts) {
        auto rows = Json::array();
        for (unsigned lane = 0; lane < receipts.size(); ++lane) {
            const auto& actual = receipts[lane];
            rows.push_back(Json{{"lane", lane}, {"operation", static_cast<unsigned>(actual.operation)},
                {"call_id", actual.call_id}, {"rows", actual.rows}, {"device_generation", actual.generation},
                {"device_epoch", actual.epoch}, {"chain_id", actual.chain_id}, {"sequence", actual.sequence},
                {"irq_events", actual.irq_events}, {"flags", actual.flags}, {"queue_ns", actual.queue_ns},
                {"prepare_ns", actual.prepare_ns}, {"device_ns", actual.device_ns}, {"retire_ns", actual.retire_ns},
                {"ioctl_ns", actual.ioctl_ns}, {"packet_bytes", actual.packet_bytes}, {"span_count", actual.span_count},
                {"packet_crc32", actual.packet_crc32}});
        }
        return rows;
    };
    return {{"first", phase(pair.first)}, {"second", phase(pair.second)}, {"first_join_ns", pair.first_join_ns},
            {"second_join_ns", pair.second_join_ns}, {"first_wall_ns", pair.first_wall_ns},
            {"second_wall_ns", pair.second_wall_ns}, {"audit_ns", pair.audit_ns}};
}
#endif

class Journal {
public:
    explicit Journal(const std::string& path) : output_(std::fopen(path.c_str(), "ab"), &std::fclose) {
        if (!output_) throw std::runtime_error("cannot open the transcript journal");
    }

    void append(std::string_view line, bool durable = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (std::fwrite(line.data(), 1, line.size(), output_.get()) != line.size() ||
            std::fputc('\n', output_.get()) == EOF || std::fflush(output_.get()) != 0) {
            throw std::runtime_error("cannot write the transcript journal");
        }
        if (!durable) return;
#ifdef _WIN32
        const int synced = _commit(_fileno(output_.get()));
#else
        const int synced = fsync(fileno(output_.get()));
#endif
        if (synced != 0) throw std::runtime_error("cannot sync the transcript journal");
    }

private:
    std::unique_ptr<std::FILE, decltype(&std::fclose)> output_;
    std::mutex mutex_;
};

const char* integrity_name(Integrity value) {
    switch (value) {
    case Integrity::pending: return "pending";
    case Integrity::complete: return "complete";
    case Integrity::with_gaps: return "with_gaps";
    case Integrity::failed: return "failed";
    }
    throw std::invalid_argument("unknown integrity value");
}

Json index_or_null(const std::optional<std::uint64_t>& value) {
    return value ? Json(*value) : Json(nullptr);
}

Json snapshot_json(const SessionSnapshot& snapshot) {
    auto revisions = Json::array();
    for (const auto& reference : snapshot.ordered_segment_revisions) {
        revisions.push_back({{"segment_id", reference.segment_id}, {"revision", reference.revision}});
    }
    return {{"snapshot_id", snapshot.snapshot_id}, {"session_id", snapshot.session_id},
            {"schema_version", snapshot.schema_version}, {"ordered_segment_revisions", revisions},
            {"stop_sample", index_or_null(snapshot.stop_sample)}, {"high_watermark", snapshot.high_watermark},
            {"content_sha256", snapshot.content_sha256}, {"created_at", snapshot.created_at},
            {"frozen", snapshot.frozen}, {"integrity", integrity_name(snapshot.integrity)}};
}

Json status_json(const session::Status& status, std::uint64_t preview_dropped) {
    Json value = {{"state", session::state_name(status.state)}, {"session_id", status.session_id},
                  {"capture_end_sample", status.capture_end_sample}, {"stop_sample", index_or_null(status.stop_sample)},
                  {"high_watermark", status.high_watermark}, {"integrity", integrity_name(status.integrity)},
                  {"processed_sample", status.processed_sample}, {"async_stop_supported", true},
                  {"audio_preview_dropped_samples", preview_dropped}, {"refinement_job_id", status.refinement_job_id}};
    if (status.snapshot) {
        value["snapshot_id"] = status.snapshot->snapshot_id;
        value["snapshot_sha256"] = status.snapshot->content_sha256;
    }
    if (!status.error_code.empty()) value["error_code"] = status.error_code;
    return value;
}

std::string required_uuid(const Json& request, const char* field) {
    const auto text = request.at(field).get<std::string>();
    if (!edge_speech::is_uuid(text)) throw std::invalid_argument("invalid control UUID");
    return text;
}

std::string handle_request(session::SessionCoordinator& coordinator, std::string_view text,
                           const std::atomic<std::uint64_t>& preview_dropped, std::atomic<bool>& async_draining,
                           const std::string& model_release) {
    const auto request = Json::parse(text.begin(), text.end(), nullptr, false);
    if (!request.is_object()) throw std::invalid_argument("invalid control JSON object");
    const auto request_id = required_uuid(request, "request_id");
    auto reply = [&](bool ok, std::string_view code) {
        auto response = status_json(coordinator.status(), preview_dropped.load());
        response["request_id"] = request_id;
        response["ok"] = ok;
        response["code"] = code;
        return response;
    };
    try {
        if (!request.contains("protocol") || !request["protocol"].is_number_integer() || request["protocol"] != 1) {
            return reply(false, "invalid_request").dump();
        }
        const auto op = request.at("op").get<std::string>();
        if (op == "start") {
            const auto asked_release = request.value("model_release", std::string(model_release));
            if (asked_release != model_release) return reply(false, "model_mismatch").dump();
            const auto receipt = coordinator.start({request_id, asked_release});
            auto response = reply(receipt.ok, receipt.code);
            response["state"] = session::state_name(receipt.state);
            response["session_id"] = receipt.session_id;
            return response.dump();
        }
        if (op == "status") return reply(true, "ok").dump();
        const auto session_id = required_uuid(request, "session_id");
        if (op == "stop") {
            const auto wait = request.value("wait", true);
            const auto receipt = wait ? coordinator.stop(request_id, session_id)
                                      : coordinator.request_stop(request_id, session_id);
            if (!wait && receipt.ok && receipt.state == session::State::draining) async_draining = true;
            auto response = reply(receipt.ok, receipt.code);
            if (wait || !receipt.ok || response["session_id"] != receipt.session_id) {
                response["state"] = session::state_name(receipt.state);
                response["session_id"] = receipt.session_id;
                response["stop_sample"] = index_or_null(receipt.stop_sample);
            } else if (response.contains("error_code")) {
                response["ok"] = false;
                response["code"] = response["error_code"];
            }
            return response.dump();
        }
        if (session_id != coordinator.status().session_id) return reply(false, "invalid_state").dump();
        if (op == "resume") {
            auto response = reply(true, "ok");
            response["replay_available"] = false;
            const auto snapshot = coordinator.status().snapshot;
            if (snapshot) {
                auto full = response;
                full["snapshot"] = snapshot_json(*snapshot);
                auto encoded = full.dump();
                if (encoded.size() <= edge_speech::fzas_json_payload_limit) return encoded;
            }
            return response.dump();
        }
        if (op == "complete") {
            const auto status = coordinator.status();
            auto receipt = status.state == session::State::faulted ? coordinator.reset() : coordinator.complete();
            if (receipt.ok && receipt.state == session::State::completed) receipt = coordinator.reset();
            return reply(receipt.ok, receipt.code).dump();
        }
        if (op == "cancel_refine") {
            const auto receipt = coordinator.cancel_refine(required_uuid(request, "refinement_job_id"));
            auto response = reply(receipt.ok, receipt.code);
            response["refinement_job_id"] = receipt.refinement_job_id;
            return response.dump();
        }
        if (op == "refine") return reply(false, "accelerator_unavailable").dump();
        return reply(false, "invalid_request").dump();
    } catch (const nlohmann::json::exception&) {
        return reply(false, "invalid_request").dump();
    } catch (const std::invalid_argument&) {
        return reply(false, "invalid_request").dump();
    }
}

const char* gap_reason(audio::CaptureGapReason reason) {
    switch (reason) {
    case audio::CaptureGapReason::device_xrun: return "device_xrun";
    case audio::CaptureGapReason::disconnect: return "disconnect";
    case audio::CaptureGapReason::timeout: return "other";
    case audio::CaptureGapReason::queue_overrun: return "queue_overrun";
    }
    return "other";
}

void verify_asset(const std::string& path, const std::string& expected_sha256) {
    std::ifstream input(path, std::ios::binary);
    if (!input || session::sha256(input) != expected_sha256) {
        throw std::runtime_error("locked recognition asset SHA-256 mismatch: " + path);
    }
}

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
std::string parameter_identity(std::uint16_t call_id) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (auto value : fz_asr_calls[call_id].parameters_sha256) {
        result.push_back(hex[value >> 4]);
        result.push_back(hex[value & 15]);
    }
    return result;
}

std::filesystem::path save_embedding_audit(const std::filesystem::path& root, const asr::ChunkStamp& stamp,
                                           const asr::AuditedAsrJob& job) {
    namespace fs = std::filesystem;
    const auto directory = root / stamp.session_id / ("generation-" + std::to_string(stamp.stream_generation));
    fs::create_directories(directory);
    const auto index = stamp.chunk_id - 1;
    const auto name = "chunk-" + std::string(index < 10 ? "0" : "") + std::to_string(index);
    const auto input = directory / (name + ".input");
    const auto output = directory / (name + ".output");
    const auto staged_input = directory / (name + ".input.tmp");
    const auto staged_output = directory / (name + ".output.tmp");
    if (fs::exists(input) || fs::exists(output) || fs::exists(staged_input) || fs::exists(staged_output)) {
        throw std::runtime_error("Embedding evidence window already exists");
    }
    const auto write = [](const fs::path& path, const std::vector<std::uint8_t>& payload) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        out.close();
        if (!out) throw std::runtime_error("cannot write Embedding evidence window");
    };
    bool input_published = false;
    try {
        write(staged_input, job.embedding_input);
        write(staged_output, job.embedding_output);
        fs::rename(staged_input, input);
        input_published = true;
        fs::rename(staged_output, output);
    } catch (...) {
        std::error_code ignored;
        fs::remove(staged_input, ignored);
        fs::remove(staged_output, ignored);
        if (input_published) fs::remove(input, ignored);
        throw;
    }
    return directory;
}
#endif

asr::OrtModelPaths verified_model(const Options& cfg) {
    std::ifstream lock_file(cfg.version_lock, std::ios::binary);
    if (!lock_file) throw std::runtime_error("cannot read the model version lock");
    const auto lock = Json::parse(lock_file);
    const auto& model = lock.at("zipformer");
    if (model.at("source").at("revision") != "e2382758de9a0219b4efe682b95af30b399db3b8") {
        throw std::runtime_error("unexpected source revision in the model lock");
    }
    const auto asset = [&](const char* name, const char* expected_filename) {
        const auto& entry = model.at("artifacts").at(name);
        if (entry.at("path") != expected_filename) throw std::runtime_error("unexpected locked asset filename");
        const auto path = cfg.model_dir + '/' + expected_filename;
        verify_asset(path, entry.at("sha256").get<std::string>());
        return path;
    };
#ifdef _WIN32
    verify_asset(cfg.ort_library, lock.at("host").at("ort_c_api_windows").at("dll_sha256").get<std::string>());
#else
    verify_asset(cfg.ort_library, lock.at("board_candidate").at("ort").at("library_sha256").get<std::string>());
#endif
    std::string encoder;
    if (cfg.mode == AsrMode::embedding) {
        verify_asset(cfg.pl_encoder, "2e357e72f9dc14c5ae4fce57d5358f1ec059a5c669c21920ada5b867013a1e27");
        encoder = cfg.pl_encoder;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    } else if (cfg.mode == AsrMode::jobs || cfg.mode == AsrMode::cpu_w8) {
        verify_asset(cfg.pl_encoder, jobs_encoder_sha256);
        encoder = cfg.pl_encoder;
#endif
    } else if (cfg.mode == AsrMode::cpu_int8) {
        verify_asset(cfg.cpu_encoder, cfg.cpu_encoder_sha256);
        encoder = cfg.cpu_encoder;
    } else {
        encoder = asset("encoder", "encoder-epoch-99-avg-1.onnx");
    }
    const auto checked_cpu_asset = [&](const std::string& override, const std::string& sha,
                                       const char* name, const char* filename) {
        if (override.empty()) return asset(name, filename);
        verify_asset(override, sha);
        return override;
    };
    return {cfg.ort_library, encoder,
        checked_cpu_asset(cfg.cpu_decoder, cfg.cpu_decoder_sha256, "decoder", "decoder-epoch-99-avg-1.onnx"),
        checked_cpu_asset(cfg.cpu_joiner, cfg.cpu_joiner_sha256, "joiner", "joiner-epoch-99-avg-1.onnx"), asset("tokens", "tokens.txt")};
}

int service(const Options& cfg) {
    const auto paths = verified_model(cfg);
    std::string model_release = cfg.mode == AsrMode::embedding ? pl_release : cpu_release;
    if (cfg.mode == AsrMode::cpu_int8) model_release = cpu_int8_release;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    if (cfg.mode == AsrMode::jobs || cfg.mode == AsrMode::cpu_w8) model_release = jobs_release;
#endif
    Journal journal(cfg.journal);
    auto trace = cfg.asr_trace.empty() ? nullptr : std::make_unique<Journal>(cfg.asr_trace);
    auto summary = cfg.asr_summary.empty() ? nullptr : std::make_unique<Journal>(cfg.asr_summary);
    struct CpuTotals {
        std::uint64_t chunks = 0, encoder_ns = 0, decoder_joiner_ns = 0;
        std::uint64_t decoder_ns = 0, joiner_ns = 0;
    } totals;
    const auto profile_dir = std::filesystem::u8path(cfg.ort_profile_dir);
    if (!profile_dir.empty()) std::filesystem::create_directories(profile_dir);
    const auto diagnostics = [&](const std::string& id) {
        asr::OrtCpuDiagnostics result;
        result.intra_threads = cfg.ort_graph_threads;
        result.require_int8_encoder = cfg.mode == AsrMode::cpu_int8;
        result.require_int8_decoder = !cfg.cpu_decoder.empty();
        result.require_int8_joiner = !cfg.cpu_joiner.empty();
        result.detailed_timing = bool(trace);
        if (!profile_dir.empty()) result.profile_prefix = profile_dir / id;
        if (trace || summary) result.on_chunk = [&](const asr::OrtCpuChunkTrace& measured) {
            if (summary) {
                ++totals.chunks;
                totals.encoder_ns += measured.encoder_ns + measured.cache_ns;
                totals.decoder_joiner_ns += measured.decoder_joiner_ns;
                totals.decoder_ns += measured.decoder_ns;
                totals.joiner_ns += measured.joiner_ns;
            }
            if (trace) trace->append(Json{{"event", "asr_cpu_chunk"}, {"session_id", measured.stamp.session_id},
                {"generation", measured.stamp.stream_generation}, {"chunk_id", measured.stamp.chunk_id},
                {"state_epoch", measured.stamp.state_epoch}, {"encoder_ns", measured.encoder_ns},
                {"decoder_joiner_ns", measured.decoder_joiner_ns}, {"marshal_ns", measured.marshal_ns},
                {"cache_ns", measured.cache_ns}, {"text_ns", measured.text_ns},
                {"decoder_ns", measured.decoder_ns}, {"joiner_ns", measured.joiner_ns}, {"greedy_ns", measured.greedy_ns},
                {"decoder_calls", measured.decoder_calls}, {"joiner_calls", measured.joiner_calls}}.dump(), false);
        };
        return result;
    };
    const bool plain_cpu = cpu_mode(cfg.mode) && cfg.mode != AsrMode::cpu_w8;
    auto cpu = plain_cpu && profile_dir.empty() ? asr::make_ort_cpu_backend(paths, diagnostics({})) : nullptr;
    std::shared_ptr<asr::ChunkBackend> profiled_cpu;
    std::atomic<std::uint64_t> preview_dropped{0};
    std::atomic<bool> async_draining{false};
    const auto spool_capture = cfg.spool_capture.value_or(false);
    const auto capture_root = std::filesystem::u8path(cfg.capture_dir.empty() ? cfg.journal + ".audio" : cfg.capture_dir);
    std::string capture_session_id;
    session::SessionCoordinator* coordinator = nullptr;
    transport::Handlers wire;
    wire.control_request = [&](std::string_view request) {
        return handle_request(*coordinator, request, preview_dropped, async_draining, model_release);
    };
    wire.persist_event = [&](const edge_speech::TranscriptEvent&, std::string_view json) { journal.append(json); };
    wire.diagnostic = [](transport::Channel channel, std::string_view diagnostic) {
        std::cerr << (channel == transport::Channel::control ? "control" : "archive") << ": " << diagnostic << '\n';
    };
    transport::Server server(cfg.server, std::move(wire));
    session::Dependencies dependencies;
    std::shared_ptr<asr::FeatureStaging> feature_staging;
    dependencies.make_source = [&](const std::string& id) -> std::unique_ptr<audio::AudioSource> {
        const auto uuid = session::wire_uuid(id);
        std::unique_ptr<audio::AudioSource> source;
        if (cfg.wav.empty()) {
            source = std::make_unique<audio::CaptureSource>(uuid, audio::make_alsa_capture(cfg.alsa_rate), spool_capture);
        } else {
            source = std::make_unique<app::WavReplaySource>(uuid, std::make_unique<std::ifstream>(cfg.wav, std::ios::binary));
        }
        if (spool_capture) {
            audio::SpoolCaptureSource::OnStored on_stored;
            if (feature_staging) {
                on_stored = [staging = feature_staging](const audio::CaptureEvent& event) {
                    try {
                        if (const auto* block = std::get_if<edge_speech::AudioBlock>(&event)) staging->offer(*block);
                        else staging->discontinuity();
                    } catch (...) {
                        std::fputs("ASR feature preparation failed during capture\n", stderr);
                    }
                };
            }
            source = audio::make_spooled_capture_source(std::move(source), uuid, capture_root / id, std::move(on_stored));
        } else {
            if (feature_staging) source = std::make_unique<PreparingCaptureSource>(std::move(source), feature_staging);
            if (cfg.wav.empty() || feature_staging) source = audio::make_buffered_capture_source(std::move(source));
        }
        preview_dropped = 0;
        async_draining = false;
        capture_session_id = id;
        return source;
    };
    dependencies.make_stream = [&](const std::string& id, const session::StartConfig& start) {
        totals = {};
        asr::StreamConfig config;
        config.trailing_silence_ms = cfg.endpoint_ms;
        config.endpoint_policy = cfg.endpoint_policy;
        config.frozen_fbank = cfg.frozen_fbank.value_or(cfg.mode == AsrMode::cpu_w8 || !cpu_mode(cfg.mode));
        feature_staging = cfg.prepare_ahead.value_or(false)
            ? std::make_shared<asr::FeatureStaging>(config.frozen_fbank) : nullptr;
        asr::StreamPreparation preparation{feature_staging, {}};
        if ((feature_staging || cpu_mode(cfg.mode)) && trace) {
            preparation.on_chunk = [&](const asr::FeaturePreparationTrace& row) {
                trace->append(Json{{"event", "asr_feature_preparation"}, {"session_id", row.stamp.session_id},
                    {"generation", row.stamp.stream_generation}, {"chunk_id", row.stamp.chunk_id},
                    {"prepared_ahead", row.prepared_ahead}, {"prepare_ns", row.prepare_ns},
                    {"queue_ns", row.queue_ns}, {"overlap_ns", row.overlap_ns}, {"backend_ns", row.backend_ns},
                    {"service_ns", row.service_ns}, {"queue_peak", row.queue_peak}}.dump(), false);
            };
        }
        auto backend = cpu;
        if (plain_cpu && !profile_dir.empty()) backend = asr::make_ort_cpu_backend(paths, diagnostics(id));
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (cfg.mode == AsrMode::cpu_w8) {
            asr::OrtAsrJobs jobs;
            if (trace) jobs.on_job = [&](const asr::OrtAsrJobTrace& job) {
                static constexpr const char* operations[] = {"ffn", "conv", "embedding"};
                Json row{{"event", "asr_job"}, {"backend", "cpu"}, {"session_id", job.stamp.session_id},
                    {"generation", job.stamp.stream_generation}, {"chunk_id", job.stamp.chunk_id},
                    {"state_epoch", job.stamp.state_epoch}, {"call_id", job.call_id}, {"rows", job.rows},
                    {"operation", operations[static_cast<unsigned>(job.operation)]}, {"elapsed_ns", job.elapsed_ns}};
                if (const auto* measured = job.w8_profile) {
                    Json phases = Json::object();
                    std::uint64_t scoped = 0;
                    const auto activation = measured->ns[static_cast<unsigned>(asr::W8Part::ffn_double_swish)];
                    for (unsigned i = 0; i < measured->ns.size(); ++i) {
                        auto ns = measured->ns[i];
                        if (!ns) continue;
                        // DoubleSwish is a nested measurement within the fused Linear1 loop.
                        if (i == static_cast<unsigned>(asr::W8Part::ffn_linear1)) ns -= activation;
                        scoped += ns;
                        phases[asr::w8_part_names[i]] = ns;
                    }
                    phases["ffn_double_swish_elements"] = measured->double_swish_elements;
                    phases["kernel_ns"] = measured->kernel_ns;
                    phases["unattributed_kernel_ns"] = measured->kernel_ns - scoped;
                    phases["outside_kernel_ns"] = job.elapsed_ns - measured->kernel_ns;
                    row["ps_w8_detail"] = std::move(phases);
                }
                trace->append(row.dump(), false);
            };
            backend = asr::make_ort_asr_jobs_backend(paths, std::move(jobs), diagnostics(id));
        }
#endif
        if (!profile_dir.empty()) profiled_cpu = backend;
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
        if (cfg.mode == AsrMode::embedding) {
            namespace pl = edge_speech::pl;
            auto transport = cfg.embedding_capture.empty() ? pl::open_linux_embedding_stream()
                                                           : pl::open_embedding_capture(cfg.embedding_capture);
            auto device = std::make_shared<pl::EmbeddingStream>(std::move(transport));
            backend = asr::make_ort_pl_backend(paths, [device, &trace](const asr::ChunkStamp& stamp,
                                                                       const asr::FeatureWindow& x) {
                auto reply = device->submit(x.data(), x.size());
                if (trace) trace->append(Json{{"event", "embedding"}, {"session_id", stamp.session_id},
                    {"generation", stamp.stream_generation}, {"chunk_id", stamp.chunk_id},
                    {"chain_id", reply.identity.chain_id}, {"sequence", reply.identity.sequence}}.dump());
                return asr::OrtEmbeddingResult{stamp, std::move(reply.output)};
            });
        }
#endif
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (cfg.mode == AsrMode::jobs) {
            asr::OrtAsrJobs jobs;
            jobs.enabled = job_selector(cfg.pl_ops);
            jobs.paired_ffn = paired_selector(cfg.paired_ffn_rows, jobs.enabled);
            std::unique_ptr<edge_speech::pl::AsrJobTransport> device;
            if (cfg.lane) {
                device = edge_speech::pl::open_linux_asr_lane(*cfg.lane
                    ? "/dev/fzspeech-asr-lane1" : "/dev/fzspeech-asr-lane0");
            } else if (jobs.paired_ffn.any()) {
                device = edge_speech::pl::open_linux_dual_asr_jobs();
            } else {
                device = edge_speech::pl::open_linux_asr_jobs();
            }
            const auto audit_calls = audit_selector(cfg);
            std::shared_ptr<asr::OrtAsrJobAudit> audit;
            if (audit_calls.any()) {
                audit = std::make_shared<asr::OrtAsrJobAudit>(std::move(device), audit_calls,
                    !cfg.embedding_audit_dir.empty(), [](std::string_view bytes) { return session::sha256(bytes); });
                jobs.transport = audit;
            } else {
                jobs.transport = std::move(device);
            }
            if (trace) {
                jobs.on_job = [&, audit](const asr::OrtAsrJobTrace& job) {
                    if (audit) audit->observe(job);
                    const auto& stamp = job.stamp;
                    const auto op = job.operation == edge_speech::pl::AsrOperation::embedding ? "embedding" :
                        job.operation == edge_speech::pl::AsrOperation::ffn ? "ffn" : "conv";
                    Json event{{"event", "asr_job"}, {"session_id", stamp.session_id},
                        {"generation", stamp.stream_generation}, {"chunk_id", stamp.chunk_id},
                        {"state_epoch", stamp.state_epoch}, {"call_id", job.call_id}, {"operation", op},
                        {"rows", job.rows}, {"backend", job.receipt || job.paired_ffn ? "pl" : "ps"},
                        {"elapsed_ns", job.elapsed_ns}};
                    if (job.receipt) {
                        const auto& actual = *job.receipt;
                        event["device_generation"] = actual.generation;
                        event["device_epoch"] = actual.epoch;
                        event["chain_id"] = actual.chain_id;
                        event["sequence"] = actual.sequence;
                        event["irq_events"] = actual.irq_events;
                        event["flags"] = actual.flags;
                        event["prepare_ns"] = actual.prepare_ns;
                        event["device_ns"] = actual.device_ns;
                        event["retire_ns"] = actual.retire_ns;
                        event["marshal_ns"] = job.marshal_ns;
                        event["submit_ns"] = job.submit_ns;
                        event["output_ns"] = job.output_ns;
                        event["ioctl_ns"] = actual.ioctl_ns;
                        event["audit_ns"] = actual.audit_ns;
                        event["packet_bytes"] = actual.packet_bytes;
                        event["span_count"] = actual.span_count;
                        event["packet_crc32"] = actual.packet_crc32;
                    }
                    if (job.paired_ffn) {
                        event["paired_ffn"] = paired_receipts(*job.paired_ffn);
                        event["marshal_ns"] = job.marshal_ns;
                        event["submit_ns"] = job.submit_ns;
                        event["output_ns"] = job.output_ns;
                    }
                    trace->append(event.dump(), false);
                };
                jobs.on_chunk = [&, audit](const asr::OrtAsrChunkTrace& completed) {
                    const auto audit_started = asr::OrtJobClock::now();
                    if (audit) {
                        for (const auto& payload : audit->take_committed(completed.stamp)) {
                            const auto& stamp = completed.stamp;
                            const auto& actual = payload.paired_ffn ? payload.paired_ffn->second[0] : payload.receipt;
                            Json event{{"event", "asr_job_payload"}, {"session_id", stamp.session_id},
                                {"generation", stamp.stream_generation}, {"chunk_id", stamp.chunk_id},
                                {"state_epoch", stamp.state_epoch}, {"call_id", actual.call_id},
                                {"layout_sha256", FZ_ASR_LAYOUT_SHA256_HEX},
                                {"parameters_sha256", parameter_identity(actual.call_id)},
                                {"operation", fz_asr_calls[actual.call_id].operation}, {"rows", actual.rows},
                                {"state_slot", fz_asr_calls[actual.call_id].state_slot},
                                {"input_bytes", payload.input_bytes}, {"output_bytes", payload.output_bytes},
                                {"history_bytes", payload.history_bytes}, {"input_sha256", payload.input_sha256},
                                {"output_sha256", payload.output_sha256}};
                            if (payload.paired_ffn) event["paired_ffn"] = paired_receipts(*payload.paired_ffn);
                            else {
                                event["device_generation"] = actual.generation;
                                event["device_epoch"] = actual.epoch;
                                event["chain_id"] = actual.chain_id;
                                event["sequence"] = actual.sequence;
                            }
                            if (payload.history_bytes) {
                                event["old_history_sha256"] = payload.old_sha256;
                                event["working_history_sha256"] = payload.working_sha256;
                            }
                            if (!payload.embedding_input.empty()) {
                                event["embedding_window_directory"] = save_embedding_audit(cfg.embedding_audit_dir,
                                    stamp, payload).u8string();
                            }
                            trace->append(event.dump(), false);
                        }
                    }
                    Json event{{"event", "asr_chunk_committed"}, {"session_id", completed.stamp.session_id},
                        {"generation", completed.stamp.stream_generation}, {"chunk_id", completed.stamp.chunk_id},
                        {"encoder_ns", completed.encoder_ns}, {"decoder_joiner_ns", completed.decoder_joiner_ns},
                        {"job_trace_ns", completed.job_trace_ns},
                        {"payload_trace_ns", audit ? asr::job_elapsed_ns(audit_started) : 0},
                        {"commit_ns", completed.commit_ns}, {"ps_calls_ffn_conv_embedding", completed.ps_calls},
                        {"pl_calls_ffn_conv_embedding", completed.pl_calls}};
                    if (completed.paired_ffn_calls) {
                        event["paired_ffn_calls"] = completed.paired_ffn_calls;
                        event["physical_pl_calls_by_lane"] = completed.physical_pl_calls;
                    }
                    trace->append(event.dump());
                };
            }
            backend = asr::make_ort_asr_jobs_backend(paths, std::move(jobs));
        }
#endif
        return std::make_unique<asr::StreamEngine>(id, start.model_release, backend, config, std::move(preparation));
    };
    dependencies.on_audio = [&](const edge_speech::AudioBlock& block) {
        if (!server.send_audio(block)) preview_dropped.fetch_add(block.sample_count);
    };
    dependencies.on_event = [&](const edge_speech::TranscriptEvent& event) { server.publish(event); };
    dependencies.on_gap = [&](const audio::CaptureGap& gap) {
        const auto event = Json{{"type", "capture_gap"}, {"session_id", capture_session_id},
            {"sample_start", gap.sample_start}, {"sample_end", index_or_null(gap.sample_end)},
            {"reason", gap_reason(gap.reason)}, {"detail", gap.detail}}.dump();
        journal.append(event);
        server.send_control_event(event);
    };
    dependencies.freeze = [&](const SessionSnapshot& snapshot, const asr::FinalizedStream& finalized) {
        if (profiled_cpu) {
            const auto start = asr::FeatureClock::now();
            const auto files = asr::finish_ort_profile(*profiled_cpu);
            profiled_cpu.reset();
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(asr::FeatureClock::now() - start);
            if (trace) trace->append(Json{{"event", "ort_profiles"}, {"session_id", snapshot.session_id},
                {"encoder", files[0]}, {"decoder", files[1]}, {"joiner", files[2]}, {"export_ns", elapsed.count()}}.dump(), false);
        }
        if (summary) {
            Json line{{"schema", "fz-asr-cpu-summary-v1"}, {"session_id", snapshot.session_id}, {"model_release", model_release},
                      {"ort_intra_threads", cfg.ort_intra_threads}, {"ort_encoder_threads", cfg.ort_graph_threads[0]},
                      {"ort_decoder_threads", cfg.ort_graph_threads[1]}, {"ort_joiner_threads", cfg.ort_graph_threads[2]},
                      {"profile_enabled", !profile_dir.empty()},
                      {"detailed_timing", bool(trace)}, {"chunks", totals.chunks}, {"stop_sample", finalized.stop_sample},
                      {"decoder_steps", finalized.decoder_steps}, {"encoder_ns", totals.encoder_ns},
                      {"encoder_includes_cache", true}, {"decoder_joiner_ns", totals.decoder_joiner_ns},
                      {"model_ns", totals.encoder_ns + totals.decoder_joiner_ns}, {"segment_tokens", finalized.segment_tokens}};
            if (trace) {
                line["decoder_ns"] = totals.decoder_ns;
                line["joiner_ns"] = totals.joiner_ns;
            }
            if (cfg.mode == AsrMode::cpu_int8) {
                Json graph_hashes{{"encoder", cfg.cpu_encoder_sha256}};
                if (!cfg.cpu_decoder.empty()) graph_hashes["decoder"] = cfg.cpu_decoder_sha256;
                if (!cfg.cpu_joiner.empty()) graph_hashes["joiner"] = cfg.cpu_joiner_sha256;
                line["int8_graph_sha256"] = std::move(graph_hashes);
            }
            summary->append(line.dump());
        }
        if (trace) trace->append(Json{{"event", "freeze"}, {"session_id", snapshot.session_id},
            {"segment_tokens", finalized.segment_tokens}, {"decoder_steps", finalized.decoder_steps}}.dump());
        auto line = snapshot_json(snapshot);
        line["type"] = "snapshot";
        journal.append(line.dump());
    };

    session::SessionCoordinator owner(std::move(dependencies));
    coordinator = &owner;
    server.start();
    std::atomic<bool> observing{true};
    std::thread observer;
    const auto stop_observer = [&] {
        observing = false;
        if (observer.joinable()) observer.join();
    };
    try {
        std::string embedding_source = "cpu";
        if (cfg.mode == AsrMode::embedding) embedding_source = cfg.embedding_capture.empty() ? "device" : "capture";
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (cfg.mode == AsrMode::jobs && job_selector(cfg.pl_ops).test(0)) embedding_source = "device";
#endif
        std::cout << Json{{"control_port", server.control_port()}, {"archive_port", server.archive_port()},
                          {"model_release", model_release}, {"embedding_source", embedding_source},
                          {"audio_source", cfg.wav.empty() ? "alsa" : "wav"}, {"async_stop_supported", true},
                          {"capture_mode", spool_capture ? "spool" : "realtime"},
                          {"capture_root", spool_capture ? Json(capture_root.u8string()) : Json(nullptr)}}.dump()
                  << std::endl;
        observer = std::thread([&] {
            session::Status previous;
            while (observing) {
                try {
                    const auto status = owner.status();
                    const bool progress_changed = async_draining && status.state == session::State::draining &&
                        (status.stop_sample != previous.stop_sample || status.processed_sample != previous.processed_sample);
                    const bool notify = status.state != previous.state || status.session_id != previous.session_id ||
                        status.error_code != previous.error_code || progress_changed;
                    if (notify && (status.state != session::State::draining || async_draining || !status.error_code.empty())) {
                        auto event = status_json(status, preview_dropped.load());
                        event["type"] = "state";
                        if (server.send_control_event(event.dump()) == transport::PublishResult::queued) previous = status;
                    }
                } catch (...) {
                    std::fputs("session state notification failed\n", stderr);
                }
                std::this_thread::sleep_for(100ms);
            }
        });
        while (running) {
            if (!owner.pump()) std::this_thread::sleep_for(2ms);
        }
        const auto status = owner.status();
        if (status.state == session::State::recording || status.state == session::State::draining) {
            owner.stop("00000000-0000-4000-8000-000000000000", status.session_id);
        }
        stop_observer();
        server.stop();
    } catch (...) {
        stop_observer();
        server.stop();
        throw;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "edge_speech_board --model-dir DIR --ort-library FILE --journal FILE [--version-lock FILE] "
                     "[--wav FILE | --alsa-rate HZ] [--endpoint-ms MS] [--endpoint-policy continuous|reset] "
                     "[--capture-mode spool|realtime] [--capture-dir DIR] "
                     "[--control-port PORT] [--archive-port PORT] [--asr-trace FILE] "
                     "[--asr-summary FILE] [--ort-profile-dir DIR] [--ort-intra-threads 1|2|3|4] [--fbank native|frozen] "
                     "[--ort-encoder-threads N] [--ort-decoder-threads N] [--ort-joiner-threads N] "
                     "[--cpu-encoder FILE --cpu-encoder-sha256 SHA256] [--cpu-decoder FILE --cpu-decoder-sha256 SHA256] "
                     "[--cpu-joiner FILE --cpu-joiner-sha256 SHA256] "
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
                     "[--asr-backend cpu|cpu-int8|cpu-w8|pl|jobs] [--pl-encoder FILE] [--embedding-capture DIR] "
                     "[--pl-ops embedding,ffn,conv|all|CALL_ID,...] "
                     "[--pl-audit-ops SELECTOR|off] [--embedding-audit-dir DIR] "
                     "[--pl-lane 0|1 | --paired-ffn-rows 2,4,8,16] [--prepare-ahead one|off]\n"
#else
                     "[--asr-backend cpu|cpu-int8|pl] [--pl-encoder FILE] [--embedding-capture DIR] [--prepare-ahead one|off]\n"
#endif
                     ;
        return 0;
    }
    std::signal(SIGINT, stop_signal);
    std::signal(SIGTERM, stop_signal);
    try {
        return service(options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "edge_speech_board: " << error.what() << '\n';
        return 1;
    }
}

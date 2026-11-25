#include "board/asr/ort_backend.hpp"
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
#include "board/asr/ort_asr_jobs.hpp"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(__MINGW32__) && !defined(_stdcall)
#define _stdcall __stdcall
#endif
#else
#include <dlfcn.h>
#endif

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <sstream>

namespace edge_speech::asr {
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
void add_ort_pl_custom_ops(Ort::SessionOptions& options, bool unified);
#endif
namespace {

using TimingClock = std::chrono::steady_clock;

std::uint64_t ns_between(TimingClock::time_point begin, TimingClock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

constexpr std::size_t state_count = 35;
constexpr std::size_t encoded_frames = 8;
constexpr std::size_t projection = 512;
constexpr std::size_t vocab_size = 6254;
constexpr auto floating = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto integer = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr char word_boundary[] = "\xE2\x96\x81";

#ifdef _WIN32
std::wstring native_path(const std::string& path) {
    const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, nullptr, 0);
    if (size <= 1) {
        throw std::invalid_argument("invalid UTF-8 ORT path");
    }
    std::wstring native(static_cast<std::size_t>(size), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, &native[0], size)) {
        throw std::invalid_argument("invalid UTF-8 ORT path");
    }
    native.pop_back();
    return native;
}
#else
const std::string& native_path(const std::string& path) {
    return path;
}
#endif

class RuntimeLibrary {
public:
    explicit RuntimeLibrary(std::string file) : file_(std::move(file)) {
        if (file_.find_first_of("/\\") == std::string::npos) {
            throw std::invalid_argument("ORT needs the explicit private library path");
        }
#ifdef _WIN32
        handle_ = LoadLibraryW(native_path(file_).c_str());
        const auto symbol = handle_ ? GetProcAddress(handle_, "OrtGetApiBase") : nullptr;
#else
        handle_ = dlopen(file_.c_str(), RTLD_LOCAL | RTLD_NOW);
        const auto symbol = handle_ ? dlsym(handle_, "OrtGetApiBase") : nullptr;
#endif
        try {
            if (!symbol) {
                throw std::runtime_error("unable to load OrtGetApiBase from " + file_);
            }
            decltype(&OrtGetApiBase) get_api_base = nullptr;
            static_assert(sizeof(get_api_base) == sizeof(symbol), "unsupported library symbol ABI");
            std::memcpy(&get_api_base, &symbol, sizeof(symbol));
            const auto* base = get_api_base();
            const std::string_view version = base->GetVersionString();
#if defined(__aarch64__) && defined(__linux__)
            if (version != "1.17.1") {
                throw std::runtime_error("board requires private ORT 1.17.1");
            }
#else
            if (version != "1.18.1" && version != "1.17.1") {
                throw std::runtime_error("unsupported ORT version");
            }
#endif
            const auto* api = base->GetApi(ORT_API_VERSION);
            if (!api) {
                throw std::runtime_error("ORT runtime does not expose the compiled C API version");
            }
            Ort::InitApi(api);
        } catch (...) {
            unload();
            throw;
        }
    }

    ~RuntimeLibrary() { unload(); }

    static void ensure(const std::string& file) {
        // ORT C++ objects store handles whose methods share this process-wide API.
        static const RuntimeLibrary runtime(file);
        if (runtime.file_ != file) {
            throw std::invalid_argument("a different private ORT runtime is already loaded");
        }
    }

private:
    void unload() {
#ifdef _WIN32
        if (handle_) {
            FreeLibrary(handle_);
        }
#else
        if (handle_) {
            dlclose(handle_);
        }
#endif
    }

    std::string file_;
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};

struct TensorSpec {
    std::string name;
    ONNXTensorElementDataType type;
    std::vector<std::int64_t> shape;
    std::vector<std::string> symbols;

    TensorSpec(std::string name, ONNXTensorElementDataType type,
               std::vector<std::int64_t> shape, std::size_t batch_axis)
        : name(std::move(name)), type(type), shape(std::move(shape)), symbols(this->shape.size()) {
        symbols.at(batch_axis) = "N";
    }

    TensorSpec(std::string name, ONNXTensorElementDataType type, std::vector<std::int64_t> shape)
        : name(std::move(name)), type(type), shape(std::move(shape)), symbols(this->shape.size()) {}
};

struct GraphSpec {
    std::vector<TensorSpec> inputs, outputs;
};

GraphSpec encoder_spec(bool pl) {
    auto input = pl ? TensorSpec("pl_embedding", floating, {1, embedding_frames, embedding_channels})
                    : TensorSpec("x", floating, {1, window_frames, feature_dim}, 0);
    GraphSpec spec{{std::move(input)},
                   {TensorSpec("encoder_out", floating, {1, encoded_frames, projection}, 0)}};
    spec.outputs.front().symbols[1] = "Addencoder_out_dim_1";
    constexpr std::array<std::int64_t, 5> left{64, 32, 16, 8, 32};
    constexpr std::array<const char*, 7> names{"len", "avg", "key", "val", "val2", "conv1", "conv2"};
    for (std::size_t kind = 0; kind < names.size(); ++kind) {
        for (std::size_t stage = 0; stage < left.size(); ++stage) {
            const auto name = std::string("cached_") + names[kind] + '_' + std::to_string(stage);
            const auto batch_axis = kind >= 2 && kind <= 4 ? 2 : 1;
            std::vector<std::int64_t> shape;
            switch (kind) {
            case 0: shape = {2, 1}; break;
            case 1: shape = {2, 1, 256}; break;
            case 2: shape = {2, left[stage], 1, 192}; break;
            case 3:
            case 4: shape = {2, left[stage], 1, 96}; break;
            default: shape = {2, 1, 256, 30}; break;
            }
            spec.inputs.emplace_back(name, kind == 0 ? integer : floating, shape, batch_axis);
            auto output = spec.inputs.back();
            output.name = "new_" + name;
            const auto symbolic = [&](std::size_t axis) {
                output.symbols.at(axis) = "Concat" + output.name + "_dim_" + std::to_string(axis);
            };
            if (kind == 1 && stage > 0) {
                symbolic(2);
            } else if (kind >= 2 && kind <= 4) {
                symbolic(1);
                symbolic(3);
            } else if (kind >= 5) {
                symbolic(3);
            }
            spec.outputs.push_back(std::move(output));
        }
    }
    return spec;
}

void validate_ports(const Ort::Session& session, const std::vector<TensorSpec>& specs, bool input) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    const auto count = input ? session.GetInputCount() : session.GetOutputCount();
    if (count != specs.size()) {
        throw std::invalid_argument("unexpected ONNX graph port count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto& spec = specs[index];
        const auto name = input ? session.GetInputNameAllocated(index, allocator) :
                                  session.GetOutputNameAllocated(index, allocator);
        const auto type = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
        if (name.get() != spec.name || type.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::invalid_argument("unexpected ONNX port " + spec.name);
        }
        const auto tensor = type.GetTensorTypeAndShapeInfo();
        const auto shape = tensor.GetShape();
        if (tensor.GetElementType() != spec.type || shape.size() != spec.shape.size()) {
            throw std::invalid_argument("unexpected ONNX dtype or rank for " + spec.name);
        }
        std::vector<const char*> symbols(shape.size());
        tensor.GetSymbolicDimensions(symbols.data(), symbols.size());
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            const auto expected = spec.symbols[axis].empty() ? spec.shape[axis] : -1;
            if (shape[axis] != expected ||
                std::string_view(symbols[axis] ? symbols[axis] : "") != spec.symbols[axis]) {
                throw std::invalid_argument("unexpected ONNX shape for " + spec.name);
            }
        }
    }
}

void validate_metadata(const Ort::Session& session,
                       std::initializer_list<std::pair<const char*, const char*>> required) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    const auto metadata = session.GetModelMetadata();
    for (const auto& [key, expected] : required) {
        const auto item = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (!item || std::string_view(item.get()) != expected) {
            throw std::invalid_argument(std::string("unexpected ONNX metadata: ") + key);
        }
    }
}

Ort::SessionOptions make_options(bool pl, bool unified, unsigned intra_threads);

class Graph {
public:
    Graph(const Ort::Env& env, const std::string& path, GraphSpec spec, const std::filesystem::path& profile_prefix,
          const char* graph_name, bool pl, bool unified, unsigned threads)
        : session(open(env, path, profile_prefix, graph_name, pl, unified, threads)), spec(std::move(spec)) {
        validate_ports(session, this->spec.inputs, true);
        validate_ports(session, this->spec.outputs, false);
        for (const auto& port : this->spec.inputs) input_names.push_back(port.name.c_str());
        for (const auto& port : this->spec.outputs) output_names.push_back(port.name.c_str());
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    std::vector<Ort::Value> run(const OrtValue* const* inputs) {
        std::vector<Ort::Value> results;
        results.reserve(output_names.size());
        std::array<OrtValue*, state_count + 1> raw{};
        Ort::RunOptions options;
        auto* status = Ort::GetApi().Run(session, options, input_names.data(), inputs, input_names.size(),
                                       output_names.data(), output_names.size(), raw.data());
        for (std::size_t index = 0; index < output_names.size(); ++index) {
            results.emplace_back(raw[index]);
        }
        Ort::ThrowOnError(status);
        for (std::size_t index = 0; index < results.size(); ++index) {
            const auto& expected = spec.outputs[index];
            if (!results[index] || !results[index].IsTensor()) {
                throw std::runtime_error("ORT output missing: " + expected.name);
            }
            const auto info = results[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != expected.type || info.GetShape() != expected.shape) {
                throw std::runtime_error("ORT runtime output shape or dtype mismatch: " + expected.name);
            }
            if (expected.type == floating) {
                const auto* data = results[index].GetTensorData<float>();
                const auto size = info.GetElementCount();
                if (!std::all_of(data, data + size, [](float f) { return std::isfinite(f); })) {
                    throw std::runtime_error("ORT returned nonfinite values: " + expected.name);
                }
            }
        }
        return results;
    }

    Ort::Session session;
    const GraphSpec spec;

private:
    static Ort::Session open(const Ort::Env& env, const std::string& path, std::filesystem::path prefix,
                             const char* graph_name, bool pl, bool unified, unsigned threads) {
        auto options = make_options(pl, unified, threads);
        if (!prefix.empty()) {
            prefix += std::string("-") + graph_name;
            options.EnableProfiling(prefix.c_str());
        }
        return Ort::Session(env, native_path(path).c_str(), options);
    }

    std::vector<const char*> input_names, output_names;
};

std::vector<std::string> load_tokens(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::invalid_argument("cannot open ONNX model tokens");
    }
    std::vector<std::string> tokens(vocab_size + 3);
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream entry(line);
        std::string token, extra;
        std::int64_t index;
        if (!(entry >> token >> index) || (entry >> extra) || index < 0 ||
            static_cast<std::size_t>(index) >= tokens.size() || !tokens[index].empty()) {
            throw std::invalid_argument("invalid Zipformer token entry");
        }
        tokens[index] = std::move(token);
    }
    if (!stream.eof() || std::find(tokens.begin(), tokens.end(), "") != tokens.end() ||
        tokens[0] != "<blk>" || tokens[1] != "<sos/eos>" || tokens[2] != "<unk>" ||
        tokens[vocab_size] != "#0" || tokens[vocab_size + 1] != "#1" || tokens[vocab_size + 2] != "#2") {
        throw std::invalid_argument("tokens do not match Zipformer vocabulary contract");
    }
    return tokens;
}

Ort::Env make_env(const std::string& library) {
    RuntimeLibrary::ensure(library);
    return Ort::Env(ORT_LOGGING_LEVEL_WARNING, "edge-speech");
}

Ort::SessionOptions make_options(bool pl, bool unified, unsigned intra_threads) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(static_cast<int>(intra_threads));
    options.SetInterOpNumThreads(1);
#ifdef EDGE_SPEECH_ENABLE_ORT_PL
    if (pl || unified) {
        options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        add_ort_pl_custom_ops(options, unified);
    }
#else
    if (pl || unified) throw std::logic_error("ORT PL support was not built");
#endif
    return options;
}

struct BackendIdentity {};

struct OrtState final : ChunkState {
    using EncoderCache = std::vector<Ort::Value>;
    std::shared_ptr<const EncoderCache> cache;
    std::shared_ptr<const BackendIdentity> owner;
    std::optional<ChunkStamp> last_chunk;
    std::array<std::int64_t, 2> context{-1, 0};
    std::array<float, projection> decoder{};
    std::vector<std::int64_t> segment_tokens;
    std::uint64_t trailing = 0;
};

class OrtBackend final : public ChunkBackend {
public:
    explicit OrtBackend(const OrtModelPaths& paths, OrtEmbeddingSubmit submit = {}, OrtCpuDiagnostics diagnostics = {}
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
                        , std::optional<OrtAsrJobs> jobs = {}
#endif
                        )
        : submit_(std::move(submit)),
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
          jobs_(std::move(jobs)), unified_(jobs_.has_value()),
#endif
          owner_(submit_ || unified_ ? std::make_shared<BackendIdentity>() : nullptr),
          diagnostics_(std::move(diagnostics)), tokens_(load_tokens(paths.tokens)),
          env_(make_env(paths.runtime_library)),
          encoder_(env_, paths.encoder, encoder_spec(bool(submit_)), diagnostics_.profile_prefix,
                   "encoder", bool(submit_), unified_, diagnostics_.intra_threads[0]),
          decoder_(env_, paths.decoder,
                   {{{"y", integer, {1, 2}, 0}}, {{"decoder_out", floating, {1, projection}, 0}}},
                   diagnostics_.profile_prefix, "decoder", bool(submit_), unified_, diagnostics_.intra_threads[1]),
          joiner_(env_, paths.joiner,
                  {{{"encoder_out", floating, {1, projection}, 0}, {"decoder_out", floating, {1, projection}, 0}},
                    {{"logit", floating, {1, vocab_size}, 0}}},
                   diagnostics_.profile_prefix, "joiner", bool(submit_), unified_, diagnostics_.intra_threads[2]),
          cpu_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        validate_metadata(encoder_.session, {{"model_type", "zipformer"}, {"version", "1"},
            {"model_author", "k2-fsa"}, {"T", "39"}, {"decode_chunk_len", "32"}, {"num_encoder_layers", "2,2,2,2,2"},
            {"encoder_dims", "256,256,256,256,256"}, {"attention_dims", "192,192,192,192,192"},
            {"cnn_module_kernels", "31,31,31,31,31"}, {"left_context_len", "64,32,16,8,32"}});
        if (submit_) validate_metadata(encoder_.session, {{"edge_speech_pl_abi", "t015-ab-embedding-pl-v1"}});
        if (diagnostics_.require_int8_encoder)
            validate_metadata(encoder_.session, {{"fz_asr_cpu_quantization", "ort-cpu-int8"}});
        if (diagnostics_.require_int8_decoder)
            validate_metadata(decoder_.session, {{"fz_asr_cpu_quantization", "ort-cpu-int8"}});
        if (diagnostics_.require_int8_joiner)
            validate_metadata(joiner_.session, {{"fz_asr_cpu_quantization", "ort-cpu-int8"}});
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (unified_) validate_metadata(encoder_.session, {{"edge_speech_pl_abi", "fz-asr-job-v3"},
            {"fz_asr_layout_sha256", FZ_ASR_LAYOUT_SHA256_HEX}, {"fz_asr_model_revision", fz_asr_model_revision},
            {"fz_asr_w8_sha256", fz_asr_w8_pack_sha256}});
#endif
        validate_metadata(decoder_.session, {{"context_size", "2"}, {"vocab_size", "6254"}});
        validate_metadata(joiner_.session, {{"joiner_dim", "512"}});
    }

    std::shared_ptr<const ChunkState> initial_state() override {
        auto state = std::make_shared<OrtState>();
        state->owner = owner_;
        auto cache = std::make_shared<OrtState::EncoderCache>();
        cache->reserve(state_count);
        auto allocator = Ort::AllocatorWithDefaultOptions();
        for (std::size_t index = 1; index < encoder_.spec.inputs.size(); ++index) {
            const auto& spec = encoder_.spec.inputs[index];
            auto tensor = Ort::Value::CreateTensor(allocator, spec.shape.data(), spec.shape.size(), spec.type);
            const auto size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
            if (spec.type == floating) {
                std::fill_n(tensor.GetTensorMutableData<float>(), size, 0.0F);
            } else {
                std::fill_n(tensor.GetTensorMutableData<std::int64_t>(), size, 0);
            }
            cache->push_back(std::move(tensor));
        }
        state->cache = std::move(cache);
        run_decoder(*state);
        return state;
    }

    std::shared_ptr<const ChunkState> continue_segment(
        const std::shared_ptr<const ChunkState>& state) override {
        const auto old = checked_state(state);
        auto next = std::make_shared<OrtState>();
        next->cache = old->cache;
        next->owner = old->owner;
        next->last_chunk = old->last_chunk;
        next->context = old->context;
        next->decoder = old->decoder;
        next->trailing = old->trailing;
        // The previous segment's exact token IDs have been returned to the
        // stream owner. Retaining only future display tokens bounds history.
        return next;
    }

    ChunkResult run(const ChunkRequest& request) override {
        const auto old = checked_state(request.input_state);
        std::unique_lock<std::mutex> device_lock(device_mutex_, std::defer_lock);
        if (submit_ || unified_) {
            device_lock.lock();
            validate_progress(request.stamp, old->last_chunk);
        }
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (jobs_) return run_jobs(request, *old);
#endif
        return infer(request, *old);
    }

    std::array<std::string, 3> finish_profile() {
        if (diagnostics_.profile_prefix.empty() || profile_finished_)
            throw std::logic_error("ORT profile is unavailable or already finished");
        profile_finished_ = true;
        Ort::AllocatorWithDefaultOptions allocator;
        std::array<std::string, 3> files;
        std::exception_ptr failure;
        const std::array<Ort::Session*, 3> sessions{&encoder_.session, &decoder_.session, &joiner_.session};
        for (std::size_t i = 0; i < files.size(); ++i) {
            try {
                const auto path = sessions[i]->EndProfilingAllocated(allocator);
                if (!path || !*path || !std::filesystem::is_regular_file(std::filesystem::u8path(path.get())))
                    throw std::runtime_error("ORT profile file was not written");
                files[i] = path.get();
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        return files;
    }

private:
    ChunkResult infer(const ChunkRequest& request, const OrtState& old) {
        const bool timing = bool(diagnostics_.on_chunk);
        const bool detailed = timing && diagnostics_.detailed_timing;
        bool jobs_timing = false;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        jobs_timing = jobs_ && bool(jobs_->on_chunk);
#endif
        OrtCpuChunkTrace measured{request.stamp};
        const auto marshal_start = detailed ? TimingClock::now() : TimingClock::time_point{};
        // ORT's tensor view is mutable. The caller's features and replay cache
        // stay independent of any subsequent ORT work.
        auto window = request.features;
        OrtEmbeddingFeatures embedded;
        if (submit_) embedded = embed(request);
        auto x = submit_ ? view(embedded.data(), embedded.size(), encoder_.spec.inputs[0])
                         : view(window.data(), window.size(), encoder_.spec.inputs[0]);
        std::array<const OrtValue*, state_count + 1> encoder_inputs{};
        encoder_inputs[0] = x;
        for (std::size_t index = 0; index < old.cache->size(); ++index) {
            encoder_inputs[index + 1] = (*old.cache)[index];
        }
        const auto encoder_start = timing || jobs_timing ? TimingClock::now() : TimingClock::time_point{};
        auto outputs = encoder_.run(encoder_inputs.data());
        const auto cache_start = detailed || jobs_timing ? TimingClock::now() : TimingClock::time_point{};
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (unified_) OrtAsrJobScope::current().end_encoder(jobs_timing ? ns_between(encoder_start, cache_start) : 0);
#endif
        auto next = std::make_shared<OrtState>();
        next->owner = owner_;
        if (submit_ || unified_) next->last_chunk = request.stamp;
        next->cache = std::make_shared<OrtState::EncoderCache>(
            std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
        next->context = old.context;
        next->decoder = old.decoder;
        next->segment_tokens = old.segment_tokens;
        next->trailing = old.trailing;
        auto* encoded = outputs[0].GetTensorMutableData<float>();
        std::size_t frame = 0;
        const auto decoder_start = timing || jobs_timing ? TimingClock::now() : TimingClock::time_point{};
        for (; frame < encoded_frames; ++frame) {
            auto row = view(encoded + frame * projection, projection, joiner_.spec.inputs[0]);
            auto decoder = view(next->decoder.data(), projection, joiner_.spec.inputs[1]);
            const std::array<const OrtValue*, 2> joiner_inputs{row, decoder};
            const auto joiner_start = detailed ? TimingClock::now() : TimingClock::time_point{};
            const auto logits = joiner_.run(joiner_inputs.data());
            const auto greedy_start = detailed ? TimingClock::now() : TimingClock::time_point{};
            if (detailed) measured.joiner_ns += ns_between(joiner_start, greedy_start);
            const auto* first = logits[0].GetTensorData<float>();
            const auto token = std::max_element(first, first + vocab_size) - first;
            if (token == 0 || token == 2) {
                ++next->trailing;
                if (detailed) measured.greedy_ns += ns_between(greedy_start, TimingClock::now());
                continue;
            }
            next->trailing = 0;
            next->context = {next->context[1], token};
            next->segment_tokens.push_back(token);
            const auto token_decoder_start = detailed ? TimingClock::now() : TimingClock::time_point{};
            run_decoder(*next);
            if (detailed) {
                measured.greedy_ns += ns_between(greedy_start, token_decoder_start);
                measured.decoder_ns += ns_between(token_decoder_start, TimingClock::now());
                ++measured.decoder_calls;
            }
        }
        const auto text_start = timing || jobs_timing ? TimingClock::now() : TimingClock::time_point{};
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (jobs_timing) OrtAsrJobScope::current().end_decoder(ns_between(decoder_start, text_start));
#endif
        std::string text;
        for (const auto token : next->segment_tokens) {
            text += tokens_.at(token);
        }
        for (std::size_t at = text.find(word_boundary); at != std::string::npos;
             at = text.find(word_boundary, at + 1)) {
            text.replace(at, sizeof(word_boundary) - 1, " ");
        }
        const auto first = text.find_first_not_of(' ');
        if (first == std::string::npos) {
            text.clear();
        } else {
            text = text.substr(first, text.find_last_not_of(' ') - first + 1);
        }
        Backend source = submit_ ? Backend::pl : Backend::cpu;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
        if (unified_ && jobs_->enabled.any()) source = jobs_->enabled.all() ? Backend::pl : Backend::mixed;
#endif
        if (timing) {
            measured.encoder_ns = ns_between(encoder_start, detailed ? cache_start : decoder_start);
            measured.decoder_joiner_ns = ns_between(decoder_start, text_start);
            if (detailed) {
                measured.marshal_ns = ns_between(marshal_start, encoder_start);
                measured.cache_ns = ns_between(cache_start, decoder_start);
                measured.text_ns = ns_between(text_start, TimingClock::now());
                measured.joiner_calls = static_cast<std::uint32_t>(frame);
            }
            diagnostics_.on_chunk(measured);
        }
        return {request.stamp, next, std::move(text), next->trailing, source,
                next->segment_tokens, frame};
    }

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    ChunkResult run_jobs(const ChunkRequest& request, const OrtState& old) {
        const auto& stamp = request.stamp;
        if (last_jobs_) {
            const auto& last = last_jobs_->result.stamp;
            if (stamp == last) {
                if (last_jobs_->source_state != request.input_state ||
                    std::memcmp(last_jobs_->input.data(), request.features.data(), sizeof(FeatureWindow)) != 0) {
                    throw std::invalid_argument("ASR job retry changed its input state or feature window");
                }
                return last_jobs_->result;
            }
            if (stamp.session_id != last.session_id || stamp.stream_generation < last.stream_generation ||
                (stamp.stream_generation == last.stream_generation &&
                 (stamp.state_epoch != last.chunk_id ||
                  old.cache != checked_state(last_jobs_->result.state)->cache))) {
                throw std::invalid_argument("ASR job request does not advance the last committed transaction");
            }
        }
        if (!std::all_of(request.features.begin(), request.features.end(), [](float value) { return std::isfinite(value); })) {
            throw std::invalid_argument("nonfinite ASR job feature window");
        }
        OrtAsrJobScope chunk(*jobs_, stamp);
        CompletedJobs completed{request.features, request.input_state, infer(request, old)};
        chunk.commit();
        // Event publication can be retried after this backend returns. Save
        // the complete result so the replay cannot advance the lease twice.
        last_jobs_.emplace(std::move(completed));
        chunk.report();
        return last_jobs_->result;
    }
#endif

    std::shared_ptr<const OrtState> checked_state(const std::shared_ptr<const ChunkState>& state) const {
        auto ort_state = std::dynamic_pointer_cast<const OrtState>(state);
        if (!ort_state || !ort_state->cache || ort_state->cache->size() != state_count || ort_state->owner != owner_) {
            throw std::invalid_argument("missing or foreign Zipformer cache");
        }
        return ort_state;
    }

    static void validate_progress(const ChunkStamp& stamp, const std::optional<ChunkStamp>& previous) {
        if (stamp.session_id.empty() || !stamp.stream_generation || !stamp.chunk_id ||
            stamp.state_epoch != stamp.chunk_id - 1 ||
            (previous ? (previous->session_id != stamp.session_id || previous->stream_generation != stamp.stream_generation ||
                         previous->chunk_id != stamp.state_epoch) : stamp.state_epoch != 0)) {
            throw std::invalid_argument("PL request does not follow its input state");
        }
    }

    const OrtEmbeddingFeatures& embed(const ChunkRequest& request) {
        if (device_failed_) throw std::runtime_error("PL Embedding session is no longer usable");
        const auto& stamp = request.stamp;
        if (completed_ && completed_->result.stamp.session_id == stamp.session_id) {
            const auto& last = completed_->result.stamp;
            if (stamp == last) {
                if (std::memcmp(completed_->input.data(), request.features.data(),
                                request.features.size() * sizeof(float)) != 0) {
                    throw std::invalid_argument("PL retry changed the submitted features");
                }
                return completed_->result.features;
            }
            if (stamp.stream_generation < last.stream_generation ||
                (stamp.stream_generation == last.stream_generation && stamp.chunk_id <= last.chunk_id)) {
                throw std::invalid_argument("PL request precedes the last completed job");
            }
        }
        if (!std::all_of(request.features.begin(), request.features.end(), [](float value) { return std::isfinite(value); })) {
            throw std::invalid_argument("nonfinite PL Embedding input");
        }
        try {
            auto result = submit_(stamp, request.features);
            if (!(result.stamp == stamp) || !std::all_of(result.features.begin(), result.features.end(),
                                                        [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("stale or nonfinite PL Embedding completion");
            }
            completed_.emplace(CompletedEmbedding{request.features, std::move(result)});
        } catch (...) {
            device_failed_ = true;
            throw;
        }
        return completed_->result.features;
    }

    template <typename T>
    Ort::Value view(T* data, std::size_t count, const TensorSpec& spec) const {
        return Ort::Value::CreateTensor(cpu_, data, count, spec.shape.data(), spec.shape.size());
    }

    void run_decoder(OrtState& state) {
        auto context = view(state.context.data(), state.context.size(), decoder_.spec.inputs[0]);
        const OrtValue* input = context;
        const auto result = decoder_.run(&input);
        std::copy_n(result[0].GetTensorData<float>(), projection, state.decoder.begin());
    }

    struct CompletedEmbedding {
        FeatureWindow input;
        OrtEmbeddingResult result;
    };

    const OrtEmbeddingSubmit submit_;
#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
    struct CompletedJobs {
        FeatureWindow input;
        std::shared_ptr<const ChunkState> source_state;
        ChunkResult result;
    };
    const std::optional<OrtAsrJobs> jobs_;
    std::optional<CompletedJobs> last_jobs_;
#endif
    const bool unified_ = false;
    const std::shared_ptr<const BackendIdentity> owner_;
    const OrtCpuDiagnostics diagnostics_;
    bool profile_finished_ = false;
    std::mutex device_mutex_;
    std::optional<CompletedEmbedding> completed_;
    bool device_failed_ = false;
    const std::vector<std::string> tokens_;
    Ort::Env env_;
    Graph encoder_, decoder_, joiner_;
    Ort::MemoryInfo cpu_;
};

}  // namespace

std::shared_ptr<ChunkBackend> make_ort_cpu_backend(const OrtModelPaths& paths, OrtCpuDiagnostics diagnostics) {
    return std::make_shared<OrtBackend>(paths, OrtEmbeddingSubmit{}, std::move(diagnostics));
}

std::array<std::string, 3> finish_ort_profile(ChunkBackend& backend) {
    return dynamic_cast<OrtBackend&>(backend).finish_profile();
}

std::shared_ptr<ChunkBackend> make_ort_pl_backend(const OrtModelPaths& paths, OrtEmbeddingSubmit submit) {
    if (!submit) throw std::invalid_argument("ORT PL backend requires an Embedding submitter");
    return std::make_shared<OrtBackend>(paths, std::move(submit));
}

#ifdef EDGE_SPEECH_ENABLE_ASR_JOBS
std::shared_ptr<ChunkBackend> make_ort_asr_jobs_backend(const OrtModelPaths& paths, OrtAsrJobs jobs,
                                                       OrtCpuDiagnostics diagnostics) {
    if (jobs.enabled.any() != bool(jobs.transport)) throw std::invalid_argument("ASR job selector and transport disagree");
    if ((jobs.paired_ffn & ~jobs.enabled).any())
        throw std::invalid_argument("paired FFN requires an enabled PL call");
    for (std::uint16_t id = 0; id < jobs.paired_ffn.size(); ++id) {
        if (jobs.paired_ffn.test(id) && (fz_asr_calls[id].operation != 0 || !jobs.transport->supports_paired_ffn(id)))
            throw std::invalid_argument("unsupported paired FFN call " + std::to_string(id));
    }
    return std::make_shared<OrtBackend>(paths, OrtEmbeddingSubmit{}, std::move(diagnostics), std::move(jobs));
}
#endif

}  // namespace edge_speech::asr

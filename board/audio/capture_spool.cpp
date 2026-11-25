#include "board/audio/capture_spool.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace edge_speech::audio {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto session_limit = std::chrono::hours(1);
constexpr std::uint64_t sample_limit = output_rate * 60 * 60;
constexpr std::uint64_t unknown_end = UINT64_MAX;
constexpr std::uint64_t max_gap_detail = 4096;
constexpr char index_magic[] = "FZPCM01\n";

using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
using Record = std::array<std::uint8_t, 5 * sizeof(std::uint64_t)>;

File open_file(const std::filesystem::path& path, bool writer) {
#ifdef _WIN32
    File file(_wfopen(path.c_str(), writer ? L"wbx" : L"rb"), &std::fclose);
#else
    File file(std::fopen(path.c_str(), writer ? "wbx" : "rb"), &std::fclose);
#endif
    if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0) {
        throw CaptureStorageError("cannot open capture file: " + path.u8string());
    }
    return file;
}

void write_bytes(const File& file, const void* bytes, std::size_t count) {
    errno = 0;
    if (std::fwrite(bytes, 1, count, file.get()) != count) {
        const auto error = errno;
        throw CaptureStorageError(error ? "cannot write captured audio: " +
            std::error_code(error, std::generic_category()).message() : "short write in captured audio");
    }
}

void read_bytes(const File& file, void* bytes, std::size_t count) {
    if (std::fread(bytes, 1, count, file.get()) != count) {
        throw CaptureStorageError("cannot read committed audio or capture index");
    }
}

void sync_file(const File& file) {
#ifdef _WIN32
    const auto result = _commit(_fileno(file.get()));
#else
    const auto result = fsync(fileno(file.get()));
#endif
    if (result != 0) throw CaptureStorageError("cannot sync captured audio or capture index");
}

// After the magic, each record contains five little-endian u64 fields.
// PCM: 0, sample_start, sample_end, audio_flags, 0.
// Gap: 1+xrun/disconnect/timeout/queue-overrun enum, start, end (UINT64_MAX if
// unknown), detection time, detail length, followed by those detail bytes.
Record record(std::array<std::uint64_t, 5> fields) {
    Record bytes{};
    for (std::size_t i = 0; i < fields.size(); ++i) {
        for (std::size_t j = 0; j < sizeof(fields[i]); ++j) {
            bytes[i * sizeof(fields[i]) + j] = static_cast<std::uint8_t>(fields[i] >> (8 * j));
        }
    }
    return bytes;
}

}  // namespace

struct SpoolCaptureSource::Impl {
    Impl(std::unique_ptr<AudioSource> capture, WireUuid id, const std::filesystem::path& directory,
         OnStored callback)
        : source(std::move(capture)), session(id), on_stored(std::move(callback)) {
        if (!source || directory.empty()) throw std::invalid_argument("capture spool requires a source and directory");
        try {
            if (directory.has_parent_path()) std::filesystem::create_directories(directory.parent_path());
            if (!std::filesystem::create_directory(directory)) {
                throw CaptureStorageError("capture directory already exists: " + directory.u8string());
            }
        } catch (const std::filesystem::filesystem_error& error) {
            throw CaptureStorageError(error.what());
        }
        const auto pcm = directory / "audio.s16le";
        const auto index = directory / "audio.index";
        pcm_out = open_file(pcm, true);
        index_out = open_file(index, true);
        write_bytes(index_out, index_magic, sizeof(index_magic) - 1);
        pcm_in = open_file(pcm, false);
        index_in = open_file(index, false);
        char magic[sizeof(index_magic) - 1];
        read_bytes(index_in, magic, sizeof(magic));
        if (std::string_view(magic, sizeof(magic)) != index_magic) throw CaptureStorageError("invalid capture index");
        worker = std::thread(&Impl::run, this);
    }

    ~Impl() { stop(); }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!stop_requested && source) source->request_stop();
            stop_requested = true;
        }
        wake.notify_one();
    }

    void stop() {
        request_stop();
        if (worker.joinable()) worker.join();
    }

    std::optional<CaptureEvent> read();
    void store(CaptureEvent event, std::uint64_t& end);
    void remember_failure(std::exception_ptr& error);
    void run() noexcept;

    std::unique_ptr<AudioSource> source;
    const WireUuid session;
    const OnStored on_stored;
    File pcm_out{nullptr, &std::fclose}, index_out{nullptr, &std::fclose};
    File pcm_in{nullptr, &std::fclose}, index_in{nullptr, &std::fclose};
    mutable std::mutex mutex;
    std::condition_variable wake;
    CaptureProgress progress;
    std::uint64_t stored_events = 0, read_events = 0, read_end = 0;
    std::exception_ptr failure;
    bool stop_requested = false;
    std::thread worker;
};

std::optional<CaptureEvent> SpoolCaptureSource::Impl::read() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (read_events == stored_events) {
            if (failure) std::rethrow_exception(failure);
            return std::nullopt;
        }
    }
    Record bytes;
    read_bytes(index_in, bytes.data(), bytes.size());
    const auto field = [&](std::size_t i) {
        return edge_speech::detail::read_le<std::uint64_t>(&bytes[i * sizeof(std::uint64_t)]);
    };
    const auto kind = field(0), start = field(1), end = field(2), extra = field(3), size = field(4);
    if (start != read_end || (end != unknown_end && (end < start || end > sample_limit))) {
        throw CaptureStorageError("inconsistent capture index timeline");
    }
    ++read_events;
    if (end != unknown_end) read_end = end;
    if (kind == 0) {
        if (end == unknown_end || end == start || end - start > samples_per_block || extra > UINT32_MAX || size) {
            throw CaptureStorageError("invalid PCM capture index record");
        }
        const auto count = static_cast<std::uint32_t>(end - start);
        AudioBlock block{session, start, count, static_cast<std::uint32_t>(extra), Bytes(2 * count)};
        read_bytes(pcm_in, block.pcm_s16le.data(), block.pcm_s16le.size());
        return CaptureEvent{std::move(block)};
    }
    if (kind > 4 || size > max_gap_detail || end == start) throw CaptureStorageError("invalid gap capture index record");
    CaptureGap gap{start, end == unknown_end ? std::nullopt : std::optional<std::uint64_t>{end},
                   static_cast<CaptureGapReason>(kind - 1), extra, std::string(static_cast<std::size_t>(size), '\0')};
    read_bytes(index_in, gap.detail.data(), gap.detail.size());
    return CaptureEvent{std::move(gap)};
}

void SpoolCaptureSource::Impl::store(CaptureEvent event, std::uint64_t& end) {
    auto next = end;
    if (auto* block = std::get_if<AudioBlock>(&event)) {
        if (block->session_id != session || block->sample_start != end || !block->sample_count ||
            block->sample_count > samples_per_block || block->pcm_s16le.size() != 2 * block->sample_count) {
            throw std::invalid_argument("invalid capture spool PCM extent");
        }
        const auto count = std::min<std::uint64_t>(block->sample_count, sample_limit - end);
        block->sample_count = static_cast<std::uint32_t>(count);
        block->pcm_s16le.resize(2 * block->sample_count);
        next = end + count;
        const auto entry = record({0, end, next, block->audio_flags, 0});
        write_bytes(pcm_out, block->pcm_s16le.data(), block->pcm_s16le.size());
        write_bytes(index_out, entry.data(), entry.size());
    } else {
        auto& gap = std::get<CaptureGap>(event);
        if (gap.sample_start != end || (gap.sample_end && *gap.sample_end <= end) ||
            gap.detail.size() > max_gap_detail) {
            throw std::invalid_argument("invalid capture spool gap extent or detail");
        }
        if (gap.sample_end) gap.sample_end = next = std::min(*gap.sample_end, sample_limit);
        const auto entry = record({1 + static_cast<std::uint64_t>(gap.reason), end,
                                   gap.sample_end ? next : unknown_end, gap.detected_at_ns, gap.detail.size()});
        write_bytes(index_out, entry.data(), entry.size());
        write_bytes(index_out, gap.detail.data(), gap.detail.size());
    }
    end = next;
    {
        std::lock_guard<std::mutex> lock(mutex);
        ++stored_events;
        progress.sample_end = end;
    }
    if (on_stored) on_stored(event);
}

void SpoolCaptureSource::Impl::remember_failure(std::exception_ptr& error) {
    if (error) return;
    error = std::current_exception();
    try {
        std::rethrow_exception(error);
    } catch (const CaptureStorageError&) {
        // The published prefix cannot move beyond the failed write. Report it
        // independently of the decoder and of the remaining close/sync work.
        std::lock_guard<std::mutex> lock(mutex);
        progress.storage_failed = true;
    } catch (...) {
    }
}

void SpoolCaptureSource::Impl::run() noexcept {
    const auto deadline = Clock::now() + session_limit;
    std::exception_ptr error;
    std::uint64_t end = 0;
    try {
        while (end < sample_limit && Clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (stop_requested) break;
            }
            if (auto event = source->read()) {
                store(std::move(*event), end);
            } else {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait_for(lock, std::chrono::milliseconds(2), [this] { return stop_requested; });
            }
        }
    } catch (...) {
        remember_failure(error);
    }
    try {
        source->stop();
        while (!error && end < sample_limit) {
            auto event = source->read();
            if (!event) break;
            store(std::move(*event), end);
        }
    } catch (...) {
        remember_failure(error);
    }
    // Keep both files for recovery, including on a short or failed write.
    for (const auto* file : {&pcm_out, &index_out}) {
        try {
            sync_file(*file);
        } catch (...) {
                remember_failure(error);
        }
    }
    pcm_out.reset();
    index_out.reset();
    if (error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& reason) {
            std::fprintf(stderr, "capture stopped: %s\n", reason.what());
        } catch (...) {
            std::fputs("capture stopped with an unknown error\n", stderr);
        }
    }
    std::lock_guard<std::mutex> lock(mutex);
    source.reset();
    failure = std::move(error);
    progress.sealed = true;
}

SpoolCaptureSource::SpoolCaptureSource(std::unique_ptr<AudioSource> source, WireUuid session,
                                       const std::filesystem::path& directory, OnStored on_stored)
    : impl_(std::make_unique<Impl>(std::move(source), session, directory, std::move(on_stored))) {}

SpoolCaptureSource::~SpoolCaptureSource() = default;
void SpoolCaptureSource::request_stop() { impl_->request_stop(); }
void SpoolCaptureSource::stop() { impl_->stop(); }
std::optional<CaptureEvent> SpoolCaptureSource::read() { return impl_->read(); }
std::optional<CaptureProgress> SpoolCaptureSource::capture_progress() const { return progress(); }

CaptureProgress SpoolCaptureSource::progress() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->progress;
}

std::unique_ptr<SpoolCaptureSource> make_spooled_capture_source(
    std::unique_ptr<AudioSource> source, WireUuid session, const std::filesystem::path& directory,
    SpoolCaptureSource::OnStored on_stored) {
    return std::make_unique<SpoolCaptureSource>(std::move(source), session, directory, std::move(on_stored));
}

}  // namespace edge_speech::audio

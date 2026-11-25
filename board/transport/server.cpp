#include "board/transport/server.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

namespace edge_speech::transport {
namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

#ifdef _WIN32
using SocketId = SOCKET;
using AddressLength = int;
constexpr SocketId invalid_socket = INVALID_SOCKET;
bool can_retry() { const int code = WSAGetLastError(); return code == WSAEWOULDBLOCK || code == WSAEINTR; }
void close_socket(SocketId id) { ::closesocket(id); }

struct NetworkRuntime {
    NetworkRuntime() {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("FZAS could not initialize Winsock");
        }
    }
    ~NetworkRuntime() { WSACleanup(); }
};

void nonblocking(SocketId id) {
    u_long mode = 1;
    if (::ioctlsocket(id, FIONBIO, &mode) != 0) {
        throw std::runtime_error("FZAS could not configure socket");
    }
}
#else
using SocketId = int;
using AddressLength = socklen_t;
constexpr SocketId invalid_socket = -1;
bool can_retry() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR; }
void close_socket(SocketId id) { ::close(id); }
struct NetworkRuntime {};

void nonblocking(SocketId id) {
    if (id >= FD_SETSIZE || ::fcntl(id, F_SETFL, O_NONBLOCK) == -1 ||
        ::fcntl(id, F_SETFD, FD_CLOEXEC) == -1) {
        throw std::runtime_error("FZAS could not configure socket");
    }
}
#endif

class Socket {
public:
    explicit Socket(SocketId id = invalid_socket) : id_(id) {}
    ~Socket() { if (*this) close_socket(id_); }
    Socket(Socket&& other) noexcept : id_(std::exchange(other.id_, invalid_socket)) {}
    Socket& operator=(Socket&& other) noexcept {
        std::swap(id_, other.id_);
        return *this;
    }
    explicit operator bool() const { return id_ != invalid_socket; }
    SocketId get() const { return id_; }

private:
    SocketId id_;
};

Socket listen_loopback(std::uint16_t port) {
    Socket listener(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (!listener) throw std::runtime_error("FZAS could not create listener");
    const int enabled = 1;
#ifdef _WIN32
    constexpr int option = SO_EXCLUSIVEADDRUSE;
#else
    constexpr int option = SO_REUSEADDR;
#endif
    if (::setsockopt(listener.get(), SOL_SOCKET, option,
                     reinterpret_cast<const char*>(&enabled), sizeof(enabled)) != 0) {
        throw std::runtime_error("FZAS could not configure listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0) {
        throw std::runtime_error("FZAS could not bind loopback listener");
    }
    nonblocking(listener.get());
    return listener;
}

std::uint16_t bound_port(const Socket& listener) {
    sockaddr_in address{};
    AddressLength length = sizeof(address);
    if (!listener || ::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        throw std::logic_error("FZAS listener is not running");
    }
    return ntohs(address.sin_port);
}

enum class IoResult { pending, closed, complete };

class WireError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ReadFrame {
    Bytes wire = Bytes(fzas_header_bytes);
    std::size_t position = 0;
    bool have_header = false;
    TimePoint deadline{};

    IoResult read(SocketId peer, std::chrono::milliseconds timeout) {
        const auto count = ::recv(peer, reinterpret_cast<char*>(wire.data() + position),
                                  static_cast<int>(wire.size() - position), 0);
        if (count < 0 && can_retry()) return IoResult::pending;
        if (count < 0) throw WireError("FZAS read failed");
        if (count == 0) {
            if (position != 0) throw WireError("FZAS stream ended during a partial frame");
            return IoResult::closed;
        }
        if (position == 0) deadline = Clock::now() + timeout;
        position += static_cast<std::size_t>(count);
        if (position == fzas_header_bytes && !have_header) {
            try {
                const auto header = decode_frame_header(wire.data(), wire.size());
                wire.resize(fzas_header_bytes + header.payload_bytes);
            } catch (const std::invalid_argument& error) {
                throw WireError(error.what());
            }
            have_header = true;
        }
        return position == wire.size() ? IoResult::complete : IoResult::pending;
    }

    Frame take() {
        try {
            auto frame = decode_frame(wire);
            wire.resize(fzas_header_bytes);
            have_header = false;
            position = 0;
            return frame;
        } catch (const std::invalid_argument& error) {
            throw WireError(error.what());
        }
    }
};

struct WriteFrame {
    Bytes wire;
    std::size_t position = 0;
    TimePoint deadline{};

    bool pending() const { return position < wire.size(); }

    void prepare(Frame frame, std::uint64_t& seq, std::chrono::milliseconds timeout) {
        if (seq == std::numeric_limits<std::uint64_t>::max()) {
            throw WireError("FZAS send sequence exhausted");
        }
        frame.transport_seq = ++seq;
        wire = encode_frame(frame);
        position = 0;
        deadline = Clock::now() + timeout;
    }

    void write(SocketId peer) {
#ifdef _WIN32
        constexpr int flags = 0;
#else
        constexpr int flags = MSG_NOSIGNAL;
#endif
        const auto count = ::send(peer, reinterpret_cast<const char*>(wire.data() + position),
                                  static_cast<int>(wire.size() - position), flags);
        if (count < 0 && can_retry()) return;
        if (count <= 0) throw WireError("FZAS write failed");
        position += static_cast<std::size_t>(count);
    }
};

Bytes bytes(std::string_view text) { return Bytes(text.begin(), text.end()); }

void append_string(std::string& json, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    json += '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            json += '\\';
            json += static_cast<char>(ch);
        } else if (ch < 0x20) {
            json += "\\u00";
            json += hex[ch >> 4];
            json += hex[ch & 0xf];
        } else {
            json += static_cast<char>(ch);
        }
    }
    json += '"';
}

std::string_view backend_name(Backend backend) {
    switch (backend) {
    case Backend::cpu: return "cpu";
    case Backend::pl: return "pl";
    case Backend::mixed: return "mixed";
    }
    throw std::invalid_argument("invalid transcript backend");
}

std::string transcript_json(const TranscriptEvent& event) {
    validate_transcript_revision(event);
    if (event.text.size() > fzas_json_payload_limit || event.model_release.size() > fzas_json_payload_limit) {
        throw std::invalid_argument("FZAS transcript exceeds JSON limit");
    }
    std::string json = "{\"type\":\"transcript\",\"event_id\":" + std::to_string(event.event_id) +
                       ",\"session_id\":";
    append_string(json, event.session_id);
    json += ",\"segment_id\":" + std::to_string(event.segment_id) +
            ",\"revision\":" + std::to_string(event.revision) +
            ",\"is_final\":" + (event.is_final ? "true" : "false") +
            ",\"sample_start\":" + std::to_string(event.sample_start) +
            ",\"sample_end\":" + std::to_string(event.sample_end) + ",\"text\":";
    append_string(json, event.text);
    json += ",\"model_release\":";
    append_string(json, event.model_release);
    json += ",\"backend\":";
    append_string(json, backend_name(event.backend));
    json += '}';
    if (json.size() > fzas_json_payload_limit) {
        throw std::invalid_argument("FZAS transcript exceeds JSON limit");
    }
    return json;
}

struct ChannelState {
    ChannelState(Channel channel, std::size_t depth) : channel(channel), depth(depth) {}

    PublishResult enqueue(Frame frame) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!connected) return PublishResult::disconnected;
        if (outgoing.size() >= depth || reset_requested) {
            if (channel == Channel::control) reset_requested = true;
            return PublishResult::backpressure;
        }
        outgoing.push_back(std::move(frame));
        return PublishResult::queued;
    }

    std::optional<Frame> next() {
        std::lock_guard<std::mutex> lock(mutex);
        if (outgoing.empty()) return std::nullopt;
        Frame frame = std::move(outgoing.front());
        outgoing.pop_front();
        return frame;
    }

    void set_connected(bool value) {
        std::lock_guard<std::mutex> lock(mutex);
        connected = value;
        reset_requested = false;
        outgoing.clear();
    }

    bool must_reset() {
        std::lock_guard<std::mutex> lock(mutex);
        return reset_requested;
    }

    const Channel channel;
    const std::size_t depth;
    Socket listener;
    std::thread worker;

private:
    std::mutex mutex;
    std::deque<Frame> outgoing;
    bool connected = false;
    bool reset_requested = false;
};

}  // namespace

struct Server::Impl {
    Impl(ServerConfig cfg, Handlers callbacks)
        : config(cfg), handlers(std::move(callbacks)), control(Channel::control, config.control_queue_depth),
          archive(Channel::archive, config.archive_queue_depth) {
        if (!config.control_queue_depth || !config.archive_queue_depth || config.frame_timeout.count() <= 0) {
            throw std::invalid_argument("FZAS queue limits and frame timeout must be positive");
        }
    }

    void diagnose(Channel channel, std::string_view reason) const noexcept {
        if (handlers.diagnostic) {
            try {
                handlers.diagnostic(channel, reason);
            } catch (...) {
                // A diagnostic callback cannot terminate transport or reveal a request body.
            }
        }
    }

    std::optional<Frame> dispatch(ChannelState& state, Frame frame) {
        if (frame.message_type == MessageType::heartbeat) {
            return frame;
        }
        const std::string_view payload = frame.payload.empty() ? std::string_view{} :
            std::string_view(reinterpret_cast<const char*>(frame.payload.data()), frame.payload.size());
        if (state.channel == Channel::control && frame.message_type == MessageType::request &&
            handlers.control_request) {
            const std::string response = handlers.control_request(payload);
            if (response.empty() || response.size() > fzas_json_payload_limit) {
                throw WireError("FZAS control handler returned an invalid response length");
            }
            return Frame{MessageType::response, 0, bytes(response)};
        } else if (state.channel == Channel::archive && frame.message_type == MessageType::ack &&
                   handlers.archive_ack) {
            handlers.archive_ack(payload);
            return std::nullopt;
        } else {
            throw WireError("FZAS message type is not supported on this channel");
        }
    }

    void run_peer(ChannelState& state, SocketId peer) {
        nonblocking(peer);
        ReadFrame input;
        WriteFrame output;
        std::optional<Frame> reply;
        std::optional<std::uint64_t> receive_seq;
        std::uint64_t send_seq = 0;
        const std::string_view hello = state.channel == Channel::control ?
            R"({"protocol":1,"channel":"control","audio_format":{"sample_rate_hz":16000,"channels":1,"encoding":"s16le"}})" :
            R"({"protocol":1,"channel":"archive","audio_format":{"sample_rate_hz":16000,"channels":1,"encoding":"s16le"}})";
        output.prepare({MessageType::hello, 0, bytes(hello)}, send_seq, config.frame_timeout);
        state.set_connected(true);

        while (running) {
            if (state.must_reset()) throw WireError("FZAS control queue saturated; reconnect required");
            if (!output.pending()) {
                // Replies take precedence over events produced while a request is handled.
                auto frame = reply ? std::exchange(reply, std::nullopt) : state.next();
                if (frame) output.prepare(std::move(*frame), send_seq, config.frame_timeout);
            }
            const auto now = Clock::now();
            if (input.position && now >= input.deadline) throw WireError("FZAS partial frame timed out");
            if (output.pending() && now >= output.deadline) throw WireError("FZAS frame write timed out");

            fd_set read_set, write_set;
            FD_ZERO(&read_set);
            FD_ZERO(&write_set);
            // One pending reply is enough. TCP applies backpressure to pipelined requests.
            if (!reply) FD_SET(peer, &read_set);
            FD_SET(state.listener.get(), &read_set);
            if (output.pending()) FD_SET(peer, &write_set);
            timeval wait{0, 20000};
            const int ready = ::select(static_cast<int>(std::max(peer, state.listener.get()) + 1),
                                       &read_set, &write_set, nullptr, &wait);
            if (ready < 0 && can_retry()) continue;
            if (ready < 0) throw WireError("FZAS socket polling failed");
            if (!ready) continue;
            if (FD_ISSET(peer, &read_set)) {
                const auto status = input.read(peer, config.frame_timeout);
                if (status == IoResult::closed) return;
                if (status == IoResult::complete) {
                    auto frame = input.take();
                    if (receive_seq && frame.transport_seq <= *receive_seq) {
                        throw WireError("FZAS transport_seq did not advance");
                    }
                    receive_seq = frame.transport_seq;
                    reply = dispatch(state, std::move(frame));
                }
            }
            if (FD_ISSET(state.listener.get(), &read_set)) {
                Socket busy(::accept(state.listener.get(), nullptr, nullptr));
            }
            if (FD_ISSET(peer, &write_set)) output.write(peer);
        }
    }

    void run(ChannelState& state) {
        const auto listener = state.listener.get();
        while (running) {
            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(listener, &read_set);
            timeval wait{0, 20000};
            const int ready = ::select(static_cast<int>(listener + 1), &read_set, nullptr, nullptr, &wait);
            if (!ready || (ready < 0 && can_retry())) continue;
            if (ready < 0) {
                diagnose(state.channel, "FZAS listener polling failed");
                return;
            }
            Socket peer(::accept(listener, nullptr, nullptr));
            if (!peer) {
                if (!can_retry()) diagnose(state.channel, "FZAS listener accept failed");
                continue;
            }
            try {
                run_peer(state, peer.get());
            } catch (const WireError& error) {
                state.set_connected(false);
                diagnose(state.channel, error.what());
                continue;
            } catch (...) {
                state.set_connected(false);
                diagnose(state.channel, "FZAS connection or application handler failed");
                continue;
            }
            state.set_connected(false);
        }
    }

    NetworkRuntime network;
    ServerConfig config;
    Handlers handlers;
    ChannelState control, archive;
    std::mutex publish_mutex;
    std::atomic<bool> running{false};
};

Server::Server(ServerConfig config, Handlers handlers)
    : impl_(new Impl(config, std::move(handlers))) {}

Server::~Server() { stop(); }

void Server::start() {
    if (impl_->running) return;
    auto control = listen_loopback(impl_->config.control_port);
    auto archive = listen_loopback(impl_->config.archive_port);
    impl_->control.listener = std::move(control);
    impl_->archive.listener = std::move(archive);
    impl_->running = true;
    try {
        impl_->control.worker = std::thread([this] { impl_->run(impl_->control); });
        impl_->archive.worker = std::thread([this] { impl_->run(impl_->archive); });
    } catch (...) {
        stop();
        throw;
    }
}

void Server::stop() {
    impl_->running = false;
    for (auto* channel : {&impl_->control, &impl_->archive}) {
        if (channel->worker.joinable()) channel->worker.join();
        channel->listener = Socket{};
    }
}

std::uint16_t Server::control_port() const { return bound_port(impl_->control.listener); }
std::uint16_t Server::archive_port() const { return bound_port(impl_->archive.listener); }

PublishResult Server::publish(const TranscriptEvent& event) {
    if (!impl_->handlers.persist_event) throw std::logic_error("FZAS transcript persistence is required");
    const std::string json = transcript_json(event);
    std::lock_guard<std::mutex> lock(impl_->publish_mutex);
    impl_->handlers.persist_event(event, json);
    return send_control_event(json);
}

PublishResult Server::send_control_event(std::string_view event_json) {
    if (event_json.empty() || event_json.size() > fzas_json_payload_limit) {
        throw std::invalid_argument("FZAS control event has invalid JSON payload length");
    }
    return impl_->control.enqueue({MessageType::event, 0, bytes(event_json)});
}

bool Server::send_audio(const AudioBlock& block) {
    if (block.pcm_s16le.size() != 2 * std::uint64_t{block.sample_count} ||
        block.pcm_s16le.size() > fzas_binary_payload_limit - pcm_header_bytes) {
        throw std::invalid_argument("FZAS PCM payload length is invalid");
    }
    checked_extent(block.sample_start, block.sample_count);
    Bytes payload(pcm_header_bytes + block.pcm_s16le.size());
    std::copy(block.session_id.begin(), block.session_id.end(), payload.begin());
    detail::write_le(payload.data() + 16, block.sample_start);
    detail::write_le(payload.data() + 24, block.sample_count);
    detail::write_le(payload.data() + 28, block.audio_flags);
    std::copy(block.pcm_s16le.begin(), block.pcm_s16le.end(), payload.begin() + pcm_header_bytes);
    return impl_->archive.enqueue({MessageType::pcm, 0, std::move(payload)}) == PublishResult::queued;
}

bool Server::send_file_chunk(Bytes payload) {
    if (payload.size() > fzas_binary_payload_limit) {
        throw std::invalid_argument("FZAS file chunk exceeds binary limit");
    }
    return impl_->archive.enqueue({MessageType::file_chunk, 0, std::move(payload)}) == PublishResult::queued;
}

}  // namespace edge_speech::transport

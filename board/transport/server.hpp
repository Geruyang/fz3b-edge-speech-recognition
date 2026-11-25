#pragma once

#include "shared/types.hpp"

#include <chrono>
#include <functional>
#include <memory>

namespace edge_speech::transport {

enum class Channel { control, archive };

struct ServerConfig {
    // Both listeners are IPv4 loopback only. Zero requests a kernel-selected port.
    std::uint16_t control_port = 18765;
    std::uint16_t archive_port = 18766;
    std::size_t control_queue_depth = 64;
    std::size_t archive_queue_depth = 8;
    std::chrono::milliseconds frame_timeout{5000};
};

struct Handlers {
    // Callbacks on the two I/O threads can run concurrently.
    // The owner validates request JSON and returns a complete response envelope.
    std::function<std::string(std::string_view)> control_request;
    std::function<void(std::string_view)> archive_ack;
    // Return only after recording the caller-assigned event ID durably. Throw on failure.
    // Called synchronously by publish; the session owner supplies its journal.
    std::function<void(const TranscriptEvent&, std::string_view)> persist_event;
    // Reports transport errors without logging request bodies or credentials.
    std::function<void(Channel, std::string_view)> diagnostic;
};

enum class PublishResult { queued, disconnected, backpressure };

class Server {
public:
    explicit Server(ServerConfig config = {}, Handlers handlers = {});
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Lifecycle calls are owned by the server's supervisor.
    void start();
    void stop();
    std::uint16_t control_port() const;
    std::uint16_t archive_port() const;

    // Persists before attempting live delivery. Missing or failed persistence throws.
    PublishResult publish(const TranscriptEvent& event);
    // Owner supplies non-transcript event JSON (status, gap, snapshot, etc.).
    PublishResult send_control_event(std::string_view event_json);
    // False means the archive peer is absent or its queue is full. Retry from the source.
    bool send_audio(const AudioBlock& block);
    bool send_file_chunk(Bytes payload);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_speech::transport

#pragma once

#include "hardware/hls/descriptor_engine.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace edge_speech::pl {

enum class Error {
    owner_busy, stale_handle, invalid_buffer, buffer_busy, invalid_descriptor, queue_full,
    invalid_port_profile, profile_busy, profile_mismatch, quarantined
};

struct DeviceException : std::runtime_error {
    explicit DeviceException(Error value) : std::runtime_error("DMA contract rejected operation"), error(value) {}
    Error error;
};

enum class Access : unsigned { read = FZ_ACCESS_READ, write = FZ_ACCESS_WRITE, read_write = FZ_ACCESS_READ_WRITE };
enum class Observation { pending, success, failed, unknown };
enum class Completion { timeout, success, failed, unknown };
enum class Isolation { idle, unknown };

struct PortState {
    unsigned read = 0, write = 0;
    bool fifo_empty = true;
};

struct Snapshot {
    Observation completion;
    bool valid;
    std::array<PortState, 4> ports;
    bool engine_idle;
};

struct Profile {
    std::uint64_t generation, epoch;
    unsigned id, mask;
};

struct DmaSegment {
    std::uint64_t address, bytes;
};

struct BufferHandle {
    std::uint64_t generation, value;
};

struct Transfer {
    std::uint64_t job_id;
    BufferHandle handle;
    std::uint64_t offset, bytes;
    unsigned pair;
    Access access;
    std::vector<unsigned> dependencies;
};

struct Chain {
    std::uint64_t id;
    Profile profile;
    std::vector<Transfer> transfers;
};

struct Receipt {
    std::uint64_t id;
    Profile profile;
    std::vector<fz_dma_span> spans;
};

// Host reference for the ioctl/device state transitions. Device observations and
// DMA segments come from the simulated provider and never from a submit request.
class Device {
public:
    Device(unsigned dma_address_bits, std::uint64_t budget_bytes);
    Profile prepare(std::uint64_t owner);
    BufferHandle register_buffer(std::uint64_t owner, std::uint64_t bytes, Access access,
                                 std::vector<DmaSegment> segments);
    void unregister(std::uint64_t owner, BufferHandle handle);
    Receipt submit(std::uint64_t owner, const Chain& chain);
    Completion wait(std::uint64_t owner, std::uint64_t id, const Snapshot& snapshot);
    Isolation quiesce(std::uint64_t owner, const Snapshot& snapshot);
    void owner_exit(std::uint64_t owner, const Snapshot& snapshot);
    Isolation recover(const Snapshot& snapshot);
    Profile set_port_profile(std::uint64_t owner, unsigned id, unsigned mask,
                             std::uint64_t expected_epoch, const Snapshot& snapshot);
    bool quarantined() const { return quarantined_; }
    bool can_remove() const { return !owner_ && !quarantined_ && buffers_.empty(); }
    std::uint64_t resident_bytes() const { return resident_; }

private:
    struct Buffer {
        std::uint64_t bytes;
        Access access;
        std::vector<DmaSegment> segments;
        bool in_flight = false;
    };
    void assert_owner(std::uint64_t owner) const;
    void release_chain();
    void clear_owner();
    const std::uint64_t dma_mask_, budget_;
    std::uint64_t owner_ = 0, resident_ = 0, next_handle_ = 0;
    std::uint64_t last_chain_ = 0, last_job_ = 0;
    Profile profile_{};
    bool quarantined_ = false, quiesced_ = false;
    std::map<std::uint64_t, Buffer> buffers_;
    std::optional<std::uint64_t> active_chain_;
};

#ifdef __linux__
class LinuxDevice {
public:
    explicit LinuxDevice(const char* path = "/dev/fzspeech");
    ~LinuxDevice();
    LinuxDevice(const LinuxDevice&) = delete;
    LinuxDevice& operator=(const LinuxDevice&) = delete;

    fz_info info() const;
    fz_info prepare();
    fz_profile set_port_profile(fz_profile request);
    fz_limits set_transport_limits(fz_limits request);
    fz_transport_stats transport_statistics() const;
    fz_probe_receipt job_receipt(std::uint64_t generation, std::uint64_t chain_id) const;
    void release_provider(std::uint64_t generation, unsigned timeout_ms);
    fz_buffer register_buffer(std::uint64_t generation, std::uint64_t bytes, Access access);
    void unregister(fz_buffer buffer);
    void write_buffer(fz_buffer buffer, std::uint64_t offset, const void* data, std::uint64_t bytes);
    void read_buffer(fz_buffer buffer, std::uint64_t offset, void* data, std::uint64_t bytes) const;
    void submit(fz_chain chain, const std::vector<fz_transfer>& transfers);
    void submit_probe(fz_chain chain, fz_buffer input, std::uint64_t input_offset,
                      fz_buffer output, std::uint64_t output_offset,
                      bool write_enable, std::uint64_t first_job);
    fz_wait wait(std::uint64_t generation, std::uint64_t chain_id, unsigned timeout_ms);
    fz_wait quiesce(std::uint64_t generation, unsigned timeout_ms);

private:
    void request(unsigned long command, void* data) const;
    const int fd_;
};
#endif

} // namespace edge_speech::pl

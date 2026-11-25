#include "board/pl/device.hpp"
#include "driver/fzspeech_job.h"

#include <algorithm>
#include <limits>
#include <utility>

#ifdef __linux__
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <system_error>
#include <unistd.h>
#endif

namespace edge_speech::pl {
namespace {

std::uint64_t address_mask(unsigned bits) {
    if (!bits || bits > 64)
        throw std::invalid_argument("invalid DMA address width");
    return std::numeric_limits<std::uint64_t>::max() >> (64 - bits);
}

bool drained(const Snapshot& snapshot) {
    return snapshot.valid && snapshot.engine_idle && snapshot.completion != Observation::unknown &&
        std::all_of(snapshot.ports.begin(), snapshot.ports.end(), [](const PortState& port) {
            return port.fifo_empty && port.read == 0 && port.write == 0;
        });
}

std::uint64_t increment(std::uint64_t value) {
    if (value == std::numeric_limits<std::uint64_t>::max())
        throw DeviceException(Error::quarantined);
    return value + 1;
}

} // namespace

static_assert(sizeof(fz_chain) == 48 && sizeof(fz_transfer) == 48);
static_assert(sizeof(fz_dma_span) == 32);

Device::Device(unsigned dma_address_bits, std::uint64_t budget_bytes)
    : dma_mask_(address_mask(dma_address_bits)), budget_(budget_bytes) {}

void Device::assert_owner(std::uint64_t owner) const {
    if (quarantined_)
        throw DeviceException(Error::quarantined);
    if (!owner || owner != owner_)
        throw DeviceException(Error::owner_busy);
}

Profile Device::prepare(std::uint64_t owner) {
    if (quarantined_)
        throw DeviceException(Error::quarantined);
    if (!owner || owner_ != 0)
        throw DeviceException(Error::owner_busy);
    profile_ = {increment(profile_.generation), increment(profile_.epoch), 4, 0xf};
    owner_ = owner;
    quiesced_ = false;
    last_chain_ = last_job_ = 0;
    return profile_;
}

BufferHandle Device::register_buffer(std::uint64_t owner, std::uint64_t bytes, Access access,
                                     std::vector<DmaSegment> segments) {
    assert_owner(owner);
    if (quiesced_ || !bytes || bytes > FZ_MAX_BUFFER_BYTES || bytes > budget_ - resident_ ||
        segments.empty() || static_cast<unsigned>(access) < 1 || static_cast<unsigned>(access) > 3)
        throw DeviceException(Error::invalid_buffer);
    std::uint64_t remaining = bytes;
    for (const auto& segment : segments) {
        if (!segment.bytes || segment.bytes > remaining || !fz_aligned(segment.address | segment.bytes) ||
            segment.address > dma_mask_ || segment.bytes - 1 > dma_mask_ - segment.address)
            throw DeviceException(Error::invalid_buffer);
        remaining -= segment.bytes;
    }
    if (remaining)
        throw DeviceException(Error::invalid_buffer);
    next_handle_ = increment(next_handle_);
    buffers_.emplace(next_handle_, Buffer{bytes, access, std::move(segments)});
    resident_ += bytes;
    return {profile_.generation, next_handle_};
}

void Device::unregister(std::uint64_t owner, BufferHandle handle) {
    assert_owner(owner);
    const auto it = buffers_.find(handle.value);
    if (handle.generation != profile_.generation || it == buffers_.end())
        throw DeviceException(Error::stale_handle);
    if (it->second.in_flight)
        throw DeviceException(Error::buffer_busy);
    resident_ -= it->second.bytes;
    buffers_.erase(it);
}

Receipt Device::submit(std::uint64_t owner, const Chain& chain) {
    assert_owner(owner);
    if (active_chain_)
        throw DeviceException(Error::queue_full);
    if (quiesced_ || chain.profile.generation != profile_.generation)
        throw DeviceException(Error::stale_handle);
    if (chain.profile.epoch != profile_.epoch || chain.profile.id != profile_.id ||
        chain.profile.mask != profile_.mask)
        throw DeviceException(Error::profile_mismatch);
    if (chain.id <= last_chain_ || chain.transfers.empty() || chain.transfers.size() > FZ_MAX_DESCRIPTORS ||
        chain.transfers.size() > (FZ_MAX_CHAIN_BYTES - sizeof(fz_chain)) / sizeof(fz_transfer))
        throw DeviceException(Error::invalid_descriptor);

    Receipt receipt{chain.id, profile_, {}};
    const auto span_limit = fz_span_limit(static_cast<fz_u32>(chain.transfers.size()));
    auto last_job = last_job_;
    for (unsigned index = 0; index < chain.transfers.size(); ++index) {
        const auto& transfer = chain.transfers[index];
        const unsigned port = fz_pair_port(profile_.mask, transfer.pair);
        if (transfer.job_id <= last_job || port >= 4 || transfer.dependencies.size() > 1 ||
            (!transfer.dependencies.empty() && transfer.dependencies[0] >= index))
            throw DeviceException(Error::invalid_descriptor);
        last_job = transfer.job_id;
        const auto it = buffers_.find(transfer.handle.value);
        if (transfer.handle.generation != profile_.generation || it == buffers_.end())
            throw DeviceException(Error::stale_handle);
        const auto& buffer = it->second;
        const unsigned access = static_cast<unsigned>(transfer.access);
        if (!fz_range(transfer.offset, transfer.bytes, buffer.bytes) ||
            !fz_aligned(transfer.offset | transfer.bytes) ||
            (access != FZ_ACCESS_READ && access != FZ_ACCESS_WRITE) ||
            !(access & static_cast<unsigned>(buffer.access)))
            throw DeviceException(Error::invalid_buffer);
        for (unsigned prior = 0; prior < index; ++prior) {
            const auto& earlier = chain.transfers[prior];
            const auto combined_access = access | static_cast<unsigned>(earlier.access);
            if (earlier.handle.value == transfer.handle.value && (combined_access & FZ_ACCESS_WRITE) &&
                earlier.offset < transfer.offset + transfer.bytes &&
                transfer.offset < earlier.offset + earlier.bytes)
                throw DeviceException(Error::invalid_descriptor);
        }

        // The chain sequencer owns dependencies; plans contain only one legal
        // reader/writer port for each transfer. Walk the requested window in SG order.
        auto skip = transfer.offset;
        auto left = transfer.bytes;
        for (const auto& segment : buffer.segments) {
            if (skip >= segment.bytes) {
                skip -= segment.bytes;
                continue;
            }
            if (receipt.spans.size() == span_limit)
                throw DeviceException(Error::invalid_descriptor);
            const auto span_bytes = std::min(left, segment.bytes - skip);
            receipt.spans.push_back({transfer.offset + transfer.bytes - left, segment.address + skip,
                                     span_bytes, index, port});
            left -= span_bytes;
            skip = 0;
            if (!left)
                break;
        }
        if (left)
            throw DeviceException(Error::invalid_buffer);
    }

    for (const auto& transfer : chain.transfers)
        buffers_.at(transfer.handle.value).in_flight = true;
    active_chain_ = chain.id;
    last_chain_ = chain.id;
    last_job_ = last_job;
    return receipt;
}

void Device::release_chain() {
    active_chain_.reset();
    for (auto& entry : buffers_)
        entry.second.in_flight = false;
}

Completion Device::wait(std::uint64_t owner, std::uint64_t id, const Snapshot& snapshot) {
    assert_owner(owner);
    if (active_chain_ != id)
        throw DeviceException(Error::stale_handle);
    if (!snapshot.valid || snapshot.completion == Observation::unknown) {
        quarantined_ = true;
        return Completion::unknown;
    }
    if (!drained(snapshot) || snapshot.completion == Observation::pending)
        return Completion::timeout;
    release_chain();
    return snapshot.completion == Observation::success ? Completion::success : Completion::failed;
}

Isolation Device::quiesce(std::uint64_t owner, const Snapshot& snapshot) {
    assert_owner(owner);
    quiesced_ = true;
    if (!drained(snapshot)) {
        quarantined_ = true;
        return Isolation::unknown;
    }
    release_chain();
    return Isolation::idle;
}

void Device::clear_owner() {
    release_chain();
    buffers_.clear();
    owner_ = resident_ = 0;
}

void Device::owner_exit(std::uint64_t owner, const Snapshot& snapshot) {
    if (!owner || owner != owner_)
        return;
    if (!quarantined_ && quiesce(owner, snapshot) == Isolation::idle)
        clear_owner();
}

Isolation Device::recover(const Snapshot& snapshot) {
    if (!quarantined_ || !drained(snapshot))
        return Isolation::unknown;
    clear_owner();
    quarantined_ = false;
    return Isolation::idle;
}

Profile Device::set_port_profile(std::uint64_t owner, unsigned id, unsigned mask,
                                 std::uint64_t expected_epoch, const Snapshot& snapshot) {
    assert_owner(owner);
    if (!mask || fz_profile_mask(id) != mask)
        throw DeviceException(Error::invalid_port_profile);
    if (expected_epoch != profile_.epoch)
        throw DeviceException(Error::profile_mismatch);
    if (active_chain_ || quiesced_)
        throw DeviceException(Error::profile_busy);
    if (!snapshot.valid || snapshot.completion == Observation::unknown) {
        quarantined_ = true;
        throw DeviceException(Error::quarantined);
    }
    if (!drained(snapshot))
        throw DeviceException(Error::profile_busy);
    profile_ = {profile_.generation, increment(profile_.epoch), id, mask};
    return profile_;
}

#ifdef __linux__
LinuxDevice::LinuxDevice(const char* path) : fd_(::open(path, O_RDWR | O_CLOEXEC)) {
    if (fd_ < 0)
        throw std::system_error(errno, std::generic_category(), "open speech accelerator");
}

LinuxDevice::~LinuxDevice() {
    ::close(fd_);
}

void LinuxDevice::request(unsigned long command, void* data) const {
    if (::ioctl(fd_, command, data) < 0)
        throw std::system_error(errno, std::generic_category(), "speech accelerator ioctl");
}

fz_info LinuxDevice::info() const {
    fz_info result{};
    request(FZ_GET_INFO, &result);
    if (result.abi_version != FZ_ABI_VERSION || result.descriptor_version != FZ_DESCRIPTOR_VERSION)
        throw std::runtime_error("unsupported speech accelerator ABI");
    return result;
}

fz_info LinuxDevice::prepare() {
    (void)info();
    fz_info result{};
    request(FZ_PREPARE, &result);
    return result;
}

fz_profile LinuxDevice::set_port_profile(fz_profile profile) {
    request(FZ_SET_PORT_PROFILE, &profile);
    return profile;
}

fz_limits LinuxDevice::set_transport_limits(fz_limits limits) {
    request(FZ_SET_TRANSPORT_LIMITS, &limits);
    return limits;
}

fz_transport_stats LinuxDevice::transport_statistics() const {
    fz_transport_stats stats{};
    request(FZ_GET_TRANSPORT_STATS, &stats);
    return stats;
}

fz_probe_receipt LinuxDevice::job_receipt(std::uint64_t generation, std::uint64_t chain_id) const {
    fz_probe_receipt receipt{};
    receipt.generation = generation;
    receipt.chain_id = chain_id;
    request(FZ_GET_JOB_RECEIPT, &receipt);
    return receipt;
}

void LinuxDevice::release_provider(std::uint64_t generation, unsigned timeout_ms) {
    fz_wait receipt{generation, 0, {}, timeout_ms, 0};
    request(FZ_RELEASE_DEVICE, &receipt);
}

fz_buffer LinuxDevice::register_buffer(std::uint64_t generation, std::uint64_t bytes, Access access) {
    fz_buffer buffer{generation, 0, bytes, static_cast<unsigned>(access), 0};
    request(FZ_ALLOC_BUFFER, &buffer);
    return buffer;
}

void LinuxDevice::unregister(fz_buffer buffer) {
    request(FZ_FREE_BUFFER, &buffer);
}

void LinuxDevice::write_buffer(fz_buffer buffer, std::uint64_t offset, const void* data, std::uint64_t bytes) {
    fz_io io{buffer.generation, buffer.handle, offset, bytes, reinterpret_cast<std::uintptr_t>(data)};
    request(FZ_WRITE_BUFFER, &io);
}

void LinuxDevice::read_buffer(fz_buffer buffer, std::uint64_t offset, void* data, std::uint64_t bytes) const {
    fz_io io{buffer.generation, buffer.handle, offset, bytes, reinterpret_cast<std::uintptr_t>(data)};
    request(FZ_READ_BUFFER, &io);
}

void LinuxDevice::submit(fz_chain chain, const std::vector<fz_transfer>& transfers) {
    if (transfers.empty() || transfers.size() > FZ_MAX_DESCRIPTORS ||
        transfers.size() > (FZ_MAX_CHAIN_BYTES - sizeof(chain)) / sizeof(fz_transfer))
        throw DeviceException(Error::invalid_descriptor);
    chain.descriptor_version = FZ_DESCRIPTOR_VERSION;
    chain.descriptor_count = static_cast<fz_u32>(transfers.size());
    const auto payload_bytes = transfers.size() * sizeof(fz_transfer);
    chain.chain_bytes = static_cast<fz_u32>(sizeof(chain) + payload_bytes);
    std::vector<unsigned char> payload(chain.chain_bytes);
    std::memcpy(payload.data(), &chain, sizeof(chain));
    std::memcpy(payload.data() + sizeof(chain), transfers.data(), payload_bytes);
    request(FZ_SUBMIT_CHAIN, payload.data());
}

void LinuxDevice::submit_probe(fz_chain chain, fz_buffer input, std::uint64_t input_offset,
                               fz_buffer output, std::uint64_t output_offset,
                               bool write_enable, std::uint64_t first_job) {
    if (input.generation != chain.generation ||
        (write_enable && output.generation != chain.generation))
        throw DeviceException(Error::stale_handle);
    std::vector<fz_transfer> entries(128);
    fz_u32 count = 0;
    if (fz_job_build_transfers(input.handle, input_offset, output.handle, output_offset,
                              write_enable, first_job, entries.data(), entries.size(), &count))
        throw DeviceException(Error::invalid_descriptor);
    entries.resize(count);
    submit(chain, entries);
}

fz_wait LinuxDevice::wait(std::uint64_t generation, std::uint64_t chain_id, unsigned timeout_ms) {
    fz_wait receipt{generation, chain_id, {}, timeout_ms, 0};
    request(FZ_WAIT, &receipt);
    return receipt;
}

fz_wait LinuxDevice::quiesce(std::uint64_t generation, unsigned timeout_ms) {
    fz_wait receipt{generation, 0, {}, timeout_ms, 0};
    request(FZ_QUIESCE, &receipt);
    return receipt;
}
#endif

} // namespace edge_speech::pl

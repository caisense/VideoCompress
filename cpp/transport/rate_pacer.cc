#include "transport/rate_pacer.h"

#include <algorithm>
#include <thread>

namespace roi_h265 {

namespace {

const size_t kIpv4UdpHeaderBytes = 28;
const size_t kEthernetWireOverheadBytes = 38;
const size_t kMinimumEthernetPayloadBytes = 46;

}  // namespace

RatePacer::RatePacer(int bitrate_bps, size_t max_burst_bytes)
    : bitrate_bps_(0), tokens_(0.0), max_burst_bytes_(0.0),
      last_refill_(std::chrono::steady_clock::now()) {
    reset(bitrate_bps, max_burst_bytes);
}

size_t RatePacer::wireBytesForUdpPayload(size_t udp_payload_bytes) {
    const size_t ip_packet_bytes = udp_payload_bytes + kIpv4UdpHeaderBytes;
    return kEthernetWireOverheadBytes + std::max(kMinimumEthernetPayloadBytes, ip_packet_bytes);
}

void RatePacer::reset(int bitrate_bps, size_t max_burst_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    bitrate_bps_ = std::max(1, bitrate_bps);
    max_burst_bytes_ = std::max(1.0, static_cast<double>(max_burst_bytes));
    // Start empty: startup is not allowed to spend a one-second burst budget.
    tokens_ = 0.0;
    last_refill_ = std::chrono::steady_clock::now();
}

int RatePacer::bitrateBps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bitrate_bps_;
}

void RatePacer::refillLocked(const std::chrono::steady_clock::time_point &now) {
    const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(
        now - last_refill_).count();
    if (elapsed_seconds > 0.0) {
        tokens_ = std::min(max_burst_bytes_, tokens_ + elapsed_seconds * bitrate_bps_ / 8.0);
        last_refill_ = now;
    }
}

void RatePacer::waitForTokens(size_t bytes) {
    size_t remaining = bytes;
    while (remaining > 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        refillLocked(now);
        const double allowed = std::min<double>(remaining, max_burst_bytes_);
        if (tokens_ >= allowed) {
            tokens_ -= allowed;
            remaining -= static_cast<size_t>(allowed);
            continue;
        }
        const double missing = allowed - tokens_;
        const double seconds = missing * 8.0 / bitrate_bps_;
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
}

}  // namespace roi_h265

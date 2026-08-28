#ifndef ROI_H265_TRANSPORT_RATE_PACER_H_
#define ROI_H265_TRANSPORT_RATE_PACER_H_

#include <stddef.h>

#include <chrono>
#include <mutex>

namespace roi_h265 {

// A byte-token bucket with a deliberately small burst allowance. It is placed
// after MPP so an oversized IDR is paced fragment-by-fragment over UDP/RTP.
class RatePacer {
public:
    RatePacer(int bitrate_bps, size_t max_burst_bytes);
    // UDP payload bytes are converted to physical Ethernet-wire bytes before
    // spending tokens.  Keeping this here makes every RTP media stream use
    // precisely the same accounting rule.
    static size_t wireBytesForUdpPayload(size_t udp_payload_bytes);
    void reset(int bitrate_bps, size_t max_burst_bytes);
    void waitForTokens(size_t bytes);
    int bitrateBps() const;

private:
    void refillLocked(const std::chrono::steady_clock::time_point &now);

    mutable std::mutex mutex_;
    int bitrate_bps_;
    double tokens_;
    double max_burst_bytes_;
    std::chrono::steady_clock::time_point last_refill_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_RATE_PACER_H_

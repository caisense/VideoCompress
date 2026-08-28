#ifndef ROI_H265_TRANSPORT_ASYNC_RTP_SENDER_H_
#define ROI_H265_TRANSPORT_ASYNC_RTP_SENDER_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/frame_meta.h"
#include "transport/packetizer.h"
#include "transport/udp_sender.h"

namespace roi_h265 {

struct RtpAccessUnit {
    FrameMeta frame;
    std::vector<uint8_t> bytes;
    bool key_frame;
    RtpStreamProfile stream_profile;

    RtpAccessUnit() : key_frame(false) {}
};

struct AsyncRtpSenderSnapshot {
    size_t queued_frames;
    size_t queued_bytes;
    uint64_t sent_frames;
    uint64_t dropped_p_frames;
    uint64_t dropped_key_frames;
    uint64_t last_sent_frame_id;
    uint64_t last_capture_to_send_us;
    uint64_t last_queue_delay_us;
    bool waiting_for_key_frame;
    bool sending;

    AsyncRtpSenderSnapshot()
        : queued_frames(0), queued_bytes(0), sent_frames(0), dropped_p_frames(0),
          dropped_key_frames(0), last_sent_frame_id(0), last_capture_to_send_us(0),
          last_queue_delay_us(0), waiting_for_key_frame(false), sending(false) {}
};

// Owns the blocking token bucket and UDP socket on a dedicated worker thread.
// Access units remain whole while queued. If an encoded P frame becomes stale,
// all dependent queued P frames are discarded and the encoder is asked for a
// recovery IDR before more P frames are accepted.
class AsyncRtpSender {
public:
    AsyncRtpSender(const std::string &host, int port, int pacing_bitrate_bps, int mtu,
                   size_t max_queue_frames, int max_queue_latency_ms,
                   const std::shared_ptr<RatePacer> &pacer = std::shared_ptr<RatePacer>());
    ~AsyncRtpSender();

    bool start(std::string *error);
    bool enqueue(RtpAccessUnit access_unit, std::string *error);
    // Finish the currently transmitting whole access unit, discard queued old
    // profile frames, reset pacing, then require a fresh IDR.
    bool switchProfile(int pacing_bitrate_bps, std::string *error);
    void stop();

    bool needsKeyFrame() const;
    bool failed(std::string *error) const;
    AsyncRtpSenderSnapshot snapshot() const;

private:
    struct QueueItem {
        RtpAccessUnit access_unit;
        std::chrono::steady_clock::time_point enqueued_at;
    };

    static uint32_t rtpTimestamp(uint64_t pts_us);
    void workerLoop();
    void dropAllPFramesLocked();
    void dropExpiredPFramesLocked(const std::chrono::steady_clock::time_point &now);
    void enterRecoveryLocked();
    void setFailureLocked(const std::string &error);

    UdpSender udp_sender_;
    H265RtpPacketizer packetizer_;
    const size_t max_queue_frames_;
    const std::chrono::milliseconds max_queue_latency_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<QueueItem> queue_;
    std::thread worker_;
    bool started_;
    bool stopping_;
    bool failed_;
    bool waiting_for_key_frame_;
    bool sending_;
    std::string failure_message_;
    uint64_t sent_frames_;
    uint64_t dropped_p_frames_;
    uint64_t dropped_key_frames_;
    uint64_t last_sent_frame_id_;
    uint64_t last_capture_to_send_us_;
    uint64_t last_queue_delay_us_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_ASYNC_RTP_SENDER_H_

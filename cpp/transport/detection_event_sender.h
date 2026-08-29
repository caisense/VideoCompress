#ifndef ROI_H265_TRANSPORT_DETECTION_EVENT_SENDER_H_
#define ROI_H265_TRANSPORT_DETECTION_EVENT_SENDER_H_

#include <stddef.h>
#include <stdint.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common/config.h"
#include "transport/detection_event_protocol.h"
#include "transport/rate_pacer.h"
#include "transport/udp_sender.h"

namespace roi_h265 {

struct DetectionEventSenderSnapshot {
    bool enabled;
    bool transmitting;
    size_t queued_events;
    uint64_t submitted_results;
    uint64_t replaced_events;
    uint64_t state_packets;
    uint64_t heartbeat_packets;
    uint64_t sent_wire_bytes;
    uint64_t failed_packets;
    uint16_t last_present_mask;
    std::string last_error;

    DetectionEventSenderSnapshot();
};

// The inference thread only publishes a compact state transition.  The worker
// owns UDP pacing and retains a bounded event queue, so semantic notification
// cannot make RKNN inference wait behind a congested video packet.
class DetectionEventSender {
public:
    DetectionEventSender(const DetectionEventConfig &config, const std::string &host,
                         int mtu, const std::shared_ptr<RatePacer> &pacer);
    ~DetectionEventSender();

    bool start(std::string *error);
    void setEnabled(bool enabled);
    bool submit(const SegResult &segmentation, std::string *error);
    bool failed(std::string *error) const;
    void stop();
    DetectionEventSenderSnapshot snapshot() const;

private:
    void workerLoop();
    void discardQueuedHeartbeatsLocked();
    void appendEventLocked(const DetectionEventPacket &packet);

    DetectionEventConfig config_;
    std::string host_;
    int mtu_;
    std::shared_ptr<RatePacer> pacer_;
    std::unique_ptr<UdpSender> udp_sender_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<DetectionEventPacket> queue_;
    std::thread worker_;
    bool started_;
    bool stopping_;
    bool enabled_;
    bool transmitting_;
    bool failed_;
    bool has_last_observed_state_;
    uint16_t last_observed_mask_;
    bool has_last_queued_event_;
    std::chrono::steady_clock::time_point last_queued_event_;
    uint32_t next_sequence_;
    DetectionEventSenderSnapshot snapshot_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_DETECTION_EVENT_SENDER_H_

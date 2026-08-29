#include "transport/detection_event_sender.h"

#include <cstdio>
#include <vector>

namespace roi_h265 {
namespace {

const size_t kMaxQueuedEvents = 8;

bool isHeartbeat(const DetectionEventPacket &packet) {
    return (packet.flags & DETECTION_EVENT_FLAG_HEARTBEAT) != 0;
}

}  // namespace

DetectionEventSenderSnapshot::DetectionEventSenderSnapshot()
    : enabled(false), transmitting(false), queued_events(0), submitted_results(0),
      replaced_events(0), state_packets(0), heartbeat_packets(0), sent_wire_bytes(0),
      failed_packets(0), last_present_mask(0) {}

DetectionEventSender::DetectionEventSender(const DetectionEventConfig &config,
                                           const std::string &host, int mtu,
                                           const std::shared_ptr<RatePacer> &pacer)
    : config_(config), host_(host), mtu_(mtu), pacer_(pacer), udp_sender_(),
      started_(false), stopping_(false), enabled_(false), transmitting_(false),
      failed_(false), has_last_observed_state_(false), last_observed_mask_(0),
      has_last_queued_event_(false), next_sequence_(1), snapshot_() {}

DetectionEventSender::~DetectionEventSender() {
    stop();
}

bool DetectionEventSender::start(std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return true;
    if (!pacer_) {
        if (error) *error = "detection event sender requires the shared video pacer";
        return false;
    }
    udp_sender_.reset(new UdpSender(host_, config_.udp_port, pacer_->bitrateBps(), mtu_, pacer_));
    if (!udp_sender_->open(error)) {
        udp_sender_.reset();
        return false;
    }
    stopping_ = false;
    failed_ = false;
    started_ = true;
    worker_ = std::thread(&DetectionEventSender::workerLoop, this);
    return true;
}

void DetectionEventSender::setEnabled(bool enabled) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = enabled;
        snapshot_.enabled = enabled;
        if (!enabled_) {
            snapshot_.replaced_events += queue_.size();
            queue_.clear();
            snapshot_.queued_events = 0;
            has_last_observed_state_ = false;
            last_observed_mask_ = 0;
            has_last_queued_event_ = false;
            snapshot_.last_present_mask = 0;
        }
    }
    condition_.notify_all();
}

void DetectionEventSender::discardQueuedHeartbeatsLocked() {
    for (std::deque<DetectionEventPacket>::iterator it = queue_.begin(); it != queue_.end();) {
        if (isHeartbeat(*it)) {
            it = queue_.erase(it);
            ++snapshot_.replaced_events;
        } else {
            ++it;
        }
    }
}

void DetectionEventSender::appendEventLocked(const DetectionEventPacket &packet) {
    if (!isHeartbeat(packet)) {
        // A new edge state supersedes any stale heartbeat waiting behind it.
        discardQueuedHeartbeatsLocked();
    }
    if (queue_.size() >= kMaxQueuedEvents) {
        bool removed_heartbeat = false;
        for (std::deque<DetectionEventPacket>::iterator it = queue_.begin();
             it != queue_.end(); ++it) {
            if (isHeartbeat(*it)) {
                queue_.erase(it);
                removed_heartbeat = true;
                break;
            }
        }
        if (!removed_heartbeat) queue_.pop_front();
        ++snapshot_.replaced_events;
    }
    queue_.push_back(packet);
    snapshot_.queued_events = queue_.size();
}

bool DetectionEventSender::submit(const SegResult &segmentation, std::string *error) {
    const DetectionEventSummary summary =
        summarizeDetectionEvent(segmentation, config_.min_confidence);
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_ || stopping_) {
            if (error) *error = "detection event sender is not running";
            return false;
        }
        if (failed_) {
            if (error) *error = snapshot_.last_error;
            return false;
        }
        if (!enabled_) return true;
        ++snapshot_.submitted_results;

        const uint16_t previous = has_last_observed_state_ ? last_observed_mask_ : 0;
        const bool changed = !has_last_observed_state_
            ? summary.present_mask != 0
            : summary.present_mask != previous;
        has_last_observed_state_ = true;
        last_observed_mask_ = summary.present_mask;
        snapshot_.last_present_mask = summary.present_mask;

        bool heartbeat = false;
        if (!changed && summary.present_mask != 0 && has_last_queued_event_) {
            heartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_queued_event_).count() >= config_.heartbeat_ms;
        }
        if (!changed && !heartbeat) return true;

        DetectionEventPacket packet;
        packet.flags = heartbeat ? DETECTION_EVENT_FLAG_HEARTBEAT : 0;
        packet.sequence = next_sequence_++;
        packet.frame_id = summary.frame_id;
        packet.pts_ms = summary.pts_ms;
        packet.present_mask = summary.present_mask;
        packet.entered_mask = changed
            ? static_cast<uint16_t>(summary.present_mask & static_cast<uint16_t>(~previous)) : 0;
        packet.exited_mask = changed
            ? static_cast<uint16_t>(previous & static_cast<uint16_t>(~summary.present_mask)) : 0;
        packet.counts = summary.counts;
        packet.max_confidence_percent = summary.max_confidence_percent;
        appendEventLocked(packet);
        last_queued_event_ = now;
        has_last_queued_event_ = true;
    }
    condition_.notify_one();
    return true;
}

bool DetectionEventSender::failed(std::string *error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ && error) *error = snapshot_.last_error;
    return failed_;
}

DetectionEventSenderSnapshot DetectionEventSender::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DetectionEventSenderSnapshot copy = snapshot_;
    copy.enabled = enabled_;
    copy.transmitting = transmitting_;
    copy.queued_events = queue_.size();
    return copy;
}

void DetectionEventSender::workerLoop() {
    for (;;) {
        DetectionEventPacket packet;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || (enabled_ && !queue_.empty()); });
            if (stopping_) break;
            if (!enabled_ || queue_.empty()) continue;
            packet = queue_.front();
            queue_.pop_front();
            transmitting_ = true;
            snapshot_.transmitting = true;
            snapshot_.queued_events = queue_.size();
        }

        std::string send_error;
        std::vector<uint8_t> encoded;
        bool sent = serializeDetectionEventPacket(packet, &encoded, &send_error);
        if (sent) {
            std::vector<std::vector<uint8_t> > packets(1, encoded);
            sent = udp_sender_->sendPackets(packets, &send_error);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            transmitting_ = false;
            snapshot_.transmitting = false;
            if (sent) {
                ++snapshot_.state_packets;
                if (isHeartbeat(packet)) ++snapshot_.heartbeat_packets;
                snapshot_.sent_wire_bytes += RatePacer::wireBytesForUdpPayload(encoded.size());
                snapshot_.last_error.clear();
            } else {
                ++snapshot_.failed_packets;
                failed_ = true;
                snapshot_.last_error = send_error.empty() ? "ROEV UDP send failed" : send_error;
            }
        }
        if (sent && !isHeartbeat(packet)) {
            std::fprintf(stderr,
                "Detection event sent: seq=%u frame=%llu present_mask=0x%04x "
                "entered_mask=0x%04x exited_mask=0x%04x\n",
                packet.sequence, static_cast<unsigned long long>(packet.frame_id),
                packet.present_mask, packet.entered_mask, packet.exited_mask);
        }
        if (!sent) {
            condition_.notify_all();
            break;
        }
    }
}

void DetectionEventSender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stopping_ = true;
        enabled_ = false;
        queue_.clear();
        snapshot_.enabled = false;
        snapshot_.queued_events = 0;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = false;
        transmitting_ = false;
        snapshot_.transmitting = false;
    }
    udp_sender_.reset();
}

}  // namespace roi_h265

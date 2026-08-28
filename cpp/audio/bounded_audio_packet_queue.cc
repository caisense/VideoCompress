#include "audio/bounded_audio_packet_queue.h"

#include <algorithm>
#include <utility>

namespace roi_h265 {

BoundedAudioPacketQueue::BoundedAudioPacketQueue()
    : max_packets_(1), max_latency_(1), packets_() {}

void BoundedAudioPacketQueue::reset(size_t max_packets, int max_latency_ms) {
    max_packets_ = std::max<size_t>(1, max_packets);
    max_latency_ = std::chrono::milliseconds(std::max(1, max_latency_ms));
    packets_.clear();
}

void BoundedAudioPacketQueue::forceMarker(BoundedAudioPacket *packet) {
    if (!packet) return;
    packet->marker = true;
    if (packet->datagram.size() > 1U) packet->datagram[1] |= 0x80U;
}

BoundedAudioPacketDrop BoundedAudioPacketQueue::discardExpiredInternal(
    const std::chrono::steady_clock::time_point &now) {
    BoundedAudioPacketDrop dropped;
    while (!packets_.empty() && now - packets_.front().enqueued_at > max_latency_) {
        dropped.codec_frames += packets_.front().codec_frames;
        dropped.marker_dropped = dropped.marker_dropped || packets_.front().marker;
        ++dropped.packets;
        packets_.pop_front();
    }
    if (dropped.marker_dropped && !packets_.empty()) forceMarker(&packets_.front());
    return dropped;
}

BoundedAudioPacketDrop BoundedAudioPacketQueue::discardExpired(
    const std::chrono::steady_clock::time_point &now) {
    return discardExpiredInternal(now);
}

BoundedAudioPacketDrop BoundedAudioPacketQueue::push(
    BoundedAudioPacket packet, const std::chrono::steady_clock::time_point &now) {
    BoundedAudioPacketDrop dropped = discardExpiredInternal(now);
    while (packets_.size() >= max_packets_) {
        dropped.codec_frames += packets_.front().codec_frames;
        dropped.marker_dropped = dropped.marker_dropped || packets_.front().marker;
        ++dropped.packets;
        packets_.pop_front();
    }
    if (dropped.marker_dropped) {
        if (!packets_.empty()) forceMarker(&packets_.front());
        else forceMarker(&packet);
    }
    packets_.push_back(std::move(packet));
    return dropped;
}

BoundedAudioPacketDrop BoundedAudioPacketQueue::pop(
    BoundedAudioPacket *packet, const std::chrono::steady_clock::time_point &now) {
    BoundedAudioPacketDrop dropped = discardExpiredInternal(now);
    if (!packet || packets_.empty()) return dropped;
    *packet = std::move(packets_.front());
    packets_.pop_front();
    return dropped;
}

BoundedAudioPacketQueueSnapshot BoundedAudioPacketQueue::snapshot(
    const std::chrono::steady_clock::time_point &now) const {
    BoundedAudioPacketQueueSnapshot value;
    value.packets = packets_.size();
    if (!packets_.empty() && now > packets_.front().enqueued_at) {
        value.oldest_age_ms = static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::milliseconds>(now - packets_.front().enqueued_at).count());
    }
    return value;
}

void BoundedAudioPacketQueue::clear() { packets_.clear(); }

bool BoundedAudioPacketQueue::empty() const { return packets_.empty(); }

}  // namespace roi_h265

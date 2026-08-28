#ifndef ROI_H265_AUDIO_BOUNDED_AUDIO_PACKET_QUEUE_H_
#define ROI_H265_AUDIO_BOUNDED_AUDIO_PACKET_QUEUE_H_

#include <stddef.h>
#include <stdint.h>

#include <chrono>
#include <deque>
#include <vector>

namespace roi_h265 {

// A complete RTP datagram and the Codec2 frames represented by it.  This
// queue deliberately works on whole datagrams: dropping a stale item never
// creates a truncated RTP packet on the wire.
struct BoundedAudioPacket {
    std::vector<uint8_t> datagram;
    uint64_t codec_frames;
    bool marker;
    std::chrono::steady_clock::time_point enqueued_at;

    BoundedAudioPacket() : datagram(), codec_frames(0), marker(false), enqueued_at() {}
};

struct BoundedAudioPacketDrop {
    uint64_t packets;
    uint64_t codec_frames;
    bool marker_dropped;

    BoundedAudioPacketDrop() : packets(0), codec_frames(0), marker_dropped(false) {}
};

struct BoundedAudioPacketQueueSnapshot {
    size_t packets;
    uint64_t oldest_age_ms;

    BoundedAudioPacketQueueSnapshot() : packets(0), oldest_age_ms(0) {}
};

// Caller supplies synchronization.  On saturation or timeout the oldest
// audio is discarded so the next packet always represents the newest speech.
class BoundedAudioPacketQueue {
public:
    BoundedAudioPacketQueue();

    void reset(size_t max_packets, int max_latency_ms);
    BoundedAudioPacketDrop push(BoundedAudioPacket packet,
                                const std::chrono::steady_clock::time_point &now);
    BoundedAudioPacketDrop pop(BoundedAudioPacket *packet,
                               const std::chrono::steady_clock::time_point &now);
    BoundedAudioPacketDrop discardExpired(const std::chrono::steady_clock::time_point &now);
    BoundedAudioPacketQueueSnapshot snapshot(
        const std::chrono::steady_clock::time_point &now) const;
    void clear();
    bool empty() const;

private:
    BoundedAudioPacketDrop discardExpiredInternal(
        const std::chrono::steady_clock::time_point &now);
    static void forceMarker(BoundedAudioPacket *packet);

    size_t max_packets_;
    std::chrono::milliseconds max_latency_;
    std::deque<BoundedAudioPacket> packets_;
};

}  // namespace roi_h265

#endif  // ROI_H265_AUDIO_BOUNDED_AUDIO_PACKET_QUEUE_H_

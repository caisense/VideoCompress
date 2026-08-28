#include <cstdlib>
#include <iostream>

#include "audio/bounded_audio_packet_queue.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

roi_h265::BoundedAudioPacket makePacket(bool marker, uint64_t frames,
                                        const std::chrono::steady_clock::time_point &when) {
    roi_h265::BoundedAudioPacket value;
    value.datagram.assign(12U, 0U);
    value.datagram[1] = marker ? 0x80U : 0U;
    value.codec_frames = frames;
    value.marker = marker;
    value.enqueued_at = when;
    return value;
}

void testCapacityDropsOldestWholePacketAndPreservesMarker() {
    roi_h265::BoundedAudioPacketQueue queue;
    queue.reset(2, 1000);
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    CHECK(queue.push(makePacket(true, 2, now), now).packets == 0U);
    CHECK(queue.push(makePacket(false, 2, now), now).packets == 0U);
    const roi_h265::BoundedAudioPacketDrop dropped = queue.push(makePacket(false, 2, now), now);
    CHECK(dropped.packets == 1U && dropped.codec_frames == 2U && dropped.marker_dropped);

    roi_h265::BoundedAudioPacket first;
    CHECK(queue.pop(&first, now).packets == 0U);
    CHECK(first.marker && (first.datagram[1] & 0x80U) != 0U);
    CHECK(first.codec_frames == 2U);
}

void testAgeLimitDropsStaleAudioBeforeItCanBeSent() {
    roi_h265::BoundedAudioPacketQueue queue;
    queue.reset(3, 100);
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    CHECK(queue.push(makePacket(true, 2, begin), begin).packets == 0U);
    const std::chrono::steady_clock::time_point later = begin + std::chrono::milliseconds(101);
    const roi_h265::BoundedAudioPacketDrop dropped = queue.push(makePacket(false, 2, later), later);
    CHECK(dropped.packets == 1U && dropped.codec_frames == 2U && dropped.marker_dropped);

    roi_h265::BoundedAudioPacket current;
    CHECK(queue.pop(&current, later).packets == 0U);
    CHECK(current.marker && (current.datagram[1] & 0x80U) != 0U);
    CHECK(queue.empty());
}

}  // namespace

int main() {
    testCapacityDropsOldestWholePacketAndPreservesMarker();
    testAgeLimitDropsStaleAudioBeforeItCanBeSent();
    if (failures) return EXIT_FAILURE;
    std::cout << "bounded audio queue tests passed\n";
    return EXIT_SUCCESS;
}

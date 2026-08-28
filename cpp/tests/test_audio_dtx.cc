#include <cstdlib>
#include <iostream>
#include <vector>

#include "audio/codec2_dtx_controller.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ << ": " #condition << "\n"; \
        ++failures; \
    } \
} while (0)

void testMillisecondFrameConversion() {
    CHECK(roi_h265::codec2FramesForDurationMs(0, 320) == 0);
    CHECK(roi_h265::codec2FramesForDurationMs(80, 320) == 2);
    CHECK(roi_h265::codec2FramesForDurationMs(80, 160) == 4);
    CHECK(roi_h265::codec2FramesForDurationMs(600, 320) == 15);
    CHECK(roi_h265::codec2FramesForDurationMs(1000, 320) == 25);
    CHECK(roi_h265::codec2FramesForDurationMs(1, 320) == 1);
}

roi_h265::Codec2DtxFrame frame(int index) {
    roi_h265::Codec2DtxFrame result;
    result.timestamp = static_cast<uint32_t>(index * 320);
    result.payload.push_back(static_cast<uint8_t>(index));
    return result;
}

void checkPacket(const roi_h265::Codec2DtxPacket &packet, int first, int second,
                 bool marker, bool keepalive) {
    CHECK(packet.timestamp == static_cast<uint32_t>(first * 320));
    CHECK(packet.frame_count == 2);
    CHECK(packet.payload.size() == 2U);
    CHECK(packet.payload[0] == static_cast<uint8_t>(first));
    CHECK(packet.payload[1] == static_cast<uint8_t>(second));
    CHECK(packet.marker == marker);
    CHECK(packet.keepalive == keepalive);
}

void testDisabledDtxPreservesContinuousPacketization() {
    roi_h265::Codec2DtxController dtx(false, 2, 2, 25, 0);
    std::vector<roi_h265::Codec2DtxPacket> packets;
    for (int index = 0; index < 4; ++index) dtx.push(frame(index), false, &packets);
    CHECK(packets.size() == 2U);
    checkPacket(packets[0], 0, 1, true, false);
    checkPacket(packets[1], 2, 3, false, false);
    const roi_h265::Codec2DtxSnapshot snapshot = dtx.snapshot();
    CHECK(snapshot.suppressed_codec_frames == 0U);
    CHECK(snapshot.speech_codec_frames == 4U);
    CHECK(snapshot.speech_rtp_packets == 2U);
}

void testDtxPreRollProtectsTalkspurtAndPacketTail() {
    roi_h265::Codec2DtxController dtx(true, 2, 2, 100, 0);
    std::vector<roi_h265::Codec2DtxPacket> packets;
    dtx.push(frame(0), false, &packets);
    dtx.push(frame(1), false, &packets);
    dtx.push(frame(2), true, &packets);
    CHECK(packets.size() == 1U);
    // Frame 1 predates VAD confirmation, so its presence proves that DTX does
    // not cut the first syllable merely because frame 0 was stale history.
    checkPacket(packets[0], 1, 2, true, false);

    dtx.push(frame(3), true, &packets);
    dtx.push(frame(4), false, &packets);
    CHECK(packets.size() == 2U);
    checkPacket(packets[1], 3, 4, false, false);
    const roi_h265::Codec2DtxSnapshot snapshot = dtx.snapshot();
    CHECK(snapshot.suppressed_codec_frames == 1U);
    CHECK(snapshot.speech_codec_frames == 4U);
    CHECK(snapshot.speech_rtp_packets == 2U);
    CHECK(!snapshot.speech_active);
}

void testDtxSendsSparseKeepaliveWithoutTimestampReuse() {
    roi_h265::Codec2DtxController dtx(true, 2, 2, 4, 0);
    std::vector<roi_h265::Codec2DtxPacket> packets;
    for (int index = 0; index < 4; ++index) dtx.push(frame(index), false, &packets);
    CHECK(packets.size() == 1U);
    checkPacket(packets[0], 2, 3, false, true);

    dtx.push(frame(4), false, &packets);
    dtx.push(frame(5), true, &packets);
    CHECK(packets.size() == 2U);
    checkPacket(packets[1], 4, 5, true, false);
    CHECK(packets[1].timestamp > packets[0].timestamp);
    const roi_h265::Codec2DtxSnapshot snapshot = dtx.snapshot();
    CHECK(snapshot.suppressed_codec_frames == 2U);
    CHECK(snapshot.keepalive_codec_frames == 2U);
    CHECK(snapshot.keepalive_rtp_packets == 1U);
}

void testDtxHangoverKeepsOneTalkspurtAcrossBriefGap() {
    // Three 40 ms non-speech frames occur between two voiced frames.  They
    // model quiet consonants or a natural short pause within one sentence.
    roi_h265::Codec2DtxController dtx(true, 2, 2, 100, 3);
    std::vector<roi_h265::Codec2DtxPacket> packets;
    dtx.push(frame(0), false, &packets);
    dtx.push(frame(1), true, &packets);
    checkPacket(packets[0], 0, 1, true, false);
    dtx.push(frame(2), false, &packets);
    dtx.push(frame(3), false, &packets);
    dtx.push(frame(4), false, &packets);
    dtx.push(frame(5), true, &packets);
    CHECK(packets.size() == 3U);
    checkPacket(packets[1], 2, 3, false, false);
    checkPacket(packets[2], 4, 5, false, false);
    const roi_h265::Codec2DtxSnapshot snapshot = dtx.snapshot();
    CHECK(snapshot.hangover_codec_frames == 3U);
    CHECK(snapshot.speech_rtp_packets == 3U);
    CHECK(snapshot.speech_active);
}

}  // namespace

int main() {
    testMillisecondFrameConversion();
    testDisabledDtxPreservesContinuousPacketization();
    testDtxPreRollProtectsTalkspurtAndPacketTail();
    testDtxSendsSparseKeepaliveWithoutTimestampReuse();
    testDtxHangoverKeepsOneTalkspurtAcrossBriefGap();
    if (failures) {
        std::cerr << failures << " audio DTX test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "audio DTX tests passed\n";
    return EXIT_SUCCESS;
}

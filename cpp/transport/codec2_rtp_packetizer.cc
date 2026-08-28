#include "transport/codec2_rtp_packetizer.h"

namespace roi_h265 {

Codec2RtpPacketizer::Codec2RtpPacketizer(uint16_t sequence_number, uint32_t ssrc,
                                         uint8_t payload_type)
    : sequence_number_(sequence_number), ssrc_(ssrc), payload_type_(payload_type) {}

std::vector<uint8_t> Codec2RtpPacketizer::packetize(const uint8_t *payload, size_t length,
                                                     uint32_t timestamp, bool marker) {
    std::vector<uint8_t> packet(12, 0);
    packet[0] = 0x80;
    packet[1] = static_cast<uint8_t>((marker ? 0x80 : 0x00) | (payload_type_ & 0x7f));
    packet[2] = static_cast<uint8_t>(sequence_number_ >> 8);
    packet[3] = static_cast<uint8_t>(sequence_number_++);
    packet[4] = static_cast<uint8_t>(timestamp >> 24);
    packet[5] = static_cast<uint8_t>(timestamp >> 16);
    packet[6] = static_cast<uint8_t>(timestamp >> 8);
    packet[7] = static_cast<uint8_t>(timestamp);
    packet[8] = static_cast<uint8_t>(ssrc_ >> 24);
    packet[9] = static_cast<uint8_t>(ssrc_ >> 16);
    packet[10] = static_cast<uint8_t>(ssrc_ >> 8);
    packet[11] = static_cast<uint8_t>(ssrc_);
    if (payload && length > 0) packet.insert(packet.end(), payload, payload + length);
    return packet;
}

}  // namespace roi_h265

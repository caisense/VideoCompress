#ifndef ROI_H265_TRANSPORT_CODEC2_RTP_PACKETIZER_H_
#define ROI_H265_TRANSPORT_CODEC2_RTP_PACKETIZER_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace roi_h265 {

// Codec2 does not have an IANA static RTP payload type.  This project uses a
// private dynamic payload type and a fixed-size Codec2 frame payload described
// in the companion audio SDP.
class Codec2RtpPacketizer {
public:
    Codec2RtpPacketizer(uint16_t sequence_number, uint32_t ssrc, uint8_t payload_type);

    std::vector<uint8_t> packetize(const uint8_t *payload, size_t length,
                                   uint32_t timestamp, bool marker = true);

private:
    uint16_t sequence_number_;
    uint32_t ssrc_;
    uint8_t payload_type_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_CODEC2_RTP_PACKETIZER_H_

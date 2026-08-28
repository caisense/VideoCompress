#ifndef ROI_H265_TRANSPORT_PACKETIZER_H_
#define ROI_H265_TRANSPORT_PACKETIZER_H_

#include <stdint.h>

#include <cstddef>
#include <vector>

namespace roi_h265 {

struct RtpStreamProfile {
    bool valid;
    uint8_t profile;
    uint16_t width;
    uint16_t height;
    uint8_t fps;
    uint8_t generation;

    RtpStreamProfile()
        : valid(false), profile(0), width(0), height(0), fps(0), generation(0) {}
};

class H265RtpPacketizer {
public:
    H265RtpPacketizer(uint16_t sequence_number, uint32_t ssrc, int mtu);
    std::vector<std::vector<uint8_t> > packetize(const uint8_t *annex_b, size_t length,
                                                  uint32_t timestamp,
                                                  const RtpStreamProfile *profile = NULL);

private:
    std::vector<uint8_t> makeHeader(bool marker, uint32_t timestamp,
                                    const RtpStreamProfile *profile);
    void appendNal(const uint8_t *nal, size_t length, uint32_t timestamp,
                   bool marker, const RtpStreamProfile *profile,
                   std::vector<std::vector<uint8_t> > *packets);

    uint16_t sequence_number_;
    uint32_t ssrc_;
    int mtu_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_PACKETIZER_H_

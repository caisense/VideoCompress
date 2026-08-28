#ifndef ROI_H265_ENCODER_MPP_H265_ENCODER_H_
#define ROI_H265_ENCODER_MPP_H265_ENCODER_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

struct EncodedAccessUnit {
    FrameMeta frame;
    std::vector<uint8_t> bytes;
    bool key_frame;
    int average_qp;
    int realtime_bitrate_bps;

    EncodedAccessUnit() : key_frame(false), average_qp(-1), realtime_bitrate_bps(0) {}
};

class MppH265Encoder {
public:
    explicit MppH265Encoder(const EncoderConfig &config, const RoiConfig &roi_config);
    ~MppH265Encoder();

    bool initialize(std::string *error);
    bool encode(const FramePacket &frame, const std::vector<RoiRegion> &regions,
                EncodedAccessUnit *output, std::string *error);
    bool requestIdr(std::string *error);
    void shutdown();
    bool available() const;

private:
    EncoderConfig config_;
    RoiConfig roi_config_;
    void *state_;
};

}  // namespace roi_h265

#endif  // ROI_H265_ENCODER_MPP_H265_ENCODER_H_

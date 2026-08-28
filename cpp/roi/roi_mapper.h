#ifndef ROI_H265_ROI_MAPPER_H_
#define ROI_H265_ROI_MAPPER_H_

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

class RoiMapper {
public:
    explicit RoiMapper(const RoiConfig &config);

    RoiMap build(const SegResult &segmentation, int encoder_width, int encoder_height) const;
    RoiMap buildBboxMap(const SegResult &segmentation, int encoder_width, int encoder_height) const;

private:
    RoiConfig config_;
};

}  // namespace roi_h265

#endif  // ROI_H265_ROI_MAPPER_H_

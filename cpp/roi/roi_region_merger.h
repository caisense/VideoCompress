#ifndef ROI_H265_ROI_REGION_MERGER_H_
#define ROI_H265_ROI_REGION_MERGER_H_

#include <vector>

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

class RoiRegionMerger {
public:
    explicit RoiRegionMerger(const RoiConfig &config);
    std::vector<RoiRegion> merge(const RoiMap &map) const;

private:
    int deltaQp(RoiLevel level) const;
    RoiConfig config_;
};

}  // namespace roi_h265

#endif  // ROI_H265_ROI_REGION_MERGER_H_

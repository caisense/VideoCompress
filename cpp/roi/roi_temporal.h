#ifndef ROI_H265_ROI_TEMPORAL_H_
#define ROI_H265_ROI_TEMPORAL_H_

#include <mutex>

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

class RoiTemporalSmoother {
public:
    explicit RoiTemporalSmoother(const RoiConfig &config);
    RoiMap update(const RoiMap &incoming);
    void reset();

private:
    RoiConfig config_;
    RoiMap previous_;
    std::vector<int> weaker_counts_;
    bool has_previous_;
};

// Stores only a fully built ROI map. The encoder never waits for inference: it
// selects the latest map that predates its frame and fades it out after max age.
class RoiManager {
public:
    explicit RoiManager(const RoiConfig &config);

    void reconfigure(const RoiConfig &config);
    void submit(const RoiMap &fresh_map);
    RoiMap select(uint64_t encoder_frame_id, uint64_t encoder_pts_us,
                  int encoder_width, int encoder_height) const;

private:
    RoiConfig config_;
    mutable std::mutex mutex_;
    RoiTemporalSmoother smoother_;
    RoiMap latest_;
    bool has_latest_;
};

}  // namespace roi_h265

#endif  // ROI_H265_ROI_TEMPORAL_H_

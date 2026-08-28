#ifndef ROI_H265_TRANSPORT_SNAPSHOT_CROP_H_
#define ROI_H265_TRANSPORT_SNAPSHOT_CROP_H_

#include "common/frame_meta.h"

namespace roi_h265 {

// A source-image rectangle with exclusive right/bottom edges.  An invalid
// rectangle tells the JPEG encoder to preserve the full source frame.
struct SnapshotCrop {
    int left;
    int top;
    int right;
    int bottom;
    bool valid;

    SnapshotCrop() : left(0), top(0), right(0), bottom(0), valid(false) {}
};

// Forms the clipped union of all relevant detection boxes and adds the same
// percentage margin on every side.  It does not upscale a small target crop;
// JPEG encode later applies only the configured maximum dimensions.
SnapshotCrop snapshotCropForRelevantDetections(const SegResult &segmentation,
                                               int source_width, int source_height,
                                               int margin_percent);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_SNAPSHOT_CROP_H_

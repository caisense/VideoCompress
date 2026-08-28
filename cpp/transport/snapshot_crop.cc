#include "transport/snapshot_crop.h"

#include <algorithm>

#include "transport/snapshot_protocol.h"

namespace roi_h265 {

SnapshotCrop snapshotCropForRelevantDetections(const SegResult &segmentation,
                                               int source_width, int source_height,
                                               int margin_percent) {
    SnapshotCrop crop;
    if (source_width <= 0 || source_height <= 0 || margin_percent < 0) return crop;

    int left = source_width;
    int top = source_height;
    int right = 0;
    int bottom = 0;
    bool found = false;
    for (size_t index = 0; index < segmentation.instances.size(); ++index) {
        const SegInstance &instance = segmentation.instances[index];
        if (!isRelevantDetectionClass(instance.class_id)) continue;
        const int instance_left = std::max(0, std::min(source_width, instance.bbox.left));
        const int instance_top = std::max(0, std::min(source_height, instance.bbox.top));
        const int instance_right = std::max(0, std::min(source_width, instance.bbox.right));
        const int instance_bottom = std::max(0, std::min(source_height, instance.bbox.bottom));
        if (instance_right <= instance_left || instance_bottom <= instance_top) continue;
        left = std::min(left, instance_left);
        top = std::min(top, instance_top);
        right = std::max(right, instance_right);
        bottom = std::max(bottom, instance_bottom);
        found = true;
    }
    if (!found) return crop;

    const int width = right - left;
    const int height = bottom - top;
    const int margin_x = (width * std::min(100, margin_percent) + 99) / 100;
    const int margin_y = (height * std::min(100, margin_percent) + 99) / 100;
    crop.left = std::max(0, left - margin_x);
    crop.top = std::max(0, top - margin_y);
    crop.right = std::min(source_width, right + margin_x);
    crop.bottom = std::min(source_height, bottom + margin_y);
    crop.valid = crop.right > crop.left && crop.bottom > crop.top;
    return crop;
}

}  // namespace roi_h265

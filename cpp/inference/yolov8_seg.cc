#include "inference/yolov8_seg.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace roi_h265 {

bool runYolov8Seg(rknn_app_context_t *app_ctx, image_buffer_t *image,
                  const FrameMeta &frame, SegResult *result) {
    if (!app_ctx || !image || !result || image->width <= 0 || image->height <= 0) return false;

    object_detect_result_list raw;
    memset(&raw, 0, sizeof(raw));
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const int ret = inference_yolov8_seg_model(app_ctx, image, &raw);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (ret != 0) {
        release_object_detect_results(&raw);
        return false;
    }

    SegResult converted;
    converted.frame = frame;
    converted.source_width = image->width;
    converted.source_height = image->height;
    converted.inference_latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    const int count = std::min(raw.count, OBJ_NUMB_MAX_SIZE);
    for (int i = 0; i < count; ++i) {
        SegInstance instance;
        instance.class_id = raw.results[i].cls_id;
        instance.confidence = raw.results[i].prop;
        instance.bbox.left = std::max(0, raw.results[i].box.left);
        instance.bbox.top = std::max(0, raw.results[i].box.top);
        instance.bbox.right = std::min(image->width, raw.results[i].box.right);
        instance.bbox.bottom = std::min(image->height, raw.results[i].box.bottom);
        instance.mask_width = image->width;
        instance.mask_height = image->height;
        if (raw.results_seg[i].seg_mask) {
            instance.mask.assign(raw.results_seg[i].seg_mask,
                raw.results_seg[i].seg_mask + static_cast<size_t>(image->width) * image->height);
        }
        converted.instances.push_back(instance);
    }
    release_object_detect_results(&raw);
    *result = converted;
    return true;
}

}  // namespace roi_h265

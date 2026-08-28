#ifndef ROI_H265_INFERENCE_YOLOV8_SEG_H_
#define ROI_H265_INFERENCE_YOLOV8_SEG_H_

#include "common/frame_meta.h"
#include "../yolov8_seg.h"

namespace roi_h265 {

// Converts the existing RKNN implementation's output to the pipeline-neutral
// SegResult. Model inference itself remains in rknpu2/yolov8_seg.cc.
bool runYolov8Seg(rknn_app_context_t *app_ctx, image_buffer_t *image,
                  const FrameMeta &frame, SegResult *result);

}  // namespace roi_h265

#endif  // ROI_H265_INFERENCE_YOLOV8_SEG_H_

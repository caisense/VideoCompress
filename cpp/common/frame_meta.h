#ifndef ROI_H265_COMMON_FRAME_META_H_
#define ROI_H265_COMMON_FRAME_META_H_

#include <cstddef>
#include <stdint.h>

#include <vector>

namespace roi_h265 {

struct BBox {
    int left;
    int top;
    int right;
    int bottom;

    BBox() : left(0), top(0), right(0), bottom(0) {}
};

struct FrameMeta {
    uint64_t frame_id;
    uint64_t pts_us;
    uint64_t capture_time_us;

    FrameMeta() : frame_id(0), pts_us(0), capture_time_us(0) {}
};

// This is deliberately independent from RKNN and MPP.  A mask is a binary,
// row-major image in source-frame coordinates (0 = background, non-zero = ROI).
struct SegInstance {
    int class_id;
    float confidence;
    BBox bbox;
    int mask_width;
    int mask_height;
    std::vector<uint8_t> mask;

    SegInstance()
        : class_id(-1), confidence(0.0f), mask_width(0), mask_height(0) {}
};

struct SegResult {
    FrameMeta frame;
    int source_width;
    int source_height;
    uint64_t inference_latency_us;
    std::vector<SegInstance> instances;

    SegResult()
        : source_width(0), source_height(0), inference_latency_us(0) {}
};

enum RoiLevel {
    ROI_BACKGROUND = 0,
    ROI_HALO = 1,
    ROI_CORE = 2,
    ROI_EDGE = 3,
};

inline int roiPriority(RoiLevel level) {
    return static_cast<int>(level);
}

inline RoiLevel weakerRoiLevel(RoiLevel level) {
    switch (level) {
    case ROI_EDGE: return ROI_CORE;
    case ROI_CORE: return ROI_HALO;
    default: return ROI_BACKGROUND;
    }
}

struct RoiMap {
    FrameMeta source_frame;
    int frame_width;
    int frame_height;
    int cell_size;
    int grid_width;
    int grid_height;
    std::vector<RoiLevel> cells;

    RoiMap()
        : frame_width(0), frame_height(0), cell_size(16), grid_width(0), grid_height(0) {}

    RoiLevel at(int x, int y) const {
        return cells[static_cast<size_t>(y) * grid_width + x];
    }

    void set(int x, int y, RoiLevel value) {
        cells[static_cast<size_t>(y) * grid_width + x] = value;
    }
};

struct RoiRegion {
    int x;
    int y;
    int width;
    int height;
    RoiLevel level;
    int delta_qp;
    bool force_intra;

    RoiRegion()
        : x(0), y(0), width(0), height(0), level(ROI_BACKGROUND),
          delta_qp(0), force_intra(false) {}
};

struct FramePacket {
    FrameMeta meta;
    int source_width;
    int source_height;
    int encoder_width;
    int encoder_height;
    // RGB is used by RKNN. NV12 is the RGA-preprocessed input to MPP.
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> nv12;

    FramePacket()
        : source_width(0), source_height(0), encoder_width(0), encoder_height(0) {}
};

}  // namespace roi_h265

#endif  // ROI_H265_COMMON_FRAME_META_H_

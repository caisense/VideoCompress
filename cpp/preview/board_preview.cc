#include "preview/board_preview.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "yolov8_seg.h"
#include "postprocess.h"

namespace roi_h265 {
namespace {

const char kWindowName[] = "YOLOv8-Seg ROI H.265";

cv::Scalar colorForClass(int class_id) {
    static const unsigned char kColors[][3] = {
        {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255},
        {49, 210, 207}, {10, 249, 72}, {23, 204, 146}, {134, 219, 61},
        {52, 147, 26}, {187, 212, 0}, {168, 153, 44}, {255, 194, 0},
        {147, 69, 52}, {255, 115, 100}, {236, 24, 0}, {255, 56, 132},
        {133, 0, 82}, {255, 56, 203}, {200, 149, 255}, {199, 55, 255},
    };
    const int index = class_id >= 0 ? class_id % N_CLASS_COLORS : 0;
    return cv::Scalar(kColors[index][0], kColors[index][1], kColors[index][2]);
}

void blendMask(cv::Mat* bgr, const cv::Mat& mask, int class_id) {
    if (!bgr || bgr->empty() || mask.empty() || mask.type() != CV_8UC1 ||
        mask.cols != bgr->cols || mask.rows != bgr->rows) {
        return;
    }
    const cv::Scalar color = colorForClass(class_id);
    const unsigned char blue = static_cast<unsigned char>(color[0]);
    const unsigned char green = static_cast<unsigned char>(color[1]);
    const unsigned char red = static_cast<unsigned char>(color[2]);
    for (int y = 0; y < bgr->rows; ++y) {
        unsigned char* pixels = bgr->ptr<unsigned char>(y);
        const unsigned char* mask_pixels = mask.ptr<unsigned char>(y);
        for (int x = 0; x < bgr->cols; ++x) {
            if (!mask_pixels[x]) continue;
            const int offset = x * 3;
            pixels[offset] = static_cast<unsigned char>((pixels[offset] + blue) / 2);
            pixels[offset + 1] = static_cast<unsigned char>((pixels[offset + 1] + green) / 2);
            pixels[offset + 2] = static_cast<unsigned char>((pixels[offset + 2] + red) / 2);
        }
    }
}

cv::Point rotatePointCounterClockwise(const cv::Point& point, int source_width) {
    return cv::Point(point.y, source_width - 1 - point.x);
}

int scalePreviewCoordinate(int coordinate, int source_size, int preview_size) {
    if (source_size <= 0 || preview_size <= 0) return 0;
    const long long scaled = static_cast<long long>(coordinate) * preview_size / source_size;
    return std::max(0, std::min(preview_size - 1, static_cast<int>(scaled)));
}

void previewBounds(const SegInstance& instance, int source_width, int source_height,
                   int preview_width, int preview_height, bool rotate_ccw,
                   int* left, int* top, int* right, int* bottom) {
    const int source_left = std::max(0, std::min(source_width - 1, instance.bbox.left));
    const int source_top = std::max(0, std::min(source_height - 1, instance.bbox.top));
    const int source_right = std::max(0, std::min(source_width - 1, instance.bbox.right));
    const int source_bottom = std::max(0, std::min(source_height - 1, instance.bbox.bottom));
    const cv::Point corners[] = {
        cv::Point(scalePreviewCoordinate(source_left, source_width, preview_width),
                  scalePreviewCoordinate(source_top, source_height, preview_height)),
        cv::Point(scalePreviewCoordinate(source_right, source_width, preview_width),
                  scalePreviewCoordinate(source_top, source_height, preview_height)),
        cv::Point(scalePreviewCoordinate(source_left, source_width, preview_width),
                  scalePreviewCoordinate(source_bottom, source_height, preview_height)),
        cv::Point(scalePreviewCoordinate(source_right, source_width, preview_width),
                  scalePreviewCoordinate(source_bottom, source_height, preview_height)),
    };
    if (!rotate_ccw) {
        *left = corners[0].x;
        *top = corners[0].y;
        *right = corners[3].x;
        *bottom = corners[3].y;
        return;
    }
    cv::Point rotated = rotatePointCounterClockwise(corners[0], preview_width);
    *left = rotated.x;
    *top = rotated.y;
    *right = rotated.x;
    *bottom = rotated.y;
    for (size_t index = 1; index < sizeof(corners) / sizeof(corners[0]); ++index) {
        rotated = rotatePointCounterClockwise(corners[index], preview_width);
        *left = std::min(*left, rotated.x);
        *top = std::min(*top, rotated.y);
        *right = std::max(*right, rotated.x);
        *bottom = std::max(*bottom, rotated.y);
    }
}

}  // namespace

BoardPreview::BoardPreview(int window_width, int window_height, bool rotate_ccw)
    : opened_(false), window_width_(window_width), window_height_(window_height),
      rotate_ccw_(rotate_ccw) {}

BoardPreview::~BoardPreview() { close(); }

void BoardPreview::prepareEnvironment() {
    const char* runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    if ((runtime_dir == NULL || *runtime_dir == '\0') &&
        (wayland_display == NULL || *wayland_display == '\0') &&
        access("/run/wayland-0", F_OK) == 0 &&
        setenv("XDG_RUNTIME_DIR", "/run", 0) == 0 &&
        setenv("WAYLAND_DISPLAY", "wayland-0", 0) == 0) {
        std::fprintf(stderr, "Board preview: using Weston socket /run/wayland-0\n");
    }
}

bool BoardPreview::open(std::string* error) {
    if (opened_) return true;
    prepareEnvironment();
    if (access("/run/wayland-0", F_OK) != 0 &&
        (!std::getenv("DISPLAY") || !*std::getenv("DISPLAY"))) {
        if (error) *error = "no Weston Wayland socket or X11 display is available";
        return false;
    }
    try {
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, window_width_, window_height_);
        std::fprintf(stderr, "Board preview: encoder input, rotate=%s, window=%dx%d\n",
                     rotate_ccw_ ? "ccw" : "none", window_width_, window_height_);
        opened_ = true;
        return true;
    } catch (const cv::Exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

bool BoardPreview::show(const FramePacket& frame, const SegResult* segmentation, std::string* error) {
    if (!opened_ && !open(error)) return false;
    if (frame.source_width <= 0 || frame.source_height <= 0 || frame.encoder_width <= 0 ||
        frame.encoder_height <= 0 || frame.nv12.size() < static_cast<size_t>(frame.encoder_width) *
            frame.encoder_height * 3 / 2) {
        if (error) *error = "invalid encoder NV12 frame for preview";
        return false;
    }
    try {
        // Display the exact image passed into MPP, rather than the 4:3 camera
        // frame.  This makes the board preview's content and 9:16 rotated
        // aspect match the H.265 receiver, including the selected rate profile.
        const cv::Mat nv12(frame.encoder_height * 3 / 2, frame.encoder_width, CV_8UC1,
                           const_cast<unsigned char*>(frame.nv12.data()));
        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        const int preview_width = bgr.cols;
        const int preview_height = bgr.rows;
        if (rotate_ccw_) {
            cv::Mat rotated_bgr;
            cv::rotate(bgr, rotated_bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
            bgr = rotated_bgr;
        }
        if (segmentation) {
            const int segmentation_width = segmentation->source_width > 0
                ? segmentation->source_width : frame.source_width;
            const int segmentation_height = segmentation->source_height > 0
                ? segmentation->source_height : frame.source_height;
            for (size_t index = 0; index < segmentation->instances.size(); ++index) {
                const SegInstance& instance = segmentation->instances[index];
                if (instance.mask_width == segmentation_width &&
                    instance.mask_height == segmentation_height &&
                    instance.mask.size() >= static_cast<size_t>(segmentation_width) * segmentation_height) {
                    const cv::Mat source_mask(instance.mask_height, instance.mask_width, CV_8UC1,
                                              const_cast<unsigned char*>(instance.mask.data()));
                    cv::Mat encoder_mask;
                    if (source_mask.cols == preview_width && source_mask.rows == preview_height) {
                        encoder_mask = source_mask;
                    } else {
                        cv::resize(source_mask, encoder_mask,
                                   cv::Size(preview_width, preview_height), 0.0, 0.0,
                                   cv::INTER_NEAREST);
                    }
                    if (rotate_ccw_) {
                        cv::Mat rotated_mask;
                        cv::rotate(encoder_mask, rotated_mask, cv::ROTATE_90_COUNTERCLOCKWISE);
                        blendMask(&bgr, rotated_mask, instance.class_id);
                    } else {
                        blendMask(&bgr, encoder_mask, instance.class_id);
                    }
                }
                const cv::Scalar color = colorForClass(instance.class_id);
                int left = 0;
                int top = 0;
                int right = 0;
                int bottom = 0;
                previewBounds(instance, segmentation_width, segmentation_height,
                              preview_width, preview_height, rotate_ccw_,
                              &left, &top, &right, &bottom);
                left = std::max(0, std::min(bgr.cols - 1, left));
                top = std::max(0, std::min(bgr.rows - 1, top));
                right = std::max(0, std::min(bgr.cols - 1, right));
                bottom = std::max(0, std::min(bgr.rows - 1, bottom));
                if (right > left && bottom > top) {
                    cv::rectangle(bgr, cv::Point(left, top), cv::Point(right, bottom), color, 2);
                }
                char label[128];
                std::snprintf(label, sizeof(label), "%s %.1f%%", coco_cls_to_name(instance.class_id),
                              instance.confidence * 100.0f);
                cv::putText(bgr, label, cv::Point(left, std::max(16, top - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
            }
        }
        char status[96];
        const unsigned long long frame_id = static_cast<unsigned long long>(frame.meta.frame_id);
        const unsigned long long seg_us = segmentation
            ? static_cast<unsigned long long>(segmentation->inference_latency_us) : 0ULL;
        std::snprintf(status, sizeof(status), "frame=%llu seg=%.1f ms", frame_id, seg_us / 1000.0);
        cv::putText(bgr, status, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::imshow(kWindowName, bgr);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            if (error) *error = "preview window closed by user";
            return false;
        }
        return true;
    } catch (const cv::Exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

void BoardPreview::close() {
    if (!opened_) return;
    try {
        cv::destroyWindow(kWindowName);
    } catch (const cv::Exception&) {
    }
    opened_ = false;
}

}  // namespace roi_h265

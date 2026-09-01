#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "common/config.h"
#include "transport/rebuild_reference.h"

namespace {

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << std::endl; \
        std::exit(1); \
    } \
} while (0)

roi_h265::SegInstance makePerson(int left, int top, int right, int bottom,
                                 bool with_mask) {
    roi_h265::SegInstance instance;
    instance.class_id = 0;
    instance.confidence = 0.95f;
    instance.bbox.left = left;
    instance.bbox.top = top;
    instance.bbox.right = right;
    instance.bbox.bottom = bottom;
    if (with_mask) {
        instance.mask_width = 640;
        instance.mask_height = 480;
        instance.mask.assign(640U * 480U, 0);
        for (int y = std::max(0, top + 10); y < std::min(480, top + 130); ++y) {
            for (int x = std::max(0, left + 20); x < std::min(640, right - 20); ++x) {
                instance.mask[static_cast<size_t>(y) * 640U + x] = 255;
            }
        }
    }
    return instance;
}

void testMaskHeadAndFullGeometry() {
    roi_h265::RebuildConfig config;
    config.reference_mode = roi_h265::REBUILD_REFERENCE_MODE_ADAPTIVE;
    const roi_h265::SegInstance person = makePerson(100, 50, 300, 450, true);
    const roi_h265::ReferenceCropSelection full =
        roi_h265::selectFullReferenceCrop(person.bbox, 640, 480, 20);
    const roi_h265::ReferenceCropSelection head =
        roi_h265::selectReferenceCrop(person, 640, 480, config);
    CHECK(full.valid);
    CHECK(head.valid);
    CHECK(full.kind == roi_h265::REFERENCE_KIND_FULL);
    CHECK(head.kind == roi_h265::REFERENCE_KIND_HEAD_PRIORITY);
    CHECK(head.used_mask);
    CHECK(head.crop.left >= person.bbox.left);
    CHECK(head.crop.top >= person.bbox.top);
    CHECK(head.crop.right <= person.bbox.right);
    CHECK(head.crop.bottom <= person.bbox.bottom);
    CHECK(head.crop.right - head.crop.left < full.crop.right - full.crop.left);
    CHECK(head.crop.bottom - head.crop.top < full.crop.bottom - full.crop.top);
}

void testFallbacksAndNonPersonStaySafe() {
    roi_h265::RebuildConfig config;
    config.reference_mode = roi_h265::REBUILD_REFERENCE_MODE_ADAPTIVE;
    const roi_h265::SegInstance tiny = makePerson(10, 10, 25, 30, false);
    const roi_h265::ReferenceCropSelection tiny_result =
        roi_h265::selectReferenceCrop(tiny, 640, 480, config);
    CHECK(tiny_result.valid);
    CHECK(tiny_result.kind == roi_h265::REFERENCE_KIND_FULL);
    CHECK(tiny_result.fallback_reason == "SMALL");
    CHECK(tiny_result.crop.left >= 0 && tiny_result.crop.top >= 0);
    CHECK(tiny_result.crop.right <= 640 && tiny_result.crop.bottom <= 480);

    const roi_h265::SegInstance edge = makePerson(-8, 30, 60, 230, true);
    const roi_h265::ReferenceCropSelection edge_result =
        roi_h265::selectReferenceCrop(edge, 640, 480, config);
    CHECK(edge_result.valid);
    CHECK(edge_result.crop.left >= 0 && edge_result.crop.top >= 0);
    CHECK(edge_result.crop.right <= 640 && edge_result.crop.bottom <= 480);

    roi_h265::SegInstance vehicle = makePerson(100, 50, 300, 450, true);
    vehicle.class_id = 2;
    const roi_h265::ReferenceCropSelection vehicle_result =
        roi_h265::selectReferenceCrop(vehicle, 640, 480, config);
    CHECK(vehicle_result.valid);
    CHECK(vehicle_result.kind == roi_h265::REFERENCE_KIND_FULL);
    CHECK(vehicle_result.fallback_reason.empty());
}

void testSeatedPartialPersonUsesProxyWhenMaskIsSparse() {
    roi_h265::RebuildConfig config;
    config.reference_mode = roi_h265::REBUILD_REFERENCE_MODE_HEAD;
    roi_h265::SegInstance person = makePerson(250, 160, 330, 280, false);
    const roi_h265::ReferenceCropSelection result =
        roi_h265::selectReferenceCrop(person, 640, 480, config);
    CHECK(result.valid);
    CHECK(result.kind == roi_h265::REFERENCE_KIND_HEAD_PRIORITY);
    CHECK(result.used_fallback_proxy);
    CHECK(result.crop.right - result.crop.left >= config.head_min_size);
    CHECK(result.crop.bottom - result.crop.top >= config.head_min_size);
}

}  // namespace

int main() {
    testMaskHeadAndFullGeometry();
    testFallbacksAndNonPersonStaySafe();
    testSeatedPartialPersonUsesProxyWhenMaskIsSparse();
    std::cout << "rebuild reference selector tests passed" << std::endl;
    return 0;
}

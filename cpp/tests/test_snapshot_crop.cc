#include <cstdlib>
#include <iostream>

#include "common/frame_meta.h"
#include "transport/snapshot_crop.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

roi_h265::SegInstance instance(int class_id, int left, int top, int right, int bottom) {
    roi_h265::SegInstance value;
    value.class_id = class_id;
    value.bbox.left = left;
    value.bbox.top = top;
    value.bbox.right = right;
    value.bbox.bottom = bottom;
    return value;
}

void testRelevantUnionWithContext() {
    roi_h265::SegResult segmentation;
    segmentation.instances.push_back(instance(1, 0, 0, 640, 480));  // ignored class
    segmentation.instances.push_back(instance(0, 100, 100, 200, 200));
    segmentation.instances.push_back(instance(2, 300, 120, 400, 220));
    const roi_h265::SnapshotCrop crop = roi_h265::snapshotCropForRelevantDetections(
        segmentation, 640, 480, 25);
    CHECK(crop.valid);
    CHECK(crop.left == 25 && crop.top == 70);
    CHECK(crop.right == 475 && crop.bottom == 250);
}

void testCropClampsAtSourceEdges() {
    roi_h265::SegResult segmentation;
    segmentation.instances.push_back(instance(8, -20, -10, 40, 40));
    const roi_h265::SnapshotCrop crop = roi_h265::snapshotCropForRelevantDetections(
        segmentation, 640, 480, 25);
    CHECK(crop.valid);
    CHECK(crop.left == 0 && crop.top == 0);
    CHECK(crop.right == 50 && crop.bottom == 50);
}

void testNoValidRelevantBoxUsesFullFrame() {
    roi_h265::SegResult segmentation;
    segmentation.instances.push_back(instance(1, 10, 10, 100, 100));
    segmentation.instances.push_back(instance(0, 50, 50, 50, 100));
    const roi_h265::SnapshotCrop crop = roi_h265::snapshotCropForRelevantDetections(
        segmentation, 640, 480, 25);
    CHECK(!crop.valid);
}

}  // namespace

int main() {
    testRelevantUnionWithContext();
    testCropClampsAtSourceEdges();
    testNoValidRelevantBoxUsesFullFrame();
    if (failures != 0) {
        std::cerr << failures << " snapshot crop test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Snapshot crop tests passed\n";
    return EXIT_SUCCESS;
}

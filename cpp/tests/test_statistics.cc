#include <cstdlib>
#include <iostream>

#include "common/statistics.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

void testAverageBitrateUsesFrameTimeline() {
    roi_h265::RuntimeStatistics statistics;
    roi_h265::FrameMeta first;
    first.frame_id = 0;
    first.pts_us = 1000000;
    first.capture_time_us = first.pts_us;
    statistics.recordEncoded(first, true, 4000, 25, 0, 4);
    CHECK(statistics.snapshot().average_bitrate_bps == 0);

    roi_h265::FrameMeta second = first;
    second.frame_id = 1;
    second.pts_us += 333333;
    second.capture_time_us = second.pts_us;
    statistics.recordEncoded(second, false, 500, 28, 0, 1);
    const roi_h265::StatisticsSnapshot snapshot = statistics.snapshot();
    CHECK(snapshot.average_bitrate_bps > 100000);
    CHECK(snapshot.average_bitrate_bps < 120000);
    CHECK(snapshot.instantaneous_bitrate_bps == (4000 + 500) * 8);
}

}  // namespace

int main() {
    testAverageBitrateUsesFrameTimeline();
    if (failures) return EXIT_FAILURE;
    std::cout << "statistics tests passed\n";
    return EXIT_SUCCESS;
}

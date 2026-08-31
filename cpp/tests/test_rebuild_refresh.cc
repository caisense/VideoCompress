#include <cstdlib>
#include <iostream>

#include "transport/rebuild_refresh.h"

namespace {

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << std::endl; \
        std::exit(1); \
    } \
} while (0)

void testSixFpsUsesTheLast334MsOpportunity() {
    const roi_h265::RebuildRefreshDecision at_0 =
        roi_h265::evaluateRebuildRefresh(true, false, 0, 0, 72, 6, 350, 450, 75);
    CHECK(!at_0.start);
    CHECK(at_0.reason == roi_h265::REBUILD_REFRESH_HOLD);

    const roi_h265::RebuildRefreshDecision at_167 =
        roi_h265::evaluateRebuildRefresh(true, false, 167, 67, 72, 6, 350, 450, 75);
    CHECK(!at_167.start);
    CHECK(at_167.reason == roi_h265::REBUILD_REFRESH_HOLD);

    const roi_h265::RebuildRefreshDecision at_334 =
        roi_h265::evaluateRebuildRefresh(true, false, 334, 234, 72, 6, 350, 450, 75);
    CHECK(at_334.start);
    CHECK(at_334.reason == roi_h265::REBUILD_REFRESH_HARD_DEADLINE);
    CHECK(at_334.refresh_threshold_ms == 220);
    CHECK(at_334.next_deadline_ms == 303);
    CHECK(at_334.scheduling_quantum_ms == 167);

    const roi_h265::RebuildRefreshDecision at_501 =
        roi_h265::evaluateRebuildRefresh(true, false, 501, 401, 72, 6, 350, 450, 75);
    CHECK(at_501.start);
    CHECK(at_501.reason == roi_h265::REBUILD_REFRESH_HARD_DEADLINE);
}

void testCaptureAgeWinsOverReadyAge() {
    const roi_h265::RebuildRefreshDecision decision =
        roi_h265::evaluateRebuildRefresh(true, false, 334, 234, 72, 6, 350, 450, 75);
    CHECK(decision.start);
    CHECK(decision.capture_age_ms == 334);
    CHECK(decision.ready_age_ms == 234);
}

void testSlowDeliveryMovesTheDecisionEarlier() {
    const roi_h265::RebuildRefreshDecision decision =
        roi_h265::evaluateRebuildRefresh(true, false, 167, 67, 150, 6, 350, 450, 75);
    CHECK(decision.start);
    CHECK(decision.reason == roi_h265::REBUILD_REFRESH_HARD_DEADLINE);
    CHECK(decision.next_deadline_ms == 225);
    CHECK(decision.refresh_threshold_ms == 142);
}

void testReferenceAndTransferGuards() {
    const roi_h265::RebuildRefreshDecision missing =
        roi_h265::evaluateRebuildRefresh(false, false, 0, 0, 72, 6, 350, 450, 75);
    CHECK(missing.start);
    CHECK(missing.reason == roi_h265::REBUILD_REFRESH_NO_REFERENCE);

    const roi_h265::RebuildRefreshDecision active =
        roi_h265::evaluateRebuildRefresh(true, true, 500, 400, 72, 6, 350, 450, 75);
    CHECK(!active.start);
    CHECK(active.reason == roi_h265::REBUILD_REFRESH_TRANSFER_ACTIVE);
}

void testBandwidthProtectionKeepsOneStartPerOpportunity() {
    const int ages[] = {0, 167, 334, 501};
    int starts = 0;
    for (int age : ages) {
        const roi_h265::RebuildRefreshDecision decision =
            roi_h265::evaluateRebuildRefresh(true, false, age, age, 72, 6,
                                              350, 450, 75);
        if (decision.start) ++starts;
    }
    CHECK(starts == 2);
    const roi_h265::RebuildRefreshDecision transfer_active =
        roi_h265::evaluateRebuildRefresh(true, true, 334, 234, 72, 6,
                                          350, 450, 75);
    CHECK(!transfer_active.start);
}

}  // namespace

int main() {
    testSixFpsUsesTheLast334MsOpportunity();
    testCaptureAgeWinsOverReadyAge();
    testSlowDeliveryMovesTheDecisionEarlier();
    testReferenceAndTransferGuards();
    testBandwidthProtectionKeepsOneStartPerOpportunity();
    std::cout << "rebuild refresh tests passed" << std::endl;
    return 0;
}

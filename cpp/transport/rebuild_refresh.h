#ifndef ROI_H265_TRANSPORT_REBUILD_REFRESH_H_
#define ROI_H265_TRANSPORT_REBUILD_REFRESH_H_

#include <algorithm>

namespace roi_h265 {

enum RebuildRefreshReason {
    REBUILD_REFRESH_NO_REFERENCE = 0,
    REBUILD_REFRESH_HARD_DEADLINE = 1,
    REBUILD_REFRESH_SOFT_CADENCE = 2,
    REBUILD_REFRESH_HOLD = 3,
    REBUILD_REFRESH_TRANSFER_ACTIVE = 4,
    REBUILD_REFRESH_NO_ACTIVE_TARGET = 5,
};

struct RebuildRefreshDecision {
    bool start;
    int capture_age_ms;
    int ready_age_ms;
    int estimated_delivery_ms;
    int scheduling_quantum_ms;
    int refresh_threshold_ms;
    // Age of the reference at which a new transfer must be started to leave
    // room for its estimated delivery and the configured guard.
    int next_deadline_ms;
    RebuildRefreshReason reason;

    RebuildRefreshDecision()
        : start(false), capture_age_ms(0), ready_age_ms(0),
          estimated_delivery_ms(0), scheduling_quantum_ms(1000),
          refresh_threshold_ms(0), next_deadline_ms(0),
          reason(REBUILD_REFRESH_NO_ACTIVE_TARGET) {}
};

inline const char *rebuildRefreshReasonName(RebuildRefreshReason reason) {
    switch (reason) {
        case REBUILD_REFRESH_NO_REFERENCE: return "NO_REFERENCE";
        case REBUILD_REFRESH_HARD_DEADLINE: return "HARD_DEADLINE";
        case REBUILD_REFRESH_SOFT_CADENCE: return "SOFT_CADENCE";
        case REBUILD_REFRESH_HOLD: return "HOLD";
        case REBUILD_REFRESH_TRANSFER_ACTIVE: return "TRANSFER_ACTIVE";
        case REBUILD_REFRESH_NO_ACTIVE_TARGET: return "NO_ACTIVE_TARGET";
    }
    return "UNKNOWN";
}

// Evaluate one track at one source-frame opportunity.  The half-quantum
// allowance makes a continuous deadline usable with a discrete 6 fps worker:
// a nominal 350 ms soft cadence becomes the 334 ms opportunity, while a slow
// estimated delivery can move the decision to the preceding opportunity.
inline RebuildRefreshDecision evaluateRebuildRefresh(
        bool has_reference, bool transfer_active, int capture_age_ms,
        int ready_age_ms, int estimated_delivery_ms, int source_fps,
        int soft_refresh_ms, int hard_deadline_ms, int refresh_guard_ms) {
    RebuildRefreshDecision decision;
    decision.capture_age_ms = std::max(0, capture_age_ms);
    decision.ready_age_ms = std::max(0, ready_age_ms);
    decision.estimated_delivery_ms = std::max(0, estimated_delivery_ms);
    const int fps = std::max(1, source_fps);
    decision.scheduling_quantum_ms = (1000 + fps - 1) / fps;
    const int half_quantum = std::max(1, decision.scheduling_quantum_ms / 2);
    const int hard_window = std::max(0, hard_deadline_ms -
        std::max(0, refresh_guard_ms));
    decision.next_deadline_ms = std::max(0, hard_window -
        decision.estimated_delivery_ms);
    const int hard_start = std::max(0, decision.next_deadline_ms - half_quantum);
    const int soft_start = std::max(0, soft_refresh_ms - half_quantum);
    decision.refresh_threshold_ms = std::min(soft_start, hard_start);

    if (!has_reference) {
        decision.start = true;
        decision.reason = REBUILD_REFRESH_NO_REFERENCE;
    } else if (transfer_active) {
        decision.reason = REBUILD_REFRESH_TRANSFER_ACTIVE;
    } else if (decision.capture_age_ms >= hard_start) {
        decision.start = true;
        decision.reason = REBUILD_REFRESH_HARD_DEADLINE;
    } else if (decision.capture_age_ms >= soft_start) {
        decision.start = true;
        decision.reason = REBUILD_REFRESH_SOFT_CADENCE;
    } else {
        decision.reason = REBUILD_REFRESH_HOLD;
    }
    return decision;
}

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_REBUILD_REFRESH_H_

#ifndef ROI_H265_COMMON_STATISTICS_H_
#define ROI_H265_COMMON_STATISTICS_H_

#include <stdint.h>

#include <deque>
#include <mutex>
#include <string>

#include "common/frame_meta.h"

namespace roi_h265 {

struct StatisticsSnapshot {
    uint64_t frame_id;
    uint64_t segmentation_latency_us;
    uint64_t segmentation_result_age_frames;
    int roi_cells;
    double background_ratio;
    double halo_ratio;
    double core_ratio;
    double edge_ratio;
    int roi_rectangles;
    std::string h265_frame_type;
    size_t encoded_frame_bytes;
    int instantaneous_bitrate_bps;
    int average_bitrate_bps;
    int encoder_average_qp;
    size_t send_queue_size;
    uint64_t end_to_end_latency_us;
};

class RuntimeStatistics {
public:
    RuntimeStatistics();

    void recordSegmentation(const SegResult &result);
    void recordRoi(const RoiMap &map, uint64_t encoder_frame_id, int rectangle_count);
    void recordEncoded(const FrameMeta &frame, bool key_frame, size_t bytes, int average_qp,
                       int encoder_realtime_bps, size_t send_queue_size);
    StatisticsSnapshot snapshot() const;
    std::string logLine() const;

private:
    struct ByteSample {
        uint64_t timestamp_us;
        size_t bytes;
    };
    static uint64_t nowMicros();
    void trimOneSecondLocked(uint64_t now_us);

    mutable std::mutex mutex_;
    StatisticsSnapshot latest_;
    std::deque<ByteSample> one_second_bytes_;
    uint64_t total_bytes_;
    uint64_t first_frame_pts_us_;
};

}  // namespace roi_h265

#endif  // ROI_H265_COMMON_STATISTICS_H_

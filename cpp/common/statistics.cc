#include "common/statistics.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>

namespace roi_h265 {

RuntimeStatistics::RuntimeStatistics()
    : total_bytes_(0), first_frame_pts_us_(0) {
    latest_.frame_id = 0;
    latest_.segmentation_latency_us = 0;
    latest_.segmentation_result_age_frames = 0;
    latest_.roi_cells = 0;
    latest_.background_ratio = 1.0;
    latest_.halo_ratio = 0.0;
    latest_.core_ratio = 0.0;
    latest_.edge_ratio = 0.0;
    latest_.roi_rectangles = 0;
    latest_.h265_frame_type = "unknown";
    latest_.encoded_frame_bytes = 0;
    latest_.instantaneous_bitrate_bps = 0;
    latest_.average_bitrate_bps = 0;
    latest_.encoder_average_qp = -1;
    latest_.send_queue_size = 0;
    latest_.end_to_end_latency_us = 0;
}

uint64_t RuntimeStatistics::nowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void RuntimeStatistics::trimOneSecondLocked(uint64_t now_us) {
    while (!one_second_bytes_.empty() && now_us - one_second_bytes_.front().timestamp_us > 1000000) {
        one_second_bytes_.pop_front();
    }
}

void RuntimeStatistics::recordSegmentation(const SegResult &result) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_.segmentation_latency_us = result.inference_latency_us;
}

void RuntimeStatistics::recordRoi(const RoiMap &map, uint64_t encoder_frame_id, int rectangle_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    int counts[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < map.cells.size(); ++i) ++counts[static_cast<int>(map.cells[i])];
    const double count = map.cells.empty() ? 1.0 : static_cast<double>(map.cells.size());
    latest_.segmentation_result_age_frames = encoder_frame_id >= map.source_frame.frame_id
        ? encoder_frame_id - map.source_frame.frame_id : 0;
    latest_.roi_cells = static_cast<int>(map.cells.size());
    latest_.background_ratio = counts[ROI_BACKGROUND] / count;
    latest_.halo_ratio = counts[ROI_HALO] / count;
    latest_.core_ratio = counts[ROI_CORE] / count;
    latest_.edge_ratio = counts[ROI_EDGE] / count;
    latest_.roi_rectangles = rectangle_count;
}

void RuntimeStatistics::recordEncoded(const FrameMeta &frame, bool key_frame, size_t bytes, int average_qp,
                                      int encoder_realtime_bps, size_t send_queue_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t now_us = nowMicros();
    if (first_frame_pts_us_ == 0) first_frame_pts_us_ = frame.pts_us;
    one_second_bytes_.push_back(ByteSample{now_us, bytes});
    trimOneSecondLocked(now_us);
    size_t one_second_total = 0;
    for (std::deque<ByteSample>::const_iterator it = one_second_bytes_.begin(); it != one_second_bytes_.end(); ++it)
        one_second_total += it->bytes;
    total_bytes_ += bytes;
    latest_.frame_id = frame.frame_id;
    latest_.h265_frame_type = key_frame ? "IDR/IRAP" : "P";
    latest_.encoded_frame_bytes = bytes;
    const uint64_t instantaneous_bps = one_second_total * 8;
    latest_.instantaneous_bitrate_bps = instantaneous_bps > static_cast<uint64_t>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max() : static_cast<int>(instantaneous_bps);
    // Use capture PTS rather than wall-clock encode completion time.  On the
    // first frame the latter spans only a few microseconds and previously
    // overflowed the signed metric even though the configured stream is low
    // bitrate.  A single frame has no duration, so report zero until a second
    // timestamp establishes the stream interval.
    const uint64_t elapsed_us = frame.pts_us > first_frame_pts_us_
        ? frame.pts_us - first_frame_pts_us_ : 0;
    const uint64_t average_bps = elapsed_us == 0 ? 0
        : total_bytes_ * 8 * 1000000ULL / elapsed_us;
    latest_.average_bitrate_bps = average_bps > static_cast<uint64_t>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max() : static_cast<int>(average_bps);
    if (encoder_realtime_bps > 0) latest_.average_bitrate_bps = encoder_realtime_bps;
    latest_.encoder_average_qp = average_qp;
    latest_.send_queue_size = send_queue_size;
    latest_.end_to_end_latency_us = frame.capture_time_us > 0 && now_us >= frame.capture_time_us
        ? now_us - frame.capture_time_us : 0;
}

StatisticsSnapshot RuntimeStatistics::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

std::string RuntimeStatistics::logLine() const {
    const StatisticsSnapshot value = snapshot();
    std::ostringstream line;
    line << "frame=" << value.frame_id
         << " seg_us=" << value.segmentation_latency_us
         << " roi_age=" << value.segmentation_result_age_frames
         << " roi_cells=" << value.roi_cells
         << " ratios[B/H/C/E]=" << std::fixed << std::setprecision(2)
         << value.background_ratio << '/' << value.halo_ratio << '/' << value.core_ratio << '/' << value.edge_ratio
         << " roi_rects=" << value.roi_rectangles
         << " type=" << value.h265_frame_type
         << " bytes=" << value.encoded_frame_bytes
         << " inst_bps=" << value.instantaneous_bitrate_bps
         << " avg_bps=" << value.average_bitrate_bps
         << " qp=" << value.encoder_average_qp
         << " send_q=" << value.send_queue_size
         << " e2e_us=" << value.end_to_end_latency_us;
    return line.str();
}

}  // namespace roi_h265

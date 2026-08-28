#include "roi/roi_temporal.h"

#include <algorithm>

namespace roi_h265 {
namespace {

RoiMap backgroundMap(uint64_t frame_id, uint64_t pts_us, int width, int height, int cell_size) {
    RoiMap map;
    map.source_frame.frame_id = frame_id;
    map.source_frame.pts_us = pts_us;
    map.frame_width = width;
    map.frame_height = height;
    map.cell_size = cell_size;
    map.grid_width = (width + cell_size - 1) / cell_size;
    map.grid_height = (height + cell_size - 1) / cell_size;
    map.cells.assign(static_cast<size_t>(map.grid_width) * map.grid_height, ROI_BACKGROUND);
    return map;
}

}  // namespace

RoiTemporalSmoother::RoiTemporalSmoother(const RoiConfig &config)
    : config_(config), has_previous_(false) {}

void RoiTemporalSmoother::reset() {
    previous_ = RoiMap();
    weaker_counts_.clear();
    has_previous_ = false;
}

RoiMap RoiTemporalSmoother::update(const RoiMap &incoming) {
    if (!has_previous_ || previous_.grid_width != incoming.grid_width ||
        previous_.grid_height != incoming.grid_height || previous_.cell_size != incoming.cell_size) {
        previous_ = incoming;
        weaker_counts_.assign(incoming.cells.size(), 0);
        has_previous_ = true;
        return previous_;
    }

    RoiMap output = incoming;
    for (size_t index = 0; index < incoming.cells.size(); ++index) {
        const RoiLevel before = previous_.cells[index];
        const RoiLevel now = incoming.cells[index];
        if (roiPriority(now) >= roiPriority(before)) {
            // A newly appearing target gets its full priority without delay.
            output.cells[index] = now;
            weaker_counts_[index] = 0;
        } else {
            ++weaker_counts_[index];
            if (weaker_counts_[index] < config_.hold_frames) {
                output.cells[index] = before;
            } else {
                // A disappearance is intentionally staged: edge -> core -> halo -> background.
                output.cells[index] = weakerRoiLevel(before);
                weaker_counts_[index] = 0;
            }
        }
    }
    previous_ = output;
    return previous_;
}

RoiManager::RoiManager(const RoiConfig &config)
    : config_(config), smoother_(config), has_latest_(false) {}

void RoiManager::reconfigure(const RoiConfig &config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    smoother_ = RoiTemporalSmoother(config);
    latest_ = RoiMap();
    has_latest_ = false;
}

void RoiManager::submit(const RoiMap &fresh_map) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_latest_ && fresh_map.source_frame.frame_id < latest_.source_frame.frame_id) return;
    latest_ = smoother_.update(fresh_map);
    has_latest_ = true;
}

RoiMap RoiManager::select(uint64_t encoder_frame_id, uint64_t encoder_pts_us,
                          int encoder_width, int encoder_height) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_latest_ || latest_.source_frame.frame_id > encoder_frame_id ||
        latest_.source_frame.pts_us > encoder_pts_us ||
        latest_.frame_width != encoder_width || latest_.frame_height != encoder_height) {
        return backgroundMap(encoder_frame_id, encoder_pts_us, encoder_width, encoder_height, config_.cell_size);
    }

    const uint64_t age = encoder_frame_id - latest_.source_frame.frame_id;
    if (age > static_cast<uint64_t>(config_.max_age_frames)) {
        return backgroundMap(encoder_frame_id, encoder_pts_us, encoder_width, encoder_height, config_.cell_size);
    }

    RoiMap selected = latest_;
    selected.source_frame.frame_id = latest_.source_frame.frame_id;
    selected.source_frame.pts_us = latest_.source_frame.pts_us;
    if (age > static_cast<uint64_t>(config_.hold_frames)) {
        const uint64_t steps = 1 + (age - config_.hold_frames - 1) / config_.hold_frames;
        for (size_t i = 0; i < selected.cells.size(); ++i) {
            for (uint64_t step = 0; step < steps; ++step) selected.cells[i] = weakerRoiLevel(selected.cells[i]);
        }
    }
    return selected;
}

}  // namespace roi_h265

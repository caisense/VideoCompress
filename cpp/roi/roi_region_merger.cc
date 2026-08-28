#include "roi/roi_region_merger.h"

#include <algorithm>
#include <map>

namespace roi_h265 {

RoiRegionMerger::RoiRegionMerger(const RoiConfig &config) : config_(config) {}

int RoiRegionMerger::deltaQp(RoiLevel level) const {
    switch (level) {
    case ROI_EDGE: return config_.edge_delta_qp;
    case ROI_CORE: return config_.core_delta_qp;
    case ROI_HALO: return config_.halo_delta_qp;
    default: return config_.background_delta_qp;
    }
}

std::vector<RoiRegion> RoiRegionMerger::merge(const RoiMap &map) const {
    struct OpenRegion {
        RoiRegion region;
        int last_row;
    };
    std::vector<RoiRegion> all;
    std::map<std::string, OpenRegion> open;

    for (int y = 0; y < map.grid_height; ++y) {
        std::map<std::string, OpenRegion> next_open;
        for (int x = 0; x < map.grid_width;) {
            const RoiLevel level = map.at(x, y);
            int end = x + 1;
            while (end < map.grid_width && map.at(end, y) == level) ++end;
            const int pixel_x = x * map.cell_size;
            const int pixel_width = std::min(map.frame_width, end * map.cell_size) - pixel_x;
            const std::string key = std::to_string(static_cast<int>(level)) + ":" +
                std::to_string(pixel_x) + ":" + std::to_string(pixel_width);
            std::map<std::string, OpenRegion>::iterator prior = open.find(key);
            if (prior != open.end() && prior->second.last_row == y - 1) {
                OpenRegion extended = prior->second;
                extended.region.height += std::min(map.cell_size, map.frame_height - y * map.cell_size);
                extended.last_row = y;
                next_open[key] = extended;
            } else {
                RoiRegion region;
                region.x = pixel_x;
                region.y = y * map.cell_size;
                region.width = pixel_width;
                region.height = std::min(map.cell_size, map.frame_height - region.y);
                region.level = level;
                region.delta_qp = deltaQp(level);
                region.force_intra = false;
                OpenRegion created;
                created.region = region;
                created.last_row = y;
                next_open[key] = created;
            }
            x = end;
        }
        for (std::map<std::string, OpenRegion>::const_iterator it = open.begin(); it != open.end(); ++it) {
            if (next_open.find(it->first) == next_open.end()) all.push_back(it->second.region);
        }
        open.swap(next_open);
    }
    for (std::map<std::string, OpenRegion>::const_iterator it = open.begin(); it != open.end(); ++it) {
        all.push_back(it->second.region);
    }

    std::stable_sort(all.begin(), all.end(), [](const RoiRegion &a, const RoiRegion &b) {
        const int priority_a = roiPriority(a.level);
        const int priority_b = roiPriority(b.level);
        if (priority_a != priority_b) return priority_a > priority_b;
        const int area_a = a.width * a.height;
        const int area_b = b.width * b.height;
        return area_a > area_b;
    });
    if (all.size() > static_cast<size_t>(config_.max_regions)) all.resize(config_.max_regions);
    return all;
}

}  // namespace roi_h265

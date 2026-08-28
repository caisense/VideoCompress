#include "roi/roi_debug.h"

#include <algorithm>
#include <fstream>
#include <vector>

namespace roi_h265 {

bool writeRoiMapPgm(const RoiMap &map, const std::string &path, std::string *error) {
    if (map.frame_width <= 0 || map.frame_height <= 0 || map.cells.empty()) {
        if (error) *error = "ROI map is empty";
        return false;
    }
    std::vector<uint8_t> image(static_cast<size_t>(map.frame_width) * map.frame_height, 0);
    for (int gy = 0; gy < map.grid_height; ++gy) {
        for (int gx = 0; gx < map.grid_width; ++gx) {
            const uint8_t intensity[] = {32, 96, 176, 255};
            const uint8_t value = intensity[static_cast<int>(map.at(gx, gy))];
            const int x_end = std::min(map.frame_width, (gx + 1) * map.cell_size);
            const int y_end = std::min(map.frame_height, (gy + 1) * map.cell_size);
            for (int y = gy * map.cell_size; y < y_end; ++y)
                for (int x = gx * map.cell_size; x < x_end; ++x)
                    image[static_cast<size_t>(y) * map.frame_width + x] = value;
        }
    }
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output) {
        if (error) *error = "cannot open debug ROI image";
        return false;
    }
    output << "P5\n" << map.frame_width << ' ' << map.frame_height << "\n255\n";
    output.write(reinterpret_cast<const char *>(image.data()), image.size());
    return static_cast<bool>(output);
}

}  // namespace roi_h265

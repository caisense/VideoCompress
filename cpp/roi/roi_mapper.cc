#include "roi/roi_mapper.h"

#include <algorithm>

namespace roi_h265 {
namespace {

typedef std::vector<uint8_t> BinaryMask;

BinaryMask resizeMask(const SegInstance &instance, int source_width, int source_height,
                      int encoder_width, int encoder_height) {
    BinaryMask mapped(static_cast<size_t>(encoder_width) * encoder_height, 0);
    if (instance.mask.empty() || instance.mask_width <= 0 || instance.mask_height <= 0 ||
        source_width <= 0 || source_height <= 0) return mapped;

    for (int y = 0; y < encoder_height; ++y) {
        const int source_y = std::min(source_height - 1,
            static_cast<int>((static_cast<int64_t>(2 * y + 1) * source_height) / (2 * encoder_height)));
        const int mask_y = std::min(instance.mask_height - 1,
            static_cast<int>((static_cast<int64_t>(2 * source_y + 1) * instance.mask_height) / (2 * source_height)));
        for (int x = 0; x < encoder_width; ++x) {
            const int source_x = std::min(source_width - 1,
                static_cast<int>((static_cast<int64_t>(2 * x + 1) * source_width) / (2 * encoder_width)));
            const int mask_x = std::min(instance.mask_width - 1,
                static_cast<int>((static_cast<int64_t>(2 * source_x + 1) * instance.mask_width) / (2 * source_width)));
            mapped[static_cast<size_t>(y) * encoder_width + x] =
                instance.mask[static_cast<size_t>(mask_y) * instance.mask_width + mask_x] ? 1 : 0;
        }
    }
    return mapped;
}

BinaryMask erode(const BinaryMask &input, int width, int height, int radius) {
    if (radius <= 0) return input;
    BinaryMask output(static_cast<size_t>(width) * height, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool kept = true;
            for (int dy = -radius; dy <= radius && kept; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= width || yy >= height ||
                        !input[static_cast<size_t>(yy) * width + xx]) {
                        kept = false;
                        break;
                    }
                }
            }
            output[static_cast<size_t>(y) * width + x] = kept ? 1 : 0;
        }
    }
    return output;
}

BinaryMask dilate(const BinaryMask &input, int width, int height, int radius) {
    if (radius <= 0) return input;
    BinaryMask output(static_cast<size_t>(width) * height, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool found = false;
            for (int dy = -radius; dy <= radius && !found; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < width && yy < height &&
                        input[static_cast<size_t>(yy) * width + xx]) {
                        found = true;
                        break;
                    }
                }
            }
            output[static_cast<size_t>(y) * width + x] = found ? 1 : 0;
        }
    }
    return output;
}

void promote(std::vector<RoiLevel> *pixels, const BinaryMask &mask, RoiLevel level) {
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] && roiPriority(level) > roiPriority((*pixels)[i])) (*pixels)[i] = level;
    }
}

RoiMap emptyMap(const SegResult &segmentation, int width, int height, int cell_size) {
    RoiMap map;
    map.source_frame = segmentation.frame;
    map.frame_width = width;
    map.frame_height = height;
    map.cell_size = cell_size;
    map.grid_width = (width + cell_size - 1) / cell_size;
    map.grid_height = (height + cell_size - 1) / cell_size;
    map.cells.assign(static_cast<size_t>(map.grid_width) * map.grid_height, ROI_BACKGROUND);
    return map;
}

void classifyCells(const std::vector<RoiLevel> &pixels, RoiMap *map, float threshold) {
    for (int cell_y = 0; cell_y < map->grid_height; ++cell_y) {
        for (int cell_x = 0; cell_x < map->grid_width; ++cell_x) {
            int counts[4] = {0, 0, 0, 0};
            const int x_begin = cell_x * map->cell_size;
            const int y_begin = cell_y * map->cell_size;
            const int x_end = std::min(map->frame_width, x_begin + map->cell_size);
            const int y_end = std::min(map->frame_height, y_begin + map->cell_size);
            const int area = (x_end - x_begin) * (y_end - y_begin);
            for (int y = y_begin; y < y_end; ++y) {
                for (int x = x_begin; x < x_end; ++x) {
                    ++counts[static_cast<int>(pixels[static_cast<size_t>(y) * map->frame_width + x])];
                }
            }
            RoiLevel chosen = ROI_BACKGROUND;
            for (int level = ROI_EDGE; level >= ROI_HALO; --level) {
                if (area > 0 && static_cast<float>(counts[level]) / area >= threshold) {
                    chosen = static_cast<RoiLevel>(level);
                    break;
                }
            }
            map->set(cell_x, cell_y, chosen);
        }
    }
}

}  // namespace

RoiMapper::RoiMapper(const RoiConfig &config) : config_(config) {}

RoiMap RoiMapper::build(const SegResult &segmentation, int encoder_width, int encoder_height) const {
    RoiMap map = emptyMap(segmentation, encoder_width, encoder_height, config_.cell_size);
    if (encoder_width <= 0 || encoder_height <= 0 || segmentation.source_width <= 0 ||
        segmentation.source_height <= 0) return map;

    std::vector<RoiLevel> pixel_levels(static_cast<size_t>(encoder_width) * encoder_height, ROI_BACKGROUND);
    for (size_t i = 0; i < segmentation.instances.size(); ++i) {
        const BinaryMask object = resizeMask(segmentation.instances[i], segmentation.source_width,
                                             segmentation.source_height, encoder_width, encoder_height);
        const BinaryMask core = erode(object, encoder_width, encoder_height, config_.erosion_radius);
        const BinaryMask expanded = dilate(object, encoder_width, encoder_height, config_.dilation_radius);
        BinaryMask edge(object.size(), 0);
        BinaryMask halo(object.size(), 0);
        for (size_t pixel = 0; pixel < object.size(); ++pixel) {
            edge[pixel] = object[pixel] && !core[pixel] ? 1 : 0;
            halo[pixel] = expanded[pixel] && !object[pixel] ? 1 : 0;
        }
        // Promote in increasing priority order. An object's edge also wins over
        // another object's halo, which is intentional for overlapping instances.
        promote(&pixel_levels, halo, ROI_HALO);
        promote(&pixel_levels, core, ROI_CORE);
        promote(&pixel_levels, edge, ROI_EDGE);
    }
    classifyCells(pixel_levels, &map, config_.occupancy_threshold);
    return map;
}

RoiMap RoiMapper::buildBboxMap(const SegResult &segmentation, int encoder_width, int encoder_height) const {
    RoiMap map = emptyMap(segmentation, encoder_width, encoder_height, config_.cell_size);
    if (segmentation.source_width <= 0 || segmentation.source_height <= 0) return map;
    for (size_t i = 0; i < segmentation.instances.size(); ++i) {
        const BBox &box = segmentation.instances[i].bbox;
        const int x0 = std::max(0, box.left * encoder_width / segmentation.source_width);
        const int y0 = std::max(0, box.top * encoder_height / segmentation.source_height);
        const int x1 = std::min(encoder_width, (box.right * encoder_width + segmentation.source_width - 1) /
            segmentation.source_width);
        const int y1 = std::min(encoder_height, (box.bottom * encoder_height + segmentation.source_height - 1) /
            segmentation.source_height);
        for (int gy = 0; gy < map.grid_height; ++gy) {
            for (int gx = 0; gx < map.grid_width; ++gx) {
                const int cell_x0 = gx * map.cell_size;
                const int cell_y0 = gy * map.cell_size;
                const int cell_x1 = std::min(encoder_width, cell_x0 + map.cell_size);
                const int cell_y1 = std::min(encoder_height, cell_y0 + map.cell_size);
                if (cell_x0 < x1 && cell_x1 > x0 && cell_y0 < y1 && cell_y1 > y0)
                    map.set(gx, gy, ROI_CORE);
            }
        }
    }
    return map;
}

}  // namespace roi_h265

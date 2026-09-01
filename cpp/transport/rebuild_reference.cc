#include "transport/rebuild_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace roi_h265 {
namespace {

BBox clampBox(const BBox &box, int source_width, int source_height) {
    BBox result;
    result.left = std::max(0, std::min(source_width, box.left));
    result.top = std::max(0, std::min(source_height, box.top));
    result.right = std::max(0, std::min(source_width, box.right));
    result.bottom = std::max(0, std::min(source_height, box.bottom));
    return result;
}

bool hasArea(const BBox &box) {
    return box.right > box.left && box.bottom > box.top;
}

int width(const BBox &box) { return std::max(0, box.right - box.left); }
int height(const BBox &box) { return std::max(0, box.bottom - box.top); }

BBox intersection(const BBox &left, const BBox &right) {
    BBox result;
    result.left = std::max(left.left, right.left);
    result.top = std::max(left.top, right.top);
    result.right = std::min(left.right, right.right);
    result.bottom = std::min(left.bottom, right.bottom);
    return result;
}

BBox fixedHeadProxy(const BBox &parent, const RebuildConfig &config) {
    BBox proxy;
    const int parent_width = width(parent);
    const int parent_height = height(parent);
    const int proxy_width = std::max(1, static_cast<int>(std::round(
        parent_width * config.head_width_ratio)));
    const int proxy_height = std::max(1, static_cast<int>(std::round(
        parent_height * config.head_height_ratio)));
    const int center_x = (parent.left + parent.right) / 2;
    proxy.left = center_x - proxy_width / 2;
    proxy.top = parent.top;
    proxy.right = proxy.left + proxy_width;
    proxy.bottom = proxy.top + proxy_height;
    return proxy;
}

bool maskUsable(const SegInstance &instance) {
    return instance.mask_width > 0 && instance.mask_height > 0 &&
        instance.mask.size() == static_cast<size_t>(instance.mask_width) *
            static_cast<size_t>(instance.mask_height);
}

bool maskAtSource(const SegInstance &instance, int source_x, int source_y,
                  int source_width, int source_height) {
    if (!maskUsable(instance) || source_width <= 0 || source_height <= 0) {
        return false;
    }
    const int mask_x = std::min(instance.mask_width - 1, std::max(0,
        static_cast<int>((static_cast<int64_t>(2 * source_x + 1) *
                          instance.mask_width) / (2 * source_width))));
    const int mask_y = std::min(instance.mask_height - 1, std::max(0,
        static_cast<int>((static_cast<int64_t>(2 * source_y + 1) *
                          instance.mask_height) / (2 * source_height))));
    return instance.mask[static_cast<size_t>(mask_y) * instance.mask_width +
                         mask_x] != 0;
}

struct ForegroundBounds {
    BBox box;
    int foreground_pixels;
    int scanned_pixels;

    ForegroundBounds() : foreground_pixels(0), scanned_pixels(0) {}
};

ForegroundBounds upperForeground(const SegInstance &instance,
                                 const BBox &parent,
                                 int source_width,
                                 int source_height) {
    ForegroundBounds result;
    result.box.left = parent.right;
    result.box.top = parent.bottom;
    result.box.right = parent.left;
    result.box.bottom = parent.top;
    const int upper_bottom = std::min(parent.bottom,
        parent.top + std::max(1, static_cast<int>(std::ceil(height(parent) * 0.55))));
    for (int y = parent.top; y < upper_bottom; ++y) {
        for (int x = parent.left; x < parent.right; ++x) {
            ++result.scanned_pixels;
            if (!maskAtSource(instance, x, y, source_width, source_height)) continue;
            ++result.foreground_pixels;
            result.box.left = std::min(result.box.left, x);
            result.box.top = std::min(result.box.top, y);
            result.box.right = std::max(result.box.right, x + 1);
            result.box.bottom = std::max(result.box.bottom, y + 1);
        }
    }
    return result;
}

float areaRatio(const BBox &visible, const BBox &wanted) {
    const int wanted_width = width(wanted);
    const int wanted_height = height(wanted);
    if (wanted_width <= 0 || wanted_height <= 0) return 0.0f;
    return static_cast<float>(width(visible) * height(visible)) /
        static_cast<float>(wanted_width * wanted_height);
}

ReferenceCropSelection invalidSelection(RebuildReferenceKind kind,
                                        const BBox &parent,
                                        const std::string &reason) {
    ReferenceCropSelection result;
    result.kind = kind;
    result.parent_bbox = parent;
    result.fallback_reason = reason;
    result.selector_reason = reason;
    result.valid = false;
    return result;
}

}  // namespace

ReferenceCropSelection::ReferenceCropSelection()
    : kind(REFERENCE_KIND_FULL), valid(false), used_mask(false),
      used_fallback_proxy(false), foreground_ratio(0.0f), visible_ratio(0.0f) {}

const char *rebuildReferenceKindName(RebuildReferenceKind kind) {
    return kind == REFERENCE_KIND_HEAD_PRIORITY ? "HEAD" : "FULL";
}

ReferenceCropSelection selectFullReferenceCrop(const BBox &bbox,
                                               int source_width,
                                               int source_height,
                                               int margin_percent) {
    const BBox parent = clampBox(bbox, source_width, source_height);
    if (source_width <= 0 || source_height <= 0 || !hasArea(parent)) {
        return invalidSelection(REFERENCE_KIND_FULL, parent, "INVALID");
    }
    const int box_width = std::max(1, bbox.right - bbox.left);
    const int box_height = std::max(1, bbox.bottom - bbox.top);
    const int margin_x = (box_width * std::max(0, margin_percent) + 99) / 100;
    const int margin_y = (box_height * std::max(0, margin_percent) + 99) / 100;
    BBox wanted;
    wanted.left = bbox.left - margin_x;
    wanted.top = bbox.top - margin_y;
    wanted.right = bbox.right + margin_x;
    wanted.bottom = bbox.bottom + margin_y;
    const BBox crop = clampBox(wanted, source_width, source_height);
    if (!hasArea(crop)) return invalidSelection(REFERENCE_KIND_FULL, parent, "INVALID");

    ReferenceCropSelection result;
    result.kind = REFERENCE_KIND_FULL;
    result.crop = crop;
    result.parent_bbox = parent;
    result.focus = intersection(fixedHeadProxy(parent, RebuildConfig()), crop);
    result.valid = hasArea(result.focus) && hasArea(result.crop);
    result.visible_ratio = areaRatio(crop, wanted);
    return result;
}

ReferenceCropSelection selectHeadPriorityCrop(const SegInstance &instance,
                                              int source_width,
                                              int source_height,
                                              const RebuildConfig &config) {
    const ReferenceCropSelection full = selectFullReferenceCrop(
        instance.bbox, source_width, source_height, config.crop_margin_percent);
    if (!full.valid) return invalidSelection(REFERENCE_KIND_FULL,
                                             full.parent_bbox, "INVALID");
    const BBox parent = full.parent_bbox;
    const int min_size = std::max(1, config.head_min_size);
    const ForegroundBounds foreground = upperForeground(
        instance, parent, source_width, source_height);
    const bool foreground_found = foreground.foreground_pixels > 0 &&
        hasArea(foreground.box);
    std::string selector_reason;
    BBox wanted;
    BBox focus;
    if (foreground_found) {
        const int margin_x = std::max(1, (width(foreground.box) *
            std::max(0, config.head_margin_percent) + 99) / 100);
        const int margin_y = std::max(1, (height(foreground.box) *
            std::max(0, config.head_margin_percent) + 99) / 100);
        focus = foreground.box;
        wanted.left = foreground.box.left - margin_x;
        wanted.top = foreground.box.top - margin_y;
        wanted.right = foreground.box.right + margin_x;
        wanted.bottom = std::min(parent.bottom,
            parent.top + std::max(1, static_cast<int>(std::round(
                height(parent) * config.head_height_ratio))));
        selector_reason = "MASK";
    } else {
        wanted = fixedHeadProxy(parent, config);
        focus = wanted;
        selector_reason = "MASK";
    }

    // Keep the selected patch inside the detector parent.  This prevents a
    // margin from becoming unrelated background at image edges.
    wanted.left = std::max(parent.left, wanted.left);
    wanted.top = std::max(parent.top, wanted.top);
    wanted.right = std::min(parent.right, wanted.right);
    wanted.bottom = std::min(parent.bottom, wanted.bottom);
    const BBox candidate = clampBox(wanted, source_width, source_height);
    const float candidate_visible_ratio = areaRatio(candidate, wanted);
    if (hasArea(candidate) && width(candidate) >= min_size &&
        height(candidate) >= min_size && candidate_visible_ratio >= 0.80f) {
        ReferenceCropSelection result;
        result.kind = REFERENCE_KIND_HEAD_PRIORITY;
        result.crop = candidate;
        result.parent_bbox = parent;
        result.focus = intersection(clampBox(focus, source_width, source_height), candidate);
        if (!hasArea(result.focus)) result.focus = candidate;
        result.valid = hasArea(result.focus);
        result.used_mask = foreground_found;
        result.used_fallback_proxy = !foreground_found;
        result.selector_reason = selector_reason;
        result.foreground_ratio = foreground.scanned_pixels > 0
            ? static_cast<float>(foreground.foreground_pixels) /
                static_cast<float>(foreground.scanned_pixels) : 0.0f;
        result.visible_ratio = candidate_visible_ratio;
        return result;
    }

    const std::string failure = !hasArea(candidate) || candidate_visible_ratio < 0.80f
        ? "EDGE" : "SMALL";
    const BBox proxy_wanted = fixedHeadProxy(parent, config);
    const BBox proxy = clampBox(proxy_wanted, source_width, source_height);
    const float proxy_visible_ratio = areaRatio(proxy, proxy_wanted);
    if (hasArea(proxy) && width(proxy) >= min_size && height(proxy) >= min_size &&
        proxy_visible_ratio >= 0.80f) {
        ReferenceCropSelection result;
        result.kind = REFERENCE_KIND_HEAD_PRIORITY;
        result.crop = proxy;
        result.parent_bbox = parent;
        result.focus = proxy;
        result.valid = true;
        result.used_fallback_proxy = true;
        result.selector_reason = failure;
        result.foreground_ratio = foreground.scanned_pixels > 0
            ? static_cast<float>(foreground.foreground_pixels) /
                static_cast<float>(foreground.scanned_pixels) : 0.0f;
        result.visible_ratio = proxy_visible_ratio;
        return result;
    }

    ReferenceCropSelection result = full;
    result.kind = REFERENCE_KIND_FULL;
    result.focus = intersection(full.focus, full.crop);
    result.fallback_reason = width(parent) < min_size || height(parent) < min_size
        ? "SMALL" : failure;
    result.selector_reason = result.fallback_reason;
    result.foreground_ratio = foreground.scanned_pixels > 0
        ? static_cast<float>(foreground.foreground_pixels) /
            static_cast<float>(foreground.scanned_pixels) : 0.0f;
    result.visible_ratio = full.visible_ratio;
    return result;
}

ReferenceCropSelection selectReferenceCrop(const SegInstance &instance,
                                           int source_width,
                                           int source_height,
                                           const RebuildConfig &config) {
    if (instance.class_id != config.head_class_id ||
        config.reference_mode == REBUILD_REFERENCE_MODE_PERSON) {
        return selectFullReferenceCrop(instance.bbox, source_width, source_height,
                                       config.crop_margin_percent);
    }
    return selectHeadPriorityCrop(instance, source_width, source_height, config);
}

}  // namespace roi_h265

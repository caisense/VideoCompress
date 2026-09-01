#ifndef ROI_H265_TRANSPORT_REBUILD_REFERENCE_H_
#define ROI_H265_TRANSPORT_REBUILD_REFERENCE_H_

#include <string>

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

enum RebuildReferenceKind {
    REFERENCE_KIND_FULL = 0,
    REFERENCE_KIND_HEAD_PRIORITY = 1,
};

// A selector result is source-frame geometry.  ``crop`` is the rectangle sent
// in RB/1; ``parent_bbox`` is the detector rectangle used by the receiver for
// registration.  ``focus`` is the comparable head region used for density
// telemetry in both FULL and HEAD modes.
struct ReferenceCropSelection {
    RebuildReferenceKind kind;
    BBox crop;
    BBox parent_bbox;
    BBox focus;
    bool valid;
    bool used_mask;
    bool used_fallback_proxy;
    std::string fallback_reason;
    std::string selector_reason;
    float foreground_ratio;
    float visible_ratio;

    ReferenceCropSelection();
};

const char *rebuildReferenceKindName(RebuildReferenceKind kind);

ReferenceCropSelection selectFullReferenceCrop(const BBox &bbox,
                                               int source_width,
                                               int source_height,
                                               int margin_percent);

ReferenceCropSelection selectHeadPriorityCrop(const SegInstance &instance,
                                              int source_width,
                                              int source_height,
                                              const RebuildConfig &config);

ReferenceCropSelection selectReferenceCrop(const SegInstance &instance,
                                           int source_width,
                                           int source_height,
                                           const RebuildConfig &config);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_REBUILD_REFERENCE_H_

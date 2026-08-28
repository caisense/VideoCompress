#ifndef ROI_H265_ROI_DEBUG_H_
#define ROI_H265_ROI_DEBUG_H_

#include <string>

#include "common/frame_meta.h"

namespace roi_h265 {

// Writes an 8-bit PGM where each visible 16x16 cell shows its final ROI level.
bool writeRoiMapPgm(const RoiMap &map, const std::string &path, std::string *error);

}  // namespace roi_h265

#endif  // ROI_H265_ROI_DEBUG_H_

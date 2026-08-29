#ifndef ROI_H265_TRANSPORT_DETECTION_EVENT_PROTOCOL_H_
#define ROI_H265_TRANSPORT_DETECTION_EVENT_PROTOCOL_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string>
#include <vector>

#include "common/frame_meta.h"

namespace roi_h265 {

// ROEV/1 is a small, independent companion datagram.  It deliberately does
// not alter RFC 7798 H.265 RTP, so standard video receivers remain unchanged.
const size_t kDetectionEventClassCount = 4;
const size_t kDetectionEventHeaderBytes = 44;
const uint16_t kDetectionEventClassMask = 0x000f;

enum DetectionEventType {
    DETECTION_EVENT_STATE = 1,
};

enum DetectionEventFlags {
    // A periodic repeat of the present state.  It recovers a lost UDP edge
    // event without continuously reporting every inference result.
    DETECTION_EVENT_FLAG_HEARTBEAT = 1 << 0,
};

// The four index positions are stable on the wire: person, car, boat,
// airplane.  They match the relevant-class mask used by snapshot/rebuild.
struct DetectionEventPacket {
    uint8_t type;
    uint16_t flags;
    uint32_t sequence;
    uint64_t frame_id;
    uint32_t pts_ms;
    uint16_t present_mask;
    uint16_t entered_mask;
    uint16_t exited_mask;
    std::array<uint8_t, kDetectionEventClassCount> counts;
    std::array<uint8_t, kDetectionEventClassCount> max_confidence_percent;
    uint32_t crc32;

    DetectionEventPacket();
};

struct DetectionEventSummary {
    uint64_t frame_id;
    uint32_t pts_ms;
    uint16_t present_mask;
    std::array<uint8_t, kDetectionEventClassCount> counts;
    std::array<uint8_t, kDetectionEventClassCount> max_confidence_percent;

    DetectionEventSummary();
};

// Returns 0..3 for person/car/boat/airplane COCO classes 0/2/8/4, or -1.
int detectionEventClassIndex(int class_id);
const char *detectionEventClassName(size_t index);

DetectionEventSummary summarizeDetectionEvent(const SegResult &result,
                                              float min_confidence);

uint32_t detectionEventCrc32(const uint8_t *bytes, size_t length);
bool serializeDetectionEventPacket(const DetectionEventPacket &packet,
                                   std::vector<uint8_t> *bytes,
                                   std::string *error);
bool parseDetectionEventPacket(const uint8_t *bytes, size_t length,
                               DetectionEventPacket *packet,
                               std::string *error);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_DETECTION_EVENT_PROTOCOL_H_

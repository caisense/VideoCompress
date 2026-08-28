#ifndef ROI_H265_TRANSPORT_SNAPSHOT_PROTOCOL_H_
#define ROI_H265_TRANSPORT_SNAPSHOT_PROTOCOL_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "common/frame_meta.h"

namespace roi_h265 {

// A small, reliable, stop-and-wait protocol for high-resolution JPEG evidence
// over the same constrained UDP link as video.  It is deliberately not RTP:
// snapshots need resumable byte offsets, whereas H.265 remains standard RTP.
enum SnapshotPacketType {
    SNAPSHOT_START = 1,
    SNAPSHOT_DATA = 2,
    SNAPSHOT_END = 3,
    SNAPSHOT_RESUME = 4,
    SNAPSHOT_ACK = 5,
    SNAPSHOT_COMPLETE = 6,
    SNAPSHOT_ABORT = 7,
};

const size_t kSnapshotHeaderBytes = 32;
const uint8_t kSnapshotProtocolVersion = 1;

struct SnapshotPacket {
    uint8_t type;
    uint16_t flags;
    uint32_t transfer_id;
    uint32_t total_bytes;
    uint32_t offset;
    uint16_t payload_length;
    uint16_t width;
    uint16_t height;
    // bit 0 person, bit 1 car, bit 2 boat, bit 3 airplane.
    uint16_t class_mask;
    uint32_t crc32;
    std::vector<uint8_t> payload;

    SnapshotPacket();
};

bool serializeSnapshotPacket(const SnapshotPacket &packet, std::vector<uint8_t> *bytes,
                             std::string *error);
bool parseSnapshotPacket(const uint8_t *bytes, size_t length, SnapshotPacket *packet,
                         std::string *error);
uint32_t snapshotCrc32(const uint8_t *bytes, size_t length);

// COCO class IDs used by the current YOLOv8-Seg model.  Filtering immediately
// after inference keeps ROI, preview, and snapshot triggering confined to the
// four scenes requested by the low-bandwidth workflow.
bool isRelevantDetectionClass(int class_id);
uint16_t relevantDetectionClassBit(int class_id);
uint16_t relevantDetectionClassMask(const SegResult &result, float min_confidence);
void filterRelevantDetections(SegResult *result, float min_confidence);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_SNAPSHOT_PROTOCOL_H_

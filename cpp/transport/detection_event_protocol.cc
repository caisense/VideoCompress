#include "transport/detection_event_protocol.h"

#include <cstring>

namespace roi_h265 {
namespace {

void writeU16(uint8_t *bytes, uint16_t value) {
    bytes[0] = static_cast<uint8_t>((value >> 8) & 0xffU);
    bytes[1] = static_cast<uint8_t>(value & 0xffU);
}

void writeU32(uint8_t *bytes, uint32_t value) {
    bytes[0] = static_cast<uint8_t>((value >> 24) & 0xffU);
    bytes[1] = static_cast<uint8_t>((value >> 16) & 0xffU);
    bytes[2] = static_cast<uint8_t>((value >> 8) & 0xffU);
    bytes[3] = static_cast<uint8_t>(value & 0xffU);
}

void writeU64(uint8_t *bytes, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        bytes[index] = static_cast<uint8_t>((value >> ((7 - index) * 8)) & 0xffU);
    }
}

uint16_t readU16(const uint8_t *bytes) {
    return static_cast<uint16_t>((static_cast<uint16_t>(bytes[0]) << 8) | bytes[1]);
}

uint32_t readU32(const uint8_t *bytes) {
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

uint64_t readU64(const uint8_t *bytes) {
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index) {
        value = (value << 8) | static_cast<uint64_t>(bytes[index]);
    }
    return value;
}

bool validMask(uint16_t mask) {
    return (mask & static_cast<uint16_t>(~kDetectionEventClassMask)) == 0;
}

bool validatePacket(const DetectionEventPacket &packet, std::string *error) {
    if (packet.type != DETECTION_EVENT_STATE ||
        (packet.flags & static_cast<uint16_t>(~DETECTION_EVENT_FLAG_HEARTBEAT)) != 0 ||
        !validMask(packet.present_mask) || !validMask(packet.entered_mask) ||
        !validMask(packet.exited_mask) ||
        (packet.entered_mask & static_cast<uint16_t>(~packet.present_mask)) != 0 ||
        (packet.exited_mask & packet.present_mask) != 0 ||
        (packet.entered_mask & packet.exited_mask) != 0 ||
        ((packet.flags & DETECTION_EVENT_FLAG_HEARTBEAT) != 0 &&
         (packet.entered_mask != 0 || packet.exited_mask != 0))) {
        if (error) *error = "invalid ROEV type, flags, or class masks";
        return false;
    }
    for (size_t index = 0; index < kDetectionEventClassCount; ++index) {
        const uint16_t bit = static_cast<uint16_t>(1U << index);
        if ((packet.present_mask & bit) == 0 &&
            (packet.counts[index] != 0 || packet.max_confidence_percent[index] != 0)) {
            if (error) *error = "ROEV count/confidence supplied for an absent class";
            return false;
        }
        if (packet.max_confidence_percent[index] > 100) {
            if (error) *error = "ROEV confidence percent exceeds 100";
            return false;
        }
    }
    return true;
}

uint8_t confidencePercent(float confidence) {
    if (confidence <= 0.0f) return 0;
    if (confidence >= 1.0f) return 100;
    return static_cast<uint8_t>(confidence * 100.0f + 0.5f);
}

}  // namespace

DetectionEventPacket::DetectionEventPacket()
    : type(DETECTION_EVENT_STATE), flags(0), sequence(0), frame_id(0), pts_ms(0),
      present_mask(0), entered_mask(0), exited_mask(0), counts(),
      max_confidence_percent(), crc32(0) {
    counts.fill(0);
    max_confidence_percent.fill(0);
}

DetectionEventSummary::DetectionEventSummary()
    : frame_id(0), pts_ms(0), present_mask(0), counts(), max_confidence_percent() {
    counts.fill(0);
    max_confidence_percent.fill(0);
}

int detectionEventClassIndex(int class_id) {
    switch (class_id) {
    case 0: return 0;  // person
    case 2: return 1;  // car
    case 8: return 2;  // boat
    case 4: return 3;  // airplane
    default: return -1;
    }
}

const char *detectionEventClassName(size_t index) {
    static const char *const kNames[kDetectionEventClassCount] = {
        "person", "car", "boat", "airplane",
    };
    return index < kDetectionEventClassCount ? kNames[index] : "unknown";
}

DetectionEventSummary summarizeDetectionEvent(const SegResult &result,
                                              float min_confidence) {
    DetectionEventSummary summary;
    summary.frame_id = result.frame.frame_id;
    summary.pts_ms = static_cast<uint32_t>((result.frame.pts_us / 1000ULL) & 0xffffffffULL);
    for (size_t index = 0; index < result.instances.size(); ++index) {
        const SegInstance &instance = result.instances[index];
        const int class_index = detectionEventClassIndex(instance.class_id);
        if (class_index < 0 || instance.confidence < min_confidence) continue;
        const size_t destination = static_cast<size_t>(class_index);
        summary.present_mask = static_cast<uint16_t>(summary.present_mask | (1U << destination));
        if (summary.counts[destination] < 255U) ++summary.counts[destination];
        const uint8_t confidence = confidencePercent(instance.confidence);
        if (confidence > summary.max_confidence_percent[destination]) {
            summary.max_confidence_percent[destination] = confidence;
        }
    }
    return summary;
}

uint32_t detectionEventCrc32(const uint8_t *bytes, size_t length) {
    uint32_t crc = 0xffffffffU;
    for (size_t index = 0; index < length; ++index) {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-
                static_cast<int32_t>(crc & 1U));
            crc = (crc >> 1) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

bool serializeDetectionEventPacket(const DetectionEventPacket &packet,
                                   std::vector<uint8_t> *bytes,
                                   std::string *error) {
    if (!bytes || !validatePacket(packet, error)) return false;
    bytes->assign(kDetectionEventHeaderBytes, 0);
    uint8_t *wire = bytes->data();
    wire[0] = 'R'; wire[1] = 'O'; wire[2] = 'E'; wire[3] = 'V';
    wire[4] = 1;
    wire[5] = packet.type;
    writeU16(wire + 6, packet.flags);
    writeU32(wire + 8, packet.sequence);
    writeU64(wire + 12, packet.frame_id);
    writeU32(wire + 20, packet.pts_ms);
    writeU16(wire + 24, packet.present_mask);
    writeU16(wire + 26, packet.entered_mask);
    writeU16(wire + 28, packet.exited_mask);
    // 30..31 is reserved and remains zero in ROEV/1.
    for (size_t index = 0; index < kDetectionEventClassCount; ++index) {
        wire[32 + index] = packet.counts[index];
        wire[36 + index] = packet.max_confidence_percent[index];
    }
    writeU32(wire + 40, detectionEventCrc32(wire, 40));
    return true;
}

bool parseDetectionEventPacket(const uint8_t *bytes, size_t length,
                               DetectionEventPacket *packet,
                               std::string *error) {
    if (!bytes || !packet || length != kDetectionEventHeaderBytes) {
        if (error) *error = "ROEV datagram must be exactly 44 bytes";
        return false;
    }
    if (bytes[0] != 'R' || bytes[1] != 'O' || bytes[2] != 'E' || bytes[3] != 'V' ||
        bytes[4] != 1 || readU16(bytes + 30) != 0) {
        if (error) *error = "invalid ROEV magic, version, or reserved field";
        return false;
    }
    const uint32_t expected_crc = readU32(bytes + 40);
    if (detectionEventCrc32(bytes, 40) != expected_crc) {
        if (error) *error = "ROEV CRC32 mismatch";
        return false;
    }
    DetectionEventPacket parsed;
    parsed.type = bytes[5];
    parsed.flags = readU16(bytes + 6);
    parsed.sequence = readU32(bytes + 8);
    parsed.frame_id = readU64(bytes + 12);
    parsed.pts_ms = readU32(bytes + 20);
    parsed.present_mask = readU16(bytes + 24);
    parsed.entered_mask = readU16(bytes + 26);
    parsed.exited_mask = readU16(bytes + 28);
    for (size_t index = 0; index < kDetectionEventClassCount; ++index) {
        parsed.counts[index] = bytes[32 + index];
        parsed.max_confidence_percent[index] = bytes[36 + index];
    }
    parsed.crc32 = expected_crc;
    if (!validatePacket(parsed, error)) return false;
    *packet = parsed;
    return true;
}

}  // namespace roi_h265

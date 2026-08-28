#include "transport/snapshot_protocol.h"

#include <algorithm>
#include <cstring>

namespace roi_h265 {
namespace {

const uint8_t kMagic[4] = {'R', 'S', 'N', 'P'};

void writeU16(uint8_t *destination, uint16_t value) {
    destination[0] = static_cast<uint8_t>(value >> 8);
    destination[1] = static_cast<uint8_t>(value & 0xffU);
}

void writeU32(uint8_t *destination, uint32_t value) {
    destination[0] = static_cast<uint8_t>(value >> 24);
    destination[1] = static_cast<uint8_t>((value >> 16) & 0xffU);
    destination[2] = static_cast<uint8_t>((value >> 8) & 0xffU);
    destination[3] = static_cast<uint8_t>(value & 0xffU);
}

uint16_t readU16(const uint8_t *source) {
    return static_cast<uint16_t>((static_cast<uint16_t>(source[0]) << 8) | source[1]);
}

uint32_t readU32(const uint8_t *source) {
    return (static_cast<uint32_t>(source[0]) << 24) |
        (static_cast<uint32_t>(source[1]) << 16) |
        (static_cast<uint32_t>(source[2]) << 8) |
        static_cast<uint32_t>(source[3]);
}

bool validType(uint8_t type) {
    return type >= SNAPSHOT_START && type <= SNAPSHOT_ABORT;
}

}  // namespace

SnapshotPacket::SnapshotPacket()
    : type(SNAPSHOT_START), flags(0), transfer_id(0), total_bytes(0), offset(0),
      payload_length(0), width(0), height(0), class_mask(0), crc32(0) {}

bool serializeSnapshotPacket(const SnapshotPacket &packet, std::vector<uint8_t> *bytes,
                             std::string *error) {
    if (!bytes) {
        if (error) *error = "missing snapshot packet output";
        return false;
    }
    if (!validType(packet.type)) {
        if (error) *error = "invalid snapshot packet type";
        return false;
    }
    if (packet.payload.size() > 65535U || packet.payload.size() != packet.payload_length ||
        packet.offset > packet.total_bytes ||
        packet.payload.size() > static_cast<size_t>(packet.total_bytes - packet.offset)) {
        if (error) *error = "invalid snapshot packet length or offset";
        return false;
    }
    bytes->assign(kSnapshotHeaderBytes + packet.payload.size(), 0);
    uint8_t *header = bytes->data();
    std::memcpy(header, kMagic, sizeof(kMagic));
    header[4] = kSnapshotProtocolVersion;
    header[5] = packet.type;
    writeU16(header + 6, packet.flags);
    writeU32(header + 8, packet.transfer_id);
    writeU32(header + 12, packet.total_bytes);
    writeU32(header + 16, packet.offset);
    writeU16(header + 20, packet.payload_length);
    writeU16(header + 22, packet.width);
    writeU16(header + 24, packet.height);
    writeU16(header + 26, packet.class_mask);
    writeU32(header + 28, packet.crc32);
    if (!packet.payload.empty()) {
        std::memcpy(bytes->data() + kSnapshotHeaderBytes, packet.payload.data(),
                    packet.payload.size());
    }
    return true;
}

bool parseSnapshotPacket(const uint8_t *bytes, size_t length, SnapshotPacket *packet,
                         std::string *error) {
    if (!bytes || !packet || length < kSnapshotHeaderBytes) {
        if (error) *error = "snapshot datagram is shorter than its header";
        return false;
    }
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0 ||
        bytes[4] != kSnapshotProtocolVersion || !validType(bytes[5])) {
        if (error) *error = "invalid snapshot protocol magic, version, or type";
        return false;
    }
    SnapshotPacket parsed;
    parsed.type = bytes[5];
    parsed.flags = readU16(bytes + 6);
    parsed.transfer_id = readU32(bytes + 8);
    parsed.total_bytes = readU32(bytes + 12);
    parsed.offset = readU32(bytes + 16);
    parsed.payload_length = readU16(bytes + 20);
    parsed.width = readU16(bytes + 22);
    parsed.height = readU16(bytes + 24);
    parsed.class_mask = readU16(bytes + 26);
    parsed.crc32 = readU32(bytes + 28);
    if (length != kSnapshotHeaderBytes + parsed.payload_length ||
        parsed.offset > parsed.total_bytes ||
        parsed.payload_length > parsed.total_bytes - parsed.offset) {
        if (error) *error = "invalid snapshot datagram payload length or offset";
        return false;
    }
    parsed.payload.assign(bytes + kSnapshotHeaderBytes, bytes + length);
    *packet = parsed;
    return true;
}

uint32_t snapshotCrc32(const uint8_t *bytes, size_t length) {
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

bool isRelevantDetectionClass(int class_id) {
    return class_id == 0 || class_id == 2 || class_id == 4 || class_id == 8;
}

uint16_t relevantDetectionClassBit(int class_id) {
    switch (class_id) {
    case 0: return 1U << 0;  // person
    case 2: return 1U << 1;  // car
    case 8: return 1U << 2;  // boat
    case 4: return 1U << 3;  // airplane
    default: return 0;
    }
}

uint16_t relevantDetectionClassMask(const SegResult &result, float min_confidence) {
    uint16_t mask = 0;
    for (size_t index = 0; index < result.instances.size(); ++index) {
        const SegInstance &instance = result.instances[index];
        if (instance.confidence >= min_confidence) {
            mask = static_cast<uint16_t>(mask | relevantDetectionClassBit(instance.class_id));
        }
    }
    return mask;
}

void filterRelevantDetections(SegResult *result, float min_confidence) {
    if (!result) return;
    std::vector<SegInstance> filtered;
    filtered.reserve(result->instances.size());
    for (size_t index = 0; index < result->instances.size(); ++index) {
        const SegInstance &instance = result->instances[index];
        if (instance.confidence >= min_confidence && isRelevantDetectionClass(instance.class_id)) {
            filtered.push_back(instance);
        }
    }
    result->instances.swap(filtered);
}

}  // namespace roi_h265

#include "transport/rebuild_protocol.h"

#include <algorithm>
#include <cstring>

namespace roi_h265 {
namespace {

const uint8_t kMagic[2] = {'R', 'B'};

void writeU16(uint8_t *destination, uint16_t value) {
    destination[0] = static_cast<uint8_t>(value >> 8);
    destination[1] = static_cast<uint8_t>(value);
}

void writeU32(uint8_t *destination, uint32_t value) {
    destination[0] = static_cast<uint8_t>(value >> 24);
    destination[1] = static_cast<uint8_t>(value >> 16);
    destination[2] = static_cast<uint8_t>(value >> 8);
    destination[3] = static_cast<uint8_t>(value);
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
    return type >= REBUILD_STATE && type <= REBUILD_HEARTBEAT;
}

}  // namespace

RebuildPacket::RebuildPacket()
    : type(REBUILD_STATE), profile(3), generation(0), flags(0), sequence(0),
      frame_id(0), pts_ms(0) {}

RebuildTargetState::RebuildTargetState()
    : track_id(0), class_id(0), confidence_percent(0), left(0), top(0), right(0),
      bottom(0), reference_generation(0), flags(0) {}

RebuildState::RebuildState()
    : source_width(0), source_height(0), output_width(0), output_height(0),
      source_fps(0), output_fps(0), flags(0) {}

RebuildPatchFragment::RebuildPatchFragment()
    : transfer_id(0), track_id(0), reference_generation(0), left(0), top(0),
      right(0), bottom(0), reference_left(0), reference_top(0),
      reference_right(0), reference_bottom(0), jpeg_width(0), jpeg_height(0),
      mask_width(0), mask_height(0), fragment_index(0), data_fragments(0),
      blob_size(0), mask_rle_bytes(0), chunk_bytes(0) {}

uint32_t rebuildCrc32(const uint8_t *bytes, size_t length) {
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

bool serializeRebuildPacket(const RebuildPacket &packet, std::vector<uint8_t> *bytes,
                            std::string *error) {
    if (!bytes || !validType(packet.type) || packet.payload.size() > 65535U) {
        if (error) *error = "invalid rebuild packet output, type, or length";
        return false;
    }
    bytes->assign(kRebuildHeaderBytes + packet.payload.size(), 0);
    uint8_t *header = bytes->data();
    std::memcpy(header, kMagic, sizeof(kMagic));
    header[2] = kRebuildProtocolVersion;
    header[3] = packet.type;
    header[4] = packet.profile;
    header[5] = packet.generation;
    writeU16(header + 6, static_cast<uint16_t>(kRebuildHeaderBytes));
    writeU32(header + 8, packet.sequence);
    writeU32(header + 12, packet.frame_id);
    writeU32(header + 16, packet.pts_ms);
    writeU16(header + 20, static_cast<uint16_t>(packet.payload.size()));
    writeU16(header + 22, packet.flags);
    if (!packet.payload.empty()) {
        std::memcpy(bytes->data() + kRebuildHeaderBytes, packet.payload.data(),
                    packet.payload.size());
    }
    std::vector<uint8_t> covered;
    covered.reserve(24 + packet.payload.size());
    covered.insert(covered.end(), bytes->begin(), bytes->begin() + 24);
    covered.insert(covered.end(), packet.payload.begin(), packet.payload.end());
    writeU32(header + 24, rebuildCrc32(covered.data(), covered.size()));
    return true;
}

bool parseRebuildPacket(const uint8_t *bytes, size_t length, RebuildPacket *packet,
                        std::string *error) {
    if (!bytes || !packet || length < kRebuildHeaderBytes) {
        if (error) *error = "rebuild datagram is shorter than its header";
        return false;
    }
    const uint16_t header_bytes = readU16(bytes + 6);
    const uint16_t payload_bytes = readU16(bytes + 20);
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0 ||
        bytes[2] != kRebuildProtocolVersion || !validType(bytes[3]) ||
        header_bytes != kRebuildHeaderBytes ||
        length != static_cast<size_t>(header_bytes) + payload_bytes) {
        if (error) *error = "invalid rebuild magic, version, type, or length";
        return false;
    }
    std::vector<uint8_t> covered;
    covered.reserve(24 + payload_bytes);
    covered.insert(covered.end(), bytes, bytes + 24);
    covered.insert(covered.end(), bytes + kRebuildHeaderBytes, bytes + length);
    if (readU32(bytes + 24) != rebuildCrc32(covered.data(), covered.size())) {
        if (error) *error = "rebuild CRC32 mismatch";
        return false;
    }
    RebuildPacket parsed;
    parsed.type = bytes[3];
    parsed.profile = bytes[4];
    parsed.generation = bytes[5];
    parsed.sequence = readU32(bytes + 8);
    parsed.frame_id = readU32(bytes + 12);
    parsed.pts_ms = readU32(bytes + 16);
    parsed.flags = readU16(bytes + 22);
    parsed.payload.assign(bytes + kRebuildHeaderBytes, bytes + length);
    *packet = parsed;
    return true;
}

bool serializeRebuildState(const RebuildState &state, std::vector<uint8_t> *payload,
                           std::string *error) {
    if (!payload || state.targets.size() > 255U || state.source_width == 0 ||
        state.source_height == 0 || state.output_width == 0 || state.output_height == 0) {
        if (error) *error = "invalid rebuild state dimensions or target count";
        return false;
    }
    payload->assign(kRebuildStateHeaderBytes +
                    state.targets.size() * kRebuildTargetStateBytes, 0);
    writeU16(payload->data(), state.source_width);
    writeU16(payload->data() + 2, state.source_height);
    writeU16(payload->data() + 4, state.output_width);
    writeU16(payload->data() + 6, state.output_height);
    (*payload)[8] = state.source_fps;
    (*payload)[9] = state.output_fps;
    (*payload)[10] = static_cast<uint8_t>(state.targets.size());
    (*payload)[11] = state.flags;
    for (size_t index = 0; index < state.targets.size(); ++index) {
        const RebuildTargetState &target = state.targets[index];
        uint8_t *record = payload->data() + kRebuildStateHeaderBytes +
            index * kRebuildTargetStateBytes;
        writeU16(record, target.track_id);
        record[2] = target.class_id;
        record[3] = target.confidence_percent;
        writeU16(record + 4, target.left);
        writeU16(record + 6, target.top);
        writeU16(record + 8, target.right);
        writeU16(record + 10, target.bottom);
        writeU16(record + 12, target.reference_generation);
        record[14] = target.flags;
    }
    return true;
}

bool parseRebuildState(const uint8_t *payload, size_t length, RebuildState *state,
                       std::string *error) {
    if (!payload || !state || length < kRebuildStateHeaderBytes) {
        if (error) *error = "rebuild state is shorter than its header";
        return false;
    }
    const uint8_t count = payload[10];
    if (length != kRebuildStateHeaderBytes +
            static_cast<size_t>(count) * kRebuildTargetStateBytes) {
        if (error) *error = "rebuild state target length mismatch";
        return false;
    }
    RebuildState parsed;
    parsed.source_width = readU16(payload);
    parsed.source_height = readU16(payload + 2);
    parsed.output_width = readU16(payload + 4);
    parsed.output_height = readU16(payload + 6);
    parsed.source_fps = payload[8];
    parsed.output_fps = payload[9];
    parsed.flags = payload[11];
    for (uint8_t index = 0; index < count; ++index) {
        const uint8_t *record = payload + kRebuildStateHeaderBytes +
            static_cast<size_t>(index) * kRebuildTargetStateBytes;
        RebuildTargetState target;
        target.track_id = readU16(record);
        target.class_id = record[2];
        target.confidence_percent = record[3];
        target.left = readU16(record + 4);
        target.top = readU16(record + 6);
        target.right = readU16(record + 8);
        target.bottom = readU16(record + 10);
        target.reference_generation = readU16(record + 12);
        target.flags = record[14];
        if (target.track_id == 0 || target.right <= target.left ||
            target.bottom <= target.top) {
            if (error) *error = "invalid rebuild target state";
            return false;
        }
        parsed.targets.push_back(target);
    }
    *state = parsed;
    return true;
}

bool serializeRebuildPatchFragment(const RebuildPatchFragment &fragment,
                                   std::vector<uint8_t> *payload, std::string *error) {
    if (!payload || fragment.track_id == 0 || fragment.data_fragments == 0 ||
        fragment.fragment_index > fragment.data_fragments || fragment.chunk_bytes == 0 ||
        fragment.blob_size == 0 || fragment.data.size() > fragment.chunk_bytes ||
        fragment.right <= fragment.left || fragment.bottom <= fragment.top ||
        fragment.reference_right <= fragment.reference_left ||
        fragment.reference_bottom <= fragment.reference_top) {
        if (error) *error = "invalid rebuild patch fragment";
        return false;
    }
    payload->assign(kRebuildPatchFragmentHeaderBytes + fragment.data.size(), 0);
    uint8_t *header = payload->data();
    writeU32(header, fragment.transfer_id);
    writeU16(header + 4, fragment.track_id);
    writeU16(header + 6, fragment.reference_generation);
    writeU16(header + 8, fragment.left);
    writeU16(header + 10, fragment.top);
    writeU16(header + 12, fragment.right);
    writeU16(header + 14, fragment.bottom);
    writeU16(header + 16, fragment.reference_left);
    writeU16(header + 18, fragment.reference_top);
    writeU16(header + 20, fragment.reference_right);
    writeU16(header + 22, fragment.reference_bottom);
    writeU16(header + 24, fragment.jpeg_width);
    writeU16(header + 26, fragment.jpeg_height);
    header[28] = fragment.mask_width;
    header[29] = fragment.mask_height;
    header[30] = fragment.fragment_index;
    header[31] = fragment.data_fragments;
    writeU32(header + 32, fragment.blob_size);
    writeU16(header + 36, fragment.mask_rle_bytes);
    writeU16(header + 38, fragment.chunk_bytes);
    if (!fragment.data.empty()) {
        std::memcpy(payload->data() + kRebuildPatchFragmentHeaderBytes,
                    fragment.data.data(), fragment.data.size());
    }
    return true;
}

bool parseRebuildPatchFragment(const uint8_t *payload, size_t length,
                               RebuildPatchFragment *fragment, std::string *error) {
    if (!payload || !fragment || length < kRebuildPatchFragmentHeaderBytes) {
        if (error) *error = "rebuild patch is shorter than fragment metadata";
        return false;
    }
    RebuildPatchFragment parsed;
    parsed.transfer_id = readU32(payload);
    parsed.track_id = readU16(payload + 4);
    parsed.reference_generation = readU16(payload + 6);
    parsed.left = readU16(payload + 8);
    parsed.top = readU16(payload + 10);
    parsed.right = readU16(payload + 12);
    parsed.bottom = readU16(payload + 14);
    parsed.reference_left = readU16(payload + 16);
    parsed.reference_top = readU16(payload + 18);
    parsed.reference_right = readU16(payload + 20);
    parsed.reference_bottom = readU16(payload + 22);
    parsed.jpeg_width = readU16(payload + 24);
    parsed.jpeg_height = readU16(payload + 26);
    parsed.mask_width = payload[28];
    parsed.mask_height = payload[29];
    parsed.fragment_index = payload[30];
    parsed.data_fragments = payload[31];
    parsed.blob_size = readU32(payload + 32);
    parsed.mask_rle_bytes = readU16(payload + 36);
    parsed.chunk_bytes = readU16(payload + 38);
    parsed.data.assign(payload + kRebuildPatchFragmentHeaderBytes, payload + length);
    if (parsed.track_id == 0 || parsed.reference_generation == 0 ||
        parsed.data_fragments == 0 ||
        parsed.fragment_index > parsed.data_fragments || parsed.chunk_bytes == 0 ||
        parsed.data.size() > parsed.chunk_bytes || parsed.blob_size == 0 ||
        parsed.mask_rle_bytes > parsed.blob_size || parsed.right <= parsed.left ||
        parsed.bottom <= parsed.top ||
        parsed.reference_right <= parsed.reference_left ||
        parsed.reference_bottom <= parsed.reference_top) {
        if (error) *error = "invalid rebuild patch fragment metadata";
        return false;
    }
    *fragment = parsed;
    return true;
}

}  // namespace roi_h265

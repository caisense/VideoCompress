#ifndef ROI_H265_TRANSPORT_REBUILD_PROTOCOL_H_
#define ROI_H265_TRANSPORT_REBUILD_PROTOCOL_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace roi_h265 {

// RB/1 is a project companion protocol, not RTP.  Standard H.265 remains on
// the RTP port; RB/1 carries semantic state and fragmented visual references.
// All integer fields are big-endian and every datagram is independently CRC32
// protected so a corrupt reference is never composited onto a decoded frame.
enum RebuildPacketType {
    REBUILD_STATE = 1,
    REBUILD_PATCH_DATA = 2,
    REBUILD_PATCH_PARITY = 3,
    REBUILD_HEARTBEAT = 4,
};

const uint8_t kRebuildProtocolVersion = 1;
// Packet flags are intentionally outside the repeated fragment metadata so the
// 40-byte RB/1 layout and protocol version remain unchanged.  Old receivers
// ignore the optional HEAD bit and continue to decode the crop coordinates.
const uint16_t kRebuildPacketFlagParity = 0x0001;
const uint16_t kRebuildPacketFlagHeadReference = 0x0002;
const size_t kRebuildHeaderBytes = 28;
const size_t kRebuildStateHeaderBytes = 12;
const size_t kRebuildTargetStateBytes = 16;
// 32-byte legacy crop metadata plus an 8-byte (four uint16) detector box
// captured with the reference.  The extra box lets the receiver perform
// scale/translation registration instead of blindly centring an old crop on
// the current target.
const size_t kRebuildPatchFragmentHeaderBytes = 40;

struct RebuildPacket {
    uint8_t type;
    uint8_t profile;
    uint8_t generation;
    uint16_t flags;
    uint32_t sequence;
    uint32_t frame_id;
    uint32_t pts_ms;
    std::vector<uint8_t> payload;

    RebuildPacket();
};

struct RebuildTargetState {
    uint16_t track_id;
    uint8_t class_id;
    uint8_t confidence_percent;
    uint16_t left;
    uint16_t top;
    uint16_t right;
    uint16_t bottom;
    uint16_t reference_generation;
    uint8_t flags;

    RebuildTargetState();
};

struct RebuildState {
    uint16_t source_width;
    uint16_t source_height;
    uint16_t output_width;
    uint16_t output_height;
    uint8_t source_fps;
    uint8_t output_fps;
    uint8_t flags;
    std::vector<RebuildTargetState> targets;

    RebuildState();
};

// The fragment metadata is repeated on every data/parity datagram.  A receiver
// can therefore join at any point, and one XOR parity fragment recovers any
// single missing data fragment in the transfer.
struct RebuildPatchFragment {
    uint32_t transfer_id;
    uint16_t track_id;
    uint16_t reference_generation;
    uint16_t left;
    uint16_t top;
    uint16_t right;
    uint16_t bottom;
    uint16_t reference_left;
    uint16_t reference_top;
    uint16_t reference_right;
    uint16_t reference_bottom;
    uint16_t jpeg_width;
    uint16_t jpeg_height;
    uint8_t mask_width;
    uint8_t mask_height;
    uint8_t fragment_index;
    uint8_t data_fragments;
    uint32_t blob_size;
    uint16_t mask_rle_bytes;
    uint16_t chunk_bytes;
    std::vector<uint8_t> data;

    RebuildPatchFragment();
};

uint32_t rebuildCrc32(const uint8_t *bytes, size_t length);
bool serializeRebuildPacket(const RebuildPacket &packet, std::vector<uint8_t> *bytes,
                            std::string *error);
bool parseRebuildPacket(const uint8_t *bytes, size_t length, RebuildPacket *packet,
                        std::string *error);
bool serializeRebuildState(const RebuildState &state, std::vector<uint8_t> *payload,
                           std::string *error);
bool parseRebuildState(const uint8_t *payload, size_t length, RebuildState *state,
                       std::string *error);
bool serializeRebuildPatchFragment(const RebuildPatchFragment &fragment,
                                   std::vector<uint8_t> *payload, std::string *error);
bool parseRebuildPatchFragment(const uint8_t *payload, size_t length,
                               RebuildPatchFragment *fragment, std::string *error);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_REBUILD_PROTOCOL_H_

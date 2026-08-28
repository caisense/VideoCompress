#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "transport/rebuild_protocol.h"

namespace {

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << std::endl; \
        std::exit(1); \
    } \
} while (0)

void testPacketRoundTripAndCrc() {
    roi_h265::RebuildPacket packet;
    packet.type = roi_h265::REBUILD_STATE;
    packet.profile = 3;
    packet.generation = 7;
    packet.flags = 2;
    packet.sequence = 1234;
    packet.frame_id = 4567;
    packet.pts_ms = 8910;
    packet.payload.push_back(1);
    packet.payload.push_back(2);
    packet.payload.push_back(3);
    std::vector<uint8_t> bytes;
    std::string error;
    CHECK(roi_h265::serializeRebuildPacket(packet, &bytes, &error));
    CHECK(bytes.size() == roi_h265::kRebuildHeaderBytes + 3U);
    roi_h265::RebuildPacket parsed;
    CHECK(roi_h265::parseRebuildPacket(bytes.data(), bytes.size(), &parsed, &error));
    CHECK(parsed.type == packet.type && parsed.profile == packet.profile);
    CHECK(parsed.generation == packet.generation && parsed.sequence == packet.sequence);
    CHECK(parsed.frame_id == packet.frame_id && parsed.pts_ms == packet.pts_ms);
    CHECK(parsed.flags == packet.flags && parsed.payload == packet.payload);
    bytes.back() ^= 1U;
    CHECK(!roi_h265::parseRebuildPacket(bytes.data(), bytes.size(), &parsed, &error));
}

void testStateRoundTrip() {
    roi_h265::RebuildState state;
    state.source_width = 640;
    state.source_height = 480;
    state.output_width = 640;
    state.output_height = 360;
    state.source_fps = 6;
    state.output_fps = 12;
    state.flags = 1;
    roi_h265::RebuildTargetState target;
    target.track_id = 9;
    target.class_id = 0;
    target.confidence_percent = 92;
    target.left = 100;
    target.top = 80;
    target.right = 220;
    target.bottom = 300;
    target.reference_generation = 4;
    target.flags = 1;
    state.targets.push_back(target);
    std::vector<uint8_t> payload;
    std::string error;
    CHECK(roi_h265::serializeRebuildState(state, &payload, &error));
    CHECK(payload.size() == roi_h265::kRebuildStateHeaderBytes +
          roi_h265::kRebuildTargetStateBytes);
    roi_h265::RebuildState parsed;
    CHECK(roi_h265::parseRebuildState(payload.data(), payload.size(), &parsed, &error));
    CHECK(parsed.source_width == 640 && parsed.source_height == 480);
    CHECK(parsed.output_width == 640 && parsed.output_height == 360);
    CHECK(parsed.source_fps == 6 && parsed.output_fps == 12);
    CHECK(parsed.targets.size() == 1U && parsed.targets[0].track_id == 9);
    CHECK(parsed.targets[0].reference_generation == 4);
}

void testPatchFragmentRoundTrip() {
    roi_h265::RebuildPatchFragment fragment;
    fragment.transfer_id = 77;
    fragment.track_id = 9;
    fragment.reference_generation = 5;
    fragment.left = 80;
    fragment.top = 40;
    fragment.right = 240;
    fragment.bottom = 320;
    fragment.reference_left = 90;
    fragment.reference_top = 50;
    fragment.reference_right = 230;
    fragment.reference_bottom = 310;
    fragment.jpeg_width = 73;
    fragment.jpeg_height = 128;
    fragment.mask_width = 32;
    fragment.mask_height = 32;
    fragment.fragment_index = 2;
    fragment.data_fragments = 4;
    fragment.blob_size = 2600;
    fragment.mask_rle_bytes = 180;
    fragment.chunk_bytes = 900;
    fragment.data.assign(800, 0x5a);
    std::vector<uint8_t> payload;
    std::string error;
    CHECK(roi_h265::serializeRebuildPatchFragment(fragment, &payload, &error));
    CHECK(payload.size() == roi_h265::kRebuildPatchFragmentHeaderBytes + 800U);
    roi_h265::RebuildPatchFragment parsed;
    CHECK(roi_h265::parseRebuildPatchFragment(
        payload.data(), payload.size(), &parsed, &error));
    CHECK(parsed.transfer_id == 77 && parsed.track_id == 9);
    CHECK(parsed.fragment_index == 2 && parsed.data_fragments == 4);
    CHECK(parsed.reference_left == 90 && parsed.reference_top == 50 &&
          parsed.reference_right == 230 && parsed.reference_bottom == 310);
    CHECK(parsed.mask_rle_bytes == 180 && parsed.data == fragment.data);
}

}  // namespace

int main() {
    testPacketRoundTripAndCrc();
    testStateRoundTrip();
    testPatchFragmentRoundTrip();
    std::cout << "rebuild protocol tests passed" << std::endl;
    return 0;
}

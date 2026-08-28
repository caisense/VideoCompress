#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common/config.h"
#include "transport/snapshot_protocol.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

void testWireRoundTrip() {
    roi_h265::SnapshotPacket source;
    source.type = roi_h265::SNAPSHOT_DATA;
    source.flags = 0x1234;
    source.transfer_id = 0x01020304U;
    source.total_bytes = 11;
    source.offset = 4;
    source.width = 1920;
    source.height = 1080;
    source.class_mask = static_cast<uint16_t>((1U << 0) | (1U << 3));
    source.payload.assign(7, 0);
    for (size_t index = 0; index < source.payload.size(); ++index) {
        source.payload[index] = static_cast<uint8_t>(index + 9);
    }
    source.payload_length = static_cast<uint16_t>(source.payload.size());
    source.crc32 = roi_h265::snapshotCrc32(source.payload.data(), source.payload.size());

    std::string error;
    std::vector<uint8_t> bytes;
    CHECK(roi_h265::serializeSnapshotPacket(source, &bytes, &error));
    CHECK(bytes.size() == roi_h265::kSnapshotHeaderBytes + source.payload.size());
    CHECK(bytes[0] == 'R' && bytes[1] == 'S' && bytes[2] == 'N' && bytes[3] == 'P');
    CHECK(bytes[8] == 1 && bytes[11] == 4);
    roi_h265::SnapshotPacket parsed;
    CHECK(roi_h265::parseSnapshotPacket(bytes.data(), bytes.size(), &parsed, &error));
    CHECK(parsed.type == source.type && parsed.flags == source.flags);
    CHECK(parsed.transfer_id == source.transfer_id && parsed.total_bytes == source.total_bytes);
    CHECK(parsed.offset == source.offset && parsed.payload == source.payload);
    CHECK(parsed.width == source.width && parsed.height == source.height);
    CHECK(parsed.class_mask == source.class_mask && parsed.crc32 == source.crc32);

    bytes.pop_back();
    CHECK(!roi_h265::parseSnapshotPacket(bytes.data(), bytes.size(), &parsed, &error));
}

void testCrcAndClassFiltering() {
    const uint8_t text[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    CHECK(roi_h265::snapshotCrc32(text, sizeof(text)) == 0xcbf43926U);
    CHECK(roi_h265::isRelevantDetectionClass(0));
    CHECK(roi_h265::isRelevantDetectionClass(2));
    CHECK(roi_h265::isRelevantDetectionClass(4));
    CHECK(roi_h265::isRelevantDetectionClass(8));
    CHECK(!roi_h265::isRelevantDetectionClass(1));
    CHECK(roi_h265::relevantDetectionClassBit(8) == (1U << 2));

    roi_h265::SegResult result;
    const int ids[] = {0, 1, 2, 4, 8};
    for (size_t index = 0; index < sizeof(ids) / sizeof(ids[0]); ++index) {
        roi_h265::SegInstance instance;
        instance.class_id = ids[index];
        instance.confidence = index == 3 ? 0.2f : 0.8f;
        result.instances.push_back(instance);
    }
    CHECK(roi_h265::relevantDetectionClassMask(result, 0.35f) ==
          static_cast<uint16_t>((1U << 0) | (1U << 1) | (1U << 2)));
    roi_h265::filterRelevantDetections(&result, 0.35f);
    CHECK(result.instances.size() == 3);
    CHECK(result.instances[0].class_id == 0);
    CHECK(result.instances[1].class_id == 2);
    CHECK(result.instances[2].class_id == 8);
}

void testImageModeConfig() {
    char app[] = "roi_sender";
    char image[] = "--transport-mode=image";
    char port[] = "--snapshot-udp-port=5010";
    char quality[] = "--snapshot-jpeg-quality=91";
    char chunk[] = "--snapshot-chunk-bytes=800";
    char confidence[] = "--snapshot-min-confidence=0.55";
    char crop[] = "--snapshot-crop=full";
    char margin[] = "--snapshot-crop-margin-percent=40";
    char rotate[] = "--snapshot-rotate=none";
    char *argv[] = {app, image, port, quality, chunk, confidence, crop, margin, rotate};
    roi_h265::AppConfig config;
    std::string error;
    CHECK(roi_h265::parseAppConfig(9, argv, &config, &error));
    CHECK(config.transport.mode == roi_h265::TRANSPORT_MODE_IMAGE);
    CHECK(config.transport.snapshot.udp_port == 5010);
    CHECK(config.transport.snapshot.jpeg_quality == 91);
    CHECK(config.transport.snapshot.chunk_payload_bytes == 800);
    CHECK(config.transport.snapshot.min_confidence == 0.55f);
    CHECK(config.transport.snapshot.crop_mode == roi_h265::SNAPSHOT_CROP_FULL);
    CHECK(config.transport.snapshot.crop_margin_percent == 40);
    CHECK(!config.transport.snapshot.rotate_ccw);
    CHECK(std::string(roi_h265::transportModeName(config.transport.mode)) == "image");

    char baseline[] = "--mode=baseline";
    char *invalid_argv[] = {app, image, baseline};
    CHECK(!roi_h265::parseAppConfig(3, invalid_argv, &config, &error));

    roi_h265::AppConfig defaults;
    CHECK(defaults.transport.snapshot.jpeg_quality == 75);
    CHECK(defaults.transport.snapshot.max_width == 1280);
    CHECK(defaults.transport.snapshot.max_height == 720);
    CHECK(defaults.transport.snapshot.chunk_payload_bytes == 1100);
    CHECK(defaults.transport.snapshot.crop_mode == roi_h265::SNAPSHOT_CROP_RELEVANT);
    CHECK(defaults.transport.snapshot.crop_margin_percent == 25);
    CHECK(defaults.transport.snapshot.rotate_ccw);
}

}  // namespace

int main() {
    testWireRoundTrip();
    testCrcAndClassFiltering();
    testImageModeConfig();
    if (failures != 0) {
        std::cerr << failures << " snapshot protocol test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Snapshot protocol tests passed\n";
    return EXIT_SUCCESS;
}

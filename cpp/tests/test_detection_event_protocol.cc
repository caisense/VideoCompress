#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common/config.h"
#include "transport/detection_event_protocol.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

void testWireRoundTrip() {
    roi_h265::DetectionEventPacket source;
    source.flags = 0;
    source.sequence = 0x01020304U;
    source.frame_id = 0x0102030405060708ULL;
    source.pts_ms = 0x11223344U;
    source.present_mask = static_cast<uint16_t>((1U << 0) | (1U << 2));
    source.entered_mask = 1U << 0;
    source.counts[0] = 2;
    source.counts[2] = 1;
    source.max_confidence_percent[0] = 91;
    source.max_confidence_percent[2] = 75;

    std::string error;
    std::vector<uint8_t> bytes;
    CHECK(roi_h265::serializeDetectionEventPacket(source, &bytes, &error));
    CHECK(bytes.size() == roi_h265::kDetectionEventHeaderBytes);
    CHECK(bytes[0] == 'R' && bytes[1] == 'O' && bytes[2] == 'E' && bytes[3] == 'V');
    CHECK(bytes[4] == 1 && bytes[5] == roi_h265::DETECTION_EVENT_STATE);
    CHECK(bytes[30] == 0 && bytes[31] == 0);
    static const uint8_t kExpectedWire[] = {
        0x52, 0x4f, 0x45, 0x56, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08, 0x11, 0x22, 0x33, 0x44,
        0x00, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x01, 0x00, 0x5b, 0x00, 0x4b, 0x00,
        0xcd, 0xb8, 0xe5, 0x59,
    };
    CHECK(bytes == std::vector<uint8_t>(kExpectedWire,
                                         kExpectedWire + sizeof(kExpectedWire)));

    roi_h265::DetectionEventPacket parsed;
    CHECK(roi_h265::parseDetectionEventPacket(bytes.data(), bytes.size(), &parsed, &error));
    CHECK(parsed.sequence == source.sequence && parsed.frame_id == source.frame_id);
    CHECK(parsed.pts_ms == source.pts_ms && parsed.flags == source.flags);
    CHECK(parsed.present_mask == source.present_mask && parsed.entered_mask == source.entered_mask);
    CHECK(parsed.exited_mask == 0 && parsed.counts == source.counts);
    CHECK(parsed.max_confidence_percent == source.max_confidence_percent);
    CHECK(parsed.crc32 == roi_h265::detectionEventCrc32(bytes.data(), 40));

    bytes[20] ^= 1U;
    CHECK(!roi_h265::parseDetectionEventPacket(bytes.data(), bytes.size(), &parsed, &error));
}

void testSummary() {
    roi_h265::SegResult result;
    result.frame.frame_id = 42;
    result.frame.pts_us = 9876543;
    const int class_ids[] = {0, 0, 2, 4, 8, 1};
    const float confidences[] = {0.88f, 0.91f, 0.30f, 0.74f, 0.99f, 0.99f};
    for (size_t index = 0; index < sizeof(class_ids) / sizeof(class_ids[0]); ++index) {
        roi_h265::SegInstance instance;
        instance.class_id = class_ids[index];
        instance.confidence = confidences[index];
        result.instances.push_back(instance);
    }
    const roi_h265::DetectionEventSummary summary =
        roi_h265::summarizeDetectionEvent(result, 0.35f);
    CHECK(summary.frame_id == 42 && summary.pts_ms == 9876U);
    CHECK(summary.present_mask == static_cast<uint16_t>((1U << 0) | (1U << 2) | (1U << 3)));
    CHECK(summary.counts[0] == 2 && summary.counts[1] == 0 && summary.counts[2] == 1 &&
          summary.counts[3] == 1);
    CHECK(summary.max_confidence_percent[0] == 91 && summary.max_confidence_percent[2] == 99 &&
          summary.max_confidence_percent[3] == 74);
    CHECK(roi_h265::detectionEventClassIndex(0) == 0);
    CHECK(roi_h265::detectionEventClassIndex(2) == 1);
    CHECK(roi_h265::detectionEventClassIndex(8) == 2);
    CHECK(roi_h265::detectionEventClassIndex(4) == 3);
    CHECK(roi_h265::detectionEventClassIndex(1) == -1);
    CHECK(std::string(roi_h265::detectionEventClassName(3)) == "airplane");
}

void testConfig() {
    char app[] = "roi_sender";
    char enabled[] = "--event-push=off";
    char port[] = "--event-udp-port=5011";
    char confidence[] = "--event-min-confidence=0.55";
    char heartbeat[] = "--event-heartbeat-ms=800";
    char *argv[] = {app, enabled, port, confidence, heartbeat};
    roi_h265::AppConfig config;
    std::string error;
    CHECK(roi_h265::parseAppConfig(5, argv, &config, &error));
    CHECK(!config.transport.event.enabled && config.transport.event.udp_port == 5011);
    CHECK(config.transport.event.min_confidence == 0.55f &&
          config.transport.event.heartbeat_ms == 800);

    roi_h265::AppConfig defaults;
    CHECK(defaults.transport.event.enabled);
    CHECK(defaults.transport.event.udp_port == 5010);
    CHECK(defaults.transport.event.min_confidence == 0.35f);
    CHECK(defaults.transport.event.heartbeat_ms == 1000);

    char collide[] = "--event-udp-port=5004";
    char *bad_argv[] = {app, collide};
    CHECK(!roi_h265::parseAppConfig(2, bad_argv, &defaults, &error));
}

}  // namespace

int main() {
    testWireRoundTrip();
    testSummary();
    testConfig();
    if (failures != 0) {
        std::cerr << failures << " detection event protocol test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Detection event protocol tests passed\n";
    return EXIT_SUCCESS;
}

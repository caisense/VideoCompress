#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/config.h"
#include "transport/detection_event_sender.h"

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

roi_h265::SegResult resultWithPerson(uint64_t frame_id, uint64_t pts_us) {
    roi_h265::SegResult result;
    result.frame.frame_id = frame_id;
    result.frame.pts_us = pts_us;
    roi_h265::SegInstance person;
    person.class_id = 0;
    person.confidence = 0.91f;
    result.instances.push_back(person);
    return result;
}

bool receivePacket(int receiver, roi_h265::DetectionEventPacket *packet,
                   std::string *error) {
    pollfd descriptor;
    descriptor.fd = receiver;
    descriptor.events = POLLIN;
    descriptor.revents = 0;
    if (poll(&descriptor, 1, 1500) <= 0 || !(descriptor.revents & POLLIN)) {
        if (error) *error = "timed out waiting for local ROEV packet";
        return false;
    }
    uint8_t bytes[256];
    const ssize_t count = recv(receiver, bytes, sizeof(bytes), 0);
    if (count <= 0) {
        if (error) *error = "failed to read local ROEV packet";
        return false;
    }
    return roi_h265::parseDetectionEventPacket(bytes, static_cast<size_t>(count), packet, error);
}

void testStateHeartbeatAndExit() {
    const int receiver = socket(AF_INET, SOCK_DGRAM, 0);
    CHECK(receiver >= 0);
    if (receiver < 0) return;

    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    CHECK(bind(receiver, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == 0);
    socklen_t address_length = sizeof(address);
    CHECK(getsockname(receiver, reinterpret_cast<sockaddr *>(&address), &address_length) == 0);
    const int port = ntohs(address.sin_port);
    if (port <= 0 || failures != 0) {
        close(receiver);
        return;
    }

    roi_h265::DetectionEventConfig config;
    config.enabled = true;
    config.udp_port = port;
    config.min_confidence = 0.35f;
    config.heartbeat_ms = 100;
    const std::shared_ptr<roi_h265::RatePacer> pacer(
        new roi_h265::RatePacer(10000000, 1200));
    roi_h265::DetectionEventSender sender(config, "127.0.0.1", 1200, pacer);
    std::string error;
    CHECK(sender.start(&error));
    sender.setEnabled(true);

    CHECK(sender.submit(resultWithPerson(10, 1000000), &error));
    roi_h265::DetectionEventPacket packet;
    CHECK(receivePacket(receiver, &packet, &error));
    CHECK(packet.present_mask == (1U << 0));
    CHECK(packet.entered_mask == (1U << 0));
    CHECK(packet.exited_mask == 0);
    CHECK((packet.flags & roi_h265::DETECTION_EVENT_FLAG_HEARTBEAT) == 0);
    CHECK(packet.counts[0] == 1 && packet.max_confidence_percent[0] == 91);

    usleep(130000);
    CHECK(sender.submit(resultWithPerson(11, 1100000), &error));
    CHECK(receivePacket(receiver, &packet, &error));
    CHECK(packet.present_mask == (1U << 0));
    CHECK(packet.entered_mask == 0 && packet.exited_mask == 0);
    CHECK((packet.flags & roi_h265::DETECTION_EVENT_FLAG_HEARTBEAT) != 0);

    roi_h265::SegResult empty;
    empty.frame.frame_id = 12;
    empty.frame.pts_us = 1200000;
    CHECK(sender.submit(empty, &error));
    CHECK(receivePacket(receiver, &packet, &error));
    CHECK(packet.present_mask == 0);
    CHECK(packet.entered_mask == 0 && packet.exited_mask == (1U << 0));
    CHECK((packet.flags & roi_h265::DETECTION_EVENT_FLAG_HEARTBEAT) == 0);
    sender.stop();
    close(receiver);
}

}  // namespace

int main() {
    testStateHeartbeatAndExit();
    if (failures != 0) {
        std::cerr << failures << " detection event sender test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Detection event sender tests passed\n";
    return EXIT_SUCCESS;
}

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "transport/async_rtp_sender.h"

namespace {
int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

int makeLoopbackReceiver(uint16_t *port) {
    const int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) return -1;
    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(socket_fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        return -1;
    }
    socklen_t length = sizeof(address);
    getsockname(socket_fd, reinterpret_cast<sockaddr *>(&address), &length);
    *port = ntohs(address.sin_port);
    return socket_fd;
}

roi_h265::RtpAccessUnit makeUnit(uint64_t frame_id, bool key, size_t payload_bytes) {
    roi_h265::RtpAccessUnit unit;
    unit.frame.frame_id = frame_id;
    unit.frame.pts_us = frame_id * 333333;
    unit.frame.capture_time_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    unit.key_frame = key;
    unit.bytes = {0, 0, 0, 1, static_cast<uint8_t>(key ? 0x26 : 0x02), 0x01};
    unit.bytes.insert(unit.bytes.end(), payload_bytes, 0x55);
    return unit;
}

void testEnqueueDoesNotWaitForPacer() {
    uint16_t port = 0;
    const int receiver = makeLoopbackReceiver(&port);
    CHECK(receiver >= 0);
    roi_h265::AsyncRtpSender sender("127.0.0.1", port, 8000, 128, 3, 500);
    std::string error;
    CHECK(sender.start(&error));
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    CHECK(sender.enqueue(makeUnit(0, true, 250), &error));
    const long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(elapsed_us < 50000);
    sender.stop();
    close(receiver);
}

void testBacklogDropsPAndRequestsRecoveryIdr() {
    uint16_t port = 0;
    const int receiver = makeLoopbackReceiver(&port);
    CHECK(receiver >= 0);
    roi_h265::AsyncRtpSender sender("127.0.0.1", port, 8000, 128, 2, 1000);
    std::string error;
    CHECK(sender.start(&error));
    CHECK(sender.enqueue(makeUnit(0, true, 400), &error));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(sender.enqueue(makeUnit(1, false, 20), &error));
    CHECK(sender.enqueue(makeUnit(2, false, 20), &error));
    CHECK(sender.enqueue(makeUnit(3, false, 20), &error));
    roi_h265::AsyncRtpSenderSnapshot snapshot = sender.snapshot();
    CHECK(snapshot.dropped_p_frames >= 3);
    CHECK(snapshot.waiting_for_key_frame);
    CHECK(sender.needsKeyFrame());
    CHECK(sender.enqueue(makeUnit(4, true, 20), &error));
    snapshot = sender.snapshot();
    CHECK(!snapshot.waiting_for_key_frame);
    sender.stop();
    close(receiver);
}

void testRuntimeProfileSwitchRequiresFreshIdr() {
    uint16_t port = 0;
    const int receiver = makeLoopbackReceiver(&port);
    CHECK(receiver >= 0);
    roi_h265::AsyncRtpSender sender("127.0.0.1", port, 8000, 128, 3, 1000);
    std::string error;
    CHECK(sender.start(&error));
    CHECK(sender.switchProfile(16000, &error));
    CHECK(sender.needsKeyFrame());
    CHECK(sender.enqueue(makeUnit(1, false, 20), &error));
    CHECK(sender.snapshot().dropped_p_frames == 1);
    CHECK(sender.enqueue(makeUnit(2, true, 20), &error));
    CHECK(!sender.needsKeyFrame());
    sender.stop();
    close(receiver);
}

}  // namespace

int main() {
    testEnqueueDoesNotWaitForPacer();
    testBacklogDropsPAndRequestsRecoveryIdr();
    testRuntimeProfileSwitchRequiresFreshIdr();
    if (failures) return EXIT_FAILURE;
    std::cout << "asynchronous RTP sender tests passed\n";
    return EXIT_SUCCESS;
}

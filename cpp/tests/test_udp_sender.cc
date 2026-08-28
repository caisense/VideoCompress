#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "transport/udp_sender.h"

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
    if (getsockname(socket_fd, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
        close(socket_fd);
        return -1;
    }
    *port = ntohs(address.sin_port);
    return socket_fd;
}

bool receiveExact(int socket_fd, const std::vector<uint8_t> &expected) {
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(socket_fd, &readable);
    timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    if (select(socket_fd + 1, &readable, NULL, NULL, &timeout) != 1) return false;
    uint8_t received[256];
    const ssize_t count = recv(socket_fd, received, sizeof(received), 0);
    return count == static_cast<ssize_t>(expected.size()) &&
        std::memcmp(received, expected.data(), expected.size()) == 0;
}

void testLoopbackAndMtu() {
    uint16_t port = 0;
    const int receiver = makeLoopbackReceiver(&port);
    CHECK(receiver >= 0);
    if (receiver < 0) return;
    roi_h265::UdpSender sender("127.0.0.1", port, 1000000, 128);
    std::string error;
    const std::vector<uint8_t> first = {0x80, 0x60, 0x00, 0x07};
    const std::vector<uint8_t> second = {0x80, 0xe0, 0x00, 0x08, 0x01};
    const std::vector<std::vector<uint8_t> > packets = {first, second};
    CHECK(sender.sendPackets(packets, &error));
    CHECK(receiveExact(receiver, first));
    CHECK(receiveExact(receiver, second));
    CHECK(sender.queuedPackets() == 0);

    const std::vector<std::vector<uint8_t> > too_large(1, std::vector<uint8_t>(101, 0));
    CHECK(!sender.sendPackets(too_large, &error));  // 128-byte IP MTU leaves 100 UDP bytes.
    CHECK(error.find("MTU") != std::string::npos);
    close(receiver);
}

void testWirePacingStartsEmpty() {
    uint16_t port = 0;
    const int receiver = makeLoopbackReceiver(&port);
    CHECK(receiver >= 0);
    if (receiver < 0) return;
    roi_h265::UdpSender sender("127.0.0.1", port, 8000, 128);
    std::string error;
    const std::vector<std::vector<uint8_t> > packet(1, std::vector<uint8_t>(20, 0x55));
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    CHECK(sender.sendPackets(packet, &error));
    const long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(receiveExact(receiver, packet[0]));
    // 20 RTP bytes occupy at least 86 Ethernet wire bytes at 8 kbps: 86 ms.
    CHECK(elapsed_us >= 60000);
    CHECK(elapsed_us < 500000);
    close(receiver);
}

void testSharedPacerCapsBothMediaSenders() {
    uint16_t first_port = 0;
    uint16_t second_port = 0;
    const int first_receiver = makeLoopbackReceiver(&first_port);
    const int second_receiver = makeLoopbackReceiver(&second_port);
    CHECK(first_receiver >= 0 && second_receiver >= 0);
    if (first_receiver < 0 || second_receiver < 0) {
        if (first_receiver >= 0) close(first_receiver);
        if (second_receiver >= 0) close(second_receiver);
        return;
    }
    const std::shared_ptr<roi_h265::RatePacer> shared(new roi_h265::RatePacer(8000, 166));
    roi_h265::UdpSender video("127.0.0.1", first_port, 8000, 128, shared);
    roi_h265::UdpSender audio("127.0.0.1", second_port, 8000, 128, shared);
    const std::vector<std::vector<uint8_t> > packet(1, std::vector<uint8_t>(20, 0x77));
    std::string error;
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    CHECK(video.sendPackets(packet, &error));
    CHECK(audio.sendPackets(packet, &error));
    const long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(receiveExact(first_receiver, packet[0]));
    CHECK(receiveExact(second_receiver, packet[0]));
    // Two 20-byte RTP datagrams cost 2 x 86 Ethernet-wire bytes.  A shared
    // 8 kbps bucket therefore needs roughly 172 ms rather than each sender
    // independently completing in 86 ms.
    CHECK(elapsed_us >= 130000);
    video.setPacingBitrate(16000);
    CHECK(audio.pacingBitrate() == 16000);
    close(first_receiver);
    close(second_receiver);
}

}  // namespace

int main() {
    testLoopbackAndMtu();
    testWirePacingStartsEmpty();
    testSharedPacerCapsBothMediaSenders();
    if (failures) return EXIT_FAILURE;
    std::cout << "UDP sender tests passed\n";
    return EXIT_SUCCESS;
}

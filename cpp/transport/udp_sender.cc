#include "transport/udp_sender.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace roi_h265 {
namespace {

const size_t kIpv4UdpHeaderBytes = 28;
const size_t kEthernetWireOverheadBytes = 38;

}  // namespace

struct UdpSender::sockaddr_storage_holder {
    sockaddr_storage address;
    socklen_t length;
};

UdpSender::UdpSender(const std::string &host, int port, int pacing_bitrate_bps, int mtu,
                     const std::shared_ptr<RatePacer> &pacer)
    : host_(host), port_(port), socket_(-1), mtu_(mtu),
      pacer_(pacer ? pacer : std::shared_ptr<RatePacer>(new RatePacer(
          pacing_bitrate_bps, static_cast<size_t>(mtu) + kEthernetWireOverheadBytes))),
      queued_packets_(0), destination_(NULL) {}

UdpSender::~UdpSender() {
    if (socket_ >= 0) close(socket_);
    delete destination_;
}

bool UdpSender::open(std::string *error) {
    if (socket_ >= 0) return true;
    char port_text[16];
    std::snprintf(port_text, sizeof(port_text), "%d", port_);
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo *result = NULL;
    if (getaddrinfo(host_.c_str(), port_text, &hints, &result) != 0 || !result) {
        if (error) *error = "cannot resolve UDP destination";
        return false;
    }
    socket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (socket_ < 0) {
        freeaddrinfo(result);
        if (error) *error = "cannot create UDP socket";
        return false;
    }
    destination_ = new sockaddr_storage_holder;
    std::memset(destination_, 0, sizeof(*destination_));
    std::memcpy(&destination_->address, result->ai_addr, result->ai_addrlen);
    destination_->length = static_cast<socklen_t>(result->ai_addrlen);
    freeaddrinfo(result);
    return true;
}

bool UdpSender::sendPackets(const std::vector<std::vector<uint8_t> > &packets, std::string *error) {
    if (!open(error)) return false;
    queued_packets_ = packets.size();
    for (size_t i = 0; i < packets.size(); ++i) {
        if (packets[i].size() > static_cast<size_t>(mtu_ - kIpv4UdpHeaderBytes)) {
            if (error) *error = "packetizer exceeded IPv4 MTU after UDP/IP headers";
            return false;
        }
        pacer_->waitForTokens(RatePacer::wireBytesForUdpPayload(packets[i].size()));
        const ssize_t written = sendto(socket_, packets[i].data(), packets[i].size(), 0,
            reinterpret_cast<const sockaddr *>(&destination_->address), destination_->length);
        if (written != static_cast<ssize_t>(packets[i].size())) {
            if (error) *error = "UDP send failed";
            return false;
        }
        --queued_packets_;
    }
    return true;
}

void UdpSender::setPacingBitrate(int bitrate_bps, size_t max_burst_bytes) {
    if (max_burst_bytes == 0) {
        max_burst_bytes = static_cast<size_t>(mtu_) + kEthernetWireOverheadBytes;
    }
    pacer_->reset(bitrate_bps, max_burst_bytes);
}

int UdpSender::pacingBitrate() const { return pacer_->bitrateBps(); }

size_t UdpSender::queuedPackets() const { return queued_packets_; }

}  // namespace roi_h265

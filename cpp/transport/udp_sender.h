#ifndef ROI_H265_TRANSPORT_UDP_SENDER_H_
#define ROI_H265_TRANSPORT_UDP_SENDER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <memory>
#include <vector>

#include "transport/rate_pacer.h"

namespace roi_h265 {

class UdpSender {
public:
    // Supplying a caller-owned pacer lets main() place this stream in a
    // centrally allocated physical-link child bucket. Without it the sender
    // owns an independent pacer, preserving the old video-only behavior for
    // tests.
    UdpSender(const std::string &host, int port, int pacing_bitrate_bps, int mtu,
              const std::shared_ptr<RatePacer> &pacer = std::shared_ptr<RatePacer>());
    ~UdpSender();

    bool open(std::string *error);
    bool sendPackets(const std::vector<std::vector<uint8_t> > &packets, std::string *error);
    void setPacingBitrate(int bitrate_bps, size_t max_burst_bytes = 0);
    int pacingBitrate() const;
    size_t queuedPackets() const;

private:
    std::string host_;
    int port_;
    int socket_;
    int mtu_;
    std::shared_ptr<RatePacer> pacer_;
    size_t queued_packets_;
    struct sockaddr_storage_holder;
    sockaddr_storage_holder *destination_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_UDP_SENDER_H_

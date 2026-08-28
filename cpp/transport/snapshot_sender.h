#ifndef ROI_H265_TRANSPORT_SNAPSHOT_SENDER_H_
#define ROI_H265_TRANSPORT_SNAPSHOT_SENDER_H_

#include <stddef.h>
#include <stdint.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"
#include "common/frame_meta.h"
#include "transport/rate_pacer.h"
#include "transport/snapshot_crop.h"
#include "transport/snapshot_protocol.h"

namespace roi_h265 {

struct SnapshotSenderSnapshot {
    bool enabled;
    bool transferring;
    size_t queued_requests;
    uint64_t submitted_requests;
    uint64_t completed_transfers;
    uint64_t failed_transfers;
    uint64_t cancelled_transfers;
    uint64_t replaced_requests;
    uint64_t sent_jpeg_bytes;
    uint64_t retransmitted_packets;
    uint64_t last_transfer_bytes;
    int last_transfer_width;
    int last_transfer_height;
    std::string last_error;

    SnapshotSenderSnapshot()
        : enabled(false), transferring(false), queued_requests(0), submitted_requests(0),
          completed_transfers(0), failed_transfers(0), cancelled_transfers(0),
          replaced_requests(0), sent_jpeg_bytes(0), retransmitted_packets(0),
          last_transfer_bytes(0), last_transfer_width(0), last_transfer_height(0) {}
};

// JPEG creation and reliable stop-and-wait transfer live outside capture and
// RKNN threads.  Only an active job plus one newest detection is retained, so
// an extremely slow link cannot build an unbounded evidence backlog.
class SnapshotSender {
public:
    SnapshotSender(const SnapshotConfig &config, const std::string &host, int mtu,
                   const std::shared_ptr<RatePacer> &pacer);
    ~SnapshotSender();

    bool start(std::string *error);
    void setEnabled(bool enabled);
    bool submit(const std::shared_ptr<FramePacket> &frame, uint16_t class_mask,
                const SnapshotCrop &crop,
                std::string *error);
    void stop();
    SnapshotSenderSnapshot snapshot() const;

private:
    struct Request {
        std::shared_ptr<FramePacket> frame;
        uint16_t class_mask;
        SnapshotCrop crop;

        Request() : class_mask(0) {}
    };

    enum TransferResult {
        TRANSFER_COMPLETED,
        TRANSFER_CANCELLED,
        TRANSFER_FAILED,
    };

    bool openSocket(std::string *error);
    bool sendPacket(const SnapshotPacket &packet, std::string *error);
    bool waitForReply(uint32_t transfer_id, SnapshotPacket *reply);
    bool isTransferAllowed() const;
    bool encodeJpeg(const Request &request, std::vector<uint8_t> *jpeg,
                    int *width, int *height, std::string *error) const;
    TransferResult transfer(const Request &request, const std::vector<uint8_t> &jpeg,
                            int width, int height, std::string *error);
    void workerLoop();

    SnapshotConfig config_;
    std::string host_;
    int mtu_;
    std::shared_ptr<RatePacer> pacer_;
    int socket_;
    struct sockaddr_storage_holder;
    sockaddr_storage_holder *destination_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<Request> queue_;
    std::thread worker_;
    bool started_;
    bool stopping_;
    bool enabled_;
    bool transferring_;
    uint32_t next_transfer_id_;
    std::chrono::steady_clock::time_point last_transfer_started_;
    bool has_last_transfer_started_;
    SnapshotSenderSnapshot snapshot_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_SNAPSHOT_SENDER_H_

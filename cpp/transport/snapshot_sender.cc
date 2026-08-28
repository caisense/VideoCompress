#include "transport/snapshot_sender.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <arpa/inet.h>
#include <netdb.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace roi_h265 {

struct SnapshotSender::sockaddr_storage_holder {
    sockaddr_storage address;
    socklen_t length;
};

SnapshotSender::SnapshotSender(const SnapshotConfig &config, const std::string &host, int mtu,
                               const std::shared_ptr<RatePacer> &pacer)
    : config_(config), host_(host), mtu_(mtu), pacer_(pacer), socket_(-1), destination_(NULL),
      started_(false), stopping_(false), enabled_(false), transferring_(false),
      next_transfer_id_(1), has_last_transfer_started_(false) {}

SnapshotSender::~SnapshotSender() { stop(); }

bool SnapshotSender::openSocket(std::string *error) {
    if (socket_ >= 0) return true;
    char port_text[16];
    std::snprintf(port_text, sizeof(port_text), "%d", config_.udp_port);
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo *result = NULL;
    if (getaddrinfo(host_.c_str(), port_text, &hints, &result) != 0 || !result) {
        if (error) *error = "cannot resolve snapshot UDP destination";
        return false;
    }
    socket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (socket_ < 0) {
        freeaddrinfo(result);
        if (error) *error = "cannot create snapshot UDP socket";
        return false;
    }
    destination_ = new sockaddr_storage_holder;
    std::memset(destination_, 0, sizeof(*destination_));
    std::memcpy(&destination_->address, result->ai_addr, result->ai_addrlen);
    destination_->length = static_cast<socklen_t>(result->ai_addrlen);
    const int connected = connect(socket_, reinterpret_cast<const sockaddr *>(&destination_->address),
                                  destination_->length);
    freeaddrinfo(result);
    if (connected != 0) {
        if (error) *error = "cannot connect snapshot UDP socket";
        close(socket_);
        socket_ = -1;
        delete destination_;
        destination_ = NULL;
        return false;
    }
    return true;
}

bool SnapshotSender::start(std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return true;
    if (!pacer_) {
        if (error) *error = "snapshot sender requires the shared video pacer";
        return false;
    }
    if (!openSocket(error)) return false;
    stopping_ = false;
    started_ = true;
    worker_ = std::thread(&SnapshotSender::workerLoop, this);
    return true;
}

void SnapshotSender::setEnabled(bool enabled) {
    std::unique_lock<std::mutex> lock(mutex_);
    enabled_ = enabled;
    snapshot_.enabled = enabled;
    if (!enabled_) {
        queue_.clear();
        // A video switch must not begin while a JPEG packet is still using the
        // shared video-side bucket.  The worker checks enabled_ between every
        // packet/ACK timeout and notifies once the current transfer unwinds.
        condition_.notify_all();
        condition_.wait(lock, [this] { return !transferring_; });
    }
    condition_.notify_all();
}

bool SnapshotSender::submit(const std::shared_ptr<FramePacket> &frame, uint16_t class_mask,
                            const SnapshotCrop &crop, std::string *error) {
    if (!frame || class_mask == 0) {
        if (error) *error = "snapshot requires a frame and at least one relevant class";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stopping_) {
        if (error) *error = "snapshot sender is not running";
        return false;
    }
    if (!enabled_) return true;
    Request request;
    request.frame = frame;
    request.class_mask = class_mask;
    request.crop = crop;
    ++snapshot_.submitted_requests;
    if (queue_.empty()) {
        queue_.push_back(request);
    } else {
        queue_[0] = request;
        ++snapshot_.replaced_requests;
    }
    snapshot_.queued_requests = queue_.size();
    condition_.notify_one();
    return true;
}

bool SnapshotSender::isTransferAllowed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return started_ && !stopping_ && enabled_;
}

bool SnapshotSender::sendPacket(const SnapshotPacket &packet, std::string *error) {
    std::vector<uint8_t> bytes;
    if (!serializeSnapshotPacket(packet, &bytes, error)) return false;
    if (bytes.size() > static_cast<size_t>(mtu_ - 28)) {
        if (error) *error = "snapshot packet exceeds configured IPv4 MTU";
        return false;
    }
    pacer_->waitForTokens(RatePacer::wireBytesForUdpPayload(bytes.size()));
    const ssize_t written = send(socket_, bytes.data(), bytes.size(), 0);
    if (written != static_cast<ssize_t>(bytes.size())) {
        if (error) *error = "snapshot UDP send failed";
        return false;
    }
    return true;
}

bool SnapshotSender::waitForReply(uint32_t transfer_id, SnapshotPacket *reply) {
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config_.ack_timeout_ms);
    while (isTransferAllowed() && std::chrono::steady_clock::now() < deadline) {
        const std::chrono::milliseconds remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        pollfd descriptor;
        descriptor.fd = socket_;
        descriptor.events = POLLIN;
        descriptor.revents = 0;
        const int timeout = std::max(1, std::min(50, static_cast<int>(remaining.count())));
        const int status = poll(&descriptor, 1, timeout);
        if (status <= 0 || !(descriptor.revents & POLLIN)) continue;
        uint8_t buffer[2048];
        const ssize_t count = recv(socket_, buffer, sizeof(buffer), 0);
        if (count <= 0) continue;
        SnapshotPacket candidate;
        std::string ignored_error;
        if (!parseSnapshotPacket(buffer, static_cast<size_t>(count), &candidate, &ignored_error) ||
            candidate.transfer_id != transfer_id) {
            continue;
        }
        if (candidate.type == SNAPSHOT_ACK || candidate.type == SNAPSHOT_RESUME ||
            candidate.type == SNAPSHOT_COMPLETE || candidate.type == SNAPSHOT_ABORT) {
            if (reply) *reply = candidate;
            return true;
        }
    }
    return false;
}

bool SnapshotSender::encodeJpeg(const Request &request, std::vector<uint8_t> *jpeg,
                                int *width, int *height, std::string *error) const {
    if (!jpeg || !width || !height || !request.frame || request.frame->source_width <= 0 ||
        request.frame->source_height <= 0 ||
        request.frame->rgb.size() != static_cast<size_t>(request.frame->source_width) *
            request.frame->source_height * 3U) {
        if (error) *error = "invalid source RGB frame for snapshot JPEG";
        return false;
    }
    const cv::Mat rgb(request.frame->source_height, request.frame->source_width, CV_8UC3,
                      const_cast<uint8_t *>(request.frame->rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (request.crop.valid) {
        const int left = std::max(0, std::min(bgr.cols - 1, request.crop.left));
        const int top = std::max(0, std::min(bgr.rows - 1, request.crop.top));
        const int right = std::max(left + 1, std::min(bgr.cols, request.crop.right));
        const int bottom = std::max(top + 1, std::min(bgr.rows, request.crop.bottom));
        bgr = bgr(cv::Rect(left, top, right - left, bottom - top)).clone();
    }
    if (config_.max_width > 0 && (bgr.cols > config_.max_width || bgr.rows > config_.max_height)) {
        const double scale = std::min(static_cast<double>(config_.max_width) / bgr.cols,
                                      static_cast<double>(config_.max_height) / bgr.rows);
        const int resized_width = std::max(1, static_cast<int>(bgr.cols * scale + 0.5));
        const int resized_height = std::max(1, static_cast<int>(bgr.rows * scale + 0.5));
        cv::resize(bgr, bgr, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_AREA);
    }
    if (config_.rotate_ccw) {
        cv::Mat rotated;
        cv::rotate(bgr, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        bgr = rotated;
    }
    std::vector<int> jpeg_options;
    jpeg_options.push_back(cv::IMWRITE_JPEG_QUALITY);
    jpeg_options.push_back(config_.jpeg_quality);
    // Entropy-table optimization is lossless and normally trims several
    // percent from a JPEG before it consumes the 60 kbps wire budget.
    jpeg_options.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
    jpeg_options.push_back(1);
    if (!cv::imencode(".jpg", bgr, *jpeg, jpeg_options) || jpeg->empty()) {
        if (error) *error = "OpenCV JPEG encoding failed";
        return false;
    }
    if (jpeg->size() > 0xffffffffU || bgr.cols > 65535 || bgr.rows > 65535) {
        if (error) *error = "snapshot JPEG exceeds protocol dimensions or byte limit";
        return false;
    }
    *width = bgr.cols;
    *height = bgr.rows;
    return true;
}

SnapshotSender::TransferResult SnapshotSender::transfer(const Request &request,
                                                        const std::vector<uint8_t> &jpeg,
                                                        int width, int height,
                                                        std::string *error) {
    if (!isTransferAllowed()) return TRANSFER_CANCELLED;
    const uint32_t transfer_id = next_transfer_id_++;
    SnapshotPacket base;
    base.transfer_id = transfer_id;
    base.total_bytes = static_cast<uint32_t>(jpeg.size());
    base.width = static_cast<uint16_t>(width);
    base.height = static_cast<uint16_t>(height);
    base.class_mask = request.class_mask;
    base.crc32 = snapshotCrc32(jpeg.data(), jpeg.size());

    SnapshotPacket start = base;
    start.type = SNAPSHOT_START;
    SnapshotPacket reply;
    bool started = false;
    for (int attempt = 0; attempt < config_.max_retries && !started; ++attempt) {
        if (!isTransferAllowed()) return TRANSFER_CANCELLED;
        if (!sendPacket(start, error)) return TRANSFER_FAILED;
        if (attempt > 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++snapshot_.retransmitted_packets;
        }
        if (waitForReply(transfer_id, &reply)) {
            if (reply.type == SNAPSHOT_ABORT) {
                if (error) *error = "snapshot receiver aborted transfer";
                return TRANSFER_FAILED;
            }
            if ((reply.type == SNAPSHOT_RESUME || reply.type == SNAPSHOT_ACK) &&
                reply.offset <= base.total_bytes) {
                started = true;
            }
        }
    }
    if (!started) {
        if (!isTransferAllowed()) return TRANSFER_CANCELLED;
        if (error) *error = "snapshot receiver did not acknowledge START";
        return TRANSFER_FAILED;
    }
    uint32_t offset = reply.offset;
    for (;;) {
        while (offset < base.total_bytes) {
            const size_t available = static_cast<size_t>(base.total_bytes - offset);
            const size_t payload_bytes = std::min<size_t>(config_.chunk_payload_bytes, available);
            SnapshotPacket data = base;
            data.type = SNAPSHOT_DATA;
            data.offset = offset;
            data.payload.assign(jpeg.begin() + offset, jpeg.begin() + offset + payload_bytes);
            data.payload_length = static_cast<uint16_t>(payload_bytes);
            bool advanced = false;
            for (int attempt = 0; attempt < config_.max_retries && !advanced; ++attempt) {
                if (!isTransferAllowed()) return TRANSFER_CANCELLED;
                if (!sendPacket(data, error)) return TRANSFER_FAILED;
                if (attempt > 0) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++snapshot_.retransmitted_packets;
                }
                if (waitForReply(transfer_id, &reply)) {
                    if (reply.type == SNAPSHOT_ABORT) {
                        if (error) *error = "snapshot receiver aborted transfer";
                        return TRANSFER_FAILED;
                    }
                    if ((reply.type == SNAPSHOT_ACK || reply.type == SNAPSHOT_RESUME) &&
                        reply.offset > offset && reply.offset <= base.total_bytes) {
                        offset = reply.offset;
                        advanced = true;
                    }
                }
            }
            if (!advanced) {
                if (!isTransferAllowed()) return TRANSFER_CANCELLED;
                if (error) *error = "snapshot receiver did not acknowledge JPEG data";
                return TRANSFER_FAILED;
            }
        }

        SnapshotPacket end = base;
        end.type = SNAPSHOT_END;
        end.offset = base.total_bytes;
        bool ended = false;
        for (int attempt = 0; attempt < config_.max_retries && !ended; ++attempt) {
            if (!isTransferAllowed()) return TRANSFER_CANCELLED;
            if (!sendPacket(end, error)) return TRANSFER_FAILED;
            if (attempt > 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                ++snapshot_.retransmitted_packets;
            }
            if (!waitForReply(transfer_id, &reply)) continue;
            if (reply.type == SNAPSHOT_COMPLETE && reply.offset == base.total_bytes) {
                return TRANSFER_COMPLETED;
            }
            if ((reply.type == SNAPSHOT_RESUME || reply.type == SNAPSHOT_ACK) &&
                reply.offset <= base.total_bytes) {
                offset = reply.offset;
                ended = true;
            } else if (reply.type == SNAPSHOT_ABORT) {
                if (error) *error = "snapshot receiver aborted transfer";
                return TRANSFER_FAILED;
            }
        }
        if (offset < base.total_bytes) continue;
        if (!isTransferAllowed()) return TRANSFER_CANCELLED;
        if (error) *error = "snapshot receiver did not confirm JPEG completion";
        return TRANSFER_FAILED;
    }
}

void SnapshotSender::workerLoop() {
    for (;;) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || (enabled_ && !queue_.empty()); });
            if (stopping_) break;
            if (!enabled_ || queue_.empty()) continue;
            if (has_last_transfer_started_) {
                const std::chrono::steady_clock::time_point earliest = last_transfer_started_ +
                    std::chrono::milliseconds(config_.min_interval_ms);
                if (std::chrono::steady_clock::now() < earliest) {
                    condition_.wait_until(lock, earliest, [this] { return stopping_ || !enabled_; });
                    if (stopping_) break;
                    if (!enabled_) continue;
                }
            }
            if (queue_.empty()) continue;
            request = queue_[0];
            queue_.clear();
            snapshot_.queued_requests = 0;
            transferring_ = true;
            snapshot_.transferring = true;
            last_transfer_started_ = std::chrono::steady_clock::now();
            has_last_transfer_started_ = true;
        }

        std::vector<uint8_t> jpeg;
        int width = 0;
        int height = 0;
        std::string transfer_error;
        TransferResult result = TRANSFER_FAILED;
        if (isTransferAllowed() && encodeJpeg(request, &jpeg, &width, &height, &transfer_error)) {
            result = transfer(request, jpeg, width, height, &transfer_error);
        } else if (!isTransferAllowed()) {
            result = TRANSFER_CANCELLED;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            transferring_ = false;
            snapshot_.transferring = false;
            if (result == TRANSFER_COMPLETED) {
                ++snapshot_.completed_transfers;
                snapshot_.sent_jpeg_bytes += jpeg.size();
                snapshot_.last_transfer_bytes = jpeg.size();
                snapshot_.last_transfer_width = width;
                snapshot_.last_transfer_height = height;
                snapshot_.last_error.clear();
                std::fprintf(stderr, "Snapshot complete: %dx%d %zu bytes, classes=0x%04x\n",
                             width, height, jpeg.size(), request.class_mask);
            } else if (result == TRANSFER_CANCELLED) {
                ++snapshot_.cancelled_transfers;
            } else {
                ++snapshot_.failed_transfers;
                snapshot_.last_error = transfer_error;
                std::fprintf(stderr, "Snapshot transfer deferred: %s\n", transfer_error.c_str());
            }
            condition_.notify_all();
        }
    }
}

void SnapshotSender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stopping_ = true;
        enabled_ = false;
        queue_.clear();
        condition_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    if (socket_ >= 0) {
        close(socket_);
        socket_ = -1;
    }
    delete destination_;
    destination_ = NULL;
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
    transferring_ = false;
    snapshot_.enabled = false;
    snapshot_.transferring = false;
    snapshot_.queued_requests = 0;
}

SnapshotSenderSnapshot SnapshotSender::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SnapshotSenderSnapshot value = snapshot_;
    value.enabled = enabled_;
    value.transferring = transferring_;
    value.queued_requests = queue_.size();
    return value;
}

}  // namespace roi_h265

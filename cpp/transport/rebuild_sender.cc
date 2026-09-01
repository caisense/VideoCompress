#include "transport/rebuild_sender.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include <arpa/inet.h>
#include <netdb.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include "transport/snapshot_protocol.h"

namespace roi_h265 {
namespace {

float intersectionOverUnion(const BBox &left, const BBox &right) {
    const int x0 = std::max(left.left, right.left);
    const int y0 = std::max(left.top, right.top);
    const int x1 = std::min(left.right, right.right);
    const int y1 = std::min(left.bottom, right.bottom);
    const int intersection = std::max(0, x1 - x0) * std::max(0, y1 - y0);
    const int left_area = std::max(0, left.right - left.left) *
        std::max(0, left.bottom - left.top);
    const int right_area = std::max(0, right.right - right.left) *
        std::max(0, right.bottom - right.top);
    const int union_area = left_area + right_area - intersection;
    return union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
}

uint16_t clampU16(int value) {
    return static_cast<uint16_t>(std::max(0, std::min(65535, value)));
}

// FrameMeta::capture_time_us is produced from steady_clock.  Keep sender
// scheduling and latency samples on the same clock domain; system_clock would
// make every completed reference look billions of microseconds old.
uint64_t steadyNowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t percentile(const std::vector<uint64_t> &values, double fraction) {
    if (values.empty()) return 0;
    std::vector<uint64_t> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    const size_t index = std::min(sorted.size() - 1,
        static_cast<size_t>(std::ceil(fraction * sorted.size()) - 1.0));
    return sorted[index];
}

std::vector<uint8_t> runLengthEncodeMask(const cv::Mat &mask) {
    std::vector<uint8_t> output;
    if (mask.empty()) return output;
    const size_t count = mask.total();
    const uint8_t *pixels = mask.ptr<uint8_t>(0);
    size_t cursor = 0;
    while (cursor < count) {
        const uint8_t value = pixels[cursor] ? 1U : 0U;
        size_t run = 1;
        while (cursor + run < count && run < 255U &&
               (pixels[cursor + run] ? 1U : 0U) == value) {
            ++run;
        }
        output.push_back(static_cast<uint8_t>(run));
        output.push_back(value);
        cursor += run;
    }
    return output;
}

}  // namespace

struct RebuildSender::sockaddr_storage_holder {
    sockaddr_storage address;
    socklen_t length;
};

RebuildSenderSnapshot::RebuildSenderSnapshot()
    : enabled(false), transmitting(false), queued_requests(0), submitted_requests(0),
      replaced_requests(0), state_packets(0), patch_transfers(0), patch_packets(0),
      parity_packets(0), patch_jpeg_bytes(0), sent_wire_bytes(0),
      cancelled_requests(0), last_reference_generation(0),
      last_state_capture_time_us(0), last_state_send_start_time_us(0),
      last_state_send_time_us(0), last_state_pacer_delay_us(0),
      last_state_capture_to_send_us(0), state_capture_to_send_p50_us(0),
      state_capture_to_send_p95_us(0),
      last_reference_capture_time_us(0), last_reference_encode_finish_time_us(0),
      last_reference_queue_enter_time_us(0),
      last_reference_first_packet_send_time_us(0),
      last_reference_last_packet_send_time_us(0),
      last_reference_queue_delay_us(0), last_reference_capture_to_send_us(0),
      reference_capture_to_send_p50_us(0),
      reference_capture_to_send_p95_us(0), last_reference_delivery_us(0),
      reference_delivery_p50_us(0), reference_delivery_p95_us(0),
      last_reference_interval_us(0),
      reference_interval_p50_us(0), reference_interval_p95_us(0),
      reference_interval_max_us(0), last_reference_blob_bytes(0),
      last_reference_chunk_count(0), last_reference_fec_bytes(0),
      last_refresh_track_id(0), last_reference_capture_age_ms(0),
      last_reference_ready_age_ms(0), last_refresh_threshold_ms(0),
      last_estimated_delivery_ms(0), last_refresh_deadline_ms(0),
      last_refresh_quantum_ms(0), last_refresh_decision_start(false),
      last_refresh_reason("NO_ACTIVE_TARGET") {}

RebuildSender::Track::Track()
    : id(0), class_id(-1), confidence(0.0f), last_seen_frame(0),
      reference_generation(0), has_reference(false), last_reference_capture_time_us(0),
      last_reference_ready_time_us(0) {}

RebuildSender::RebuildSender(const RebuildConfig &config, const std::string &host, int mtu,
                             const std::shared_ptr<RatePacer> &pacer)
    : config_(config), host_(host), mtu_(mtu), pacer_(pacer), socket_(-1),
      destination_(NULL), started_(false), stopping_(false), enabled_(false),
      transmitting_(false), failed_(false), active_generation_(0),
      has_generation_(false), next_track_id_(1), next_transfer_id_(1),
      next_packet_sequence_(1) {}

RebuildSender::~RebuildSender() { stop(); }

bool RebuildSender::openSocket(std::string *error) {
    if (socket_ >= 0) return true;
    char port_text[16];
    std::snprintf(port_text, sizeof(port_text), "%d", config_.udp_port);
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo *result = NULL;
    if (getaddrinfo(host_.c_str(), port_text, &hints, &result) != 0 || !result) {
        if (error) *error = "cannot resolve rebuild UDP destination";
        return false;
    }
    socket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (socket_ < 0) {
        freeaddrinfo(result);
        if (error) *error = "cannot create rebuild UDP socket";
        return false;
    }
    destination_ = new sockaddr_storage_holder;
    std::memset(destination_, 0, sizeof(*destination_));
    std::memcpy(&destination_->address, result->ai_addr, result->ai_addrlen);
    destination_->length = static_cast<socklen_t>(result->ai_addrlen);
    freeaddrinfo(result);
    return true;
}

bool RebuildSender::start(std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return true;
    if (!pacer_) {
        if (error) *error = "rebuild sender requires the shared video pacer";
        return false;
    }
    if (!openSocket(error)) return false;
    stopping_ = false;
    failed_ = false;
    started_ = true;
    worker_ = std::thread(&RebuildSender::workerLoop, this);
    return true;
}

void RebuildSender::setEnabled(bool enabled) {
    std::unique_lock<std::mutex> lock(mutex_);
    enabled_ = enabled;
    snapshot_.enabled = enabled;
    if (!enabled_) {
        queue_.clear();
        snapshot_.queued_requests = 0;
        condition_.notify_all();
        condition_.wait(lock, [this] { return !transmitting_; });
        pending_reference_.clear();
        tracks_.clear();
        has_generation_ = false;
        next_track_id_ = 1;
    }
    condition_.notify_all();
}

bool RebuildSender::submit(const std::shared_ptr<FramePacket> &frame,
                           const SegResult &segmentation, uint8_t profile_generation,
                           int source_fps, std::string *error) {
    if (!frame || frame->source_width <= 0 || frame->source_height <= 0 ||
        frame->rgb.size() != static_cast<size_t>(frame->source_width) *
            frame->source_height * 3U || source_fps <= 0) {
        if (error) *error = "invalid rebuild source frame";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stopping_ || failed_) {
        if (error) *error = "rebuild sender is not running";
        return false;
    }
    if (!enabled_) return true;
    Request request;
    request.frame = frame;
    request.segmentation = segmentation;
    request.generation = profile_generation;
    request.source_fps = source_fps;
    ++snapshot_.submitted_requests;
    if (queue_.empty()) queue_.push_back(request);
    else {
        queue_[0] = request;
        ++snapshot_.replaced_requests;
    }
    snapshot_.queued_requests = queue_.size();
    condition_.notify_one();
    return true;
}

bool RebuildSender::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return started_ && !stopping_ && enabled_ && !failed_;
}

bool RebuildSender::sendPacket(uint8_t type, const Request &request,
                               const std::vector<uint8_t> &payload, uint16_t flags,
                               std::string *error) {
    if (!isEnabled()) {
        if (error) *error = "rebuild transfer cancelled";
        return false;
    }
    RebuildPacket packet;
    packet.type = type;
    packet.profile = static_cast<uint8_t>(RATE_PROFILE_REBUILD);
    packet.generation = request.generation;
    packet.flags = flags;
    packet.sequence = next_packet_sequence_++;
    packet.frame_id = static_cast<uint32_t>(request.frame->meta.frame_id);
    packet.pts_ms = static_cast<uint32_t>(request.frame->meta.pts_us / 1000U);
    packet.payload = payload;
    std::vector<uint8_t> bytes;
    if (!serializeRebuildPacket(packet, &bytes, error)) return false;
    if (bytes.size() > static_cast<size_t>(mtu_ - 28)) {
        if (error) *error = "rebuild packet exceeds configured IPv4 MTU";
        return false;
    }
    const size_t wire_bytes = RatePacer::wireBytesForUdpPayload(bytes.size());
    const uint64_t send_start_us = type == REBUILD_STATE ? steadyNowMicros() : 0;
    pacer_->waitForTokens(wire_bytes);
    if (!isEnabled()) {
        if (error) *error = "rebuild transfer cancelled";
        return false;
    }
    // Keep the UDP socket unconnected, matching UdpSender.  A receiver that is
    // started later may cause an ICMP port-unreachable response; connected UDP
    // would surface that transient as ECONNREFUSED on a subsequent send and
    // incorrectly terminate the whole A/V pipeline.
    const ssize_t written = sendto(socket_, bytes.data(), bytes.size(), 0,
        reinterpret_cast<const sockaddr *>(&destination_->address), destination_->length);
    if (written != static_cast<ssize_t>(bytes.size())) {
        if (error) *error = "rebuild UDP send failed";
        return false;
    }
    const uint64_t sent_us = type == REBUILD_STATE ? steadyNowMicros() : 0;
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.sent_wire_bytes += wire_bytes;
    if (type == REBUILD_STATE) {
        ++snapshot_.state_packets;
        const uint64_t capture_time_us = request.frame->meta.capture_time_us;
        const uint64_t capture_to_send_us = capture_time_us > 0 &&
            sent_us >= capture_time_us ? sent_us - capture_time_us : 0;
        snapshot_.last_state_capture_time_us = capture_time_us;
        snapshot_.last_state_send_start_time_us = send_start_us;
        snapshot_.last_state_send_time_us = sent_us;
        snapshot_.last_state_pacer_delay_us = sent_us >= send_start_us
            ? sent_us - send_start_us : 0;
        snapshot_.last_state_capture_to_send_us = capture_to_send_us;
        if (capture_to_send_us > 0) {
            state_capture_to_send_samples_us_.push_back(capture_to_send_us);
            if (state_capture_to_send_samples_us_.size() > 64U) {
                state_capture_to_send_samples_us_.erase(
                    state_capture_to_send_samples_us_.begin());
            }
            snapshot_.state_capture_to_send_p50_us = percentile(
                state_capture_to_send_samples_us_, 0.50);
            snapshot_.state_capture_to_send_p95_us = percentile(
                state_capture_to_send_samples_us_, 0.95);
        }
    }
    else if (type == REBUILD_PATCH_DATA) ++snapshot_.patch_packets;
    else if (type == REBUILD_PATCH_PARITY) ++snapshot_.parity_packets;
    return true;
}

std::vector<RebuildSender::ActiveTarget> RebuildSender::updateTracks(
        const Request &request) {
    if (!has_generation_ || request.generation != active_generation_) {
        pending_reference_.clear();
        tracks_.clear();
        next_track_id_ = 1;
        active_generation_ = request.generation;
        has_generation_ = true;
    }
    std::vector<size_t> candidates;
    for (size_t index = 0; index < request.segmentation.instances.size(); ++index) {
        const SegInstance &instance = request.segmentation.instances[index];
        if (isRelevantDetectionClass(instance.class_id) &&
            instance.confidence >= config_.min_confidence &&
            instance.bbox.right > instance.bbox.left &&
            instance.bbox.bottom > instance.bbox.top) {
            candidates.push_back(index);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&request](size_t left, size_t right) {
        return request.segmentation.instances[left].confidence >
            request.segmentation.instances[right].confidence;
    });
    if (candidates.size() > static_cast<size_t>(config_.max_targets)) {
        candidates.resize(static_cast<size_t>(config_.max_targets));
    }
    std::vector<bool> used(tracks_.size(), false);
    std::vector<ActiveTarget> active;
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
        const size_t instance_index = candidates[candidate_index];
        const SegInstance &instance = request.segmentation.instances[instance_index];
        size_t best = tracks_.size();
        float best_iou = 0.25f;
        for (size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            if (used[track_index] || tracks_[track_index].class_id != instance.class_id) continue;
            const float iou = intersectionOverUnion(tracks_[track_index].bbox, instance.bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = track_index;
            }
        }
        if (best == tracks_.size()) {
            Track track;
            track.id = next_track_id_++;
            if (next_track_id_ == 0) next_track_id_ = 1;
            track.class_id = instance.class_id;
            track.bbox = instance.bbox;
            track.confidence = instance.confidence;
            track.last_seen_frame = request.frame->meta.frame_id;
            tracks_.push_back(track);
            used.push_back(true);
            best = tracks_.size() - 1;
        } else {
            used[best] = true;
            tracks_[best].bbox = instance.bbox;
            tracks_[best].confidence = instance.confidence;
            tracks_[best].last_seen_frame = request.frame->meta.frame_id;
        }
        ActiveTarget target;
        target.instance_index = instance_index;
        target.track_index = best;
        active.push_back(target);
    }
    // Retain tracks briefly for IoU matching, but bound storage even if IDs
    // churn under occlusion.  Only current detections are transmitted in state.
    const uint64_t oldest = request.frame->meta.frame_id >
        static_cast<uint64_t>(request.source_fps * 2)
        ? request.frame->meta.frame_id - static_cast<uint64_t>(request.source_fps * 2) : 0;
    if (tracks_.size() > 64U) {
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [oldest](const Track &track) {
            return track.last_seen_frame < oldest;
        }), tracks_.end());
        // Indices may have changed; this rare path is handled by rebuilding
        // associations on the next frame instead of risking a wrong overlay.
        active.clear();
    }
    return active;
}

bool RebuildSender::sendState(const Request &request,
                              const std::vector<ActiveTarget> &active,
                              std::string *error) {
    RebuildState state;
    state.source_width = clampU16(request.frame->source_width);
    state.source_height = clampU16(request.frame->source_height);
    state.output_width = clampU16(config_.output_width);
    state.output_height = clampU16(config_.output_height);
    state.source_fps = static_cast<uint8_t>(std::min(255, request.source_fps));
    state.output_fps = static_cast<uint8_t>(std::min(255, config_.output_fps));
    state.flags = config_.parity ? 1U : 0U;
    for (size_t index = 0; index < active.size(); ++index) {
        const Track &track = tracks_[active[index].track_index];
        RebuildTargetState target;
        target.track_id = track.id;
        target.class_id = static_cast<uint8_t>(std::max(0, std::min(255, track.class_id)));
        target.confidence_percent = static_cast<uint8_t>(std::max(0, std::min(100,
            static_cast<int>(track.confidence * 100.0f + 0.5f))));
        target.left = clampU16(std::max(0, std::min(request.frame->source_width, track.bbox.left)));
        target.top = clampU16(std::max(0, std::min(request.frame->source_height, track.bbox.top)));
        target.right = clampU16(std::max(0, std::min(request.frame->source_width, track.bbox.right)));
        target.bottom = clampU16(std::max(0, std::min(request.frame->source_height, track.bbox.bottom)));
        if (target.right <= target.left || target.bottom <= target.top) continue;
        target.reference_generation = track.reference_generation;
        target.flags = track.has_reference ? 1U : 0U;
        state.targets.push_back(target);
    }
    std::vector<uint8_t> payload;
    if (!serializeRebuildState(state, &payload, error)) return false;
    return sendPacket(REBUILD_STATE, request, payload, 0, error);
}

bool RebuildSender::buildReference(const Request &request, const ActiveTarget &active,
                                   RebuildPatchFragment *metadata,
                                   std::vector<uint8_t> *blob, size_t *jpeg_bytes,
                                   std::string *error) const {
    if (!metadata || !blob || !jpeg_bytes || active.instance_index >=
        request.segmentation.instances.size() || active.track_index >= tracks_.size()) {
        if (error) *error = "invalid rebuild reference target";
        return false;
    }
    const SegInstance &instance = request.segmentation.instances[active.instance_index];
    const Track &track = tracks_[active.track_index];
    const int box_width = std::max(1, instance.bbox.right - instance.bbox.left);
    const int box_height = std::max(1, instance.bbox.bottom - instance.bbox.top);
    const int margin_x = (box_width * config_.crop_margin_percent + 99) / 100;
    const int margin_y = (box_height * config_.crop_margin_percent + 99) / 100;
    const int left = std::max(0, instance.bbox.left - margin_x);
    const int top = std::max(0, instance.bbox.top - margin_y);
    const int right = std::min(request.frame->source_width, instance.bbox.right + margin_x);
    const int bottom = std::min(request.frame->source_height, instance.bbox.bottom + margin_y);
    if (right <= left || bottom <= top) {
        if (error) *error = "empty rebuild reference crop";
        return false;
    }
    const cv::Mat rgb(request.frame->source_height, request.frame->source_width, CV_8UC3,
                      const_cast<uint8_t *>(request.frame->rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(rgb(cv::Rect(left, top, right - left, bottom - top)),
                 bgr, cv::COLOR_RGB2BGR);
    double scale = std::min(1.0, static_cast<double>(config_.patch_max_side) /
        std::max(bgr.cols, bgr.rows));
    int quality = config_.patch_jpeg_quality;
    std::vector<uint8_t> jpeg;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const int resized_width = std::max(16, static_cast<int>(bgr.cols * scale + 0.5));
        const int resized_height = std::max(16, static_cast<int>(bgr.rows * scale + 0.5));
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
                   scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
        std::vector<int> options;
        options.push_back(cv::IMWRITE_JPEG_QUALITY);
        options.push_back(quality);
        options.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
        options.push_back(1);
        if (!cv::imencode(".jpg", resized, jpeg, options) || jpeg.empty()) {
            if (error) *error = "OpenCV rebuild JPEG encoding failed";
            return false;
        }
        metadata->jpeg_width = clampU16(resized_width);
        metadata->jpeg_height = clampU16(resized_height);
        if (jpeg.size() <= static_cast<size_t>(config_.patch_max_bytes)) break;
        if (quality > 40) quality = std::max(40, quality - 7);
        else scale *= 0.82;
    }
    if (jpeg.size() > static_cast<size_t>(config_.patch_max_bytes)) {
        if (error) *error = "rebuild JPEG cannot meet configured byte cap";
        return false;
    }

    // 64x64 retains the subject silhouette at the receiver.  The previous
    // 32x32 mask made the low-resolution edge cover stale background pixels,
    // which showed up as a coloured halo/ghost around moving targets.
    const int mask_side = 64;
    // Never treat the whole crop as foreground when a segmentation mask is
    // unavailable.  Painting the margin as foreground copies stale reference
    // background into the current frame and is indistinguishable from a
    // floating/ghosted object.  The detector box is the conservative fallback.
    cv::Mat source_mask(request.frame->source_height, request.frame->source_width, CV_8UC1,
                        cv::Scalar(0));
    if (!instance.mask.empty() && instance.mask_width > 0 && instance.mask_height > 0 &&
        instance.mask.size() == static_cast<size_t>(instance.mask_width) * instance.mask_height) {
        const cv::Mat compact(instance.mask_height, instance.mask_width, CV_8UC1,
                              const_cast<uint8_t *>(instance.mask.data()));
        cv::resize(compact, source_mask,
                   cv::Size(request.frame->source_width, request.frame->source_height),
                   0.0, 0.0, cv::INTER_NEAREST);
    } else {
        const int box_left = std::max(0, std::min(request.frame->source_width,
            instance.bbox.left));
        const int box_top = std::max(0, std::min(request.frame->source_height,
            instance.bbox.top));
        const int box_right = std::max(0, std::min(request.frame->source_width,
            instance.bbox.right));
        const int box_bottom = std::max(0, std::min(request.frame->source_height,
            instance.bbox.bottom));
        if (box_right > box_left && box_bottom > box_top) {
            source_mask(cv::Rect(box_left, box_top,
                                 box_right - box_left, box_bottom - box_top)).setTo(255);
        }
    }
    cv::Mat crop_mask = source_mask(cv::Rect(left, top, right - left, bottom - top));
    cv::Mat small_mask;
    cv::resize(crop_mask, small_mask, cv::Size(mask_side, mask_side),
               0.0, 0.0, cv::INTER_NEAREST);
    cv::threshold(small_mask, small_mask, 0, 255, cv::THRESH_BINARY);
    const std::vector<uint8_t> mask_rle = runLengthEncodeMask(small_mask);
    if (mask_rle.size() > 65535U || mask_rle.size() + jpeg.size() > 65535U) {
        if (error) *error = "rebuild mask/JPEG blob exceeds protocol limit";
        return false;
    }
    blob->clear();
    blob->reserve(mask_rle.size() + jpeg.size());
    blob->insert(blob->end(), mask_rle.begin(), mask_rle.end());
    blob->insert(blob->end(), jpeg.begin(), jpeg.end());
    *jpeg_bytes = jpeg.size();
    metadata->track_id = track.id;
    metadata->reference_generation = static_cast<uint16_t>(track.reference_generation + 1U);
    if (metadata->reference_generation == 0) metadata->reference_generation = 1;
    metadata->left = clampU16(left);
    metadata->top = clampU16(top);
    metadata->right = clampU16(right);
    metadata->bottom = clampU16(bottom);
    // Preserve the detector box at reference capture time.  The receiver
    // uses this box, rather than assuming the crop is centred, to map the
    // reference crop onto the current target (including edge-clipped crops).
    const int reference_left = std::max(0, std::min(request.frame->source_width,
        instance.bbox.left));
    const int reference_top = std::max(0, std::min(request.frame->source_height,
        instance.bbox.top));
    const int reference_right = std::max(0, std::min(request.frame->source_width,
        instance.bbox.right));
    const int reference_bottom = std::max(0, std::min(request.frame->source_height,
        instance.bbox.bottom));
    if (reference_right <= reference_left || reference_bottom <= reference_top) {
        if (error) *error = "invalid rebuild reference detector box";
        return false;
    }
    metadata->reference_left = clampU16(reference_left);
    metadata->reference_top = clampU16(reference_top);
    metadata->reference_right = clampU16(reference_right);
    metadata->reference_bottom = clampU16(reference_bottom);
    metadata->mask_width = mask_side;
    metadata->mask_height = mask_side;
    metadata->blob_size = static_cast<uint32_t>(blob->size());
    metadata->mask_rle_bytes = static_cast<uint16_t>(mask_rle.size());
    metadata->chunk_bytes = static_cast<uint16_t>(config_.patch_chunk_bytes);
    return true;
}

int RebuildSender::estimatedReferenceDeliveryMs() const {
    // Before the first completed transfer, use a conservative mask allowance
    // in addition to the configured JPEG cap.  Afterwards the last complete
    // blob size gives a much tighter estimate without pretending to exceed
    // the shared RatePacer budget.
    uint64_t last_blob_bytes = 0;
    uint64_t observed_delivery_p95_us = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_blob_bytes = snapshot_.last_reference_blob_bytes;
        observed_delivery_p95_us = snapshot_.reference_delivery_p95_us;
    }
    const size_t estimated_blob = std::min<size_t>(
        12U * 1024U, last_blob_bytes > 0
            ? static_cast<size_t>(last_blob_bytes)
            : static_cast<size_t>(config_.patch_max_bytes) + 1024U);
    const size_t chunk = static_cast<size_t>(std::max(1, config_.patch_chunk_bytes));
    const size_t data_fragments = (estimated_blob + chunk - 1U) / chunk;
    const size_t packet_count = data_fragments +
        (config_.parity && data_fragments > 1U ? 1U : 0U);
    const size_t udp_payload = kRebuildHeaderBytes + kRebuildPatchFragmentHeaderBytes + chunk;
    const size_t wire_bytes = packet_count * RatePacer::wireBytesForUdpPayload(udp_payload);
    const int bitrate = std::max(1, pacer_->bitrateBps());
    const int theoretical_ms = static_cast<int>(
        (wire_bytes * 8000U + static_cast<size_t>(bitrate) - 1U) /
        static_cast<size_t>(bitrate));
    const int observed_ms = observed_delivery_p95_us > 0
        ? static_cast<int>((observed_delivery_p95_us + 999U) / 1000U) : 0;
    return std::max(theoretical_ms, observed_ms);
}

void RebuildSender::recordRefreshDecision(
        uint16_t track_id, const RebuildRefreshDecision &decision) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.last_refresh_track_id = track_id;
    snapshot_.last_reference_capture_age_ms = decision.capture_age_ms;
    snapshot_.last_reference_ready_age_ms = decision.ready_age_ms;
    snapshot_.last_refresh_threshold_ms = decision.refresh_threshold_ms;
    snapshot_.last_estimated_delivery_ms = decision.estimated_delivery_ms;
    snapshot_.last_refresh_deadline_ms = decision.next_deadline_ms;
    snapshot_.last_refresh_quantum_ms = decision.scheduling_quantum_ms;
    snapshot_.last_refresh_decision_start = decision.start;
    snapshot_.last_refresh_reason = rebuildRefreshReasonName(decision.reason);
}

bool RebuildSender::beginReference(const Request &request, const ActiveTarget &active,
                                   std::string *error) {
    if (pending_reference_.active) {
        if (error) *error = "rebuild reference transfer already active";
        return false;
    }
    RebuildPatchFragment metadata;
    std::vector<uint8_t> blob;
    size_t jpeg_bytes = 0;
    if (!buildReference(request, active, &metadata, &blob, &jpeg_bytes, error)) return false;
    const size_t chunk_bytes = static_cast<size_t>(metadata.chunk_bytes);
    const size_t fragment_count = (blob.size() + chunk_bytes - 1U) / chunk_bytes;
    if (fragment_count == 0 || fragment_count > 255U) {
        if (error) *error = "rebuild reference needs too many fragments";
        return false;
    }
    metadata.transfer_id = next_transfer_id_++;
    metadata.data_fragments = static_cast<uint8_t>(fragment_count);
    std::vector<uint8_t> parity(chunk_bytes, 0);
    for (size_t index = 0; index < fragment_count; ++index) {
        const size_t begin = index * chunk_bytes;
        const size_t count = std::min(chunk_bytes, blob.size() - begin);
        for (size_t byte = 0; byte < count; ++byte) parity[byte] ^= blob[begin + byte];
    }
    pending_reference_.active = true;
    pending_reference_.request = request;
    pending_reference_.metadata = metadata;
    pending_reference_.blob.swap(blob);
    pending_reference_.parity.swap(parity);
    pending_reference_.next_data_index = 0;
    pending_reference_.jpeg_bytes = jpeg_bytes;
    pending_reference_.capture_time_us = request.frame->meta.capture_time_us;
    pending_reference_.encode_finish_time_us = steadyNowMicros();
    pending_reference_.queue_enter_time_us = pending_reference_.encode_finish_time_us;
    pending_reference_.chunk_count = fragment_count;
    pending_reference_.fec_bytes = config_.parity && fragment_count > 1U
        ? pending_reference_.parity.size() : 0U;
    // Put parity before data.  The receiver can then recover one missing data
    // packet, while an intact transfer completes on its final data packet and
    // cannot be followed by a phantom parity-only incomplete transfer.
    pending_reference_.parity_sent = !config_.parity || fragment_count <= 1U;
    return true;
}

bool RebuildSender::sendPendingReferencePacket(std::string *error) {
    if (!pending_reference_.active) return true;
    RebuildPatchFragment metadata = pending_reference_.metadata;
    uint8_t type = REBUILD_PATCH_DATA;
    uint16_t flags = 0;
    if (!pending_reference_.parity_sent) {
        metadata.fragment_index = metadata.data_fragments;
        metadata.data = pending_reference_.parity;
        type = REBUILD_PATCH_PARITY;
        flags = 1U;
    } else {
        const size_t index = pending_reference_.next_data_index;
        if (index >= metadata.data_fragments) {
            if (error) *error = "invalid rebuild pending fragment index";
            return false;
        }
        const size_t begin = index * metadata.chunk_bytes;
        const size_t count = std::min<size_t>(
            metadata.chunk_bytes, pending_reference_.blob.size() - begin);
        metadata.fragment_index = static_cast<uint8_t>(index);
        metadata.data.assign(pending_reference_.blob.begin() + begin,
                             pending_reference_.blob.begin() + begin + count);
    }
    std::vector<uint8_t> payload;
    if (!serializeRebuildPatchFragment(metadata, &payload, error) ||
        !sendPacket(type, pending_reference_.request, payload, flags, error)) {
        return false;
    }
    const uint64_t packet_sent_us = steadyNowMicros();
    if (pending_reference_.first_packet_send_time_us == 0) {
        pending_reference_.first_packet_send_time_us = packet_sent_us;
    }
    pending_reference_.last_packet_send_time_us = packet_sent_us;
    if (type == REBUILD_PATCH_PARITY) pending_reference_.parity_sent = true;
    else ++pending_reference_.next_data_index;

    if (!pending_reference_.parity_sent ||
        pending_reference_.next_data_index < pending_reference_.metadata.data_fragments) {
        return true;
    }

    const uint16_t track_id = pending_reference_.metadata.track_id;
    for (size_t index = 0; index < tracks_.size(); ++index) {
        if (tracks_[index].id != track_id) continue;
        tracks_[index].reference_generation =
            pending_reference_.metadata.reference_generation;
        tracks_[index].has_reference = true;
        tracks_[index].last_reference_capture_time_us =
            pending_reference_.capture_time_us;
        tracks_[index].last_reference_ready_time_us =
            pending_reference_.last_packet_send_time_us;
        break;
    }
    const uint64_t capture_to_send_us =
        pending_reference_.capture_time_us > 0 &&
        pending_reference_.last_packet_send_time_us >= pending_reference_.capture_time_us
            ? pending_reference_.last_packet_send_time_us - pending_reference_.capture_time_us : 0;
    const uint64_t delivery_us =
        pending_reference_.first_packet_send_time_us > 0 &&
        pending_reference_.last_packet_send_time_us >=
            pending_reference_.first_packet_send_time_us
            ? pending_reference_.last_packet_send_time_us -
                pending_reference_.first_packet_send_time_us : 0;
    const uint64_t queue_delay_us =
        pending_reference_.first_packet_send_time_us >= pending_reference_.queue_enter_time_us
            ? pending_reference_.first_packet_send_time_us - pending_reference_.queue_enter_time_us : 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++snapshot_.patch_transfers;
        snapshot_.patch_jpeg_bytes += pending_reference_.jpeg_bytes;
        snapshot_.last_reference_generation =
            pending_reference_.metadata.reference_generation;
        snapshot_.last_reference_capture_time_us = pending_reference_.capture_time_us;
        snapshot_.last_reference_encode_finish_time_us =
            pending_reference_.encode_finish_time_us;
        snapshot_.last_reference_queue_enter_time_us = pending_reference_.queue_enter_time_us;
        snapshot_.last_reference_first_packet_send_time_us =
            pending_reference_.first_packet_send_time_us;
        snapshot_.last_reference_last_packet_send_time_us =
            pending_reference_.last_packet_send_time_us;
        snapshot_.last_reference_queue_delay_us = queue_delay_us;
        snapshot_.last_reference_capture_to_send_us = capture_to_send_us;
        snapshot_.last_reference_delivery_us = delivery_us;
        reference_delivery_samples_us_.push_back(delivery_us);
        if (reference_delivery_samples_us_.size() > 64U) {
            reference_delivery_samples_us_.erase(reference_delivery_samples_us_.begin());
        }
        snapshot_.reference_delivery_p50_us = percentile(
            reference_delivery_samples_us_, 0.50);
        snapshot_.reference_delivery_p95_us = percentile(
            reference_delivery_samples_us_, 0.95);
        snapshot_.last_reference_blob_bytes = pending_reference_.blob.size();
        snapshot_.last_reference_chunk_count = pending_reference_.chunk_count;
        snapshot_.last_reference_fec_bytes = pending_reference_.fec_bytes;
        const uint16_t completed_track_id = pending_reference_.metadata.track_id;
        const uint64_t previous_capture_time_us =
            previous_reference_capture_times_us_[completed_track_id];
        if (previous_capture_time_us > 0 &&
                pending_reference_.capture_time_us > previous_capture_time_us) {
            const uint64_t interval_us = pending_reference_.capture_time_us -
                previous_capture_time_us;
            reference_interval_samples_us_.push_back(interval_us);
            if (reference_interval_samples_us_.size() > 64U) {
                reference_interval_samples_us_.erase(reference_interval_samples_us_.begin());
            }
            snapshot_.last_reference_interval_us = interval_us;
            snapshot_.reference_interval_p50_us = percentile(
                reference_interval_samples_us_, 0.50);
            snapshot_.reference_interval_p95_us = percentile(
                reference_interval_samples_us_, 0.95);
            snapshot_.reference_interval_max_us = *std::max_element(
                reference_interval_samples_us_.begin(), reference_interval_samples_us_.end());
        }
        previous_reference_capture_times_us_[completed_track_id] =
            pending_reference_.capture_time_us;
        if (capture_to_send_us > 0) {
            reference_capture_to_send_samples_us_.push_back(capture_to_send_us);
            if (reference_capture_to_send_samples_us_.size() > 64U) {
                reference_capture_to_send_samples_us_.erase(
                    reference_capture_to_send_samples_us_.begin());
            }
            snapshot_.reference_capture_to_send_p50_us = percentile(
                reference_capture_to_send_samples_us_, 0.50);
            snapshot_.reference_capture_to_send_p95_us = percentile(
                reference_capture_to_send_samples_us_, 0.95);
        }
    }
    pending_reference_.clear();
    return true;
}

bool RebuildSender::process(const Request &request, std::string *error) {
    const std::vector<ActiveTarget> active = updateTracks(request);
    if (!sendState(request, active, error)) return false;
    const int packets_per_frame = std::max(1, std::min(2,
        config_.patch_packets_per_frame));
    const auto advance_reference = [this, error, packets_per_frame, &request, &active]() {
        // A small bounded burst shortens a two-fragment transfer without
        // allowing the JPEG queue to monopolise the shared pacer.
        bool completed = false;
        for (int index = 0; index < packets_per_frame && pending_reference_.active;
             ++index) {
            const bool was_active = pending_reference_.active;
            if (!sendPendingReferencePacket(error)) return false;
            completed = completed || (was_active && !pending_reference_.active);
        }
        if (completed) {
            // The first STATE of this source frame was sent before the last
            // reference fragment.  Publish the new generation immediately
            // after completion instead of forcing the PC to wait an entire
            // additional 6 fps source interval before the crop is eligible.
            // This small control packet shares the same RatePacer and does
            // not bypass the physical link budget.
            if (!sendState(request, active, error)) return false;
        }
        return true;
    };
    if (pending_reference_.active) {
        bool still_active = pending_reference_.request.generation == request.generation;
        for (size_t index = 0; still_active && index < active.size(); ++index) {
            if (tracks_[active[index].track_index].id ==
                    pending_reference_.metadata.track_id) {
                const Track &track = tracks_[active[index].track_index];
                const uint64_t now_us = steadyNowMicros();
                const uint64_t capture_time_us = request.frame->meta.capture_time_us;
                const int capture_age = track.last_reference_capture_time_us > 0 &&
                    capture_time_us >= track.last_reference_capture_time_us
                    ? static_cast<int>((capture_time_us -
                        track.last_reference_capture_time_us) / 1000U) : 0;
                const int ready_age = track.last_reference_ready_time_us > 0 &&
                    now_us >= track.last_reference_ready_time_us
                    ? static_cast<int>((now_us - track.last_reference_ready_time_us) / 1000U) : 0;
                recordRefreshDecision(track.id, evaluateRebuildRefresh(
                    true, true, capture_age, ready_age, estimatedReferenceDeliveryMs(),
                    request.source_fps, config_.patch_soft_refresh_ms,
                    config_.patch_hard_deadline_ms, config_.patch_refresh_guard_ms));
                return advance_reference();
            }
        }
        pending_reference_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        ++snapshot_.cancelled_requests;
    }
    if (active.empty()) {
        recordRefreshDecision(0, RebuildRefreshDecision());
        return true;
    }
    // References are intentionally paced on the same physical token bucket
    // as H.265.  The refresh decision uses the capture-time content age and
    // rounds its continuous deadline back by half a source-frame quantum, so
    // the last safe 6 fps opportunity is not missed.
    const uint64_t now_us = steadyNowMicros();
    const uint64_t capture_time_us = request.frame->meta.capture_time_us;
    const int estimated_delivery_ms = estimatedReferenceDeliveryMs();
    size_t selected = active.size();
    RebuildRefreshDecision selected_decision;
    bool have_decision = false;
    int selected_priority = -1;
    for (size_t index = 0; index < active.size(); ++index) {
        const Track &track = tracks_[active[index].track_index];
        const int capture_age = track.last_reference_capture_time_us > 0 &&
            capture_time_us >= track.last_reference_capture_time_us
            ? static_cast<int>((capture_time_us -
                track.last_reference_capture_time_us) / 1000U) : 0;
        const int ready_age = track.last_reference_ready_time_us > 0 &&
            now_us >= track.last_reference_ready_time_us
            ? static_cast<int>((now_us - track.last_reference_ready_time_us) / 1000U) : 0;
        const RebuildRefreshDecision decision = evaluateRebuildRefresh(
            track.has_reference, false, capture_age, ready_age,
            estimated_delivery_ms, request.source_fps,
            config_.patch_soft_refresh_ms, config_.patch_hard_deadline_ms,
            config_.patch_refresh_guard_ms);
        int priority = decision.start ? 1 : 0;
        if (decision.reason == REBUILD_REFRESH_NO_REFERENCE) priority = 3;
        else if (decision.reason == REBUILD_REFRESH_HARD_DEADLINE) priority = 2;
        if (!have_decision || priority > selected_priority ||
                (priority == selected_priority &&
                 decision.capture_age_ms > selected_decision.capture_age_ms)) {
            have_decision = true;
            selected_priority = priority;
            selected = index;
            selected_decision = decision;
        }
    }
    if (have_decision) {
        recordRefreshDecision(tracks_[active[selected].track_index].id, selected_decision);
    }
    if (!have_decision || !selected_decision.start) {
        return true;
    }
    // Start at most one reference transfer here, then emit a bounded burst.
    // Later source frames advance it in the same bounded bursts, giving H.265
    // and STATE regular opportunities on the shared 100 kbps bucket.
    if (!beginReference(request, active[selected], error)) return false;
    return advance_reference();
}

void RebuildSender::workerLoop() {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || (enabled_ && !queue_.empty()); });
            if (stopping_) break;
            request = queue_.back();
            queue_.clear();
            transmitting_ = true;
            snapshot_.transmitting = true;
            snapshot_.queued_requests = 0;
        }
        std::string error;
        const bool ok = process(request, &error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            transmitting_ = false;
            snapshot_.transmitting = false;
            if (!ok) {
                if (!enabled_ || stopping_) ++snapshot_.cancelled_requests;
                else {
                    failed_ = true;
                    snapshot_.last_error = error;
                }
            }
            condition_.notify_all();
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    transmitting_ = false;
    snapshot_.transmitting = false;
    condition_.notify_all();
}

bool RebuildSender::failed(std::string *error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ && error) *error = snapshot_.last_error;
    return failed_;
}

void RebuildSender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stopping_ = true;
        enabled_ = false;
        queue_.clear();
        condition_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
    snapshot_.enabled = false;
    snapshot_.queued_requests = 0;
    if (socket_ >= 0) {
        close(socket_);
        socket_ = -1;
    }
    delete destination_;
    destination_ = NULL;
}

RebuildSenderSnapshot RebuildSender::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RebuildSenderSnapshot value = snapshot_;
    value.enabled = enabled_;
    value.transmitting = transmitting_;
    value.queued_requests = queue_.size();
    return value;
}

}  // namespace roi_h265

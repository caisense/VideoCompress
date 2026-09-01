#ifndef ROI_H265_TRANSPORT_REBUILD_SENDER_H_
#define ROI_H265_TRANSPORT_REBUILD_SENDER_H_

#include <stddef.h>
#include <stdint.h>

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"
#include "common/frame_meta.h"
#include "transport/rebuild_refresh.h"
#include "transport/rebuild_reference.h"
#include "transport/rate_pacer.h"
#include "transport/rebuild_protocol.h"

namespace roi_h265 {

struct RebuildSenderSnapshot {
    bool enabled;
    bool transmitting;
    size_t queued_requests;
    uint64_t submitted_requests;
    uint64_t replaced_requests;
    uint64_t state_packets;
    // STATE timing stays on FrameMeta's steady-clock domain so it can be
    // compared directly with the video sender's tx_send_e2e_us telemetry.
    uint64_t last_state_capture_time_us;
    uint64_t last_state_send_start_time_us;
    uint64_t last_state_send_time_us;
    uint64_t last_state_pacer_delay_us;
    uint64_t last_state_capture_to_send_us;
    uint64_t state_capture_to_send_p50_us;
    uint64_t state_capture_to_send_p95_us;
    uint64_t patch_transfers;
    uint64_t patch_packets;
    uint64_t parity_packets;
    uint64_t patch_jpeg_bytes;
    uint64_t sent_wire_bytes;
    uint64_t cancelled_requests;
    uint16_t last_reference_generation;
    uint64_t last_reference_capture_time_us;
    uint64_t last_reference_encode_finish_time_us;
    uint64_t last_reference_queue_enter_time_us;
    uint64_t last_reference_first_packet_send_time_us;
    uint64_t last_reference_last_packet_send_time_us;
    uint64_t last_reference_queue_delay_us;
    // Sender-side capture-to-last-send timing.  This is not a receiver ACK
    // or PC-complete timestamp; a PC-side delivery metric needs both clocks.
    uint64_t last_reference_capture_to_send_us;
    uint64_t reference_capture_to_send_p50_us;
    uint64_t reference_capture_to_send_p95_us;
    // Pure reference transfer span, from the first to the last paced packet.
    // Keep this separate from capture-to-send, which includes RKNN/JPEG work.
    uint64_t last_reference_delivery_us;
    uint64_t reference_delivery_p50_us;
    uint64_t reference_delivery_p95_us;
    uint64_t last_reference_interval_us;
    uint64_t reference_interval_p50_us;
    uint64_t reference_interval_p95_us;
    uint64_t reference_interval_max_us;
    uint64_t last_reference_blob_bytes;
    uint64_t last_reference_chunk_count;
    uint64_t last_reference_fec_bytes;
    std::string reference_mode;
    uint64_t full_reference_transfers;
    uint64_t head_reference_transfers;
    uint64_t head_fallback_transfers;
    uint64_t full_jpeg_bytes;
    uint64_t head_jpeg_bytes;
    std::string last_reference_kind;
    std::string last_head_selector_reason;
    std::string last_head_fallback_reason;
    int last_reference_crop_src_width;
    int last_reference_crop_src_height;
    int last_reference_jpeg_width;
    int last_reference_jpeg_height;
    int last_reference_jpeg_quality;
    int last_reference_head_pixels_width;
    int last_reference_head_pixels_height;
    uint64_t last_reference_head_pixels_area;
    double last_reference_jpeg_scale;
    double last_reference_head_pixel_gain_linear;
    double last_reference_head_pixel_gain_area;
    uint64_t reference_jpeg_bytes_p50;
    uint64_t reference_jpeg_bytes_p95;
    uint64_t reference_jpeg_quality_p50;
    uint64_t reference_jpeg_quality_p95;
    uint64_t reference_head_pixels_width_p50;
    uint64_t reference_head_pixels_width_p95;
    uint64_t reference_head_pixels_height_p50;
    uint64_t reference_head_pixels_height_p95;
    uint64_t reference_head_pixels_area_p50;
    uint64_t reference_head_pixels_area_p95;
    double reference_head_pixel_gain_linear_p50;
    double reference_head_pixel_gain_linear_p95;
    double reference_head_pixel_gain_area_p50;
    double reference_head_pixel_gain_area_p95;
    uint16_t last_refresh_track_id;
    int last_reference_capture_age_ms;
    int last_reference_ready_age_ms;
    int last_refresh_threshold_ms;
    int last_estimated_delivery_ms;
    int last_refresh_deadline_ms;
    int last_refresh_quantum_ms;
    bool last_refresh_decision_start;
    std::string last_refresh_reason;
    std::string last_error;

    RebuildSenderSnapshot();
};

// A latest-only semantic/reference worker.  Inference never waits for JPEG or
// UDP pacing, and old reference work is cancelled when a profile switch turns
// rebuild off.  The supplied pacer is also used by AsyncRtpSender.
class RebuildSender {
public:
    RebuildSender(const RebuildConfig &config, const std::string &host, int mtu,
                  const std::shared_ptr<RatePacer> &pacer);
    ~RebuildSender();

    bool start(std::string *error);
    void setEnabled(bool enabled);
    bool submit(const std::shared_ptr<FramePacket> &frame, const SegResult &segmentation,
                uint8_t profile_generation, int source_fps, std::string *error);
    bool failed(std::string *error) const;
    void stop();
    RebuildSenderSnapshot snapshot() const;

private:
    struct Request {
        std::shared_ptr<FramePacket> frame;
        SegResult segmentation;
        uint8_t generation;
        int source_fps;

        Request() : generation(0), source_fps(0) {}
    };

    struct Track {
        uint16_t id;
        int class_id;
        float confidence;
        BBox bbox;
        uint64_t last_seen_frame;
        uint16_t reference_generation;
        bool has_reference;
        uint64_t last_reference_capture_time_us;
        uint64_t last_reference_ready_time_us;

        Track();
    };

    struct ActiveTarget {
        size_t instance_index;
        size_t track_index;
    };

    struct ReferenceBuildTelemetry {
        RebuildReferenceKind kind;
        std::string selector_reason;
        std::string fallback_reason;
        int crop_src_width;
        int crop_src_height;
        int jpeg_width;
        int jpeg_height;
        int jpeg_quality;
        int head_pixels_width;
        int head_pixels_height;
        uint64_t head_pixels_area;
        double jpeg_scale;
        double head_pixel_gain_linear;
        double head_pixel_gain_area;

        ReferenceBuildTelemetry();
    };

    struct PendingReference {
        bool active;
        Request request;
        RebuildPatchFragment metadata;
        std::vector<uint8_t> blob;
        std::vector<uint8_t> parity;
        size_t next_data_index;
        size_t jpeg_bytes;
        uint64_t capture_time_us;
        uint64_t encode_finish_time_us;
        uint64_t queue_enter_time_us;
        uint64_t first_packet_send_time_us;
        uint64_t last_packet_send_time_us;
        size_t chunk_count;
        size_t fec_bytes;
        bool parity_sent;
        ReferenceBuildTelemetry telemetry;

        PendingReference()
            : active(false), next_data_index(0), jpeg_bytes(0), capture_time_us(0),
              encode_finish_time_us(0), queue_enter_time_us(0),
              first_packet_send_time_us(0), last_packet_send_time_us(0),
              chunk_count(0), fec_bytes(0), parity_sent(true) {}
        void clear() { *this = PendingReference(); }
    };

    bool openSocket(std::string *error);
    bool sendPacket(uint8_t type, const Request &request, const std::vector<uint8_t> &payload,
                    uint16_t flags, std::string *error);
    bool process(const Request &request, std::string *error);
    std::vector<ActiveTarget> updateTracks(const Request &request);
    bool sendState(const Request &request, const std::vector<ActiveTarget> &active,
                   std::string *error);
    bool beginReference(const Request &request, const ActiveTarget &active,
                        std::string *error);
    bool sendPendingReferencePacket(std::string *error);
    bool buildReference(const Request &request, const ActiveTarget &active,
                        RebuildPatchFragment *metadata, std::vector<uint8_t> *blob,
                        size_t *jpeg_bytes, ReferenceBuildTelemetry *telemetry,
                        std::string *error) const;
    void recordReferenceTelemetry(const ReferenceBuildTelemetry &telemetry,
                                  size_t jpeg_bytes);
    int estimatedReferenceDeliveryMs() const;
    void recordRefreshDecision(uint16_t track_id,
                               const RebuildRefreshDecision &decision);
    bool isEnabled() const;
    void workerLoop();

    RebuildConfig config_;
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
    bool transmitting_;
    bool failed_;
    uint8_t active_generation_;
    bool has_generation_;
    uint16_t next_track_id_;
    uint32_t next_transfer_id_;
    uint32_t next_packet_sequence_;
    std::vector<Track> tracks_;
    PendingReference pending_reference_;
    RebuildSenderSnapshot snapshot_;
    std::vector<uint64_t> reference_capture_to_send_samples_us_;
    std::vector<uint64_t> state_capture_to_send_samples_us_;
    std::vector<uint64_t> reference_delivery_samples_us_;
    std::vector<uint64_t> reference_interval_samples_us_;
    std::vector<uint64_t> reference_jpeg_bytes_samples_;
    std::vector<uint64_t> reference_jpeg_quality_samples_;
    std::vector<uint64_t> reference_head_pixels_width_samples_;
    std::vector<uint64_t> reference_head_pixels_height_samples_;
    std::vector<uint64_t> reference_head_pixels_area_samples_;
    std::vector<double> reference_head_pixel_gain_linear_samples_;
    std::vector<double> reference_head_pixel_gain_area_samples_;
    std::map<uint16_t, uint64_t> previous_reference_capture_times_us_;
    mutable size_t debug_reference_samples_written_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_REBUILD_SENDER_H_

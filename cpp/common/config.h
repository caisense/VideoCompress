#ifndef ROI_H265_COMMON_CONFIG_H_
#define ROI_H265_COMMON_CONFIG_H_

#include <stdint.h>

#include <string>

namespace roi_h265 {

enum PipelineMode {
    PIPELINE_BASELINE = 0,
    PIPELINE_BBOX_ROI = 1,
    PIPELINE_SEGMENTATION_ROI = 2,
};

enum RateProfile {
    RATE_PROFILE_LOW = 0,
    RATE_PROFILE_MEDIUM = 1,
    RATE_PROFILE_HIGH = 2,
    // A low-rate live base layer plus semantic state and sparse source-camera
    // references.  The PC reconstructs a 640x360@12 display from this stream.
    RATE_PROFILE_REBUILD = 3,
};

// Video is the normal H.265/RTP path.  Image mode leaves capture and YOLO
// running, but suppresses H.265 output and sends a reliable high-resolution
// JPEG only after a relevant detection.
enum TransportMode {
    TRANSPORT_MODE_VIDEO = 0,
    TRANSPORT_MODE_IMAGE = 1,
};

// The low-rate evidence path may retain the full source image or send the
// union of relevant detections with a configurable context margin.
enum SnapshotCropMode {
    SNAPSHOT_CROP_RELEVANT = 0,
    SNAPSHOT_CROP_FULL = 1,
};

enum RebuildReferenceMode {
    REBUILD_REFERENCE_MODE_PERSON = 0,
    REBUILD_REFERENCE_MODE_HEAD = 1,
    REBUILD_REFERENCE_MODE_ADAPTIVE = 2,
};

struct SnapshotConfig {
    int udp_port;
    int jpeg_quality;
    int min_interval_ms;
    // Zero preserves the camera source dimensions.  A non-zero pair limits
    // the JPEG to this bounding box while preserving its aspect ratio.
    int max_width;
    int max_height;
    // JPEG bytes carried by one reliable UDP data packet.  The protocol header
    // is added separately and validation keeps the full datagram under MTU.
    int chunk_payload_bytes;
    int ack_timeout_ms;
    int max_retries;
    float min_confidence;
    SnapshotCropMode crop_mode;
    // Margin added on each side of the relevant-detection union, expressed as
    // a percentage of that union's width/height.
    int crop_margin_percent;
    // The saved JPEG is rotated after crop/resize so portrait evidence matches
    // the receiver and board-preview orientation.
    bool rotate_ccw;

    SnapshotConfig();
};

// Optional companion channel used only by RATE_PROFILE_REBUILD.  H.265 stays
// standards-compliant on the normal RTP port; these datagrams carry stable
// target state and MTU-safe JPEG/mask reference fragments on a separate port.
// Both channels spend from the same physical-wire RatePacer.
struct RebuildConfig {
    int udp_port;
    int output_width;
    int output_height;
    int output_fps;
    float min_confidence;
    int max_targets;
    int patch_max_side;
    int patch_jpeg_quality;
    int patch_max_bytes;
    int patch_soft_refresh_ms;
    int patch_hard_deadline_ms;
    int patch_refresh_guard_ms;
    int patch_chunk_bytes;
    int patch_packets_per_frame;
    int crop_margin_percent;
    bool parity;
    RebuildReferenceMode reference_mode;
    int head_class_id;
    float head_height_ratio;
    float head_width_ratio;
    int head_margin_percent;
    int head_min_size;
    int head_jpeg_quality;
    int head_min_jpeg_quality;
    std::string debug_reference_dir;
    int debug_reference_max_samples;
    bool debug_head_roi;

    RebuildConfig();
};

// Lightweight semantic state notifications are intentionally separate from
// H.265 RTP, RB/1 and RSNP.  They are emitted on target state transitions and
// periodic while-present heartbeats, then spend from the shared video pacer.
struct DetectionEventConfig {
    bool enabled;
    int udp_port;
    float min_confidence;
    int heartbeat_ms;

    DetectionEventConfig();
};

struct RoiConfig {
    int cell_size;
    int background_delta_qp;
    int halo_delta_qp;
    int core_delta_qp;
    int edge_delta_qp;
    float occupancy_threshold;
    int erosion_radius;
    int dilation_radius;
    int hold_frames;
    int max_age_frames;
    int max_regions;

    RoiConfig();
};

struct EncoderConfig {
    int width;
    int height;
    int fps;
    int target_bitrate_bps;
    int gop;
    int qp_min;
    int qp_max;
    int qp_init;
    int qp_min_i;
    int qp_max_i;
    bool intra_refresh;
    int intra_refresh_rows;
    int max_reencode_times;
    int super_i_frame_bits;
    int super_p_frame_bits;
    bool grayscale_encode;

    EncoderConfig();
};

struct TransportConfig {
    std::string udp_host;
    int udp_port;
    int pacing_bitrate_bps;
    int mtu;
    int send_queue_frames;
    int send_max_latency_ms;
    // Optional receiver SDP emitted from the first IDR access unit.  It carries
    // the H.265 VPS/SPS/PPS required by FFmpeg to determine frame dimensions.
    std::string rtp_sdp_path;
    // Runtime profile commands are written as low/medium/high/rebuild lines to this
    // FIFO. image/snapshot switches to detection-triggered JPEG transfer;
    // video returns to H.265. Empty disables live switching.
    std::string profile_control_path;
    TransportMode mode;
    SnapshotConfig snapshot;
    RebuildConfig rebuild;
    DetectionEventConfig event;

    TransportConfig();
};

// Controls for the sender-side speech path before Codec2.  The soft gate is
// intentionally distinct from the board codec's hard ADC mute gate so quiet
// speech is attenuated rather than discarded.
struct AudioPreprocessConfig {
    bool enabled;
    int highpass_hz;
    int lowpass_hz;
    int noise_suppression_db;
    int noise_gate_snr_db;
    int noise_gate_attenuation_db;
    int noise_gate_hangover_ms;
    int noise_warmup_ms;
    int agc_target_dbfs;
    int agc_max_gain_db;
    // A lightweight voiced-speech detector prevents broad-band room sounds
    // from opening the gate and being amplified by AGC.  It is separate from
    // the soft gate threshold, which remains the installation SNR control.
    bool voice_activity_detection;
    int voice_start_frames;
    int voice_min_voicing_percent;

    AudioPreprocessConfig();
};

// Audio is deliberately separate from H.265 RTP, but it shares the same
// physical-wire rate pacer.  Codec2 always consumes 8 kHz mono PCM after
// capture conversion; the capture side may use a different ALSA rate/channel
// layout when an ALSA plug device is not available.
struct AudioConfig {
    bool enabled;
    std::string device;
    int udp_port;
    int capture_rate_hz;
    int capture_channels;
    int codec2_mode_bps;
    int frames_per_packet;
    std::string rtp_sdp_path;
    // RTP is sent continuously while speech is active.  During silence, a
    // bounded pre-roll protects the first syllable and an optional marker-clear
    // Codec2 keepalive prevents the receiver path from going completely idle.
    bool dtx_enabled;
    int dtx_preroll_ms;
    // Once a speech burst is open, continue transmitting for this long after
    // the software gate briefly closes. This protects unvoiced consonants and
    // natural intra-sentence pauses from being split into new talkspurts.
    int dtx_hangover_ms;
    int dtx_keepalive_ms;
    // Zero selects the exact physical-wire rate implied by Codec2 packet
    // geometry.  The pacer retains one such datagram for audio before video
    // spends shared-link tokens.
    int reserve_wire_bitrate_bps;
    // A complete RTP packet older than this is stale live audio and is
    // discarded rather than replayed after a network or scheduler stall.
    int max_latency_ms;
    AudioPreprocessConfig preprocess;

    AudioConfig();
};

struct CameraConfig {
    std::string device;
    std::string input_video;
    int width;
    int height;
    int queue_depth;
    int max_frames;

    CameraConfig();
};

struct AppConfig {
    std::string model_path;
    PipelineMode mode;
    // Mirrors the original YOLOv8-Seg demo: show the annotated source frame
    // on the board's Weston display unless explicitly disabled.
    bool preview;
    // Initial BoardPreview client area for the verified 720x1280 portrait DSI
    // output.  The preview is rendered from the encoder's 16:9 NV12 input,
    // then rotated CCW into a matching 9:16 portrait image.
    int preview_width;
    int preview_height;
    // Rotate encoder pixels before drawing annotations. This keeps labels and
    // status text upright on the portrait display.
    bool preview_rotate_ccw;
    bool debug_roi;
    std::string debug_roi_path;
    RateProfile rate_profile;
    CameraConfig camera;
    EncoderConfig encoder;
    RoiConfig roi;
    TransportConfig transport;
    AudioConfig audio;

    AppConfig();
};

// Parses --key=value command-line options. Every GOAL.md runtime parameter is
// exposed here; unknown options cause parsing to fail instead of being ignored.
bool parseAppConfig(int argc, char **argv, AppConfig *config, std::string *error);
const char *pipelineModeName(PipelineMode mode);
const char *rateProfileName(RateProfile profile);
bool parseRateProfile(const std::string &name, RateProfile *profile);
void applyRateProfile(RateProfile profile, AppConfig *config);
const char *rebuildReferenceModeName(RebuildReferenceMode mode);
bool parseRebuildReferenceMode(const std::string &name, RebuildReferenceMode *mode);
const char *transportModeName(TransportMode mode);
bool parseTransportMode(const std::string &name, TransportMode *mode);

}  // namespace roi_h265

#endif  // ROI_H265_COMMON_CONFIG_H_

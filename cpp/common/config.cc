#include "common/config.h"

#include <cstdlib>
#include <sstream>

namespace roi_h265 {

RoiConfig::RoiConfig()
    : cell_size(16), background_delta_qp(12), halo_delta_qp(2),
      core_delta_qp(-6), edge_delta_qp(-10), occupancy_threshold(0.10f),
      erosion_radius(2), dilation_radius(3), hold_frames(3),
      max_age_frames(9), max_regions(64) {}

EncoderConfig::EncoderConfig()
    : width(320), height(180), fps(10), target_bitrate_bps(42000),
      gop(50), qp_min(10), qp_max(51), qp_init(38), qp_min_i(36), qp_max_i(48),
      intra_refresh(true), intra_refresh_rows(1), max_reencode_times(3),
      super_i_frame_bits(12000), super_p_frame_bits(5500), grayscale_encode(true) {}

SnapshotConfig::SnapshotConfig()
    // Balanced 60 kbps evidence defaults: cap payload bytes before they enter
    // the physical rate limiter.  A user can still request the source-size
    // full frame with --snapshot-crop=full --snapshot-max-width=0
    // --snapshot-max-height=0.
    : udp_port(5008), jpeg_quality(75), min_interval_ms(5000), max_width(1280),
      max_height(720), chunk_payload_bytes(1100), ack_timeout_ms(300),
      max_retries(20), min_confidence(0.35f), crop_mode(SNAPSHOT_CROP_RELEVANT),
      crop_margin_percent(25), rotate_ccw(true) {}

RebuildConfig::RebuildConfig()
    : udp_port(5009), output_width(640), output_height(360), output_fps(12),
      min_confidence(0.35f), max_targets(2), patch_max_side(128),
      // A reference older than about one second is deliberately rejected by
      // the receiver.  Refreshing at 700 ms keeps the geometry registration
      // useful while still sending references progressively on the shared
      // 100 kbps bucket.
      patch_jpeg_quality(72), patch_max_bytes(1600), patch_refresh_ms(700),
      patch_chunk_bytes(1100), patch_packets_per_frame(2),
      crop_margin_percent(20), parity(true) {}

DetectionEventConfig::DetectionEventConfig()
    // A 44-byte ROEV/1 datagram is only emitted on a target-state transition;
    // while a target remains present, one heartbeat per second lets the PC
    // recover from a lost UDP edge event without turning every inference into
    // additional traffic on the shared video-side budget.
    : enabled(true), udp_port(5010), min_confidence(0.35f), heartbeat_ms(1000) {}

TransportConfig::TransportConfig()
    : udp_host("127.0.0.1"), udp_port(5004), pacing_bitrate_bps(60000),
      mtu(1200), send_queue_frames(3), send_max_latency_ms(250), rtp_sdp_path(""),
      profile_control_path("/tmp/roi-rate-profile"), mode(TRANSPORT_MODE_VIDEO), snapshot(),
      rebuild(), event() {}

AudioPreprocessConfig::AudioPreprocessConfig()
    // Voice-first board default.  The ES8388 hard ADC mute gate must stay off:
    // the software VAD suppresses non-speech without truncating the ADC signal.
    : enabled(true), highpass_hz(80), lowpass_hz(3600), noise_suppression_db(6),
      noise_gate_snr_db(2), noise_gate_attenuation_db(30), noise_gate_hangover_ms(600),
      noise_warmup_ms(600), agc_target_dbfs(-16), agc_max_gain_db(20),
      voice_activity_detection(true), voice_start_frames(2), voice_min_voicing_percent(42) {}

AudioConfig::AudioConfig()
    : enabled(false), device("hw:3,0"), udp_port(5006), capture_rate_hz(44100),
      // CODEC2 2400 has 20 ms frames.  Four frames retain the former 80 ms
      // packet interval, so the voice-quality increase does not double RTP /
       // Ethernet header cost on the shared 60 kbps physical pacer.
       capture_channels(2), codec2_mode_bps(2400), frames_per_packet(4),
       rtp_sdp_path("/tmp/roi-audio.sdp"), dtx_enabled(true), dtx_preroll_ms(80),
       dtx_hangover_ms(600), dtx_keepalive_ms(1000), reserve_wire_bitrate_bps(0),
       max_latency_ms(160) {}

CameraConfig::CameraConfig()
    : device("/dev/video0"), input_video(""), width(640), height(480), queue_depth(2), max_frames(0) {}

AppConfig::AppConfig()
    : model_path("model/yolov8_seg.rknn"), mode(PIPELINE_SEGMENTATION_ROI),
      preview(true), preview_width(720), preview_height(1280), preview_rotate_ccw(true),
      debug_roi(false), debug_roi_path("roi_map.pgm"), rate_profile(RATE_PROFILE_LOW) {}

namespace {

bool splitOption(const char *arg, std::string *key, std::string *value) {
    std::string option(arg ? arg : "");
    if (option.compare(0, 2, "--") != 0) return false;
    const std::string::size_type equal = option.find('=');
    if (equal == std::string::npos || equal == 2) return false;
    *key = option.substr(2, equal - 2);
    *value = option.substr(equal + 1);
    return true;
}

bool parseInt(const std::string &text, int *value) {
    char *end = NULL;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0') return false;
    *value = static_cast<int>(parsed);
    return true;
}

bool parseFloat(const std::string &text, float *value) {
    char *end = NULL;
    const float parsed = std::strtof(text.c_str(), &end);
    if (!end || *end != '\0') return false;
    *value = parsed;
    return true;
}

bool parseBool(const std::string &text, bool *value) {
    if (text == "1" || text == "true" || text == "on") { *value = true; return true; }
    if (text == "0" || text == "false" || text == "off") { *value = false; return true; }
    return false;
}

void applyRateProfileValues(RateProfile profile, AppConfig *config) {
    if (profile == RATE_PROFILE_LOW) {
        config->rate_profile = RATE_PROFILE_LOW;
        config->encoder.width = 320; config->encoder.height = 180; config->encoder.fps = 10;
        config->encoder.target_bitrate_bps = 42000; config->encoder.gop = 50;
        config->encoder.qp_init = 38; config->encoder.qp_min_i = 36; config->encoder.qp_max_i = 48;
        config->encoder.super_i_frame_bits = 12000; config->encoder.super_p_frame_bits = 5500;
        config->encoder.grayscale_encode = true;
        config->roi.background_delta_qp = 12; config->roi.core_delta_qp = -6; config->roi.edge_delta_qp = -10;
        config->transport.pacing_bitrate_bps = 60000; config->transport.send_max_latency_ms = 250;
        return;
    }
    if (profile == RATE_PROFILE_MEDIUM) {
        config->rate_profile = RATE_PROFILE_MEDIUM;
        config->encoder.width = 480; config->encoder.height = 270; config->encoder.fps = 15;
        config->encoder.target_bitrate_bps = 110000; config->encoder.gop = 75;
        config->encoder.qp_init = 34; config->encoder.qp_min_i = 32; config->encoder.qp_max_i = 44;
        config->encoder.super_i_frame_bits = 35000; config->encoder.super_p_frame_bits = 14000;
        config->encoder.grayscale_encode = false;
        config->roi.background_delta_qp = 10; config->roi.core_delta_qp = -6; config->roi.edge_delta_qp = -10;
        config->transport.pacing_bitrate_bps = 150000; config->transport.send_max_latency_ms = 250;
        return;
    }
    if (profile == RATE_PROFILE_HIGH) {
        config->rate_profile = RATE_PROFILE_HIGH;
        config->encoder.width = 640; config->encoder.height = 360; config->encoder.fps = 20;
        config->encoder.target_bitrate_bps = 240000; config->encoder.gop = 100;
        config->encoder.qp_init = 30; config->encoder.qp_min_i = 28; config->encoder.qp_max_i = 42;
        config->encoder.super_i_frame_bits = 60000; config->encoder.super_p_frame_bits = 24000;
        config->encoder.grayscale_encode = false;
        config->roi.background_delta_qp = 6; config->roi.core_delta_qp = -6; config->roi.edge_delta_qp = -10;
        config->transport.pacing_bitrate_bps = 300000; config->transport.send_max_latency_ms = 200;
        return;
    }
    if (profile == RATE_PROFILE_REBUILD) {
        config->rate_profile = RATE_PROFILE_REBUILD;
        // The wire stream is deliberately smaller/slower than the reconstructed
        // display.  Sparse source-camera target references restore useful
        // subject detail without pretending that 640x360 pixels crossed the link.
        config->encoder.width = 256; config->encoder.height = 144; config->encoder.fps = 6;
        config->encoder.target_bitrate_bps = 28000; config->encoder.gop = 36;
        config->encoder.qp_init = 39; config->encoder.qp_min_i = 36; config->encoder.qp_max_i = 49;
        config->encoder.super_i_frame_bits = 10000; config->encoder.super_p_frame_bits = 4500;
        config->encoder.grayscale_encode = false;
        config->roi.background_delta_qp = 14; config->roi.core_delta_qp = -7;
        config->roi.edge_delta_qp = -11;
        config->transport.pacing_bitrate_bps = 100000;
        config->transport.send_max_latency_ms = 250;
        return;
    }
}

}  // namespace

const char *pipelineModeName(PipelineMode mode) {
    switch (mode) {
    case PIPELINE_BASELINE: return "baseline";
    case PIPELINE_BBOX_ROI: return "bbox";
    default: return "segmentation";
    }
}

const char *rateProfileName(RateProfile profile) {
    switch (profile) {
    case RATE_PROFILE_LOW: return "low";
    case RATE_PROFILE_MEDIUM: return "medium";
    case RATE_PROFILE_HIGH: return "high";
    case RATE_PROFILE_REBUILD: return "rebuild";
    }
    return "unknown";
}

bool parseRateProfile(const std::string &name, RateProfile *profile) {
    if (!profile) return false;
    if (name == "low") *profile = RATE_PROFILE_LOW;
    else if (name == "medium") *profile = RATE_PROFILE_MEDIUM;
    else if (name == "high") *profile = RATE_PROFILE_HIGH;
    else if (name == "rebuild") *profile = RATE_PROFILE_REBUILD;
    else return false;
    return true;
}

const char *transportModeName(TransportMode mode) {
    return mode == TRANSPORT_MODE_IMAGE ? "image" : "video";
}

bool parseTransportMode(const std::string &name, TransportMode *mode) {
    if (!mode) return false;
    if (name == "video") *mode = TRANSPORT_MODE_VIDEO;
    else if (name == "image" || name == "snapshot") *mode = TRANSPORT_MODE_IMAGE;
    else return false;
    return true;
}

void applyRateProfile(RateProfile profile, AppConfig *config) {
    if (config) applyRateProfileValues(profile, config);
}

bool parseAppConfig(int argc, char **argv, AppConfig *config, std::string *error) {
    if (!config) return false;
    for (int i = 1; i < argc; ++i) {
        std::string key, value;
        if (!splitOption(argv[i], &key, &value)) {
            if (error) *error = "invalid option: " + std::string(argv[i]);
            return false;
        }
        int integer = 0;
        float decimal = 0.0f;
        bool boolean = false;
        if (key == "model") config->model_path = value;
        else if (key == "camera-device") config->camera.device = value;
        else if (key == "input-video") config->camera.input_video = value;
        else if (key == "udp-host") config->transport.udp_host = value;
        else if (key == "rtp-sdp-path") config->transport.rtp_sdp_path = value;
        else if (key == "profile-control") config->transport.profile_control_path = value;
        else if (key == "transport-mode") {
            if (!parseTransportMode(value, &config->transport.mode)) {
                if (error) *error = "transport mode must be video or image";
                return false;
            }
        }
        else if (key == "audio-device") config->audio.device = value;
        else if (key == "audio-rtp-sdp-path") config->audio.rtp_sdp_path = value;
        else if (key == "debug-roi-path") config->debug_roi_path = value;
        else if (key == "mode") {
            if (value == "baseline") config->mode = PIPELINE_BASELINE;
            else if (value == "bbox") config->mode = PIPELINE_BBOX_ROI;
            else if (value == "segmentation") config->mode = PIPELINE_SEGMENTATION_ROI;
            else { if (error) *error = "mode must be baseline, bbox, or segmentation"; return false; }
        } else if (key == "preview" && parseBool(value, &boolean)) config->preview = boolean;
        else if (key == "preview-width" && parseInt(value, &integer)) config->preview_width = integer;
        else if (key == "preview-height" && parseInt(value, &integer)) config->preview_height = integer;
        else if (key == "preview-rotate") {
            if (value == "ccw") config->preview_rotate_ccw = true;
            else if (value == "none") config->preview_rotate_ccw = false;
            else {
                if (error) *error = "preview rotation must be ccw or none";
                return false;
            }
        }
        else if (key == "debug-roi" && parseBool(value, &boolean)) config->debug_roi = boolean;
        else if (key == "audio" && parseBool(value, &boolean)) config->audio.enabled = boolean;
        else if (key == "rate-profile" || key == "profile") {
            RateProfile profile;
            if (!parseRateProfile(value, &profile)) {
                if (error) *error = "rate profile must be low, medium, high, or rebuild";
                return false;
            }
            applyRateProfile(profile, config);
        }
        else if (key == "encoder-width" && parseInt(value, &integer)) config->encoder.width = integer;
        else if (key == "encoder-height" && parseInt(value, &integer)) config->encoder.height = integer;
        else if (key == "fps" && parseInt(value, &integer)) config->encoder.fps = integer;
        else if (key == "target-bitrate" && parseInt(value, &integer)) config->encoder.target_bitrate_bps = integer;
        else if (key == "gop" && parseInt(value, &integer)) config->encoder.gop = integer;
        else if (key == "qp-min" && parseInt(value, &integer)) config->encoder.qp_min = integer;
        else if (key == "qp-max" && parseInt(value, &integer)) config->encoder.qp_max = integer;
        else if (key == "qp-init" && parseInt(value, &integer)) config->encoder.qp_init = integer;
        else if (key == "qp-min-i" && parseInt(value, &integer)) config->encoder.qp_min_i = integer;
        else if (key == "qp-max-i" && parseInt(value, &integer)) config->encoder.qp_max_i = integer;
        else if (key == "intra-refresh" && parseBool(value, &boolean)) config->encoder.intra_refresh = boolean;
        else if (key == "intra-refresh-rows" && parseInt(value, &integer)) config->encoder.intra_refresh_rows = integer;
        else if (key == "max-reencode-times" && parseInt(value, &integer)) config->encoder.max_reencode_times = integer;
        else if (key == "super-i-frame-bits" && parseInt(value, &integer)) config->encoder.super_i_frame_bits = integer;
        else if (key == "super-p-frame-bits" && parseInt(value, &integer)) config->encoder.super_p_frame_bits = integer;
        else if (key == "grayscale-encode" && parseBool(value, &boolean)) config->encoder.grayscale_encode = boolean;
        else if (key == "background-delta-qp" && parseInt(value, &integer)) config->roi.background_delta_qp = integer;
        else if (key == "halo-delta-qp" && parseInt(value, &integer)) config->roi.halo_delta_qp = integer;
        else if (key == "core-delta-qp" && parseInt(value, &integer)) config->roi.core_delta_qp = integer;
        else if (key == "edge-delta-qp" && parseInt(value, &integer)) config->roi.edge_delta_qp = integer;
        else if (key == "mask-occupancy-threshold" && parseFloat(value, &decimal)) config->roi.occupancy_threshold = decimal;
        else if (key == "erosion-radius" && parseInt(value, &integer)) config->roi.erosion_radius = integer;
        else if (key == "dilation-radius" && parseInt(value, &integer)) config->roi.dilation_radius = integer;
        else if (key == "roi-hold-frames" && parseInt(value, &integer)) config->roi.hold_frames = integer;
        else if (key == "roi-max-age" && parseInt(value, &integer)) config->roi.max_age_frames = integer;
        else if (key == "max-roi-region" && parseInt(value, &integer)) config->roi.max_regions = integer;
        else if (key == "udp-port" && parseInt(value, &integer)) config->transport.udp_port = integer;
        else if (key == "pacing-bitrate" && parseInt(value, &integer)) config->transport.pacing_bitrate_bps = integer;
        else if (key == "mtu" && parseInt(value, &integer)) config->transport.mtu = integer;
        else if (key == "send-queue-frames" && parseInt(value, &integer)) config->transport.send_queue_frames = integer;
        else if (key == "send-max-latency-ms" && parseInt(value, &integer)) config->transport.send_max_latency_ms = integer;
        else if (key == "snapshot-udp-port" && parseInt(value, &integer)) config->transport.snapshot.udp_port = integer;
        else if (key == "snapshot-jpeg-quality" && parseInt(value, &integer)) config->transport.snapshot.jpeg_quality = integer;
        else if (key == "snapshot-min-interval-ms" && parseInt(value, &integer)) config->transport.snapshot.min_interval_ms = integer;
        else if (key == "snapshot-max-width" && parseInt(value, &integer)) config->transport.snapshot.max_width = integer;
        else if (key == "snapshot-max-height" && parseInt(value, &integer)) config->transport.snapshot.max_height = integer;
        else if (key == "snapshot-chunk-bytes" && parseInt(value, &integer)) config->transport.snapshot.chunk_payload_bytes = integer;
        else if (key == "snapshot-ack-timeout-ms" && parseInt(value, &integer)) config->transport.snapshot.ack_timeout_ms = integer;
        else if (key == "snapshot-max-retries" && parseInt(value, &integer)) config->transport.snapshot.max_retries = integer;
        else if (key == "snapshot-min-confidence" && parseFloat(value, &decimal)) config->transport.snapshot.min_confidence = decimal;
        else if (key == "snapshot-crop") {
            if (value == "relevant") config->transport.snapshot.crop_mode = SNAPSHOT_CROP_RELEVANT;
            else if (value == "full") config->transport.snapshot.crop_mode = SNAPSHOT_CROP_FULL;
            else {
                if (error) *error = "snapshot crop must be relevant or full";
                return false;
            }
        }
        else if (key == "snapshot-crop-margin-percent" && parseInt(value, &integer)) {
            config->transport.snapshot.crop_margin_percent = integer;
        }
        else if (key == "snapshot-rotate") {
            if (value == "ccw") config->transport.snapshot.rotate_ccw = true;
            else if (value == "none") config->transport.snapshot.rotate_ccw = false;
            else {
                if (error) *error = "snapshot rotation must be ccw or none";
                return false;
            }
        }
        else if (key == "rebuild-udp-port" && parseInt(value, &integer)) config->transport.rebuild.udp_port = integer;
        else if (key == "rebuild-output-width" && parseInt(value, &integer)) config->transport.rebuild.output_width = integer;
        else if (key == "rebuild-output-height" && parseInt(value, &integer)) config->transport.rebuild.output_height = integer;
        else if (key == "rebuild-output-fps" && parseInt(value, &integer)) config->transport.rebuild.output_fps = integer;
        else if (key == "rebuild-min-confidence" && parseFloat(value, &decimal)) config->transport.rebuild.min_confidence = decimal;
        else if (key == "rebuild-max-targets" && parseInt(value, &integer)) config->transport.rebuild.max_targets = integer;
        else if (key == "rebuild-patch-max-side" && parseInt(value, &integer)) config->transport.rebuild.patch_max_side = integer;
        else if (key == "rebuild-jpeg-quality" && parseInt(value, &integer)) config->transport.rebuild.patch_jpeg_quality = integer;
        else if (key == "rebuild-patch-max-bytes" && parseInt(value, &integer)) config->transport.rebuild.patch_max_bytes = integer;
        else if (key == "rebuild-patch-refresh-ms" && parseInt(value, &integer)) config->transport.rebuild.patch_refresh_ms = integer;
        else if (key == "rebuild-chunk-bytes" && parseInt(value, &integer)) config->transport.rebuild.patch_chunk_bytes = integer;
        else if (key == "rebuild-patch-packets-per-frame" && parseInt(value, &integer)) config->transport.rebuild.patch_packets_per_frame = integer;
        else if (key == "rebuild-crop-margin-percent" && parseInt(value, &integer)) config->transport.rebuild.crop_margin_percent = integer;
        else if (key == "rebuild-parity" && parseBool(value, &boolean)) config->transport.rebuild.parity = boolean;
        else if (key == "event-push" && parseBool(value, &boolean)) config->transport.event.enabled = boolean;
        else if (key == "event-udp-port" && parseInt(value, &integer)) config->transport.event.udp_port = integer;
        else if (key == "event-min-confidence" && parseFloat(value, &decimal)) config->transport.event.min_confidence = decimal;
        else if (key == "event-heartbeat-ms" && parseInt(value, &integer)) config->transport.event.heartbeat_ms = integer;
        else if (key == "audio-udp-port" && parseInt(value, &integer)) config->audio.udp_port = integer;
        else if (key == "audio-capture-rate" && parseInt(value, &integer)) config->audio.capture_rate_hz = integer;
        else if (key == "audio-channels" && parseInt(value, &integer)) config->audio.capture_channels = integer;
        else if (key == "audio-codec2-mode" && parseInt(value, &integer)) config->audio.codec2_mode_bps = integer;
        else if (key == "audio-frames-per-packet" && parseInt(value, &integer)) config->audio.frames_per_packet = integer;
        else if (key == "audio-dtx" && parseBool(value, &boolean)) config->audio.dtx_enabled = boolean;
        else if (key == "audio-dtx-preroll-ms" && parseInt(value, &integer)) config->audio.dtx_preroll_ms = integer;
        else if (key == "audio-dtx-hangover-ms" && parseInt(value, &integer)) config->audio.dtx_hangover_ms = integer;
        else if (key == "audio-dtx-keepalive-ms" && parseInt(value, &integer)) config->audio.dtx_keepalive_ms = integer;
        else if (key == "audio-reserve-bitrate" && parseInt(value, &integer)) config->audio.reserve_wire_bitrate_bps = integer;
        else if (key == "audio-max-latency-ms" && parseInt(value, &integer)) config->audio.max_latency_ms = integer;
        else if (key == "audio-preprocess" && parseBool(value, &boolean)) config->audio.preprocess.enabled = boolean;
        else if (key == "audio-highpass-hz" && parseInt(value, &integer)) config->audio.preprocess.highpass_hz = integer;
        else if (key == "audio-lowpass-hz" && parseInt(value, &integer)) config->audio.preprocess.lowpass_hz = integer;
        else if (key == "audio-noise-suppression-db" && parseInt(value, &integer)) config->audio.preprocess.noise_suppression_db = integer;
        else if (key == "audio-noise-gate-snr-db" && parseInt(value, &integer)) config->audio.preprocess.noise_gate_snr_db = integer;
        else if (key == "audio-noise-gate-attenuation-db" && parseInt(value, &integer)) config->audio.preprocess.noise_gate_attenuation_db = integer;
        else if (key == "audio-noise-gate-hangover-ms" && parseInt(value, &integer)) config->audio.preprocess.noise_gate_hangover_ms = integer;
        else if (key == "audio-noise-warmup-ms" && parseInt(value, &integer)) config->audio.preprocess.noise_warmup_ms = integer;
        else if (key == "audio-agc-target-dbfs" && parseInt(value, &integer)) config->audio.preprocess.agc_target_dbfs = integer;
        else if (key == "audio-agc-max-gain-db" && parseInt(value, &integer)) config->audio.preprocess.agc_max_gain_db = integer;
        else if (key == "audio-vad" && parseBool(value, &boolean)) config->audio.preprocess.voice_activity_detection = boolean;
        else if (key == "audio-vad-start-frames" && parseInt(value, &integer)) config->audio.preprocess.voice_start_frames = integer;
        else if (key == "audio-vad-min-voicing" && parseInt(value, &integer)) config->audio.preprocess.voice_min_voicing_percent = integer;
        else if (key == "camera-width" && parseInt(value, &integer)) config->camera.width = integer;
        else if (key == "camera-height" && parseInt(value, &integer)) config->camera.height = integer;
        else if (key == "queue-depth" && parseInt(value, &integer)) config->camera.queue_depth = integer;
        else if (key == "max-frames" && parseInt(value, &integer)) config->camera.max_frames = integer;
        else { if (error) *error = "invalid value or unknown option: --" + key; return false; }
    }

    if (config->rate_profile == RATE_PROFILE_REBUILD &&
        (config->encoder.width != 256 || config->encoder.height != 144 ||
         config->encoder.fps != 6 || config->encoder.target_bitrate_bps != 28000 ||
         config->transport.pacing_bitrate_bps != 100000 ||
         config->encoder.grayscale_encode)) {
        if (error) {
            *error = "rebuild is an atomic 256x144@6 color/28 kbps/100 kbps profile; "
                     "remove old low/medium encoder and pacing overrides";
        }
        return false;
    }

    if (config->encoder.width <= 0 || config->encoder.height <= 0 || config->encoder.fps <= 0 ||
        config->encoder.target_bitrate_bps <= 0 || config->encoder.gop <= 0 ||
        config->transport.pacing_bitrate_bps <= 0 || config->transport.udp_port < 1 ||
        config->transport.udp_port > 65535 || config->transport.mtu < 64 ||
        config->transport.send_queue_frames <= 0 || config->transport.send_max_latency_ms <= 0 ||
        config->transport.snapshot.udp_port < 1 || config->transport.snapshot.udp_port > 65535 ||
        config->transport.snapshot.udp_port == config->transport.udp_port ||
        config->transport.snapshot.jpeg_quality < 1 || config->transport.snapshot.jpeg_quality > 100 ||
        config->transport.snapshot.min_interval_ms < 0 ||
        config->transport.snapshot.min_interval_ms > 3600000 ||
        config->transport.snapshot.max_width < 0 || config->transport.snapshot.max_height < 0 ||
        ((config->transport.snapshot.max_width == 0) !=
         (config->transport.snapshot.max_height == 0)) ||
        config->transport.snapshot.chunk_payload_bytes <= 0 ||
        config->transport.snapshot.chunk_payload_bytes > config->transport.mtu - 60 ||
        config->transport.snapshot.ack_timeout_ms < 20 ||
        config->transport.snapshot.ack_timeout_ms > 5000 ||
        config->transport.snapshot.max_retries < 1 || config->transport.snapshot.max_retries > 100 ||
        config->transport.snapshot.min_confidence < 0.0f ||
        config->transport.snapshot.min_confidence > 1.0f ||
        (config->transport.snapshot.crop_mode != SNAPSHOT_CROP_RELEVANT &&
         config->transport.snapshot.crop_mode != SNAPSHOT_CROP_FULL) ||
        config->transport.snapshot.crop_margin_percent < 0 ||
        config->transport.snapshot.crop_margin_percent > 100 ||
        config->transport.rebuild.udp_port < 1 || config->transport.rebuild.udp_port > 65535 ||
        config->transport.rebuild.udp_port == config->transport.udp_port ||
        config->transport.rebuild.udp_port == config->transport.snapshot.udp_port ||
        config->transport.rebuild.output_width <= 0 ||
        config->transport.rebuild.output_height <= 0 ||
        config->transport.rebuild.output_fps <= 0 || config->transport.rebuild.output_fps > 60 ||
        config->transport.rebuild.min_confidence < 0.0f ||
        config->transport.rebuild.min_confidence > 1.0f ||
        config->transport.rebuild.max_targets < 1 || config->transport.rebuild.max_targets > 16 ||
        config->transport.rebuild.patch_max_side < 32 ||
        config->transport.rebuild.patch_max_side > 512 ||
        config->transport.rebuild.patch_jpeg_quality < 20 ||
        config->transport.rebuild.patch_jpeg_quality > 95 ||
        config->transport.rebuild.patch_max_bytes < 256 ||
        config->transport.rebuild.patch_max_bytes > 8192 ||
        config->transport.rebuild.patch_refresh_ms < 100 ||
        config->transport.rebuild.patch_refresh_ms > 10000 ||
        config->transport.rebuild.patch_chunk_bytes < 128 ||
        config->transport.rebuild.patch_chunk_bytes > config->transport.mtu - 100 ||
        config->transport.rebuild.patch_packets_per_frame < 1 ||
        config->transport.rebuild.patch_packets_per_frame > 2 ||
        config->transport.rebuild.crop_margin_percent < 0 ||
        config->transport.rebuild.crop_margin_percent > 100 ||
        config->transport.event.udp_port < 1 || config->transport.event.udp_port > 65535 ||
        config->transport.event.udp_port == config->transport.udp_port ||
        config->transport.event.udp_port == config->transport.snapshot.udp_port ||
        config->transport.event.udp_port == config->transport.rebuild.udp_port ||
        config->transport.event.min_confidence < 0.0f ||
        config->transport.event.min_confidence > 1.0f ||
        config->transport.event.heartbeat_ms < 100 ||
        config->transport.event.heartbeat_ms > 60000 ||
        config->audio.udp_port < 1 || config->audio.udp_port > 65535 ||
        config->audio.capture_rate_hz < 8000 || config->audio.capture_rate_hz > 192000 ||
        config->audio.capture_channels < 1 || config->audio.capture_channels > 8 ||
        config->audio.frames_per_packet < 1 || config->audio.frames_per_packet > 10 ||
        config->audio.dtx_preroll_ms < 0 || config->audio.dtx_preroll_ms > 500 ||
        config->audio.dtx_hangover_ms < 0 || config->audio.dtx_hangover_ms > 2000 ||
        config->audio.dtx_keepalive_ms < 0 || config->audio.dtx_keepalive_ms > 10000 ||
        config->audio.reserve_wire_bitrate_bps < 0 ||
        config->audio.reserve_wire_bitrate_bps > 1000000 ||
        config->audio.max_latency_ms < 20 || config->audio.max_latency_ms > 2000 ||
        config->audio.preprocess.highpass_hz < 0 || config->audio.preprocess.highpass_hz > 1000 ||
        config->audio.preprocess.lowpass_hz < 1000 || config->audio.preprocess.lowpass_hz > 3600 ||
        config->audio.preprocess.highpass_hz >= config->audio.preprocess.lowpass_hz ||
        config->audio.preprocess.noise_suppression_db < 0 ||
        config->audio.preprocess.noise_suppression_db > 36 ||
        config->audio.preprocess.noise_gate_snr_db < 0 ||
        config->audio.preprocess.noise_gate_snr_db > 24 ||
        config->audio.preprocess.noise_gate_attenuation_db < 0 ||
        config->audio.preprocess.noise_gate_attenuation_db > 60 ||
        config->audio.preprocess.noise_gate_hangover_ms < 0 ||
        config->audio.preprocess.noise_gate_hangover_ms > 2000 ||
        config->audio.preprocess.noise_warmup_ms < 0 ||
        config->audio.preprocess.noise_warmup_ms > 5000 ||
        config->audio.preprocess.agc_target_dbfs < -45 ||
        config->audio.preprocess.agc_target_dbfs > -3 ||
        config->audio.preprocess.agc_max_gain_db < 0 ||
        config->audio.preprocess.agc_max_gain_db > 36 ||
        config->audio.preprocess.voice_start_frames < 1 ||
        config->audio.preprocess.voice_start_frames > 10 ||
        config->audio.preprocess.voice_min_voicing_percent < 0 ||
        config->audio.preprocess.voice_min_voicing_percent > 100 ||
        (config->audio.codec2_mode_bps != 1200 && config->audio.codec2_mode_bps != 1300 &&
         config->audio.codec2_mode_bps != 1400 && config->audio.codec2_mode_bps != 1600 &&
         config->audio.codec2_mode_bps != 2400 && config->audio.codec2_mode_bps != 3200) ||
        (config->audio.enabled && (config->audio.device.empty() ||
                                   config->audio.udp_port == config->transport.udp_port ||
                                   config->audio.udp_port == config->transport.snapshot.udp_port ||
                                   config->audio.udp_port == config->transport.rebuild.udp_port ||
                                   config->audio.udp_port == config->transport.event.udp_port)) ||
        (config->transport.mode == TRANSPORT_MODE_IMAGE && config->mode == PIPELINE_BASELINE) ||
        (config->rate_profile == RATE_PROFILE_REBUILD && config->mode == PIPELINE_BASELINE) ||
        config->roi.cell_size != 16 || config->roi.max_regions <= 0 || config->roi.max_age_frames < 0 ||
        config->roi.hold_frames <= 0 || config->roi.erosion_radius < 0 || config->roi.dilation_radius < 0 ||
        config->camera.width <= 0 || config->camera.height <= 0 || config->camera.queue_depth <= 0 ||
        config->camera.max_frames < 0 ||
        config->preview_width <= 0 || config->preview_height <= 0 ||
        config->roi.occupancy_threshold < 0.0f ||
        config->roi.occupancy_threshold > 1.0f || config->encoder.qp_min < 0 ||
        config->encoder.qp_max > 51 || config->encoder.qp_min > config->encoder.qp_max ||
        config->encoder.qp_init < config->encoder.qp_min || config->encoder.qp_init > config->encoder.qp_max ||
        config->encoder.qp_min_i < config->encoder.qp_min || config->encoder.qp_max_i > config->encoder.qp_max ||
        config->encoder.qp_min_i > config->encoder.qp_max_i || config->encoder.intra_refresh_rows <= 0 ||
        config->encoder.max_reencode_times < 0 || config->encoder.max_reencode_times > 3 ||
        config->encoder.super_i_frame_bits <= 0 || config->encoder.super_p_frame_bits <= 0 ||
        config->roi.background_delta_qp < -51 || config->roi.background_delta_qp > 51 ||
        config->roi.halo_delta_qp < -51 || config->roi.halo_delta_qp > 51 ||
        config->roi.core_delta_qp < -51 || config->roi.core_delta_qp > 51 ||
        config->roi.edge_delta_qp < -51 || config->roi.edge_delta_qp > 51) {
        if (error) *error = "invalid encoder, ROI, or transport range";
        return false;
    }
    return true;
}

}  // namespace roi_h265

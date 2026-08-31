#include <cstdlib>
#include <iostream>
#include <string>

#include "common/config.h"
#include "roi/roi_mapper.h"
#include "roi/roi_region_merger.h"
#include "roi/roi_temporal.h"

using roi_h265::AppConfig;
using roi_h265::BBox;
using roi_h265::ROI_BACKGROUND;
using roi_h265::ROI_CORE;
using roi_h265::ROI_EDGE;
using roi_h265::ROI_HALO;
using roi_h265::RoiConfig;
using roi_h265::RoiLevel;
using roi_h265::RoiMap;
using roi_h265::RoiMapper;
using roi_h265::RoiManager;
using roi_h265::RoiRegionMerger;
using roi_h265::RoiTemporalSmoother;
using roi_h265::SegInstance;
using roi_h265::SegResult;

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

SegResult rectSeg(int width, int height, int left, int top, int right, int bottom, uint64_t frame_id) {
    SegResult result;
    result.frame.frame_id = frame_id;
    result.frame.pts_us = frame_id * 333333;
    result.source_width = width;
    result.source_height = height;
    SegInstance instance;
    instance.class_id = 0;
    instance.confidence = 0.9f;
    instance.bbox.left = left;
    instance.bbox.top = top;
    instance.bbox.right = right;
    instance.bbox.bottom = bottom;
    instance.mask_width = width;
    instance.mask_height = height;
    instance.mask.assign(static_cast<size_t>(width) * height, 0);
    for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) instance.mask[static_cast<size_t>(y) * width + x] = 1;
    }
    result.instances.push_back(instance);
    return result;
}

RoiMap uniformMap(uint64_t frame_id, RoiLevel level) {
    RoiMap map;
    map.source_frame.frame_id = frame_id;
    map.source_frame.pts_us = frame_id * 333333;
    map.frame_width = 16;
    map.frame_height = 16;
    map.cell_size = 16;
    map.grid_width = 1;
    map.grid_height = 1;
    map.cells.assign(1, level);
    return map;
}

void testSegmentationHierarchy() {
    RoiConfig config;
    config.erosion_radius = 3;
    config.dilation_radius = 3;
    config.occupancy_threshold = 0.10f;
    RoiMapper mapper(config);
    const SegResult segmentation = rectSeg(128, 128, 32, 32, 96, 96, 1);
    const RoiMap map = mapper.build(segmentation, 128, 128);
    CHECK(map.cell_size == 16);
    CHECK(map.grid_width == 8 && map.grid_height == 8);
    CHECK(map.at(0, 0) == ROI_BACKGROUND);
    CHECK(map.at(1, 3) == ROI_HALO);
    CHECK(map.at(2, 3) == ROI_EDGE);
    CHECK(map.at(4, 4) == ROI_CORE);
}

void testSourceToEncoderMapping() {
    RoiConfig config;
    config.erosion_radius = 0;
    config.dilation_radius = 0;
    config.occupancy_threshold = 0.5f;
    RoiMapper mapper(config);
    const SegResult segmentation = rectSeg(128, 128, 64, 32, 128, 96, 2);
    const RoiMap map = mapper.build(segmentation, 64, 64);
    CHECK(map.grid_width == 4 && map.grid_height == 4);
    CHECK(map.at(1, 1) == ROI_BACKGROUND);
    CHECK(map.at(2, 1) == ROI_CORE);
    CHECK(map.at(3, 2) == ROI_CORE);
}

void testTemporalHysteresisAndAge() {
    RoiConfig config;
    config.hold_frames = 2;
    config.max_age_frames = 6;
    RoiTemporalSmoother smoother(config);
    const RoiMap edge = uniformMap(10, ROI_EDGE);
    const RoiMap background = uniformMap(11, ROI_BACKGROUND);
    CHECK(smoother.update(edge).cells[0] == ROI_EDGE);
    CHECK(smoother.update(background).cells[0] == ROI_EDGE);
    CHECK(smoother.update(background).cells[0] == ROI_CORE);
    CHECK(smoother.update(background).cells[0] == ROI_CORE);
    CHECK(smoother.update(background).cells[0] == ROI_HALO);
    CHECK(smoother.update(background).cells[0] == ROI_HALO);
    CHECK(smoother.update(background).cells[0] == ROI_BACKGROUND);

    RoiManager manager(config);
    manager.submit(edge);
    CHECK(manager.select(11, 3333329, 16, 16).cells[0] == ROI_BACKGROUND);
    CHECK(manager.select(11, 3666666, 16, 16).cells[0] == ROI_EDGE);
    CHECK(manager.select(13, 4333333, 16, 16).cells[0] == ROI_CORE);
    CHECK(manager.select(17, 5666666, 16, 16).cells[0] == ROI_BACKGROUND);
}

void testRegionMergeAndPriorityLimit() {
    RoiConfig config;
    config.max_regions = 2;
    RoiMap map;
    map.frame_width = 64;
    map.frame_height = 16;
    map.cell_size = 16;
    map.grid_width = 4;
    map.grid_height = 1;
    map.cells.push_back(ROI_EDGE);
    map.cells.push_back(ROI_CORE);
    map.cells.push_back(ROI_HALO);
    map.cells.push_back(ROI_BACKGROUND);
    RoiRegionMerger merger(config);
    const std::vector<roi_h265::RoiRegion> regions = merger.merge(map);
    CHECK(regions.size() == 2);
    CHECK(regions[0].level == ROI_EDGE && regions[0].delta_qp == config.edge_delta_qp);
    CHECK(regions[1].level == ROI_CORE && regions[1].delta_qp == config.core_delta_qp);

    map.cells.assign(4, ROI_CORE);
    const std::vector<roi_h265::RoiRegion> merged = merger.merge(map);
    CHECK(merged.size() == 1);
    CHECK(merged[0].x == 0 && merged[0].width == 64 && merged[0].height == 16);
}

void testCommandLineConfig() {
    char app[] = "roi_sender";
    char width[] = "--encoder-width=320";
    char height[] = "--encoder-height=180";
    char fps[] = "--fps=10";
    char bit_rate[] = "--target-bitrate=42000";
    char bg[] = "--background-delta-qp=12";
    char age[] = "--roi-max-age=9";
    char destination[] = "--udp-host=192.168.1.9";
    char pacing[] = "--pacing-bitrate=60000";
    char grayscale[] = "--grayscale-encode=off";
    char preview_off[] = "--preview=off";
    char preview_width[] = "--preview-width=960";
    char preview_height[] = "--preview-height=720";
    char preview_rotate_none[] = "--preview-rotate=none";
    char qp_init[] = "--qp-init=41";
    char send_latency[] = "--send-max-latency-ms=400";
    char *argv[] = {app, width, height, fps, bit_rate, bg, age, destination, pacing,
                    grayscale, preview_off, preview_width, preview_height, preview_rotate_none,
                    qp_init, send_latency};
    AppConfig config;
    std::string error;
    CHECK(roi_h265::parseAppConfig(16, argv, &config, &error));
    CHECK(config.encoder.width == 320 && config.encoder.height == 180);
    CHECK(config.encoder.fps == 10);
    CHECK(config.encoder.target_bitrate_bps == 42000);
    CHECK(config.transport.pacing_bitrate_bps == 60000);
    CHECK(!config.encoder.grayscale_encode);
    CHECK(config.transport.udp_host == "192.168.1.9");
    CHECK(config.encoder.qp_init == 41);
    CHECK(config.transport.send_max_latency_ms == 400);
    CHECK(!config.preview);
    CHECK(config.preview_width == 960 && config.preview_height == 720);
    CHECK(!config.preview_rotate_ccw);

    AppConfig defaults;
    CHECK(defaults.encoder.width == 320 && defaults.encoder.height == 180);
    CHECK(defaults.encoder.fps == 10 && defaults.encoder.target_bitrate_bps == 42000);
    CHECK(defaults.transport.pacing_bitrate_bps == 60000);
    CHECK(defaults.encoder.grayscale_encode);
    CHECK(defaults.transport.profile_control_path == "/tmp/roi-rate-profile");
    CHECK(defaults.preview_width == 720 && defaults.preview_height == 1280);
    CHECK(defaults.preview_rotate_ccw);
    CHECK(!defaults.audio.enabled && defaults.audio.device == "hw:3,0");
    CHECK(defaults.audio.udp_port == 5006 && defaults.audio.codec2_mode_bps == 2400);
    CHECK(defaults.audio.frames_per_packet == 4);
    CHECK(defaults.audio.dtx_enabled && defaults.audio.dtx_preroll_ms == 80);
    CHECK(defaults.audio.dtx_hangover_ms == 600);
    CHECK(defaults.audio.dtx_keepalive_ms == 1000);
    CHECK(defaults.audio.reserve_wire_bitrate_bps == 0 && defaults.audio.max_latency_ms == 160);
    CHECK(defaults.audio.capture_rate_hz == 44100 && defaults.audio.capture_channels == 2);
    CHECK(defaults.audio.preprocess.enabled);
    CHECK(defaults.audio.preprocess.highpass_hz == 80);
    CHECK(defaults.audio.preprocess.lowpass_hz == 3600);
    CHECK(defaults.audio.preprocess.noise_suppression_db == 6);
    CHECK(defaults.audio.preprocess.noise_gate_snr_db == 2);
    CHECK(defaults.audio.preprocess.noise_gate_attenuation_db == 30);
    CHECK(defaults.audio.preprocess.noise_gate_hangover_ms == 600);
    CHECK(defaults.audio.preprocess.noise_warmup_ms == 600);
    CHECK(defaults.audio.preprocess.agc_target_dbfs == -16);
    CHECK(defaults.audio.preprocess.agc_max_gain_db == 20);
    CHECK(defaults.audio.preprocess.voice_activity_detection);
    CHECK(defaults.audio.preprocess.voice_start_frames == 2);
    CHECK(defaults.audio.preprocess.voice_min_voicing_percent == 42);

    char audio_on[] = "--audio=on";
    char audio_port[] = "--audio-udp-port=5006";
    char audio_rate[] = "--audio-capture-rate=44100";
    char audio_channels[] = "--audio-channels=2";
    char audio_mode[] = "--audio-codec2-mode=1300";
    char audio_frames[] = "--audio-frames-per-packet=2";
    char audio_dtx[] = "--audio-dtx=on";
    char audio_dtx_preroll[] = "--audio-dtx-preroll-ms=120";
    char audio_dtx_hangover[] = "--audio-dtx-hangover-ms=640";
    char audio_dtx_keepalive[] = "--audio-dtx-keepalive-ms=800";
    char audio_reserve[] = "--audio-reserve-bitrate=9200";
    char audio_max_latency[] = "--audio-max-latency-ms=120";
    char audio_preprocess[] = "--audio-preprocess=on";
    char audio_highpass[] = "--audio-highpass-hz=100";
    char audio_lowpass[] = "--audio-lowpass-hz=3300";
    char audio_denoise[] = "--audio-noise-suppression-db=15";
    char audio_gate_snr[] = "--audio-noise-gate-snr-db=4";
    char audio_gate_attenuation[] = "--audio-noise-gate-attenuation-db=28";
    char audio_gate_hangover[] = "--audio-noise-gate-hangover-ms=300";
    char audio_warmup[] = "--audio-noise-warmup-ms=500";
    char audio_agc_target[] = "--audio-agc-target-dbfs=-18";
    char audio_agc_gain[] = "--audio-agc-max-gain-db=20";
    char audio_vad[] = "--audio-vad=off";
    char audio_vad_start[] = "--audio-vad-start-frames=3";
    char audio_vad_voicing[] = "--audio-vad-min-voicing=65";
    char *audio_argv[] = {app, audio_on, audio_port, audio_rate, audio_channels, audio_mode,
                          audio_frames, audio_dtx, audio_dtx_preroll, audio_dtx_hangover,
                          audio_dtx_keepalive,
                          audio_reserve, audio_max_latency,
                          audio_preprocess, audio_highpass, audio_lowpass,
                          audio_denoise, audio_gate_snr, audio_gate_attenuation,
                          audio_gate_hangover, audio_warmup, audio_agc_target, audio_agc_gain,
                          audio_vad, audio_vad_start, audio_vad_voicing};
    AppConfig audio;
    error.clear();
    CHECK(roi_h265::parseAppConfig(static_cast<int>(sizeof(audio_argv) / sizeof(audio_argv[0])),
                                   audio_argv, &audio, &error));
    CHECK(audio.audio.enabled && audio.audio.udp_port == 5006);
    CHECK(audio.audio.capture_rate_hz == 44100 && audio.audio.capture_channels == 2);
    CHECK(audio.audio.codec2_mode_bps == 1300 && audio.audio.frames_per_packet == 2);
    CHECK(audio.audio.dtx_enabled && audio.audio.dtx_preroll_ms == 120);
    CHECK(audio.audio.dtx_hangover_ms == 640);
    CHECK(audio.audio.dtx_keepalive_ms == 800);
    CHECK(audio.audio.reserve_wire_bitrate_bps == 9200 && audio.audio.max_latency_ms == 120);
    CHECK(audio.audio.preprocess.enabled && audio.audio.preprocess.highpass_hz == 100);
    CHECK(audio.audio.preprocess.lowpass_hz == 3300);
    CHECK(audio.audio.preprocess.noise_suppression_db == 15);
    CHECK(audio.audio.preprocess.noise_gate_snr_db == 4);
    CHECK(audio.audio.preprocess.noise_gate_attenuation_db == 28);
    CHECK(audio.audio.preprocess.noise_gate_hangover_ms == 300);
    CHECK(audio.audio.preprocess.noise_warmup_ms == 500);
    CHECK(audio.audio.preprocess.agc_target_dbfs == -18);
    CHECK(audio.audio.preprocess.agc_max_gain_db == 20);
    CHECK(!audio.audio.preprocess.voice_activity_detection);
    CHECK(audio.audio.preprocess.voice_start_frames == 3);
    CHECK(audio.audio.preprocess.voice_min_voicing_percent == 65);

    char profile_app[] = "roi_sender";
    char profile_high[] = "--rate-profile=high";
    char *profile_argv[] = {profile_app, profile_high};
    AppConfig high;
    CHECK(roi_h265::parseAppConfig(2, profile_argv, &high, &error));
    CHECK(high.encoder.width == 640 && high.encoder.height == 360);
    CHECK(high.encoder.fps == 20 && high.transport.pacing_bitrate_bps == 300000);
    CHECK(!high.encoder.grayscale_encode);

    char profile_medium[] = "--profile=medium";
    char *medium_argv[] = {profile_app, profile_medium};
    AppConfig medium;
    error.clear();
    CHECK(roi_h265::parseAppConfig(2, medium_argv, &medium, &error));
    CHECK(medium.encoder.width == 480 && medium.encoder.height == 270);
    CHECK(medium.encoder.fps == 15 && medium.transport.pacing_bitrate_bps == 150000);

    char profile_rebuild[] = "--profile=rebuild";
    char *rebuild_argv[] = {profile_app, profile_rebuild};
    AppConfig rebuild;
    error.clear();
    CHECK(roi_h265::parseAppConfig(2, rebuild_argv, &rebuild, &error));
    CHECK(rebuild.rate_profile == roi_h265::RATE_PROFILE_REBUILD);
    CHECK(rebuild.encoder.width == 256 && rebuild.encoder.height == 144);
    CHECK(rebuild.encoder.fps == 6 && rebuild.encoder.target_bitrate_bps == 28000);
    CHECK(rebuild.transport.pacing_bitrate_bps == 100000);
    CHECK(rebuild.transport.rebuild.udp_port == 5009);
    CHECK(rebuild.transport.rebuild.output_width == 640);
    CHECK(rebuild.transport.rebuild.output_height == 360);
    CHECK(rebuild.transport.rebuild.output_fps == 12);
    CHECK(rebuild.transport.rebuild.patch_max_side == 96);
    CHECK(rebuild.transport.rebuild.patch_jpeg_quality == 50);
    CHECK(rebuild.transport.rebuild.patch_max_bytes == 800);
    CHECK(rebuild.transport.rebuild.patch_soft_refresh_ms == 350);
    CHECK(rebuild.transport.rebuild.patch_hard_deadline_ms == 450);
    CHECK(rebuild.transport.rebuild.patch_refresh_guard_ms == 75);
    CHECK(!rebuild.encoder.grayscale_encode);
    CHECK(std::string(roi_h265::rateProfileName(rebuild.rate_profile)) == "rebuild");

    char rebuild_soft_refresh[] = "--rebuild-reference-soft-refresh-ms=180";
    char rebuild_hard_deadline[] = "--rebuild-reference-hard-deadline-ms=450";
    char rebuild_refresh_guard[] = "--rebuild-reference-refresh-guard-ms=60";
    char *rebuild_timing_argv[] = {profile_app, profile_rebuild, rebuild_soft_refresh,
                                   rebuild_hard_deadline, rebuild_refresh_guard};
    AppConfig rebuild_timing;
    error.clear();
    CHECK(roi_h265::parseAppConfig(5, rebuild_timing_argv, &rebuild_timing, &error));
    CHECK(rebuild_timing.transport.rebuild.patch_soft_refresh_ms == 180);
    CHECK(rebuild_timing.transport.rebuild.patch_hard_deadline_ms == 450);
    CHECK(rebuild_timing.transport.rebuild.patch_refresh_guard_ms == 60);

    char rebuild_bad_soft_refresh[] = "--rebuild-reference-soft-refresh-ms=450";
    char *rebuild_bad_timing_argv[] = {profile_app, profile_rebuild,
                                       rebuild_bad_soft_refresh};
    AppConfig rebuild_bad_timing;
    error.clear();
    CHECK(!roi_h265::parseAppConfig(3, rebuild_bad_timing_argv, &rebuild_bad_timing,
                                    &error));

    char rebuild_override[] = "--encoder-width=320";
    char *invalid_rebuild_argv[] = {profile_app, profile_rebuild, rebuild_override};
    AppConfig invalid_rebuild;
    error.clear();
    CHECK(!roi_h265::parseAppConfig(3, invalid_rebuild_argv, &invalid_rebuild, &error));
    CHECK(error.find("rebuild is an atomic") != std::string::npos);

    roi_h265::RateProfile parsed_profile = roi_h265::RATE_PROFILE_LOW;
    CHECK(roi_h265::parseRateProfile("high", &parsed_profile));
    CHECK(parsed_profile == roi_h265::RATE_PROFILE_HIGH);
    roi_h265::applyRateProfile(parsed_profile, &medium);
    CHECK(medium.rate_profile == roi_h265::RATE_PROFILE_HIGH);
    CHECK(medium.encoder.width == 640 && medium.encoder.fps == 20);
    CHECK(std::string(roi_h265::rateProfileName(medium.rate_profile)) == "high");

    char bad_mtu[] = "--mtu=63";
    char *bad_argv[] = {app, bad_mtu};
    CHECK(!roi_h265::parseAppConfig(2, bad_argv, &config, &error));

    char bad_preview_width[] = "--preview-width=0";
    char *bad_preview_argv[] = {app, bad_preview_width};
    CHECK(!roi_h265::parseAppConfig(2, bad_preview_argv, &config, &error));

    char bad_preview_rotate[] = "--preview-rotate=cw";
    char *bad_preview_rotate_argv[] = {app, bad_preview_rotate};
    CHECK(!roi_h265::parseAppConfig(2, bad_preview_rotate_argv, &config, &error));

    char audio_same_port[] = "--audio=on";
    char audio_bad_port[] = "--audio-udp-port=5004";
    char *bad_audio_argv[] = {app, audio_same_port, audio_bad_port};
    CHECK(!roi_h265::parseAppConfig(3, bad_audio_argv, &config, &error));

    char audio_bad_lowpass[] = "--audio-lowpass-hz=3800";
    char *bad_audio_preprocess_argv[] = {app, audio_bad_lowpass};
    CHECK(!roi_h265::parseAppConfig(2, bad_audio_preprocess_argv, &config, &error));

    char audio_bad_dtx_preroll[] = "--audio-dtx-preroll-ms=501";
    char *bad_audio_dtx_argv[] = {app, audio_bad_dtx_preroll};
    CHECK(!roi_h265::parseAppConfig(2, bad_audio_dtx_argv, &config, &error));

    char audio_bad_dtx_hangover[] = "--audio-dtx-hangover-ms=2001";
    char *bad_audio_dtx_hangover_argv[] = {app, audio_bad_dtx_hangover};
    CHECK(!roi_h265::parseAppConfig(2, bad_audio_dtx_hangover_argv, &config, &error));

    char audio_bad_latency[] = "--audio-max-latency-ms=19";
    char *bad_audio_latency_argv[] = {app, audio_bad_latency};
    CHECK(!roi_h265::parseAppConfig(2, bad_audio_latency_argv, &config, &error));
}

}  // namespace

int main() {
    testSegmentationHierarchy();
    testSourceToEncoderMapping();
    testTemporalHysteresisAndAge();
    testRegionMergeAndPriorityLimit();
    testCommandLineConfig();
    if (failures != 0) {
        std::cerr << failures << " ROI core test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "ROI core tests passed\n";
    return EXIT_SUCCESS;
}

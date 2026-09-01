#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "audio/codec2_audio_sender.h"
#include "capture/camera_capture.h"
#include "common/config.h"
#include "common/statistics.h"
#include "encoder/mpp_h265_encoder.h"
#include "inference/yolov8_seg.h"
#include "preview/board_preview.h"
#include "roi/roi_debug.h"
#include "roi/roi_mapper.h"
#include "roi/roi_region_merger.h"
#include "roi/roi_temporal.h"
#include "transport/h265_sdp.h"
#include "transport/async_rtp_sender.h"
#include "transport/codec2_sdp.h"
#include "transport/detection_event_sender.h"
#include "transport/rate_pacer.h"
#include "transport/rebuild_sender.h"
#include "transport/snapshot_protocol.h"
#include "transport/snapshot_sender.h"

namespace roi_h265 {
namespace {

template <typename T>
class LatestQueue {
public:
    explicit LatestQueue(size_t capacity) : capacity_(capacity), stopped_(false) {}

    void push(const T &item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) return;
        while (items_.size() >= capacity_) items_.pop_front();
        items_.push_back(item);
        condition_.notify_one();
    }

    bool pop(T *item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopped_ || !items_.empty(); });
        if (items_.empty()) return false;
        *item = items_.front();
        items_.pop_front();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        condition_.notify_all();
    }

private:
    size_t capacity_;
    bool stopped_;
    std::deque<T> items_;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::string debugPathForFrame(const std::string &pattern, uint64_t frame_id) {
    const std::string token("{frame_id}");
    const std::string::size_type position = pattern.find(token);
    if (position == std::string::npos) return pattern;
    return pattern.substr(0, position) + std::to_string(frame_id) +
        pattern.substr(position + token.size());
}

void printUsage(const char *program) {
    std::fprintf(stderr,
        "Usage: %s [--rate-profile=low|medium|high|rebuild] [--model=PATH] [--camera-device=/dev/video0] [--mode=baseline|bbox|segmentation]\n"
        "          [--input-video=PATH --max-frames=N]\n"
        "          [--encoder-width=320 --encoder-height=180 --fps=10 --target-bitrate=42000]\n"
        "          [--gop=50 --qp-min=10 --qp-max=51 --qp-init=38 --qp-min-i=36 --qp-max-i=48]\n"
        "          [--intra-refresh=on --intra-refresh-rows=1 --max-reencode-times=3]\n"
        "          [--super-i-frame-bits=12000 --super-p-frame-bits=5500]\n"
        "          [--grayscale-encode=on|off]\n"
        "          [--background-delta-qp=12 --halo-delta-qp=2]\n"
        "          [--core-delta-qp=-6 --edge-delta-qp=-10 --mask-occupancy-threshold=0.10]\n"
        "          [--erosion-radius=2 --dilation-radius=3 --roi-hold-frames=3 --roi-max-age=9]\n"
        "          [--max-roi-region=64 --udp-host=HOST --udp-port=5004 --pacing-bitrate=60000]\n"
        "          [--send-queue-frames=3 --send-max-latency-ms=250]\n"
        "          [--rtp-sdp-path=/tmp/roi-live.sdp]\n"
        "          [--audio=on|off --audio-device=hw:3,0 --audio-udp-port=5006]\n"
        "          [--audio-capture-rate=44100 --audio-channels=2 --audio-codec2-mode=2400]\n"
        "          [--audio-frames-per-packet=4 --audio-rtp-sdp-path=/tmp/roi-audio.sdp]\n"
        "          [--audio-dtx=on --audio-dtx-preroll-ms=80 --audio-dtx-keepalive-ms=1000]\n"
        "          [--audio-reserve-bitrate=0 --audio-max-latency-ms=160]\n"
        "          [--audio-preprocess=on --audio-highpass-hz=80 --audio-lowpass-hz=3600]\n"
        "          [--audio-noise-suppression-db=6 --audio-noise-gate-snr-db=3]\n"
        "          [--audio-noise-gate-attenuation-db=30 --audio-noise-gate-hangover-ms=240]\n"
        "          [--audio-noise-warmup-ms=600 --audio-agc-target-dbfs=-18 --audio-agc-max-gain-db=12]\n"
        "          [--audio-vad=on --audio-vad-start-frames=2 --audio-vad-min-voicing=52]\n"
        "          [--profile-control=/tmp/roi-rate-profile --transport-mode=video|image]\n"
        "          [--snapshot-udp-port=5008 --snapshot-jpeg-quality=75 --snapshot-min-interval-ms=5000]\n"
        "          [--snapshot-crop=relevant|full --snapshot-crop-margin-percent=25 --snapshot-rotate=ccw|none]\n"
        "          [--snapshot-max-width=1280 --snapshot-max-height=720 --snapshot-chunk-bytes=1100]\n"
        "          [--snapshot-ack-timeout-ms=300 --snapshot-max-retries=20 --snapshot-min-confidence=0.35]\n"
        "          [--rebuild-udp-port=5009 --rebuild-output-width=640 --rebuild-output-height=360]\n"
        "          [--rebuild-output-fps=12 --rebuild-min-confidence=0.35 --rebuild-max-targets=2]\n"
        "          [--rebuild-patch-max-side=96 --rebuild-jpeg-quality=50 --rebuild-patch-max-bytes=800]\n"
        "          [--rebuild-reference-soft-refresh-ms=350 --rebuild-reference-hard-deadline-ms=450]\n"
        "          [--rebuild-reference-refresh-guard-ms=75 --rebuild-chunk-bytes=1100]\n"
        "          [--rebuild-patch-packets-per-frame=2]\n"
        "          [--rebuild-crop-margin-percent=20 --rebuild-parity=on|off]\n"
        "          [--event-push=on|off --event-udp-port=5010 --event-min-confidence=0.35]\n"
        "          [--event-heartbeat-ms=1000]\n"
        "          [--preview=on|off --preview-rotate=ccw|none --preview-width=720 --preview-height=1280]\n"
        "          [--debug-roi=on --debug-roi-path=roi_map.pgm]\n", program);
}

void profileControlLoop(const std::string &path, std::atomic<bool> *running,
                        std::atomic<int> *requested_profile,
                        std::atomic<int> *requested_transport_mode) {
    if (path.empty() || !running || !requested_profile || !requested_transport_mode) return;
    struct stat info;
    if (::lstat(path.c_str(), &info) == 0) {
        if (!S_ISFIFO(info.st_mode)) {
            std::fprintf(stderr, "Profile control path exists but is not a FIFO: %s\n", path.c_str());
            running->store(false);
            return;
        }
    } else if (errno != ENOENT || ::mkfifo(path.c_str(), 0666) != 0) {
        std::fprintf(stderr, "Cannot create profile control FIFO %s: %s\n",
                     path.c_str(), std::strerror(errno));
        running->store(false);
        return;
    }
    const int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "Cannot open profile control FIFO %s: %s\n",
                     path.c_str(), std::strerror(errno));
        running->store(false);
        return;
    }
    std::fprintf(stderr,
        "Runtime control ready: echo low|medium|high|rebuild|image|video > %s\n", path.c_str());
    std::string pending;
    while (running->load()) {
        pollfd descriptor;
        descriptor.fd = fd;
        descriptor.events = POLLIN;
        descriptor.revents = 0;
        const int ready = ::poll(&descriptor, 1, 200);
        if (ready <= 0 || !(descriptor.revents & POLLIN)) continue;
        char buffer[128];
        const ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count <= 0) continue;
        pending.append(buffer, static_cast<size_t>(count));
        std::string::size_type newline = std::string::npos;
        while ((newline = pending.find('\n')) != std::string::npos) {
            const std::string command = pending.substr(0, newline);
            pending.erase(0, newline + 1);
            std::istringstream input(command);
            std::string name;
            input >> name;
            RateProfile profile;
            if (parseRateProfile(name, &profile)) {
                requested_profile->store(static_cast<int>(profile));
                requested_transport_mode->store(static_cast<int>(TRANSPORT_MODE_VIDEO));
                std::fprintf(stderr, "Runtime profile requested: %s\n", rateProfileName(profile));
            } else {
                TransportMode mode;
                if (parseTransportMode(name, &mode)) {
                    requested_transport_mode->store(static_cast<int>(mode));
                    std::fprintf(stderr, "Runtime transport mode requested: %s\n",
                                 transportModeName(mode));
                } else if (!name.empty()) {
                    std::fprintf(stderr, "Ignoring invalid runtime control '%s'\n", name.c_str());
                }
            }
        }
    }
    ::close(fd);
    ::unlink(path.c_str());
}

}  // namespace
}  // namespace roi_h265

int main(int argc, char **argv) {
    using namespace roi_h265;
    AppConfig config;
    std::string error;
    if (!parseAppConfig(argc, argv, &config, &error)) {
        std::fprintf(stderr, "configuration error: %s\n", error.c_str());
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (config.transport.pacing_bitrate_bps > 300000) {
        std::fprintf(stderr, "configuration error: pacing bitrate may not exceed the 300 kbps physical link\n");
        return EXIT_FAILURE;
    }

    std::unique_ptr<MppH265Encoder> encoder;
    if (config.transport.mode == TRANSPORT_MODE_VIDEO) {
        encoder.reset(new MppH265Encoder(config.encoder, config.roi));
        if (!encoder->initialize(&error)) {
            std::fprintf(stderr, "MPP H.265 initialization failed: %s\n", error.c_str());
            return EXIT_FAILURE;
        }
    }
    CameraCapture camera(config.camera, config.encoder);
    if (!camera.open(&error)) {
        std::fprintf(stderr, "camera initialization failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    if (config.preview) {
        BoardPreview::prepareEnvironment();
    }
    // The physical A/V ceiling is centrally partitioned into two child
    // buckets.  Their configured rates always add up to this one link cap;
    // each bucket still charges physical Ethernet bytes, not payload bytes.
    const size_t physical_burst_bytes = static_cast<size_t>(config.transport.mtu) + 38U;
    const std::shared_ptr<RatePacer> video_pacer(new RatePacer(
        config.transport.pacing_bitrate_bps, physical_burst_bytes));
    const std::shared_ptr<RatePacer> audio_pacer(new RatePacer(
        config.transport.pacing_bitrate_bps, physical_burst_bytes));
    std::unique_ptr<Codec2AudioSender> audio_sender;
    int video_pacing_bps = config.transport.pacing_bitrate_bps;
    size_t video_burst_bytes = physical_burst_bytes;
    if (config.audio.enabled) {
        audio_sender.reset(new Codec2AudioSender(config.audio, config.transport.udp_host,
            config.transport.mtu, config.transport.pacing_bitrate_bps, audio_pacer));
        if (!audio_sender->start(&error)) {
            std::fprintf(stderr, "Codec2 audio initialization failed: %s\n", error.c_str());
            return EXIT_FAILURE;
        }
        const Codec2AudioSenderSnapshot audio = audio_sender->snapshot();
        video_pacing_bps -= audio.reserved_wire_bitrate_bps;
        if (video_pacing_bps <= 0) {
            std::fprintf(stderr, "Audio reservation leaves no physical capacity for H.265 video\n");
            return EXIT_FAILURE;
        }
        video_burst_bytes = std::max<size_t>(1U,
            physical_burst_bytes - std::min(physical_burst_bytes - 1U,
                audio.reserved_burst_wire_bytes));
        video_pacer->reset(video_pacing_bps, video_burst_bytes);
        if (!config.audio.rtp_sdp_path.empty()) {
            Codec2RtpSdpConfig audio_sdp;
            audio_sdp.udp_port = config.audio.udp_port;
            audio_sdp.mode_bps = config.audio.codec2_mode_bps;
            audio_sdp.frames_per_packet = audio.frames_per_packet;
            audio_sdp.samples_per_frame = audio.samples_per_codec_frame;
            audio_sdp.bits_per_frame = audio.bits_per_codec_frame;
            audio_sdp.bytes_per_frame = audio.bytes_per_codec_frame;
            audio_sdp.dtx_enabled = config.audio.dtx_enabled;
            audio_sdp.dtx_keepalive_ms = config.audio.dtx_keepalive_ms;
            if (!writeCodec2RtpSdp(audio_sdp, config.audio.rtp_sdp_path, &error)) {
                std::fprintf(stderr, "Codec2 audio SDP export failed: %s\n", error.c_str());
                return EXIT_FAILURE;
            }
        }
        std::fprintf(stderr,
            "Codec2 audio enabled: %d bps, %d samples/frame, %d bytes/frame, %d frames/RTP, port %d\n",
            config.audio.codec2_mode_bps, audio.samples_per_codec_frame,
            audio.bytes_per_codec_frame, audio.frames_per_packet, config.audio.udp_port);
        std::fprintf(stderr,
            "Audio real-time transport: reserve %d bps, audio/video burst %zu/%zu B, whole-packet latency cap %d ms\n",
            audio.reserved_wire_bitrate_bps, audio.reserved_burst_wire_bytes,
            video_burst_bytes, config.audio.max_latency_ms);
        std::fprintf(stderr,
            "Audio preprocessing: %s, HP %d Hz, LP %d Hz, denoise %d dB, gate %d dB SNR/%d dB/%d ms, AGC %d dBFS max +%d dB, VAD %s/%d frames/%d%%\n",
            config.audio.preprocess.enabled ? "on" : "off",
            config.audio.preprocess.highpass_hz, config.audio.preprocess.lowpass_hz,
            config.audio.preprocess.noise_suppression_db,
            config.audio.preprocess.noise_gate_snr_db,
            config.audio.preprocess.noise_gate_attenuation_db,
            config.audio.preprocess.noise_gate_hangover_ms,
            config.audio.preprocess.agc_target_dbfs,
            config.audio.preprocess.agc_max_gain_db,
            config.audio.preprocess.voice_activity_detection ? "on" : "off",
            config.audio.preprocess.voice_start_frames,
            config.audio.preprocess.voice_min_voicing_percent);
        std::fprintf(stderr, "Audio DTX: %s, pre-roll %d ms, keepalive %d ms\n",
            config.audio.dtx_enabled ? "on" : "off", config.audio.dtx_preroll_ms,
            config.audio.dtx_keepalive_ms);
    }
    AsyncRtpSender sender(config.transport.udp_host, config.transport.udp_port,
                          video_pacing_bps, config.transport.mtu,
                          static_cast<size_t>(config.transport.send_queue_frames),
                          config.transport.send_max_latency_ms, video_pacer);
    if (!sender.start(&error)) {
        std::fprintf(stderr, "UDP initialization failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    SnapshotSender snapshot_sender(config.transport.snapshot, config.transport.udp_host,
                                   config.transport.mtu, video_pacer);
    if (!snapshot_sender.start(&error)) {
        std::fprintf(stderr, "snapshot UDP initialization failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    snapshot_sender.setEnabled(config.transport.mode == TRANSPORT_MODE_IMAGE);
    RebuildSender rebuild_sender(config.transport.rebuild, config.transport.udp_host,
                                 config.transport.mtu, video_pacer);
    if (!rebuild_sender.start(&error)) {
        std::fprintf(stderr, "rebuild UDP initialization failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    rebuild_sender.setEnabled(config.transport.mode == TRANSPORT_MODE_VIDEO &&
                              config.rate_profile == RATE_PROFILE_REBUILD);
    DetectionEventSender event_sender(config.transport.event, config.transport.udp_host,
                                      config.transport.mtu, video_pacer);
    if (!event_sender.start(&error)) {
        std::fprintf(stderr, "detection event UDP initialization failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    event_sender.setEnabled(config.transport.event.enabled);
    if (config.transport.event.enabled) {
        std::fprintf(stderr,
            "Detection event push enabled: person/car/boat/airplane UDP %d, confidence >= %.2f, heartbeat %d ms\n",
            config.transport.event.udp_port, config.transport.event.min_confidence,
            config.transport.event.heartbeat_ms);
    }
    if (config.rate_profile == RATE_PROFILE_REBUILD) {
        std::fprintf(stderr,
            "Rebuild profile enabled: base %dx%d@%d, output %dx%d@%d, semantic/reference UDP %d, combined physical ceiling %d bps, reference soft/hard/guard %d/%d/%d ms\n",
            config.encoder.width, config.encoder.height, config.encoder.fps,
            config.transport.rebuild.output_width, config.transport.rebuild.output_height,
            config.transport.rebuild.output_fps, config.transport.rebuild.udp_port,
            config.transport.pacing_bitrate_bps,
            config.transport.rebuild.patch_soft_refresh_ms,
            config.transport.rebuild.patch_hard_deadline_ms,
            config.transport.rebuild.patch_refresh_guard_ms);
    }
    if (config.transport.mode == TRANSPORT_MODE_IMAGE) {
        std::fprintf(stderr,
            "Image mode enabled: YOLO detects person/car/boat/airplane only; JPEG UDP %d crop=%s margin=%d%% rotate=%s\n",
            config.transport.snapshot.udp_port,
            config.transport.snapshot.crop_mode == SNAPSHOT_CROP_RELEVANT ? "relevant" : "full",
            config.transport.snapshot.crop_margin_percent,
            config.transport.snapshot.rotate_ccw ? "ccw" : "none");
    }

    rknn_app_context_t rknn_context;
    std::memset(&rknn_context, 0, sizeof(rknn_context));
    const bool use_inference = config.mode != PIPELINE_BASELINE;
    if (use_inference) {
        if (init_post_process() != 0 || init_yolov8_seg_model(config.model_path.c_str(), &rknn_context) != 0) {
            std::fprintf(stderr, "YOLOv8-Seg initialization failed for %s\n", config.model_path.c_str());
            deinit_post_process();
            return EXIT_FAILURE;
        }
    }

    LatestQueue<std::shared_ptr<FramePacket> > inference_queue(config.camera.queue_depth);
    LatestQueue<std::shared_ptr<FramePacket> > encoder_queue(config.camera.queue_depth);
    struct PreviewItem {
        std::shared_ptr<FramePacket> frame;
        SegResult segmentation;
        bool has_segmentation;

        PreviewItem() : has_segmentation(false) {}
    };
    LatestQueue<PreviewItem> preview_queue(1);
    RoiManager roi_manager(config.roi);
    RuntimeStatistics statistics;
    bool rtp_sdp_written = config.transport.rtp_sdp_path.empty();
    std::atomic<bool> running(true);
    std::atomic<int> requested_profile(static_cast<int>(config.rate_profile));
    std::atomic<int> active_profile(static_cast<int>(config.rate_profile));
    std::atomic<int> requested_transport_mode(static_cast<int>(config.transport.mode));
    std::atomic<int> active_transport_mode(static_cast<int>(config.transport.mode));
    std::atomic<unsigned int> profile_generation(0);
    std::mutex runtime_config_mutex;
    AppConfig runtime_config = config;
    const auto runtimeConfig = [&]() {
        std::lock_guard<std::mutex> lock(runtime_config_mutex);
        return runtime_config;
    };
    std::thread profile_control_thread;
    if (!config.transport.profile_control_path.empty()) {
        profile_control_thread = std::thread(profileControlLoop,
            config.transport.profile_control_path, &running, &requested_profile,
            &requested_transport_mode);
    }

    std::thread capture_thread([&] {
        std::string capture_error;
        std::chrono::steady_clock::time_point next_capture = std::chrono::steady_clock::now();
        while (running.load()) {
            std::this_thread::sleep_until(next_capture);
            // Advance from the current time rather than trying to catch up
            // after a slow V4L2/RGA read. This preserves the configured output
            // rate and avoids a burst into the latest-only queues.
            const AppConfig live = runtimeConfig();
            const std::chrono::microseconds frame_interval(1000000 / live.encoder.fps);
            next_capture = std::chrono::steady_clock::now() + frame_interval;
            std::shared_ptr<FramePacket> frame(new FramePacket);
            if (!camera.read(frame.get(), &capture_error)) {
                std::fprintf(stderr, "capture error: %s\n", capture_error.c_str());
                running.store(false);
                break;
            }
            if (config.camera.max_frames > 0 && frame->meta.frame_id >= static_cast<uint64_t>(config.camera.max_frames)) {
                break;
            }
            if (use_inference) inference_queue.push(frame);
            else if (config.preview) {
                PreviewItem preview_item;
                preview_item.frame = frame;
                preview_queue.push(preview_item);
            }
            encoder_queue.push(frame);
        }
        inference_queue.stop();
        encoder_queue.stop();
        if (!use_inference) preview_queue.stop();
    });

    std::thread inference_thread;
    if (use_inference) {
        inference_thread = std::thread([&] {
            std::shared_ptr<FramePacket> frame;
            while (running.load() && inference_queue.pop(&frame)) {
                image_buffer_t image;
                std::memset(&image, 0, sizeof(image));
                image.width = frame->source_width;
                image.height = frame->source_height;
                image.width_stride = image.width;
                image.height_stride = image.height;
                image.format = IMAGE_FORMAT_RGB888;
                image.virt_addr = frame->rgb.data();
                image.size = static_cast<int>(frame->rgb.size());
                image.fd = -1;
                SegResult result;
                if (!runYolov8Seg(&rknn_context, &image, frame->meta, &result)) {
                    std::fprintf(stderr, "YOLOv8-Seg inference failed on frame %llu\n",
                                 static_cast<unsigned long long>(frame->meta.frame_id));
                    continue;
                }
                const AppConfig live = runtimeConfig();
                // Publish semantic events from the raw RKNN result.  ROI/image
                // filtering may use a different threshold, so it must not hide
                // a configured event from the independent ROEV/1 channel.
                std::string event_error;
                if (!event_sender.submit(result, &event_error)) {
                    std::fprintf(stderr, "Detection event submit failed: %s\n",
                                 event_error.c_str());
                    running.store(false);
                    break;
                }
                const bool rebuild_active =
                    static_cast<TransportMode>(active_transport_mode.load()) ==
                        TRANSPORT_MODE_VIDEO &&
                    static_cast<RateProfile>(active_profile.load()) == RATE_PROFILE_REBUILD;
                filterRelevantDetections(&result, rebuild_active
                    ? live.transport.rebuild.min_confidence
                    : live.transport.snapshot.min_confidence);
                statistics.recordSegmentation(result);
                if (config.preview) {
                    PreviewItem preview_item;
                    preview_item.frame = frame;
                    preview_item.segmentation = result;
                    preview_item.has_segmentation = true;
                    preview_queue.push(preview_item);
                }
                const uint16_t class_mask = relevantDetectionClassMask(result, 0.0f);
                if (class_mask != 0 && static_cast<TransportMode>(
                    active_transport_mode.load()) == TRANSPORT_MODE_IMAGE) {
                    std::string snapshot_error;
                    const SnapshotCrop crop = live.transport.snapshot.crop_mode ==
                        SNAPSHOT_CROP_RELEVANT ? snapshotCropForRelevantDetections(
                            result, frame->source_width, frame->source_height,
                            live.transport.snapshot.crop_margin_percent) : SnapshotCrop();
                    if (!snapshot_sender.submit(frame, class_mask, crop, &snapshot_error)) {
                        std::fprintf(stderr, "Snapshot submit failed: %s\n", snapshot_error.c_str());
                    }
                }
                // In image mode the pipeline is detection plus evidence upload
                // only: no ROI/QP map is built because MPP is shut down.  Video
                // mode retains the original asynchronous ROI handoff.
                if (static_cast<TransportMode>(active_transport_mode.load()) ==
                    TRANSPORT_MODE_VIDEO) {
                    const RoiMapper mapper(live.roi);
                    const RoiMap map = config.mode == PIPELINE_BBOX_ROI
                        ? mapper.buildBboxMap(result, live.encoder.width, live.encoder.height)
                        : mapper.build(result, live.encoder.width, live.encoder.height);
                    roi_manager.submit(map);
                    if (static_cast<RateProfile>(active_profile.load()) ==
                        RATE_PROFILE_REBUILD) {
                        std::string rebuild_error;
                        if (!rebuild_sender.submit(frame, result,
                            static_cast<uint8_t>(profile_generation.load() & 0xffU),
                            live.encoder.fps, &rebuild_error)) {
                            std::fprintf(stderr, "Rebuild submit failed: %s\n",
                                         rebuild_error.c_str());
                        }
                    }
                }
            }
            preview_queue.stop();
        });
    }

    std::thread encoder_thread([&] {
        std::string encoder_error;
        std::shared_ptr<FramePacket> frame;
        uint64_t encoded_frame_count = 0;
        while (running.load() && encoder_queue.pop(&frame)) {
            const RateProfile desired = static_cast<RateProfile>(requested_profile.load());
            const TransportMode desired_transport = static_cast<TransportMode>(
                requested_transport_mode.load());
            const TransportMode active_transport = static_cast<TransportMode>(
                active_transport_mode.load());
            if (desired_transport != active_transport ||
                (desired_transport == TRANSPORT_MODE_VIDEO &&
                 desired != static_cast<RateProfile>(active_profile.load()))) {
                AppConfig next = runtimeConfig();
                if (desired != static_cast<RateProfile>(active_profile.load())) {
                    applyRateProfile(desired, &next);
                }
                next.transport.mode = desired_transport;
                const int audio_reserve_bps = audio_sender
                    ? audio_sender->snapshot().reserved_wire_bitrate_bps : 0;
                const int next_video_pacing_bps = next.transport.pacing_bitrate_bps -
                    audio_reserve_bps;
                if (next_video_pacing_bps <= 0) {
                    if (encoder_error.empty()) {
                        encoder_error = "audio reservation leaves no capacity in the shared media bucket";
                    }
                    std::fprintf(stderr, "Runtime transport switch failed: %s\n", encoder_error.c_str());
                    running.store(false);
                    break;
                }
                if (desired_transport == TRANSPORT_MODE_IMAGE) {
                    rebuild_sender.setEnabled(false);
                    // Finish at most one H.265 access unit, discard the rest,
                    // then hand the same media bucket to the screenshot worker.
                    if (!sender.switchProfile(next_video_pacing_bps, &encoder_error)) {
                        std::fprintf(stderr, "RTP image-mode switch failed: %s\n",
                                     encoder_error.c_str());
                        running.store(false);
                        break;
                    }
                    video_pacer->reset(next_video_pacing_bps, video_burst_bytes);
                    if (encoder) {
                        encoder->shutdown();
                        encoder.reset();
                    }
                    camera.updateEncoderConfig(next.encoder);
                    roi_manager.reconfigure(next.roi);
                    snapshot_sender.setEnabled(true);
                    {
                        std::lock_guard<std::mutex> lock(runtime_config_mutex);
                        runtime_config = next;
                    }
                    active_profile.store(static_cast<int>(desired));
                    active_transport_mode.store(static_cast<int>(TRANSPORT_MODE_IMAGE));
                    const unsigned int generation = profile_generation.fetch_add(1) + 1;
                    encoded_frame_count = 0;
                    rtp_sdp_written = next.transport.rtp_sdp_path.empty();
                    std::fprintf(stderr,
                        "Runtime image mode applied: YOLO-only detection, JPEG port %d, source %dx%d, pacing=%d generation=%u\n",
                        next.transport.snapshot.udp_port, next.camera.width, next.camera.height,
                        next_video_pacing_bps, generation);
                    continue;
                }

                // Stop-and-wait snapshot traffic is cancelled and joined before
                // H.265 is allowed to reclaim the shared media bucket.
                snapshot_sender.setEnabled(false);
                rebuild_sender.setEnabled(false);
                if (!sender.switchProfile(next_video_pacing_bps, &encoder_error)) {
                    std::fprintf(stderr, "RTP profile switch failed: %s\n", encoder_error.c_str());
                    running.store(false);
                    break;
                }
                // switchProfile restores UdpSender's default burst. Reinstate
                // the centrally partitioned burst before replacement frames.
                video_pacer->reset(next_video_pacing_bps, video_burst_bytes);
                if (encoder) encoder->shutdown();
                std::unique_ptr<MppH265Encoder> replacement(
                    new MppH265Encoder(next.encoder, next.roi));
                if (!replacement->initialize(&encoder_error)) {
                    std::fprintf(stderr, "MPP video-mode switch failed: %s\n", encoder_error.c_str());
                    running.store(false);
                    break;
                }
                encoder.swap(replacement);
                camera.updateEncoderConfig(next.encoder);
                roi_manager.reconfigure(next.roi);
                {
                    std::lock_guard<std::mutex> lock(runtime_config_mutex);
                    runtime_config = next;
                }
                active_profile.store(static_cast<int>(desired));
                active_transport_mode.store(static_cast<int>(TRANSPORT_MODE_VIDEO));
                const unsigned int generation = profile_generation.fetch_add(1) + 1;
                rebuild_sender.setEnabled(desired == RATE_PROFILE_REBUILD);
                encoded_frame_count = 0;
                rtp_sdp_written = next.transport.rtp_sdp_path.empty();
                std::fprintf(stderr,
                    "Runtime video mode applied: %s %dx%d %d fps encoder=%d pacing=%d generation=%u\n",
                    rateProfileName(desired), next.encoder.width, next.encoder.height,
                    next.encoder.fps, next.encoder.target_bitrate_bps,
                    next.transport.pacing_bitrate_bps, generation);
                continue;
            }
            if (active_transport == TRANSPORT_MODE_IMAGE) {
                if (event_sender.failed(&encoder_error)) {
                    std::fprintf(stderr, "Detection event transport error: %s\n",
                                 encoder_error.c_str());
                    running.store(false);
                    break;
                }
                if (audio_sender && audio_sender->failed(&encoder_error)) {
                    std::fprintf(stderr, "Codec2 audio transport error: %s\n", encoder_error.c_str());
                    running.store(false);
                    break;
                }
                continue;
            }
            const AppConfig live = runtimeConfig();
            if (frame->encoder_width != live.encoder.width ||
                frame->encoder_height != live.encoder.height) {
                continue;
            }
            if (sender.failed(&encoder_error)) {
                std::fprintf(stderr, "UDP transport error: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            if (audio_sender && audio_sender->failed(&encoder_error)) {
                std::fprintf(stderr, "Codec2 audio transport error: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            if (rebuild_sender.failed(&encoder_error)) {
                std::fprintf(stderr, "Rebuild transport error: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            if (event_sender.failed(&encoder_error)) {
                std::fprintf(stderr, "Detection event transport error: %s\n",
                             encoder_error.c_str());
                running.store(false);
                break;
            }
            const bool periodic_idr = encoded_frame_count > 0 &&
                encoded_frame_count % static_cast<uint64_t>(live.encoder.gop) == 0;
            if ((!encoder || ((sender.needsKeyFrame() || periodic_idr) &&
                !encoder->requestIdr(&encoder_error)))) {
                if (encoder_error.empty()) encoder_error = "H.265 encoder is unavailable in video mode";
                std::fprintf(stderr, "MPP recovery IDR request failed: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            RoiMap map = roi_manager.select(frame->meta.frame_id, frame->meta.pts_us,
                                            live.encoder.width, live.encoder.height);
            std::vector<RoiRegion> regions;
            if (config.mode != PIPELINE_BASELINE) {
                const RoiRegionMerger merger(live.roi);
                regions = merger.merge(map);
            }
            if (config.debug_roi && !writeRoiMapPgm(map,
                debugPathForFrame(config.debug_roi_path, frame->meta.frame_id), &encoder_error)) {
                std::fprintf(stderr, "ROI debug output failed: %s\n", encoder_error.c_str());
            }
            EncodedAccessUnit access_unit;
            if (!encoder->encode(*frame, regions, &access_unit, &encoder_error)) {
                std::fprintf(stderr, "MPP encode error: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            ++encoded_frame_count;
            if (!rtp_sdp_written && access_unit.key_frame) {
                if (writeH265RtpSdp(access_unit.bytes, live.transport.udp_port,
                                    live.transport.rtp_sdp_path, &encoder_error)) {
                    rtp_sdp_written = true;
                    std::fprintf(stderr, "Wrote H.265 RTP SDP: %s\n",
                                 live.transport.rtp_sdp_path.c_str());
                } else {
                    std::fprintf(stderr, "H.265 RTP SDP export deferred: %s\n",
                                 encoder_error.c_str());
                }
            }
            const size_t encoded_bytes = access_unit.bytes.size();
            RtpAccessUnit transport_unit;
            transport_unit.frame = access_unit.frame;
            transport_unit.bytes = std::move(access_unit.bytes);
            transport_unit.key_frame = access_unit.key_frame;
            transport_unit.stream_profile.valid = true;
            transport_unit.stream_profile.profile = static_cast<uint8_t>(live.rate_profile);
            transport_unit.stream_profile.width = static_cast<uint16_t>(live.encoder.width);
            transport_unit.stream_profile.height = static_cast<uint16_t>(live.encoder.height);
            transport_unit.stream_profile.fps = static_cast<uint8_t>(live.encoder.fps);
            transport_unit.stream_profile.generation =
                static_cast<uint8_t>(profile_generation.load() & 0xffU);
            if (!sender.enqueue(std::move(transport_unit), &encoder_error)) {
                std::fprintf(stderr, "UDP transport enqueue error: %s\n", encoder_error.c_str());
                running.store(false);
                break;
            }
            const AsyncRtpSenderSnapshot tx = sender.snapshot();
            statistics.recordRoi(map, frame->meta.frame_id, static_cast<int>(regions.size()));
            statistics.recordEncoded(frame->meta, access_unit.key_frame, encoded_bytes,
                                     access_unit.average_qp, access_unit.realtime_bitrate_bps,
                                     tx.queued_frames);
            const Codec2AudioSenderSnapshot audio_tx = audio_sender
                ? audio_sender->snapshot() : Codec2AudioSenderSnapshot();
            const RebuildSenderSnapshot rebuild_tx = rebuild_sender.snapshot();
            const DetectionEventSenderSnapshot event_tx = event_sender.snapshot();
            std::printf("%s tx_bytes=%zu tx_drop_p=%llu tx_drop_idr=%llu tx_wait_idr=%d "
                        "tx_last_frame=%llu tx_send_e2e_us=%llu audio_rtp=%llu audio_frames=%llu "
                        "audio_q=%zu audio_q_age_ms=%llu audio_q_drop=%llu audio_q_drop_frames=%llu "
                        "audio_q_wait_ms=%llu audio_reserve_bps=%d "
                        "audio_noise_dbfs=%.1f audio_agc_db=%.1f audio_speech_snr_db=%.1f "
                        "audio_voicing=%.0f audio_gate=%d audio_voice=%d audio_dtx=%d "
                        "audio_dtx_drop=%llu audio_dtx_hold=%llu audio_dtx_speech=%llu "
                        "audio_dtx_keepalive=%llu rebuild_state=%llu "
                        "rebuild_state_capture_us=%llu rebuild_state_send_start_us=%llu "
                        "rebuild_state_send_us=%llu rebuild_state_pacer_us=%llu "
                        "rebuild_state_capture_send_us=%llu rebuild_state_capture_p50_us=%llu "
                        "rebuild_state_capture_p95_us=%llu rebuild_refs=%llu "
                        "rebuild_patch_packets=%llu rebuild_parity=%llu rebuild_jpeg_bytes=%llu "
                        "rebuild_wire_bytes=%llu rebuild_q=%zu rebuild_refgen=%u "
                        "rebuild_ref_capture_us=%llu rebuild_ref_encode_us=%llu "
                        "rebuild_ref_queue_us=%llu rebuild_ref_first_send_us=%llu "
                        "rebuild_ref_last_send_us=%llu rebuild_ref_q_delay_us=%llu "
                        "rebuild_ref_capture_send_us=%llu rebuild_ref_capture_p50_us=%llu "
                        "rebuild_ref_capture_p95_us=%llu "
                        "rebuild_ref_delivery_us=%llu rebuild_ref_delivery_p50_us=%llu "
                        "rebuild_ref_delivery_p95_us=%llu rebuild_ref_bytes=%llu "
                        "rebuild_ref_chunks=%llu rebuild_ref_fec_bytes=%llu "
                        "rebuild_ref_interval_us=%llu rebuild_ref_interval_p50_us=%llu "
                        "rebuild_ref_interval_p95_us=%llu rebuild_ref_interval_max_us=%llu "
                        "rebuild_refresh_track=%u rebuild_ref_capture_age_ms=%d "
                        "rebuild_ref_ready_age_ms=%d rebuild_refresh_threshold_ms=%d "
                        "rebuild_est_delivery_ms=%d rebuild_next_deadline_ms=%d "
                        "rebuild_refresh_quantum_ms=%d rebuild_refresh_decision=%s "
                        "rebuild_refresh_reason=%s "
                        "event_packets=%llu event_heartbeat=%llu "
                        "event_replace=%llu event_wire_bytes=%llu event_mask=0x%04x event_q=%zu\n",
                        statistics.logLine().c_str(), tx.queued_bytes,
                        static_cast<unsigned long long>(tx.dropped_p_frames),
                        static_cast<unsigned long long>(tx.dropped_key_frames),
                        tx.waiting_for_key_frame ? 1 : 0,
                        static_cast<unsigned long long>(tx.last_sent_frame_id),
                        static_cast<unsigned long long>(tx.last_capture_to_send_us),
                        static_cast<unsigned long long>(audio_tx.sent_rtp_packets),
                        static_cast<unsigned long long>(audio_tx.sent_codec_frames),
                        audio_tx.queued_rtp_packets,
                        static_cast<unsigned long long>(audio_tx.oldest_queue_age_ms),
                        static_cast<unsigned long long>(audio_tx.dropped_stale_rtp_packets),
                        static_cast<unsigned long long>(audio_tx.dropped_stale_codec_frames),
                        static_cast<unsigned long long>(audio_tx.last_queue_delay_ms),
                        audio_tx.reserved_wire_bitrate_bps,
                        audio_tx.preprocess_noise_floor_dbfs,
                        audio_tx.preprocess_agc_gain_db,
                        audio_tx.preprocess_speech_snr_db,
                        audio_tx.preprocess_voicing_percent,
                        audio_tx.preprocess_gate_open ? 1 : 0,
                        audio_tx.preprocess_voice_detected ? 1 : 0,
                        audio_tx.dtx_speech_active ? 1 : 0,
                        static_cast<unsigned long long>(audio_tx.dtx_suppressed_codec_frames),
                        static_cast<unsigned long long>(audio_tx.dtx_hangover_codec_frames),
                        static_cast<unsigned long long>(audio_tx.dtx_speech_rtp_packets),
                        static_cast<unsigned long long>(audio_tx.dtx_keepalive_rtp_packets),
                        static_cast<unsigned long long>(rebuild_tx.state_packets),
                        static_cast<unsigned long long>(rebuild_tx.last_state_capture_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_state_send_start_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_state_send_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_state_pacer_delay_us),
                        static_cast<unsigned long long>(rebuild_tx.last_state_capture_to_send_us),
                        static_cast<unsigned long long>(rebuild_tx.state_capture_to_send_p50_us),
                        static_cast<unsigned long long>(rebuild_tx.state_capture_to_send_p95_us),
                        static_cast<unsigned long long>(rebuild_tx.patch_transfers),
                        static_cast<unsigned long long>(rebuild_tx.patch_packets),
                        static_cast<unsigned long long>(rebuild_tx.parity_packets),
                        static_cast<unsigned long long>(rebuild_tx.patch_jpeg_bytes),
                        static_cast<unsigned long long>(rebuild_tx.sent_wire_bytes),
                        rebuild_tx.queued_requests,
                        static_cast<unsigned int>(rebuild_tx.last_reference_generation),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_capture_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_encode_finish_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_queue_enter_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_first_packet_send_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_last_packet_send_time_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_queue_delay_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_capture_to_send_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_capture_to_send_p50_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_capture_to_send_p95_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_delivery_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_delivery_p50_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_delivery_p95_us),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_blob_bytes),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_chunk_count),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_fec_bytes),
                        static_cast<unsigned long long>(rebuild_tx.last_reference_interval_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_interval_p50_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_interval_p95_us),
                        static_cast<unsigned long long>(rebuild_tx.reference_interval_max_us),
                        static_cast<unsigned int>(rebuild_tx.last_refresh_track_id),
                        rebuild_tx.last_reference_capture_age_ms,
                        rebuild_tx.last_reference_ready_age_ms,
                        rebuild_tx.last_refresh_threshold_ms,
                        rebuild_tx.last_estimated_delivery_ms,
                        rebuild_tx.last_refresh_deadline_ms,
                        rebuild_tx.last_refresh_quantum_ms,
                        rebuild_tx.last_refresh_decision_start ? "START" : "HOLD",
                        rebuild_tx.last_refresh_reason.c_str(),
                        static_cast<unsigned long long>(event_tx.state_packets),
                        static_cast<unsigned long long>(event_tx.heartbeat_packets),
                        static_cast<unsigned long long>(event_tx.replaced_events),
                        static_cast<unsigned long long>(event_tx.sent_wire_bytes),
                        event_tx.last_present_mask, event_tx.queued_events);
            // nohup redirects stdout to a regular file on Buildroot. Flush
            // every status record so tail/grep reflects the live A/V queues
            // rather than a delayed stdio buffer.
            std::fflush(stdout);
        }
        running.store(false);
        inference_queue.stop();
    });

    // Qt HighGUI, like the original YOLOv8-Seg application, is driven by the
    // process main thread.  Capture, inference, encoder, and UDP continue in
    // their own threads while this loop consumes only the latest preview item.
    if (config.preview) {
        BoardPreview preview(config.preview_width, config.preview_height, config.preview_rotate_ccw);
        std::string preview_error;
        if (!preview.open(&preview_error)) {
            std::fprintf(stderr, "Board preview disabled: %s\n", preview_error.c_str());
            preview_queue.stop();
        } else {
            PreviewItem item;
            while (preview_queue.pop(&item)) {
                if (!item.frame) continue;
                if (!preview.show(*item.frame,
                                  item.has_segmentation ? &item.segmentation : NULL,
                                  &preview_error)) {
                    if (!preview_error.empty()) {
                        std::fprintf(stderr, "Board preview stopped: %s\n", preview_error.c_str());
                    }
                    running.store(false);
                    inference_queue.stop();
                    encoder_queue.stop();
                    preview_queue.stop();
                    break;
                }
            }
            preview.close();
        }
    }
    capture_thread.join();
    if (inference_thread.joinable()) inference_thread.join();
    encoder_thread.join();
    running.store(false);
    if (profile_control_thread.joinable()) profile_control_thread.join();
    if (audio_sender) audio_sender->stop();
    event_sender.stop();
    rebuild_sender.stop();
    snapshot_sender.stop();
    sender.stop();
    preview_queue.stop();
    camera.close();
    if (encoder) encoder->shutdown();
    if (use_inference) {
        release_yolov8_seg_model(&rknn_context);
        deinit_post_process();
    }
    return EXIT_SUCCESS;
}

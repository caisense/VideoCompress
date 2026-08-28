#include "audio/codec2_audio_sender.h"

#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <utility>
#include <vector>

#include <codec2.h>

#include "audio/audio_preprocessor.h"
#include "audio/codec2_dtx_controller.h"

namespace roi_h265 {
namespace {

const uint8_t kCodec2PayloadType = 97;

}  // namespace

Codec2AudioSender::Codec2AudioSender(const AudioConfig &config, const std::string &host, int mtu,
                                     int physical_pacing_bps,
                                     const std::shared_ptr<RatePacer> &audio_pacer)
    : config_(config), mtu_(mtu),
      udp_sender_(host, config.udp_port, physical_pacing_bps, mtu, audio_pacer),
      packetizer_(0, 0x524f4932U, kCodec2PayloadType), capture_(), codec2_(NULL),
      packet_queue_(), capture_worker_(), send_worker_(), stopping_(false), started_(false),
      failed_(false), sending_(false) {}

Codec2AudioSender::~Codec2AudioSender() { stop(); }

int Codec2AudioSender::codec2ModeForBitrate(int bitrate_bps) {
    switch (bitrate_bps) {
    case 1200: return CODEC2_MODE_1200;
    case 1300: return CODEC2_MODE_1300;
    case 1400: return CODEC2_MODE_1400;
    case 1600: return CODEC2_MODE_1600;
    case 2400: return CODEC2_MODE_2400;
    case 3200: return CODEC2_MODE_3200;
    default: return -1;
    }
}

bool Codec2AudioSender::start(std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return !failed_;
    if (sizeof(short) != sizeof(int16_t)) {
        if (error) *error = "Codec2 requires a 16-bit C++ short PCM type";
        return false;
    }
    const int codec2_mode = codec2ModeForBitrate(config_.codec2_mode_bps);
    codec2_ = codec2_mode < 0 ? NULL : codec2_create(codec2_mode);
    if (!codec2_) {
        if (error) *error = "cannot initialize requested Codec2 mode";
        return false;
    }
    snapshot_ = Codec2AudioSenderSnapshot();
    snapshot_.samples_per_codec_frame = codec2_samples_per_frame(codec2_);
    snapshot_.bits_per_codec_frame = codec2_bits_per_frame(codec2_);
    snapshot_.bytes_per_codec_frame = codec2_bytes_per_frame(codec2_);
    snapshot_.frames_per_packet = config_.frames_per_packet;
    const size_t udp_payload_bytes = 12U + static_cast<size_t>(snapshot_.bytes_per_codec_frame) *
        static_cast<size_t>(config_.frames_per_packet);
    if (snapshot_.samples_per_codec_frame <= 0 || snapshot_.bits_per_codec_frame <= 0 ||
        snapshot_.bytes_per_codec_frame <= 0 || udp_payload_bytes > static_cast<size_t>(mtu_ - 28) ||
        udp_payload_bytes > 65507U) {
        codec2_destroy(codec2_);
        codec2_ = NULL;
        if (error) *error = "invalid Codec2 packet geometry";
        return false;
    }
    const size_t packet_wire_bytes = RatePacer::wireBytesForUdpPayload(udp_payload_bytes);
    const int64_t packet_samples = static_cast<int64_t>(snapshot_.samples_per_codec_frame) *
        static_cast<int64_t>(config_.frames_per_packet);
    const int calculated_reserve_bps = static_cast<int>((static_cast<int64_t>(packet_wire_bytes) *
        8LL * 8000LL + packet_samples - 1LL) / packet_samples);
    if (config_.reserve_wire_bitrate_bps > 0 &&
        config_.reserve_wire_bitrate_bps < calculated_reserve_bps) {
        codec2_destroy(codec2_);
        codec2_ = NULL;
        if (error) *error = "audio reserve bitrate is below the Codec2 RTP wire rate";
        return false;
    }
    const int reserve_bps = config_.reserve_wire_bitrate_bps > 0
        ? config_.reserve_wire_bitrate_bps : calculated_reserve_bps;
    if (reserve_bps >= udp_sender_.pacingBitrate()) {
        codec2_destroy(codec2_);
        codec2_ = NULL;
        if (error) *error = "audio reserve bitrate must leave physical capacity for video";
        return false;
    }
    const int packet_duration_ms = static_cast<int>((packet_samples * 1000LL + 7999LL) / 8000LL);
    const size_t max_queued_packets = std::max<size_t>(1,
        static_cast<size_t>((config_.max_latency_ms + packet_duration_ms - 1) / packet_duration_ms));
    if (!capture_.open(config_.device, config_.capture_rate_hz, config_.capture_channels, error)) {
        codec2_destroy(codec2_);
        codec2_ = NULL;
        return false;
    }
    if (!udp_sender_.open(error)) {
        capture_.close();
        codec2_destroy(codec2_);
        codec2_ = NULL;
        return false;
    }
    // Audio receives its own child bucket from main(). Its fixed rate is
    // deducted from the video child bucket, so the two rates still add up to
    // one physical-link ceiling without cross-thread token starvation. Keep
    // its burst to one small RTP packet; main() allocates the rest of the
    // original physical burst to the video child.
    const size_t physical_burst_bytes = static_cast<size_t>(mtu_) + 38U;
    const size_t audio_burst_bytes = std::min(packet_wire_bytes,
        physical_burst_bytes > 1U ? physical_burst_bytes - 1U : 1U);
    udp_sender_.setPacingBitrate(reserve_bps, audio_burst_bytes);
    packet_queue_.reset(max_queued_packets, config_.max_latency_ms);
    snapshot_.reserved_wire_bitrate_bps = reserve_bps;
    snapshot_.reserved_burst_wire_bytes = audio_burst_bytes;
    stopping_.store(false);
    failed_ = false;
    failure_message_.clear();
    started_ = true;
    sending_ = false;
    capture_worker_ = std::thread(&Codec2AudioSender::captureEncodeLoop, this);
    send_worker_ = std::thread(&Codec2AudioSender::sendLoop, this);
    return true;
}

void Codec2AudioSender::setFailure(const std::string &error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failed_) {
        failed_ = true;
        failure_message_ = error;
    }
    stopping_.store(true);
    packet_queue_.clear();
    queue_condition_.notify_all();
}

void Codec2AudioSender::recordQueueDropLocked(const BoundedAudioPacketDrop &dropped) {
    snapshot_.dropped_stale_rtp_packets += dropped.packets;
    snapshot_.dropped_stale_codec_frames += dropped.codec_frames;
}

void Codec2AudioSender::enqueuePacket(BoundedAudioPacket packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_.load()) return;
    const BoundedAudioPacketDrop dropped = packet_queue_.push(
        std::move(packet), std::chrono::steady_clock::now());
    recordQueueDropLocked(dropped);
    queue_condition_.notify_one();
}

void Codec2AudioSender::captureEncodeLoop() {
    const size_t input_frames_per_read = static_cast<size_t>(
        std::max(80, config_.capture_rate_hz / 50));  // about 20 ms
    std::vector<int16_t> captured(input_frames_per_read * static_cast<size_t>(config_.capture_channels));
    const int samples_per_frame = snapshot_.samples_per_codec_frame;
    const int bytes_per_frame = snapshot_.bytes_per_codec_frame;
    AudioPreprocessor preprocessor(config_.capture_rate_hz, config_.capture_channels,
                                   samples_per_frame, config_.preprocess);
    Codec2DtxController dtx(config_.dtx_enabled, config_.frames_per_packet,
                             codec2FramesForDurationMs(config_.dtx_preroll_ms, samples_per_frame),
                             codec2FramesForDurationMs(config_.dtx_keepalive_ms, samples_per_frame),
                             codec2FramesForDurationMs(config_.dtx_hangover_ms, samples_per_frame));
    std::vector<short> speech;
    uint32_t rtp_timestamp = 0;
    while (!stopping_.load()) {
        std::string capture_error;
        if (!capture_.readInterleaved(captured.data(), input_frames_per_read, &capture_error)) {
            if (!stopping_.load()) setFailure(capture_error);
            break;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.captured_input_frames += input_frames_per_read;
        }
        preprocessor.append(captured.data(), input_frames_per_read);
        while (!stopping_.load() && preprocessor.availableFrames() > 0) {
            if (!preprocessor.popFrame(&speech)) break;
            const AudioPreprocessSnapshot preprocess = preprocessor.snapshot();
            Codec2DtxFrame encoded;
            encoded.timestamp = rtp_timestamp;
            encoded.payload.resize(static_cast<size_t>(bytes_per_frame), 0);
            codec2_encode(codec2_, encoded.payload.data(), speech.data());
            rtp_timestamp += static_cast<uint32_t>(samples_per_frame);

            std::vector<Codec2DtxPacket> dtx_packets;
            // DTX follows the instantaneous VAD decision, while the separate
            // DTX hangover carries the packet stream across short misses. The
            // preprocessor's soft-gate hangover keeps the corresponding PCM
            // audible during exactly that same interval. With preprocessing
            // intentionally disabled there is no valid VAD decision, so DTX
            // falls back to the legacy continuous stream.
            const bool dtx_speech_open = !config_.preprocess.enabled || preprocess.voice_detected;
            dtx.push(std::move(encoded), dtx_speech_open, &dtx_packets);
            const Codec2DtxSnapshot dtx_snapshot = dtx.snapshot();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                snapshot_.preprocess_noise_floor_dbfs = preprocess.noise_floor_dbfs;
                snapshot_.preprocess_agc_gain_db = preprocess.agc_gain_db;
                snapshot_.preprocess_speech_snr_db = preprocess.speech_snr_db;
                snapshot_.preprocess_voicing_percent = preprocess.voicing_percent;
                snapshot_.preprocess_gate_open = preprocess.gate_open;
                snapshot_.preprocess_voice_detected = preprocess.voice_detected;
                snapshot_.preprocess_processed_frames = preprocess.processed_frames;
                snapshot_.preprocess_gated_frames = preprocess.gated_frames;
                snapshot_.dtx_suppressed_codec_frames = dtx_snapshot.suppressed_codec_frames;
                snapshot_.dtx_speech_rtp_packets = dtx_snapshot.speech_rtp_packets;
                snapshot_.dtx_keepalive_rtp_packets = dtx_snapshot.keepalive_rtp_packets;
                snapshot_.dtx_hangover_codec_frames = dtx_snapshot.hangover_codec_frames;
                snapshot_.dtx_speech_active = dtx_snapshot.speech_active;
            }
            for (size_t packet_index = 0; packet_index < dtx_packets.size(); ++packet_index) {
                const Codec2DtxPacket &dtx_packet = dtx_packets[packet_index];
                BoundedAudioPacket packet;
                packet.datagram = packetizer_.packetize(dtx_packet.payload.data(),
                    dtx_packet.payload.size(), dtx_packet.timestamp, dtx_packet.marker);
                packet.codec_frames = static_cast<uint64_t>(dtx_packet.frame_count);
                packet.marker = dtx_packet.marker;
                packet.enqueued_at = std::chrono::steady_clock::now();
                enqueuePacket(std::move(packet));
            }
        }
    }
}

void Codec2AudioSender::sendLoop() {
    for (;;) {
        BoundedAudioPacket packet;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_condition_.wait(lock, [this] {
                return stopping_.load() || !packet_queue_.empty();
            });
            if (stopping_.load()) break;
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const BoundedAudioPacketDrop dropped = packet_queue_.pop(&packet, now);
            recordQueueDropLocked(dropped);
            if (packet.datagram.empty()) continue;
            snapshot_.last_queue_delay_ms = static_cast<uint64_t>(std::max<int64_t>(0,
                std::chrono::duration_cast<std::chrono::milliseconds>(now - packet.enqueued_at).count()));
            sending_ = true;
        }

        const std::vector<std::vector<uint8_t> > packets(1, packet.datagram);
        std::string send_error;
        const bool sent = udp_sender_.sendPackets(packets, &send_error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sending_ = false;
            queue_condition_.notify_all();
            if (sent) {
                snapshot_.sent_codec_frames += packet.codec_frames;
                ++snapshot_.sent_rtp_packets;
            }
        }
        if (!sent) {
            if (!stopping_.load()) setFailure(send_error);
            return;
        }
    }
}

void Codec2AudioSender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stopping_.store(true);
        packet_queue_.clear();
        queue_condition_.notify_all();
    }
    capture_.requestStop();
    if (capture_worker_.joinable()) capture_worker_.join();
    if (send_worker_.joinable()) send_worker_.join();
    capture_.close();
    std::lock_guard<std::mutex> lock(mutex_);
    if (codec2_) {
        codec2_destroy(codec2_);
        codec2_ = NULL;
    }
    started_ = false;
}

bool Codec2AudioSender::failed(std::string *error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ && error) *error = failure_message_;
    return failed_;
}

Codec2AudioSenderSnapshot Codec2AudioSender::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Codec2AudioSenderSnapshot value = snapshot_;
    const BoundedAudioPacketQueueSnapshot queue = packet_queue_.snapshot(
        std::chrono::steady_clock::now());
    value.queued_rtp_packets = queue.packets;
    value.oldest_queue_age_ms = queue.oldest_age_ms;
    return value;
}

}  // namespace roi_h265

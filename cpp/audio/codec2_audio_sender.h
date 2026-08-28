#ifndef ROI_H265_AUDIO_CODEC2_AUDIO_SENDER_H_
#define ROI_H265_AUDIO_CODEC2_AUDIO_SENDER_H_

#include <stdint.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "audio/arecord_capture.h"
#include "audio/bounded_audio_packet_queue.h"
#include "common/config.h"
#include "transport/codec2_rtp_packetizer.h"
#include "transport/udp_sender.h"

struct CODEC2;

namespace roi_h265 {

struct Codec2AudioSenderSnapshot {
    uint64_t captured_input_frames;
    uint64_t sent_codec_frames;
    uint64_t sent_rtp_packets;
    int samples_per_codec_frame;
    int bits_per_codec_frame;
    int bytes_per_codec_frame;
    int frames_per_packet;
    double preprocess_noise_floor_dbfs;
    double preprocess_agc_gain_db;
    double preprocess_speech_snr_db;
    double preprocess_voicing_percent;
    bool preprocess_gate_open;
    bool preprocess_voice_detected;
    uint64_t preprocess_processed_frames;
    uint64_t preprocess_gated_frames;
    uint64_t dtx_suppressed_codec_frames;
    uint64_t dtx_speech_rtp_packets;
    uint64_t dtx_keepalive_rtp_packets;
    uint64_t dtx_hangover_codec_frames;
    bool dtx_speech_active;
    size_t queued_rtp_packets;
    uint64_t oldest_queue_age_ms;
    uint64_t dropped_stale_rtp_packets;
    uint64_t dropped_stale_codec_frames;
    uint64_t last_queue_delay_ms;
    int reserved_wire_bitrate_bps;
    size_t reserved_burst_wire_bytes;

    Codec2AudioSenderSnapshot()
        : captured_input_frames(0), sent_codec_frames(0), sent_rtp_packets(0),
          samples_per_codec_frame(0), bits_per_codec_frame(0), bytes_per_codec_frame(0),
          frames_per_packet(0), preprocess_noise_floor_dbfs(-78.0),
          preprocess_agc_gain_db(0.0), preprocess_speech_snr_db(0.0),
          preprocess_voicing_percent(0.0), preprocess_gate_open(false),
          preprocess_voice_detected(false),
           preprocess_processed_frames(0), preprocess_gated_frames(0),
           dtx_suppressed_codec_frames(0), dtx_speech_rtp_packets(0),
           dtx_keepalive_rtp_packets(0), dtx_hangover_codec_frames(0),
           dtx_speech_active(false),
          queued_rtp_packets(0), oldest_queue_age_ms(0),
          dropped_stale_rtp_packets(0), dropped_stale_codec_frames(0),
          last_queue_delay_ms(0), reserved_wire_bitrate_bps(0),
          reserved_burst_wire_bytes(0) {}
};

// Captures microphone PCM through the board's documented arecord utility,
// converts it to Codec2's 8 kHz mono input, and sends private dynamic RTP
// payload type 97 through the caller-owned audio wire-pacer child bucket.
class Codec2AudioSender {
public:
    Codec2AudioSender(const AudioConfig &config, const std::string &host, int mtu,
                      int physical_pacing_bps, const std::shared_ptr<RatePacer> &audio_pacer);
    ~Codec2AudioSender();

    bool start(std::string *error);
    void stop();
    bool failed(std::string *error) const;
    Codec2AudioSenderSnapshot snapshot() const;

private:
    static int codec2ModeForBitrate(int bitrate_bps);
    void captureEncodeLoop();
    void sendLoop();
    void enqueuePacket(BoundedAudioPacket packet);
    void recordQueueDropLocked(const BoundedAudioPacketDrop &dropped);
    void setFailure(const std::string &error);

    AudioConfig config_;
    int mtu_;
    UdpSender udp_sender_;
    Codec2RtpPacketizer packetizer_;
    ArecordCapture capture_;
    ::CODEC2 *codec2_;
    BoundedAudioPacketQueue packet_queue_;
    std::thread capture_worker_;
    std::thread send_worker_;
    std::atomic<bool> stopping_;
    mutable std::mutex mutex_;
    std::condition_variable queue_condition_;
    bool started_;
    bool failed_;
    bool sending_;
    std::string failure_message_;
    Codec2AudioSenderSnapshot snapshot_;
};

}  // namespace roi_h265

#endif  // ROI_H265_AUDIO_CODEC2_AUDIO_SENDER_H_

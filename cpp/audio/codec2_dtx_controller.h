#ifndef ROI_H265_AUDIO_CODEC2_DTX_CONTROLLER_H_
#define ROI_H265_AUDIO_CODEC2_DTX_CONTROLLER_H_

#include <stddef.h>
#include <stdint.h>

#include <deque>
#include <vector>

namespace roi_h265 {

// Converts a millisecond duration to a whole number of Codec2 frames at the
// fixed 8 kHz Codec2 sample clock. The result rounds up so configured pre-roll
// and keepalive intervals never become shorter than requested.
int codec2FramesForDurationMs(int duration_ms, int samples_per_frame);

// An encoded Codec2 frame paired with the RTP clock timestamp of its first
// PCM sample.  Codec2 is still fed every captured frame, even under DTX, so
// its predictor state remains continuous when speech resumes.
struct Codec2DtxFrame {
    uint32_t timestamp;
    std::vector<uint8_t> payload;

    Codec2DtxFrame() : timestamp(0), payload() {}
};

// One complete fixed-geometry Codec2 RTP payload.  marker follows RTP audio
// convention: it is set only on the first packet of a new speech burst.
// keepalive is sender-local accounting; it is represented on the wire by an
// ordinary marker-clear Codec2 RTP packet for backward compatibility.
struct Codec2DtxPacket {
    uint32_t timestamp;
    std::vector<uint8_t> payload;
    int frame_count;
    bool marker;
    bool keepalive;

    Codec2DtxPacket()
        : timestamp(0), payload(), frame_count(0), marker(false), keepalive(false) {}
};

struct Codec2DtxSnapshot {
    uint64_t suppressed_codec_frames;
    uint64_t speech_codec_frames;
    uint64_t speech_rtp_packets;
    uint64_t keepalive_codec_frames;
    uint64_t keepalive_rtp_packets;
    uint64_t hangover_codec_frames;
    bool speech_active;

    Codec2DtxSnapshot()
        : suppressed_codec_frames(0), speech_codec_frames(0), speech_rtp_packets(0),
          keepalive_codec_frames(0), keepalive_rtp_packets(0), hangover_codec_frames(0),
          speech_active(false) {}
};

// Groups encoded Codec2 frames into fixed-size RTP payloads and suppresses
// silence without changing the RTP sampling clock.  The controller keeps a
// bounded encoded pre-roll so VAD confirmation never clips a speech onset.
class Codec2DtxController {
public:
    Codec2DtxController(bool enabled, int frames_per_packet, int pre_roll_frames,
                        int keepalive_interval_frames, int speech_hangover_frames);

    void push(Codec2DtxFrame frame, bool speech_open,
              std::vector<Codec2DtxPacket> *output);
    Codec2DtxSnapshot snapshot() const;

private:
    void discardHistoryFront();
    void trimHistoryTo(size_t limit);
    void pushHistory(Codec2DtxFrame frame);
    void appendSpeechFrame(const Codec2DtxFrame &frame,
                           std::vector<Codec2DtxPacket> *output);
    void flushHistoryAsSpeech(std::vector<Codec2DtxPacket> *output);
    void emitKeepalive(std::vector<Codec2DtxPacket> *output);

    bool enabled_;
    size_t frames_per_packet_;
    size_t pre_roll_frames_;
    size_t history_capacity_;
    size_t keepalive_interval_frames_;
    size_t speech_hangover_frames_;
    size_t speech_hangover_remaining_;
    size_t silent_frames_since_keepalive_;
    bool speech_active_;
    bool speech_marker_pending_;
    std::deque<Codec2DtxFrame> history_;
    Codec2DtxPacket pending_speech_;
    Codec2DtxSnapshot snapshot_;
};

}  // namespace roi_h265

#endif  // ROI_H265_AUDIO_CODEC2_DTX_CONTROLLER_H_

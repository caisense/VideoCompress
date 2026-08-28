#include "audio/codec2_dtx_controller.h"

#include <algorithm>
#include <utility>

namespace roi_h265 {

int codec2FramesForDurationMs(int duration_ms, int samples_per_frame) {
    if (duration_ms <= 0 || samples_per_frame <= 0) return 0;
    const int64_t duration_samples = (static_cast<int64_t>(duration_ms) * 8000LL + 999LL) /
        1000LL;
    return static_cast<int>(std::max<int64_t>(1,
        (duration_samples + samples_per_frame - 1) / samples_per_frame));
}

Codec2DtxController::Codec2DtxController(bool enabled, int frames_per_packet,
                                          int pre_roll_frames,
                                          int keepalive_interval_frames,
                                          int speech_hangover_frames)
    : enabled_(enabled),
      frames_per_packet_(static_cast<size_t>(std::max(1, frames_per_packet))),
      pre_roll_frames_(static_cast<size_t>(std::max(0, pre_roll_frames))),
      history_capacity_(std::max(frames_per_packet_, pre_roll_frames_)),
      keepalive_interval_frames_(static_cast<size_t>(std::max(0, keepalive_interval_frames))),
      speech_hangover_frames_(static_cast<size_t>(std::max(0, speech_hangover_frames))),
      speech_hangover_remaining_(0), silent_frames_since_keepalive_(0), speech_active_(false),
      speech_marker_pending_(true), history_(), pending_speech_(), snapshot_() {}

void Codec2DtxController::discardHistoryFront() {
    if (history_.empty()) return;
    history_.pop_front();
    ++snapshot_.suppressed_codec_frames;
}

void Codec2DtxController::trimHistoryTo(size_t limit) {
    while (history_.size() > limit) discardHistoryFront();
}

void Codec2DtxController::pushHistory(Codec2DtxFrame frame) {
    history_.push_back(std::move(frame));
    trimHistoryTo(history_capacity_);
}

void Codec2DtxController::appendSpeechFrame(const Codec2DtxFrame &frame,
                                            std::vector<Codec2DtxPacket> *output) {
    if (!output) return;
    if (pending_speech_.frame_count == 0) {
        pending_speech_.timestamp = frame.timestamp;
        pending_speech_.payload.clear();
        pending_speech_.marker = speech_marker_pending_;
        pending_speech_.keepalive = false;
        speech_marker_pending_ = false;
    }
    pending_speech_.payload.insert(pending_speech_.payload.end(), frame.payload.begin(),
                                   frame.payload.end());
    ++pending_speech_.frame_count;
    if (pending_speech_.frame_count < static_cast<int>(frames_per_packet_)) return;

    output->push_back(pending_speech_);
    snapshot_.speech_codec_frames += static_cast<uint64_t>(pending_speech_.frame_count);
    ++snapshot_.speech_rtp_packets;
    pending_speech_ = Codec2DtxPacket();
}

void Codec2DtxController::flushHistoryAsSpeech(std::vector<Codec2DtxPacket> *output) {
    while (!history_.empty()) {
        appendSpeechFrame(history_.front(), output);
        history_.pop_front();
    }
}

void Codec2DtxController::emitKeepalive(std::vector<Codec2DtxPacket> *output) {
    if (!output || history_.size() < frames_per_packet_) return;
    trimHistoryTo(frames_per_packet_);
    Codec2DtxPacket packet;
    packet.timestamp = history_.front().timestamp;
    packet.marker = false;
    packet.keepalive = true;
    while (!history_.empty()) {
        const Codec2DtxFrame &frame = history_.front();
        packet.payload.insert(packet.payload.end(), frame.payload.begin(), frame.payload.end());
        ++packet.frame_count;
        history_.pop_front();
    }
    output->push_back(packet);
    snapshot_.keepalive_codec_frames += static_cast<uint64_t>(packet.frame_count);
    ++snapshot_.keepalive_rtp_packets;
}

void Codec2DtxController::push(Codec2DtxFrame frame, bool speech_open,
                               std::vector<Codec2DtxPacket> *output) {
    if (!output) return;
    if (!enabled_) {
        appendSpeechFrame(frame, output);
        return;
    }

    if (speech_open) {
        silent_frames_since_keepalive_ = 0;
        speech_hangover_remaining_ = speech_hangover_frames_;
        if (!speech_active_) {
            speech_active_ = true;
            snapshot_.speech_active = true;
            speech_marker_pending_ = true;
            if (pre_roll_frames_ == 0U) {
                trimHistoryTo(0U);
                appendSpeechFrame(frame, output);
            } else {
                pushHistory(std::move(frame));
                trimHistoryTo(pre_roll_frames_);
                flushHistoryAsSpeech(output);
            }
        } else {
            appendSpeechFrame(frame, output);
        }
        return;
    }

    if (speech_active_ && speech_hangover_remaining_ > 0U) {
        // The acoustic VAD works per Codec2 frame.  A quiet consonant or a
        // short intra-sentence pause must not immediately terminate the RTP
        // talkspurt and force the receiver to wait for a new pre-roll.
        --speech_hangover_remaining_;
        ++snapshot_.hangover_codec_frames;
        appendSpeechFrame(frame, output);
        return;
    }

    if (speech_active_) {
        if (pending_speech_.frame_count != 0) {
            // Finish an already-started packet with real, low-level tail
            // audio. This bounds suppression-induced tail loss to less than
            // one packet without adding a full extra silence packet.
            appendSpeechFrame(frame, output);
            if (pending_speech_.frame_count != 0) return;
            speech_active_ = false;
            snapshot_.speech_active = false;
            silent_frames_since_keepalive_ = 0;
            return;
        }
        speech_active_ = false;
        snapshot_.speech_active = false;
        silent_frames_since_keepalive_ = 0;
    }

    pushHistory(std::move(frame));
    ++silent_frames_since_keepalive_;
    if (keepalive_interval_frames_ > 0U &&
        silent_frames_since_keepalive_ >= keepalive_interval_frames_ &&
        history_.size() >= frames_per_packet_) {
        emitKeepalive(output);
        silent_frames_since_keepalive_ = 0;
    }
}

Codec2DtxSnapshot Codec2DtxController::snapshot() const { return snapshot_; }

}  // namespace roi_h265

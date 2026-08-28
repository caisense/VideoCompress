#ifndef ROI_H265_AUDIO_AUDIO_PREPROCESSOR_H_
#define ROI_H265_AUDIO_AUDIO_PREPROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "common/config.h"

namespace roi_h265 {

struct AudioPreprocessSnapshot {
    double noise_floor_dbfs;
    double agc_gain_db;
    double speech_snr_db;
    double voicing_percent;
    bool gate_open;
    bool voice_detected;
    uint64_t processed_frames;
    uint64_t gated_frames;

    AudioPreprocessSnapshot();
};

// Streaming capture converter for Codec2's fixed 8 kHz mono PCM input.
// It uses a windowed-sinc low-pass resampler before applying a DC/high-pass
// stage, adaptive frame-energy noise attenuation, speech-only AGC, and a
// softly attenuating gate with hangover.  The class owns no OS resources and
// is therefore directly testable on the host without ALSA, Codec2, or RK3588.
class AudioPreprocessor {
public:
    AudioPreprocessor(int input_rate_hz, int input_channels, int output_frame_samples,
                      const AudioPreprocessConfig &config);

    void append(const int16_t *interleaved, size_t frames);
    size_t availableFrames() const;
    bool popFrame(std::vector<short> *output);
    AudioPreprocessSnapshot snapshot() const;

private:
    void resampleAvailable();
    double resampleAt(double source_position) const;
    void processFrame(const std::vector<float> &input, std::vector<short> *output);
    void analyzeVoice(const std::vector<float> &samples, double *speech_power,
                      double *low_power, double *voicing_percent) const;
    static double dbToLinear(double db);
    static double dbToPower(double db);
    static double clamp(double value, double lower, double upper);

    int input_rate_hz_;
    int input_channels_;
    int output_frame_samples_;
    AudioPreprocessConfig config_;
    double input_per_output_;
    double next_input_position_;
    std::vector<float> input_;
    std::vector<float> resampled_;

    double highpass_alpha_;
    double highpass_previous_input_;
    double highpass_previous_output_;
    double noise_floor_power_;
    double speech_noise_power_;
    int warmup_frames_required_;
    int warmup_frames_seen_;
    int gate_hangover_frames_;
    int gate_hangover_remaining_;
    int voice_candidate_frames_;
    double agc_gain_;
    double gate_gain_;
    double denoise_gain_;
    double applied_gain_;
    AudioPreprocessSnapshot snapshot_;
};

}  // namespace roi_h265

#endif  // ROI_H265_AUDIO_AUDIO_PREPROCESSOR_H_

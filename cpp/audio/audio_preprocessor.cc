#include "audio/audio_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace roi_h265 {
namespace {

const int kCodec2SampleRateHz = 8000;
const int kResampleRadius = 24;
const double kPi = 3.14159265358979323846;
const double kMinimumNoiseDbfs = -78.0;

double rmsPower(const std::vector<float> &samples) {
    if (samples.empty()) return 0.0;
    double total = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        total += static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
    }
    return total / static_cast<double>(samples.size());
}

}  // namespace

AudioPreprocessSnapshot::AudioPreprocessSnapshot()
    : noise_floor_dbfs(kMinimumNoiseDbfs), agc_gain_db(0.0), speech_snr_db(0.0),
      voicing_percent(0.0), gate_open(false), voice_detected(false),
      processed_frames(0), gated_frames(0) {}

AudioPreprocessor::AudioPreprocessor(int input_rate_hz, int input_channels,
                                     int output_frame_samples,
                                     const AudioPreprocessConfig &config)
    : input_rate_hz_(std::max(kCodec2SampleRateHz, input_rate_hz)),
      input_channels_(std::max(1, input_channels)),
      output_frame_samples_(std::max(1, output_frame_samples)), config_(config),
      input_per_output_(static_cast<double>(std::max(kCodec2SampleRateHz, input_rate_hz)) /
                        kCodec2SampleRateHz),
      next_input_position_(0.0), highpass_alpha_(1.0), highpass_previous_input_(0.0),
      highpass_previous_output_(0.0),
      noise_floor_power_(dbToPower(kMinimumNoiseDbfs)),
      speech_noise_power_(dbToPower(kMinimumNoiseDbfs)), warmup_frames_required_(0),
      warmup_frames_seen_(0), gate_hangover_frames_(0), gate_hangover_remaining_(0),
      voice_candidate_frames_(0), agc_gain_(1.0),
      gate_gain_(dbToLinear(-std::max(0, config.noise_gate_attenuation_db))),
      denoise_gain_(1.0), applied_gain_(gate_gain_) {
    const int safe_highpass = std::max(0, config_.highpass_hz);
    if (safe_highpass > 0) {
        const double rc = 1.0 / (2.0 * kPi * static_cast<double>(safe_highpass));
        const double dt = 1.0 / static_cast<double>(kCodec2SampleRateHz);
        highpass_alpha_ = rc / (rc + dt);
    }
    warmup_frames_required_ = std::max(0,
        (config_.noise_warmup_ms * kCodec2SampleRateHz +
         output_frame_samples_ * 1000 - 1) /
        (output_frame_samples_ * 1000));
    gate_hangover_frames_ = std::max(0,
        (config_.noise_gate_hangover_ms * kCodec2SampleRateHz +
         output_frame_samples_ * 1000 - 1) /
        (output_frame_samples_ * 1000));
}

void AudioPreprocessor::append(const int16_t *interleaved, size_t frames) {
    if (!interleaved || frames == 0) return;
    for (size_t frame = 0; frame < frames; ++frame) {
        double total = 0.0;
        for (int channel = 0; channel < input_channels_; ++channel) {
            total += interleaved[frame * static_cast<size_t>(input_channels_) + channel];
        }
        input_.push_back(static_cast<float>(total /
            (static_cast<double>(input_channels_) * 32768.0)));
    }
    resampleAvailable();
}

size_t AudioPreprocessor::availableFrames() const {
    return resampled_.size() / static_cast<size_t>(output_frame_samples_);
}

bool AudioPreprocessor::popFrame(std::vector<short> *output) {
    if (!output || availableFrames() == 0) return false;
    std::vector<float> frame(resampled_.begin(),
                             resampled_.begin() + output_frame_samples_);
    resampled_.erase(resampled_.begin(), resampled_.begin() + output_frame_samples_);
    processFrame(frame, output);
    return true;
}

AudioPreprocessSnapshot AudioPreprocessor::snapshot() const { return snapshot_; }

void AudioPreprocessor::resampleAvailable() {
    while (next_input_position_ + kResampleRadius < static_cast<double>(input_.size())) {
        resampled_.push_back(static_cast<float>(resampleAt(next_input_position_)));
        next_input_position_ += input_per_output_;
    }
    const size_t consumed = next_input_position_ > kResampleRadius
        ? static_cast<size_t>(next_input_position_) - kResampleRadius : 0U;
    if (consumed > 0 && consumed < input_.size()) {
        input_.erase(input_.begin(), input_.begin() + consumed);
        next_input_position_ -= static_cast<double>(consumed);
    }
}

double AudioPreprocessor::resampleAt(double source_position) const {
    if (input_.empty()) return 0.0;
    const double cutoff_hz = std::min(static_cast<double>(config_.lowpass_hz),
                                      0.45 * static_cast<double>(kCodec2SampleRateHz));
    const double normalized_cutoff = cutoff_hz / static_cast<double>(input_rate_hz_);
    const int center = static_cast<int>(std::floor(source_position));
    double weighted = 0.0;
    double coefficient_total = 0.0;
    for (int index = center - kResampleRadius + 1;
         index <= center + kResampleRadius; ++index) {
        const double delta = static_cast<double>(index) - source_position;
        const double distance = std::fabs(delta);
        if (distance >= kResampleRadius) continue;
        const double x = 2.0 * normalized_cutoff * delta;
        const double sinc = std::fabs(x) < 1e-12 ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double window = 0.5 * (1.0 + std::cos(kPi * delta / kResampleRadius));
        const double coefficient = 2.0 * normalized_cutoff * sinc * window;
        const int bounded_index = std::max(0, std::min(index,
            static_cast<int>(input_.size()) - 1));
        weighted += coefficient * input_[static_cast<size_t>(bounded_index)];
        coefficient_total += coefficient;
    }
    return std::fabs(coefficient_total) < 1e-12 ? 0.0 : weighted / coefficient_total;
}

void AudioPreprocessor::analyzeVoice(const std::vector<float> &samples,
                                     double *speech_power, double *low_power,
                                     double *voicing_percent) const {
    if (speech_power) *speech_power = 0.0;
    if (low_power) *low_power = 0.0;
    if (voicing_percent) *voicing_percent = 0.0;
    if (samples.size() < 64U) return;

    // A short, Hann-windowed spectrum separates low mechanical hum from the
    // 250--3400 Hz speech band.  The normalisation keeps its values stable
    // when Codec2 changes between 20 ms and 40 ms frame modes.
    const double window_total = static_cast<double>(samples.size() - 1U) / 2.0;
    if (window_total <= 0.0) return;
    const int maximum_bin = static_cast<int>(samples.size() / 2U);
    double voice_energy = 0.0;
    double hum_energy = 0.0;
    for (int bin = 1; bin <= maximum_bin; ++bin) {
        const double frequency = static_cast<double>(bin * kCodec2SampleRateHz) /
            static_cast<double>(samples.size());
        if (frequency < 80.0 || frequency > 3400.0) continue;
        double real = 0.0;
        double imaginary = 0.0;
        for (size_t index = 0; index < samples.size(); ++index) {
            const double window = 0.5 * (1.0 - std::cos(
                2.0 * kPi * static_cast<double>(index) /
                static_cast<double>(samples.size() - 1U)));
            const double phase = 2.0 * kPi * static_cast<double>(bin * index) /
                static_cast<double>(samples.size());
            const double value = static_cast<double>(samples[index]) * window;
            real += value * std::cos(phase);
            imaginary -= value * std::sin(phase);
        }
        const double power = (real * real + imaginary * imaginary) /
            (window_total * window_total);
        if (frequency < 250.0) hum_energy += power;
        else voice_energy += power;
    }
    if (speech_power) *speech_power = voice_energy;
    if (low_power) *low_power = hum_energy;

    // Peak normalised autocorrelation over a human-pitch range.  Pre-emphasis
    // avoids treating an AC hum as a voiced vowel.  This is intentionally only
    // an admission test: the soft-gate hangover keeps unvoiced consonants and
    // short pauses once a real voice has opened the path.
    std::vector<double> emphasized(samples.size(), 0.0);
    for (size_t index = 0; index < samples.size(); ++index) {
        const double previous = index == 0 ? 0.0 : static_cast<double>(samples[index - 1U]);
        emphasized[index] = static_cast<double>(samples[index]) - 0.85 * previous;
    }
    const int maximum_lag = std::min(100, static_cast<int>(samples.size()) - 32);
    double best_correlation = 0.0;
    for (int lag = 24; lag <= maximum_lag; ++lag) {
        double product = 0.0;
        double left_power = 0.0;
        double right_power = 0.0;
        for (size_t index = static_cast<size_t>(lag); index < emphasized.size(); ++index) {
            const double left = emphasized[index];
            const double right = emphasized[index - static_cast<size_t>(lag)];
            product += left * right;
            left_power += left * left;
            right_power += right * right;
        }
        const double denominator = std::sqrt(left_power * right_power);
        if (denominator > 1e-18) best_correlation = std::max(best_correlation, product / denominator);
    }
    if (voicing_percent) *voicing_percent = 100.0 * clamp(best_correlation, 0.0, 1.0);
}

void AudioPreprocessor::processFrame(const std::vector<float> &input,
                                     std::vector<short> *output) {
    std::vector<float> filtered(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const double sample = input[i];
        if (config_.enabled && config_.highpass_hz > 0) {
            const double highpassed = highpass_alpha_ *
                (highpass_previous_output_ + sample - highpass_previous_input_);
            highpass_previous_input_ = sample;
            highpass_previous_output_ = highpassed;
            filtered[i] = static_cast<float>(highpassed);
        } else {
            filtered[i] = input[i];
        }
    }

    if (!config_.enabled) {
        output->resize(filtered.size());
        for (size_t i = 0; i < filtered.size(); ++i) {
            const double bounded = clamp(filtered[i], -1.0, 1.0);
            (*output)[i] = static_cast<short>(std::lrint(bounded * 32767.0));
        }
        return;
    }

    const double frame_power = rmsPower(filtered);
    double speech_power = 0.0;
    double low_power = 0.0;
    double voicing_percent = 0.0;
    analyzeVoice(filtered, &speech_power, &low_power, &voicing_percent);
    const double minimum_noise_power = dbToPower(kMinimumNoiseDbfs);
    if (warmup_frames_seen_ < warmup_frames_required_) {
        const double count = static_cast<double>(warmup_frames_seen_);
        noise_floor_power_ = (noise_floor_power_ * count + frame_power) / (count + 1.0);
        speech_noise_power_ = (speech_noise_power_ * count + speech_power) / (count + 1.0);
        ++warmup_frames_seen_;
    }
    noise_floor_power_ = std::max(noise_floor_power_, minimum_noise_power);
    speech_noise_power_ = std::max(speech_noise_power_, minimum_noise_power);

    const double snr_power_ratio = dbToPower(static_cast<double>(config_.noise_gate_snr_db));
    const bool warmed_up = warmup_frames_seen_ >= warmup_frames_required_;
    const bool energy_above_noise = frame_power > noise_floor_power_ * snr_power_ratio;
    const bool speech_band_above_noise = speech_power > speech_noise_power_ * snr_power_ratio;
    const double speech_snr_db = 10.0 * std::log10(
        std::max(speech_power, minimum_noise_power) /
        std::max(speech_noise_power_, minimum_noise_power));
    // A male voice can have a strong 100--220 Hz fundamental, so require the
    // speech band to be present rather than to dominate that fundamental.  A
    // pure low-frequency fan/transformer hum still fails this -8 dB test.
    const bool midband_dominant = speech_power >= low_power * dbToPower(-8.0);
    const double voicing_threshold = static_cast<double>(
        std::max(0, std::min(100, config_.voice_min_voicing_percent)));
    const bool voiced = voicing_percent >= voicing_threshold;
    // A voiced harmonic structure remains a useful speech cue even when a
    // short burst of broad-band room sound temporarily raises the adaptive
    // energy floor.  Permit it with a 5 dB-relaxed SNR threshold; unvoiced
    // sounds must still clear the full threshold and therefore do not get fed
    // into AGC.
    const double voiced_snr_ratio = dbToPower(
        static_cast<double>(config_.noise_gate_snr_db - 5));
    const bool voiced_energy_above_noise = frame_power > noise_floor_power_ * voiced_snr_ratio;
    const bool voiced_speech_band_above_noise =
        speech_power > speech_noise_power_ * voiced_snr_ratio;
    const bool continuing_voice = gate_hangover_remaining_ > 0 &&
        speech_snr_db >= static_cast<double>(config_.noise_gate_snr_db - 1) &&
        voicing_percent >= voicing_threshold * 0.65;
    bool speech_candidate = warmed_up && energy_above_noise;
    if (config_.voice_activity_detection) {
        const bool normal_admission = energy_above_noise && speech_band_above_noise;
        const bool voiced_admission = voiced && voiced_energy_above_noise &&
            voiced_speech_band_above_noise;
        speech_candidate = warmed_up && midband_dominant &&
            (normal_admission || voiced_admission || continuing_voice) &&
            (voiced || continuing_voice);
    }
    if (speech_candidate) {
        ++voice_candidate_frames_;
    } else {
        voice_candidate_frames_ = std::max(0, voice_candidate_frames_ - 1);
    }
    const bool voice_detected = speech_candidate &&
        voice_candidate_frames_ >= std::max(1, config_.voice_start_frames);
    if (voice_detected) {
        gate_hangover_remaining_ = gate_hangover_frames_;
    } else if (gate_hangover_remaining_ > 0) {
        --gate_hangover_remaining_;
    }
    const bool gate_open = voice_detected || gate_hangover_remaining_ > 0;

    if (warmed_up && !gate_open) {
        // Rise slowly.  A passing truck, keyboard burst, or other temporary
        // non-speech sound must not be learned as the new voice threshold and
        // then make the next quiet spoken phrase disappear.
        const double adaptation = frame_power > noise_floor_power_ ? 0.01 : 0.04;
        noise_floor_power_ += adaptation * (frame_power - noise_floor_power_);
        speech_noise_power_ += adaptation * (speech_power - speech_noise_power_);
        noise_floor_power_ = std::max(noise_floor_power_, minimum_noise_power);
        speech_noise_power_ = std::max(speech_noise_power_, minimum_noise_power);
    }

    const double minimum_denoise_gain =
        dbToLinear(-static_cast<double>(config_.noise_suppression_db));
    const double wiener_gain = frame_power > minimum_noise_power
        ? std::max(0.0, 1.0 - noise_floor_power_ / frame_power) : 0.0;
    double target_denoise_gain = std::max(minimum_denoise_gain, wiener_gain);
    // Once the gate is open, retain at most 3 dB of broad-band attenuation for
    // its whole hangover.  Testing only voice_detected here made unvoiced
    // consonants (and the tail of a syllable) abruptly dull between VAD hits.
    if (gate_open) target_denoise_gain = std::max(target_denoise_gain, dbToLinear(-3.0));
    const double denoise_adaptation = target_denoise_gain < denoise_gain_ ? 0.18 : 0.05;
    denoise_gain_ += denoise_adaptation * (target_denoise_gain - denoise_gain_);
    const double cleaned_rms = std::sqrt(frame_power) * denoise_gain_;
    if (voice_detected && cleaned_rms > std::sqrt(minimum_noise_power)) {
        const double target_rms = dbToLinear(static_cast<double>(config_.agc_target_dbfs));
        const double desired_gain = clamp(target_rms / cleaned_rms, 0.125,
                                          dbToLinear(static_cast<double>(config_.agc_max_gain_db)));
        const double adaptation = desired_gain < agc_gain_ ? 0.50 : 0.06;
        agc_gain_ += adaptation * (desired_gain - agc_gain_);
    } else if (!gate_open) {
        // Let the gain recover only after the speech gate has really closed.
        // Holding it during the hangover avoids a gain collapse on low-energy
        // consonants and short pauses within otherwise continuous speech.
        agc_gain_ += 0.18 * (1.0 - agc_gain_);
    }

    const double closed_gate_gain =
        dbToLinear(-static_cast<double>(config_.noise_gate_attenuation_db));
    const double target_gate_gain = gate_open ? 1.0 : closed_gate_gain;
    const double gate_adaptation = gate_open ? 0.65 : 0.25;
    gate_gain_ += gate_adaptation * (target_gate_gain - gate_gain_);
    const double target_gain = denoise_gain_ * agc_gain_ * gate_gain_;
    output->resize(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        const double fraction = static_cast<double>(i + 1) /
            static_cast<double>(filtered.size());
        const double gain = applied_gain_ + fraction * (target_gain - applied_gain_);
        const double bounded = clamp(filtered[i] * gain, -1.0, 1.0);
        (*output)[i] = static_cast<short>(std::lrint(bounded * 32767.0));
    }
    applied_gain_ = target_gain;

    snapshot_.noise_floor_dbfs = 10.0 * std::log10(
        std::max(noise_floor_power_, minimum_noise_power));
    snapshot_.agc_gain_db = 20.0 * std::log10(std::max(agc_gain_, 1e-12));
    snapshot_.speech_snr_db = speech_snr_db;
    snapshot_.voicing_percent = voicing_percent;
    snapshot_.gate_open = gate_open;
    snapshot_.voice_detected = voice_detected;
    ++snapshot_.processed_frames;
    if (!gate_open) ++snapshot_.gated_frames;
}

double AudioPreprocessor::dbToLinear(double db) { return std::pow(10.0, db / 20.0); }

double AudioPreprocessor::dbToPower(double db) { return std::pow(10.0, db / 10.0); }

double AudioPreprocessor::clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

}  // namespace roi_h265

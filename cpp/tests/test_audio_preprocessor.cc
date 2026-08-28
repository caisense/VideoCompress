#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "audio/audio_preprocessor.h"

namespace {

int failures = 0;
const double kPi = 3.14159265358979323846;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

std::vector<int16_t> sine(int sample_rate, int samples, double frequency, double amplitude,
                          double phase_offset = 0.0) {
    std::vector<int16_t> result(static_cast<size_t>(samples));
    for (int index = 0; index < samples; ++index) {
        const double phase = 2.0 * kPi * frequency * index / sample_rate + phase_offset;
        const double value = std::max(-1.0, std::min(1.0, amplitude * std::sin(phase)));
        result[static_cast<size_t>(index)] = static_cast<int16_t>(std::lrint(value * 32767.0));
    }
    return result;
}

void append(std::vector<int16_t> *destination, const std::vector<int16_t> &source) {
    destination->insert(destination->end(), source.begin(), source.end());
}

std::vector<int16_t> add(const std::vector<int16_t> &left,
                         const std::vector<int16_t> &right) {
    std::vector<int16_t> result(std::min(left.size(), right.size()));
    for (size_t index = 0; index < result.size(); ++index) {
        const int mixed = static_cast<int>(left[index]) + static_cast<int>(right[index]);
        result[index] = static_cast<int16_t>(std::max(-32768, std::min(32767, mixed)));
    }
    return result;
}

std::vector<int16_t> whiteNoise(int samples, double amplitude, uint32_t seed) {
    std::vector<int16_t> result(static_cast<size_t>(samples));
    uint32_t state = seed;
    for (int index = 0; index < samples; ++index) {
        state = state * 1664525U + 1013904223U;
        const double unit = static_cast<double>((state >> 8) & 0xffffU) / 32767.5 - 1.0;
        result[static_cast<size_t>(index)] = static_cast<int16_t>(std::lrint(
            std::max(-1.0, std::min(1.0, unit * amplitude)) * 32767.0));
    }
    return result;
}

std::vector<short> drain(roi_h265::AudioPreprocessor *preprocessor) {
    std::vector<short> result;
    std::vector<short> frame;
    while (preprocessor->popFrame(&frame)) result.insert(result.end(), frame.begin(), frame.end());
    return result;
}

double toneMagnitude(const std::vector<short> &samples, int sample_rate, double frequency) {
    if (samples.empty()) return 0.0;
    const double coefficient = 2.0 * std::cos(2.0 * kPi * frequency / sample_rate);
    double first = 0.0;
    double second = 0.0;
    for (size_t index = 0; index < samples.size(); ++index) {
        const double current = static_cast<double>(samples[index]) / 32768.0 +
            coefficient * first - second;
        second = first;
        first = current;
    }
    const double power = first * first + second * second - coefficient * first * second;
    return 2.0 * std::sqrt(std::max(0.0, power)) / samples.size();
}

double rms(const std::vector<short> &samples, size_t begin, size_t end) {
    if (begin >= end || end > samples.size()) return 0.0;
    double total = 0.0;
    for (size_t index = begin; index < end; ++index) {
        const double value = static_cast<double>(samples[index]) / 32768.0;
        total += value * value;
    }
    return std::sqrt(total / static_cast<double>(end - begin));
}

void testBandLimitedResamplerRejectsAlias() {
    roi_h265::AudioPreprocessConfig config;
    config.enabled = false;
    const int source_samples = 44100;

    roi_h265::AudioPreprocessor speech_band(44100, 1, 320, config);
    const std::vector<int16_t> speech = sine(44100, source_samples, 1000.0, 0.70);
    speech_band.append(speech.data(), speech.size());
    const std::vector<short> speech_output = drain(&speech_band);
    CHECK(speech_output.size() >= 8U * 320U);
    const std::vector<short> speech_steady(speech_output.begin() + 320, speech_output.end());
    const double speech_magnitude = toneMagnitude(speech_steady, 8000, 1000.0);
    CHECK(speech_magnitude > 0.45);

    roi_h265::AudioPreprocessor out_of_band(44100, 1, 320, config);
    const std::vector<int16_t> high = sine(44100, source_samples, 6000.0, 0.70);
    out_of_band.append(high.data(), high.size());
    const std::vector<short> high_output = drain(&out_of_band);
    CHECK(high_output.size() >= 8U * 320U);
    const std::vector<short> high_steady(high_output.begin() + 320, high_output.end());
    // 6 kHz would alias to 2 kHz after 8 kHz conversion without the low-pass.
    CHECK(toneMagnitude(high_steady, 8000, 2000.0) < speech_magnitude * 0.12);
}

void testHighpassRemovesDc() {
    roi_h265::AudioPreprocessConfig config;
    config.highpass_hz = 80;
    config.noise_suppression_db = 0;
    config.noise_gate_attenuation_db = 0;
    config.noise_gate_snr_db = 0;
    config.noise_warmup_ms = 0;
    config.agc_max_gain_db = 0;
    roi_h265::AudioPreprocessor preprocessor(8000, 1, 320, config);
    const std::vector<int16_t> dc(8000, static_cast<int16_t>(0.50 * 32767.0));
    preprocessor.append(dc.data(), dc.size());
    const std::vector<short> output = drain(&preprocessor);
    CHECK(output.size() >= 6U * 320U);
    CHECK(rms(output, output.size() - 320U, output.size()) < 0.015);
}

void testAdaptiveGateAndAgcPreserveSpeech() {
    roi_h265::AudioPreprocessConfig config;
    config.highpass_hz = 0;
    config.noise_warmup_ms = 200;
    config.noise_suppression_db = 8;
    config.noise_gate_snr_db = 1;
    config.noise_gate_attenuation_db = 10;
    config.noise_gate_hangover_ms = 500;
    config.agc_target_dbfs = -16;
    config.agc_max_gain_db = 24;
    // The documented weak-microphone diagnostic profile deliberately retains
    // the legacy energy-only admission rule.  It proves that an installer can
    // still prefer weak speech over environmental rejection when necessary.
    config.voice_activity_detection = false;
    config.voice_start_frames = 1;
    roi_h265::AudioPreprocessor preprocessor(8000, 1, 320, config);

    std::vector<int16_t> input;
    const std::vector<int16_t> room_noise = sine(8000, 320, 300.0, 0.003);
    // This is just 1.1 dB above the noise power: it exercises the failure mode
    // reported on the ES8388 board, where a 3 dB hard gate erased quiet speech.
    const std::vector<int16_t> weak_voice = add(room_noise, sine(8000, 320, 600.0, 0.0016));
    for (int frame = 0; frame < 20; ++frame) append(&input, room_noise);
    for (int frame = 0; frame < 40; ++frame) append(&input, weak_voice);
    for (int frame = 0; frame < 20; ++frame) append(&input, room_noise);
    preprocessor.append(input.data(), input.size());
    const std::vector<short> output = drain(&preprocessor);
    CHECK(output.size() >= 78U * 320U);

    double speech_peak_rms = 0.0;
    for (size_t offset = 26U * 320U; offset + 320U <= 56U * 320U; offset += 320U) {
        speech_peak_rms = std::max(speech_peak_rms, rms(output, offset, offset + 320U));
    }
    const double quiet_rms = rms(output, output.size() - 4U * 320U, output.size());
    const roi_h265::AudioPreprocessSnapshot snapshot = preprocessor.snapshot();
    // The useful behavior is not merely opening the gate: the profile must
    // retain a voice that is only slightly above the learned noise floor, then
    // let AGC lift it while keeping the final quiet segment lower.
    CHECK(speech_peak_rms > 0.010);
    CHECK(speech_peak_rms > quiet_rms * 6.0);
    CHECK(snapshot.processed_frames > 60U);
    CHECK(snapshot.gated_frames > 0U);
    CHECK(!snapshot.gate_open);
}

void testVoiceAwareGateRejectsBroadbandRoomSound() {
    roi_h265::AudioPreprocessConfig config;
    config.highpass_hz = 0;
    config.lowpass_hz = 3600;
    config.noise_warmup_ms = 400;
    config.noise_suppression_db = 6;
    config.noise_gate_snr_db = 3;
    config.noise_gate_attenuation_db = 30;
    config.noise_gate_hangover_ms = 240;
    config.agc_target_dbfs = -18;
    config.agc_max_gain_db = 12;
    config.voice_activity_detection = true;
    config.voice_start_frames = 2;
    config.voice_min_voicing_percent = 52;
    roi_h265::AudioPreprocessor preprocessor(8000, 1, 320, config);

    const int section_samples = 20 * 320;
    const std::vector<int16_t> quiet = whiteNoise(section_samples, 0.0008, 1U);
    const std::vector<int16_t> environmental = whiteNoise(section_samples, 0.030, 2U);
    std::vector<int16_t> voice = whiteNoise(section_samples, 0.0008, 3U);
    const std::vector<int16_t> fundamental = sine(8000, section_samples, 180.0, 0.018);
    const std::vector<int16_t> harmonic1 = sine(8000, section_samples, 540.0, 0.010);
    const std::vector<int16_t> harmonic2 = sine(8000, section_samples, 1080.0, 0.006);
    voice = add(add(voice, fundamental), add(harmonic1, harmonic2));

    std::vector<int16_t> input;
    append(&input, quiet);
    append(&input, environmental);
    append(&input, voice);
    append(&input, quiet);
    preprocessor.append(input.data(), input.size());
    std::vector<short> output;
    std::vector<short> frame;
    std::vector<roi_h265::AudioPreprocessSnapshot> snapshots;
    while (preprocessor.popFrame(&frame)) {
        output.insert(output.end(), frame.begin(), frame.end());
        snapshots.push_back(preprocessor.snapshot());
    }
    CHECK(output.size() >= 75U * 320U);

    // Ignore the 24-sample resampler look-ahead and gate transitions by using
    // the middle half of each 0.8 s section.
    const size_t environmental_begin = 25U * 320U;
    const size_t voice_begin = 45U * 320U;
    const size_t quiet_begin = 65U * 320U;
    const double environmental_rms = rms(output, environmental_begin, environmental_begin + 8U * 320U);
    const double voice_rms = rms(output, voice_begin, voice_begin + 8U * 320U);
    const double tail_rms = rms(output, quiet_begin, quiet_begin + 8U * 320U);
    const roi_h265::AudioPreprocessSnapshot snapshot = preprocessor.snapshot();
    double max_voice_snr = -100.0;
    double max_voice_voicing = 0.0;
    int voice_gate_frames = 0;
    int voice_detected_frames = 0;
    int environmental_gate_frames = 0;
    for (size_t index = 25U; index < 35U && index < snapshots.size(); ++index) {
        if (snapshots[index].gate_open) ++environmental_gate_frames;
    }
    for (size_t index = 45U; index < 55U && index < snapshots.size(); ++index) {
        max_voice_snr = std::max(max_voice_snr, snapshots[index].speech_snr_db);
        max_voice_voicing = std::max(max_voice_voicing, snapshots[index].voicing_percent);
        if (snapshots[index].gate_open) ++voice_gate_frames;
        if (snapshots[index].voice_detected) ++voice_detected_frames;
    }
    CHECK(voice_rms > 0.015);
    CHECK(voice_rms > environmental_rms * 12.0);
    CHECK(voice_rms > tail_rms * 12.0);
    CHECK(max_voice_snr > 0.0 && max_voice_voicing >= 52.0);
    CHECK(environmental_gate_frames == 0);
    CHECK(voice_gate_frames >= 8 && voice_detected_frames >= 8);
    CHECK(!snapshot.gate_open);
    CHECK(!snapshot.voice_detected);
}

void testAgcHoldsAcrossSpeechGateHangover() {
    roi_h265::AudioPreprocessConfig config;
    config.highpass_hz = 0;
    config.noise_warmup_ms = 0;
    config.noise_suppression_db = 6;
    config.noise_gate_snr_db = 3;
    config.noise_gate_attenuation_db = 30;
    config.noise_gate_hangover_ms = 200;
    config.agc_target_dbfs = -16;
    config.agc_max_gain_db = 20;
    config.voice_activity_detection = false;
    config.voice_start_frames = 1;
    roi_h265::AudioPreprocessor preprocessor(8000, 1, 320, config);

    std::vector<int16_t> input;
    const std::vector<int16_t> voiced = sine(8000, 5 * 320, 600.0, 0.015);
    const std::vector<int16_t> quiet(5 * 320, 0);
    append(&input, voiced);
    append(&input, quiet);
    preprocessor.append(input.data(), input.size());

    std::vector<short> output;
    bool saw_voice = false;
    bool checked_hangover = false;
    double last_voice_gain_db = 0.0;
    while (preprocessor.popFrame(&output)) {
        const roi_h265::AudioPreprocessSnapshot snapshot = preprocessor.snapshot();
        if (snapshot.voice_detected) {
            saw_voice = true;
            last_voice_gain_db = snapshot.agc_gain_db;
        } else if (saw_voice && snapshot.gate_open) {
            // A VAD miss must not reset the gain that will carry quiet
            // consonants through the configured speech-gate hangover.
            CHECK(snapshot.agc_gain_db >= last_voice_gain_db - 0.1);
            checked_hangover = true;
            break;
        }
    }
    CHECK(saw_voice);
    CHECK(checked_hangover);
}

}  // namespace

int main() {
    testBandLimitedResamplerRejectsAlias();
    testHighpassRemovesDc();
    testAdaptiveGateAndAgcPreserveSpeech();
    testVoiceAwareGateRejectsBroadbandRoomSound();
    testAgcHoldsAcrossSpeechGateHangover();
    if (failures) {
        std::cerr << failures << " audio preprocessor test(s) failed\n";
        return 1;
    }
    std::cout << "audio preprocessor tests passed\n";
    return 0;
}

#ifndef ROI_H265_AUDIO_ARECORD_CAPTURE_H_
#define ROI_H265_AUDIO_ARECORD_CAPTURE_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

namespace roi_h265 {

// The supplied Buildroot image already exposes the ES8388 microphone through
// arecord.  Running it as a small child process avoids adding an ALSA library
// dependency to the existing RKNN/MPP application while preserving the board
// documented device selection (hw:3,0).
class ArecordCapture {
public:
    ArecordCapture();
    ~ArecordCapture();

    bool open(const std::string &device, int sample_rate_hz, int channels, std::string *error);
    bool readInterleaved(int16_t *samples, size_t frames, std::string *error);
    void requestStop();
    void close();

private:
    int read_fd_;
    pid_t child_pid_;
    int channels_;
};

}  // namespace roi_h265

#endif  // ROI_H265_AUDIO_ARECORD_CAPTURE_H_

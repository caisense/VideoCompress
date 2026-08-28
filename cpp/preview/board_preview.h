#ifndef ROI_H265_PREVIEW_BOARD_PREVIEW_H_
#define ROI_H265_PREVIEW_BOARD_PREVIEW_H_

#include <string>

#include "common/frame_meta.h"

namespace roi_h265 {

// Shows an annotated view of the encoder input on the board while keeping
// every HighGUI call in the main thread. The encoder never waits for this
// optional display path.
class BoardPreview {
public:
    BoardPreview(int window_width, int window_height, bool rotate_ccw);
    ~BoardPreview();

    // Populate the environment missing from an SSH shell when Buildroot Weston
    // is listening on its standard /run/wayland-0 socket.
    static void prepareEnvironment();

    bool open(std::string* error);
    // Returns false when the user closes the window or presses q/Esc.
    bool show(const FramePacket& frame, const SegResult* segmentation, std::string* error);
    void close();

private:
    bool opened_;
    int window_width_;
    int window_height_;
    bool rotate_ccw_;
};

}  // namespace roi_h265

#endif  // ROI_H265_PREVIEW_BOARD_PREVIEW_H_

#ifndef ROI_H265_TRANSPORT_CODEC2_SDP_H_
#define ROI_H265_TRANSPORT_CODEC2_SDP_H_

#include <string>

namespace roi_h265 {

struct Codec2RtpSdpConfig {
    int udp_port;
    int payload_type;
    int mode_bps;
    int sample_rate_hz;
    int channels;
    int frames_per_packet;
    int samples_per_frame;
    int bits_per_frame;
    int bytes_per_frame;
    bool dtx_enabled;
    int dtx_keepalive_ms;

    Codec2RtpSdpConfig()
        : udp_port(5006), payload_type(97), mode_bps(1300), sample_rate_hz(8000),
          channels(1), frames_per_packet(2), samples_per_frame(320),
          bits_per_frame(52), bytes_per_frame(7), dtx_enabled(false),
          dtx_keepalive_ms(0) {}
};

// The audio SDP is intentionally separate from the H.265 SDP.  A stock
// H.265 FFplay receiver can therefore keep binding only the video RTP port,
// while the Codec2 helper binds the audio port.
bool writeCodec2RtpSdp(const Codec2RtpSdpConfig &config, const std::string &output_path,
                       std::string *error);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_CODEC2_SDP_H_

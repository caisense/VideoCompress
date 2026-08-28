#include "transport/codec2_sdp.h"

#include <cstdio>
#include <fstream>

namespace roi_h265 {

bool writeCodec2RtpSdp(const Codec2RtpSdpConfig &config, const std::string &output_path,
                       std::string *error) {
    if (output_path.empty() || config.udp_port < 1 || config.udp_port > 65535 ||
        config.payload_type < 96 || config.payload_type > 127 || config.mode_bps <= 0 ||
        config.sample_rate_hz <= 0 || config.channels != 1 || config.frames_per_packet <= 0 ||
        config.samples_per_frame <= 0 || config.bits_per_frame <= 0 || config.bytes_per_frame <= 0 ||
        config.dtx_keepalive_ms < 0 || config.dtx_keepalive_ms > 10000) {
        if (error) *error = "invalid Codec2 RTP SDP configuration";
        return false;
    }

    const std::string temporary_path = output_path + ".tmp";
    std::ofstream output(temporary_path.c_str(), std::ios::out | std::ios::trunc);
    if (!output) {
        if (error) *error = "cannot create Codec2 RTP SDP file: " + temporary_path;
        return false;
    }
    output << "v=0\n"
           << "o=- 0 0 IN IP4 127.0.0.1\n"
           << "s=RK3588 Codec2 audio\n"
           << "c=IN IP4 0.0.0.0\n"
           << "t=0 0\n"
           << "m=audio " << config.udp_port << " RTP/AVP " << config.payload_type << "\n"
           << "a=rtpmap:" << config.payload_type << " CODEC2/" << config.sample_rate_hz
           << "/" << config.channels << "\n"
           << "a=fmtp:" << config.payload_type << " mode=" << config.mode_bps
           << ";frames-per-packet=" << config.frames_per_packet
           << ";bits-per-frame=" << config.bits_per_frame
           << ";bytes-per-frame=" << config.bytes_per_frame
           << ";dtx=" << (config.dtx_enabled ? 1 : 0)
           << ";dtx-keepalive-ms=" << config.dtx_keepalive_ms << "\n"
           << "a=ptime:" << (1000 * config.frames_per_packet * config.samples_per_frame /
                               config.sample_rate_hz) << "\n";
    output.close();
    if (!output) {
        std::remove(temporary_path.c_str());
        if (error) *error = "cannot write Codec2 RTP SDP file: " + output_path;
        return false;
    }
    if (std::rename(temporary_path.c_str(), output_path.c_str()) != 0) {
        std::remove(temporary_path.c_str());
        if (error) *error = "cannot publish Codec2 RTP SDP file: " + output_path;
        return false;
    }
    return true;
}

}  // namespace roi_h265

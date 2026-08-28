#include "transport/h265_sdp.h"
#include "transport/codec2_sdp.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool contains(const std::string& text, const std::string& expected) {
    return text.find(expected) != std::string::npos;
}

void expect(bool condition, const char* message, int* failures) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++*failures;
    }
}

}  // namespace

int main() {
    const char* path = "/tmp/roi_h265_sdp_test.sdp";
    std::remove(path);
    const std::vector<unsigned char> annex_b = {
        0, 0, 0, 1, 0x40, 0x01, 0x01,
        0, 0, 1, 0x42, 0x01, 0x02, 0x03,
        0, 0, 1, 0x44, 0x01, 0x04};
    std::string error;
    int failures = 0;
    expect(roi_h265::writeH265RtpSdp(annex_b, 5004, path, &error), error.c_str(), &failures);

    std::ifstream input(path);
    const std::string sdp((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    expect(contains(sdp, "m=video 5004 RTP/AVP 96"), "SDP has RTP port", &failures);
    expect(contains(sdp, "sprop-vps=QAEB"), "SDP has VPS", &failures);
    expect(contains(sdp, "sprop-sps=QgECAw=="), "SDP has SPS", &failures);
    expect(contains(sdp, "sprop-pps=RAEE"), "SDP has PPS", &failures);
    std::remove(path);

    const char* audio_path = "/tmp/roi_codec2_sdp_test.sdp";
    std::remove(audio_path);
    roi_h265::Codec2RtpSdpConfig audio;
    audio.udp_port = 5006;
    audio.mode_bps = 1300;
    audio.frames_per_packet = 2;
    audio.samples_per_frame = 320;
    audio.bits_per_frame = 52;
    audio.bytes_per_frame = 7;
    audio.dtx_enabled = true;
    audio.dtx_keepalive_ms = 1000;
    error.clear();
    expect(roi_h265::writeCodec2RtpSdp(audio, audio_path, &error), error.c_str(), &failures);
    std::ifstream audio_input(audio_path);
    const std::string audio_sdp((std::istreambuf_iterator<char>(audio_input)), std::istreambuf_iterator<char>());
    expect(contains(audio_sdp, "m=audio 5006 RTP/AVP 97"), "audio SDP has RTP port", &failures);
    expect(contains(audio_sdp, "a=rtpmap:97 CODEC2/8000/1"), "audio SDP has Codec2 map", &failures);
    expect(contains(audio_sdp, "mode=1300;frames-per-packet=2;bits-per-frame=52;bytes-per-frame=7"),
           "audio SDP has fixed Codec2 frame geometry", &failures);
    expect(contains(audio_sdp, "dtx=1;dtx-keepalive-ms=1000"),
           "audio SDP declares DTX keepalive behavior", &failures);
    expect(contains(audio_sdp, "a=ptime:80"), "audio SDP has 80 ms packet time", &failures);
    std::remove(audio_path);
    return failures == 0 ? 0 : 1;
}

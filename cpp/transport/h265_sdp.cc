#include "transport/h265_sdp.h"

#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

namespace roi_h265 {
namespace {

size_t startCodeLength(const unsigned char* data, size_t length, size_t offset) {
    if (offset + 3 <= length && data[offset] == 0 && data[offset + 1] == 0 && data[offset + 2] == 1) return 3;
    if (offset + 4 <= length && data[offset] == 0 && data[offset + 1] == 0 &&
        data[offset + 2] == 0 && data[offset + 3] == 1) return 4;
    return 0;
}

std::string base64Encode(const unsigned char* data, size_t length) {
    static const char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((length + 2) / 3) * 4);
    for (size_t offset = 0; offset < length; offset += 3) {
        const unsigned int group = static_cast<unsigned int>(data[offset]) << 16 |
            (offset + 1 < length ? static_cast<unsigned int>(data[offset + 1]) << 8 : 0) |
            (offset + 2 < length ? static_cast<unsigned int>(data[offset + 2]) : 0);
        output.push_back(kAlphabet[(group >> 18) & 0x3f]);
        output.push_back(kAlphabet[(group >> 12) & 0x3f]);
        output.push_back(offset + 1 < length ? kAlphabet[(group >> 6) & 0x3f] : '=');
        output.push_back(offset + 2 < length ? kAlphabet[group & 0x3f] : '=');
    }
    return output;
}

}  // namespace

bool writeH265RtpSdp(const std::vector<unsigned char>& annex_b, int udp_port,
                     const std::string& output_path, std::string* error) {
    if (udp_port < 1 || udp_port > 65535 || output_path.empty()) {
        if (error) *error = "invalid RTP SDP output path or UDP port";
        return false;
    }

    std::string vps;
    std::string sps;
    std::string pps;
    size_t cursor = 0;
    while (cursor < annex_b.size()) {
        const size_t start_code = startCodeLength(annex_b.data(), annex_b.size(), cursor);
        if (!start_code) {
            ++cursor;
            continue;
        }
        const size_t nal_start = cursor + start_code;
        size_t next = nal_start;
        while (next < annex_b.size() && !startCodeLength(annex_b.data(), annex_b.size(), next)) ++next;
        if (next >= nal_start + 2) {
            const unsigned char* nal = annex_b.data() + nal_start;
            const unsigned char nal_type = static_cast<unsigned char>((nal[0] >> 1) & 0x3f);
            if (nal_type == 32 && vps.empty()) vps = base64Encode(nal, next - nal_start);
            if (nal_type == 33 && sps.empty()) sps = base64Encode(nal, next - nal_start);
            if (nal_type == 34 && pps.empty()) pps = base64Encode(nal, next - nal_start);
        }
        cursor = next;
    }
    if (vps.empty() || sps.empty() || pps.empty()) {
        if (error) *error = "IDR access unit did not contain all H.265 VPS/SPS/PPS NAL units";
        return false;
    }

    const std::string temporary_path = output_path + ".tmp";
    std::ofstream output(temporary_path.c_str(), std::ios::out | std::ios::trunc);
    if (!output) {
        if (error) *error = "cannot create RTP SDP file: " + temporary_path;
        return false;
    }
    output << "v=0\n"
           << "o=- 0 0 IN IP4 127.0.0.1\n"
           << "s=RK3588 H265\n"
           << "c=IN IP4 0.0.0.0\n"
           << "t=0 0\n"
           << "m=video " << udp_port << " RTP/AVP 96\n"
           << "a=rtpmap:96 H265/90000\n"
           << "a=fmtp:96 sprop-vps=" << vps << ";sprop-sps=" << sps
           << ";sprop-pps=" << pps << "\n";
    output.close();
    if (!output) {
        std::remove(temporary_path.c_str());
        if (error) *error = "cannot write RTP SDP file: " + temporary_path;
        return false;
    }
    if (std::rename(temporary_path.c_str(), output_path.c_str()) != 0) {
        std::remove(temporary_path.c_str());
        if (error) *error = "cannot publish RTP SDP file: " + output_path;
        return false;
    }
    return true;
}

}  // namespace roi_h265

#include "transport/packetizer.h"

#include <algorithm>
#include <utility>

namespace roi_h265 {
namespace {

size_t startCodeLength(const uint8_t *data, size_t length, size_t offset) {
    if (offset + 3 <= length && data[offset] == 0 && data[offset + 1] == 0 && data[offset + 2] == 1) return 3;
    if (offset + 4 <= length && data[offset] == 0 && data[offset + 1] == 0 &&
        data[offset + 2] == 0 && data[offset + 3] == 1) return 4;
    return 0;
}

}  // namespace

H265RtpPacketizer::H265RtpPacketizer(uint16_t sequence_number, uint32_t ssrc, int mtu)
    // 12-byte RTP header + 3-byte H.265 FU header leaves at least one byte of
    // fragment payload. Honour larger caller-provided MTUs exactly.
    : sequence_number_(sequence_number), ssrc_(ssrc), mtu_(std::max(28, mtu)) {}

std::vector<uint8_t> H265RtpPacketizer::makeHeader(bool marker, uint32_t timestamp,
                                                   const RtpStreamProfile *profile) {
    std::vector<uint8_t> header(12, 0);
    const bool include_profile = profile && profile->valid;
    header[0] = static_cast<uint8_t>(0x80 | (include_profile ? 0x10 : 0x00));
    header[1] = static_cast<uint8_t>((marker ? 0x80 : 0x00) | 96);  // dynamic H.265 payload type
    header[2] = static_cast<uint8_t>(sequence_number_ >> 8);
    header[3] = static_cast<uint8_t>(sequence_number_++);
    header[4] = static_cast<uint8_t>(timestamp >> 24);
    header[5] = static_cast<uint8_t>(timestamp >> 16);
    header[6] = static_cast<uint8_t>(timestamp >> 8);
    header[7] = static_cast<uint8_t>(timestamp);
    header[8] = static_cast<uint8_t>(ssrc_ >> 24);
    header[9] = static_cast<uint8_t>(ssrc_ >> 16);
    header[10] = static_cast<uint8_t>(ssrc_ >> 8);
    header[11] = static_cast<uint8_t>(ssrc_);
    if (include_profile) {
        // RFC 3550 header extension. 0x524f ("RO") identifies this project's
        // eight-byte profile record. It is repeated on every packet so the
        // receiver can restart its decoder before forwarding the first VPS of
        // a new profile; standard RTP/H.265 receivers simply skip it.
        header.push_back(0x52);
        header.push_back(0x4f);
        header.push_back(0x00);
        header.push_back(0x02);
        header.push_back(0x01);  // metadata version
        header.push_back(profile->profile);
        header.push_back(static_cast<uint8_t>(profile->width >> 8));
        header.push_back(static_cast<uint8_t>(profile->width));
        header.push_back(static_cast<uint8_t>(profile->height >> 8));
        header.push_back(static_cast<uint8_t>(profile->height));
        header.push_back(profile->fps);
        header.push_back(profile->generation);
    }
    return header;
}

void H265RtpPacketizer::appendNal(const uint8_t *nal, size_t length, uint32_t timestamp,
                                  bool marker, const RtpStreamProfile *profile,
                                  std::vector<std::vector<uint8_t> > *packets) {
    if (length < 2) return;
    const size_t header_bytes = profile && profile->valid ? 24 : 12;
    const size_t max_payload = static_cast<size_t>(mtu_) - header_bytes;
    if (length <= max_payload) {
        std::vector<uint8_t> packet = makeHeader(marker, timestamp, profile);
        packet.insert(packet.end(), nal, nal + length);
        packets->push_back(packet);
        return;
    }

    // RFC 7798 FU payload: 2-byte payload header (type 49) and 1-byte FU header.
    const uint8_t nal_type = static_cast<uint8_t>((nal[0] >> 1) & 0x3f);
    const uint8_t fu_header_base = nal_type;
    const uint8_t fu_indicator0 = static_cast<uint8_t>((nal[0] & 0x81) | (49 << 1));
    const uint8_t fu_indicator1 = nal[1];
    const size_t fragment_capacity = max_payload - 3;
    size_t offset = 2;
    bool first = true;
    while (offset < length) {
        const size_t take = std::min(fragment_capacity, length - offset);
        const bool last = offset + take == length;
        std::vector<uint8_t> packet = makeHeader(marker && last, timestamp, profile);
        packet.push_back(fu_indicator0);
        packet.push_back(fu_indicator1);
        packet.push_back(static_cast<uint8_t>(fu_header_base | (first ? 0x80 : 0) | (last ? 0x40 : 0)));
        packet.insert(packet.end(), nal + offset, nal + offset + take);
        packets->push_back(packet);
        offset += take;
        first = false;
    }
}

std::vector<std::vector<uint8_t> > H265RtpPacketizer::packetize(const uint8_t *annex_b, size_t length,
                                                                  uint32_t timestamp,
                                                                  const RtpStreamProfile *profile) {
    std::vector<std::vector<uint8_t> > packets;
    if (!annex_b || length == 0) return packets;
    std::vector<std::pair<const uint8_t *, size_t> > nals;
    size_t cursor = 0;
    while (cursor < length) {
        const size_t code = startCodeLength(annex_b, length, cursor);
        if (!code) { ++cursor; continue; }
        const size_t nal_start = cursor + code;
        size_t next = nal_start;
        while (next < length && !startCodeLength(annex_b, length, next)) ++next;
        if (next > nal_start) nals.push_back(std::make_pair(annex_b + nal_start, next - nal_start));
        cursor = next;
    }
    for (size_t i = 0; i < nals.size(); ++i) {
        appendNal(nals[i].first, nals[i].second, timestamp, i + 1 == nals.size(), profile, &packets);
    }
    return packets;
}

}  // namespace roi_h265

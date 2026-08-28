#ifndef ROI_H265_TRANSPORT_H265_SDP_H_
#define ROI_H265_TRANSPORT_H265_SDP_H_

#include <string>
#include <vector>

namespace roi_h265 {

// Writes an RFC 7798 SDP description from VPS/SPS/PPS NAL units found in an
// Annex-B IDR access unit.  FFmpeg needs these out-of-band parameters to know
// the HEVC stream dimensions before it can show RTP video.
bool writeH265RtpSdp(const std::vector<unsigned char>& annex_b, int udp_port,
                     const std::string& output_path, std::string* error);

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_H265_SDP_H_

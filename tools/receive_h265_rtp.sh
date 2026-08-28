#!/usr/bin/env bash
# Save the standard H.265 RTP stream using a stock FFmpeg build on the PC.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <udp-port> <duration-seconds> <output.h265> <sender-generated.sdp>" >&2
  exit 2
fi

port="$1"
duration="$2"
output="$3"
sdp="$4"

if [[ ! -f "$sdp" ]]; then
  echo "SDP not found: $sdp" >&2
  exit 2
fi
if ! grep -Fq "m=video ${port} RTP/AVP 96" "$sdp"; then
  echo "UDP port ${port} does not match SDP: $sdp" >&2
  exit 2
fi

timeout --signal=INT "$duration" ffmpeg -y -protocol_whitelist file,udp,rtp \
  -analyzeduration 1000000 -probesize 1000000 \
  -i "$sdp" -map 0:v:0 -c copy -f hevc "$output"

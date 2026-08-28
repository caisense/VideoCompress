#!/usr/bin/env bash
# Run A/B/C on the RK3588 with exactly the same source video and sender config.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <sender> <model.rknn> <input-video> <frames> <output-dir> [udp-host]" >&2
  exit 2
fi

sender="$1"
model="$2"
input="$3"
frames="$4"
out="$5"
host="${6:-127.0.0.1}"
mkdir -p "$out"

for item in A:baseline B:bbox C:segmentation; do
  label="${item%%:*}"
  mode="${item#*:}"
  mkdir -p "$out/$label/roi"
  "$sender" \
    --model="$model" --input-video="$input" --max-frames="$frames" --mode="$mode" \
    --encoder-width=320 --encoder-height=180 --fps=10 --target-bitrate=42000 --gop=50 \
    --qp-min=10 --qp-max=51 --background-delta-qp=12 --halo-delta-qp=2 \
    --core-delta-qp=-6 --edge-delta-qp=-10 --mask-occupancy-threshold=0.10 \
    --erosion-radius=2 --dilation-radius=3 --roi-hold-frames=3 --roi-max-age=9 \
    --max-roi-region=64 --udp-host="$host" --udp-port=5004 --pacing-bitrate=60000 \
    --qp-init=38 --qp-min-i=36 --qp-max-i=48 --intra-refresh=on --intra-refresh-rows=1 \
    --max-reencode-times=3 --super-i-frame-bits=12000 --super-p-frame-bits=5500 \
    --grayscale-encode=on \
    --send-queue-frames=3 --send-max-latency-ms=250 \
    --debug-roi=on --debug-roi-path="$out/$label/roi/roi_{frame_id}.pgm" \
    >"$out/$label/sender.log" 2>&1 &
  sender_pid="$!"
  sampler_pid=""
  if [[ -n "${RESOURCE_SAMPLER:-}" ]]; then
    "$RESOURCE_SAMPLER" "$sender_pid" "$out/$label/resources.csv" &
    sampler_pid="$!"
  fi
  wait "$sender_pid"
  if [[ -n "$sampler_pid" ]]; then
    wait "$sampler_pid"
  fi
done

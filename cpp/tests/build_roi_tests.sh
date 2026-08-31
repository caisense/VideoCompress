#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
cpp_dir="$(cd "$script_dir/.." && pwd)"
out_dir="${TMPDIR:-/tmp}/roi_h265_tests"
mkdir -p "$out_dir"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/common/config.cc" \
  "$cpp_dir/roi/roi_mapper.cc" \
  "$cpp_dir/roi/roi_temporal.cc" \
  "$cpp_dir/roi/roi_region_merger.cc" \
  "$cpp_dir/tests/test_roi_core.cc" \
  -o "$out_dir/test_roi_core"
"$out_dir/test_roi_core"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/common/config.cc" \
  "$cpp_dir/audio/audio_preprocessor.cc" \
  "$cpp_dir/tests/test_audio_preprocessor.cc" \
  -o "$out_dir/test_audio_preprocessor"
"$out_dir/test_audio_preprocessor"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/audio/codec2_dtx_controller.cc" \
  "$cpp_dir/tests/test_audio_dtx.cc" \
  -o "$out_dir/test_audio_dtx"
"$out_dir/test_audio_dtx"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/audio/bounded_audio_packet_queue.cc" \
  "$cpp_dir/tests/test_audio_packet_queue.cc" \
  -o "$out_dir/test_audio_packet_queue"
"$out_dir/test_audio_packet_queue"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/rate_pacer.cc" \
  "$cpp_dir/transport/packetizer.cc" \
  "$cpp_dir/transport/codec2_rtp_packetizer.cc" \
  "$cpp_dir/tests/test_transport.cc" \
  -o "$out_dir/test_transport"
"$out_dir/test_transport"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/h265_sdp.cc" \
  "$cpp_dir/transport/codec2_sdp.cc" \
  "$cpp_dir/tests/test_h265_sdp.cc" \
  -o "$out_dir/test_h265_sdp"
"$out_dir/test_h265_sdp"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/rate_pacer.cc" \
  "$cpp_dir/transport/udp_sender.cc" \
  "$cpp_dir/tests/test_udp_sender.cc" \
  -o "$out_dir/test_udp_sender"
"$out_dir/test_udp_sender"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/common/config.cc" \
  "$cpp_dir/transport/snapshot_protocol.cc" \
  "$cpp_dir/tests/test_snapshot_protocol.cc" \
  -o "$out_dir/test_snapshot_protocol"
"$out_dir/test_snapshot_protocol"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/snapshot_protocol.cc" \
  "$cpp_dir/transport/snapshot_crop.cc" \
  "$cpp_dir/tests/test_snapshot_crop.cc" \
  -o "$out_dir/test_snapshot_crop"
"$out_dir/test_snapshot_crop"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/rebuild_protocol.cc" \
  "$cpp_dir/tests/test_rebuild_protocol.cc" \
  -o "$out_dir/test_rebuild_protocol"
"$out_dir/test_rebuild_protocol"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/tests/test_rebuild_refresh.cc" \
  -o "$out_dir/test_rebuild_refresh"
"$out_dir/test_rebuild_refresh"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/transport/rate_pacer.cc" \
  "$cpp_dir/transport/packetizer.cc" \
  "$cpp_dir/transport/udp_sender.cc" \
  "$cpp_dir/transport/async_rtp_sender.cc" \
  "$cpp_dir/tests/test_async_rtp_sender.cc" \
  -o "$out_dir/test_async_rtp_sender"
"$out_dir/test_async_rtp_sender"

g++ -std=c++11 -Wall -Wextra -Werror -pthread -I"$cpp_dir" \
  "$cpp_dir/common/statistics.cc" \
  "$cpp_dir/tests/test_statistics.cc" \
  -o "$out_dir/test_statistics"
"$out_dir/test_statistics"

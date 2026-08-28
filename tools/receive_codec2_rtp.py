#!/usr/bin/env python3
"""Receive the project's fixed-frame Codec2 RTP stream.

Codec2 has no broadly implemented RTP payload format in stock FFmpeg.  The
board sender therefore writes a small audio-only SDP and this helper removes
the RTP header, restores short packet gaps with neutral Codec2 frames, and can
either save the raw Codec2 bitstream or play it through c2dec and ffplay.  It
also understands the sender's optional DTX keepalive declaration.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple


IPV4_UDP_HEADER_BYTES = 28
ETHERNET_WIRE_OVERHEAD_BYTES = 38
MINIMUM_ETHERNET_PAYLOAD_BYTES = 46


def wire_bytes_for_udp_payload(udp_payload_bytes: int) -> int:
    """Return Ethernet bytes consumed by one UDP payload/RTP datagram.

    This matches the sender's shared RatePacer: 14-byte Ethernet header,
    IPv4/UDP headers, minimum Ethernet payload padding, FCS, preamble, and
    inter-frame gap.  ``udp_payload_bytes`` is therefore the complete RTP
    datagram length, including its 12-byte fixed RTP header.
    """
    if udp_payload_bytes < 0:
        raise ValueError("UDP payload bytes must not be negative")
    return ETHERNET_WIRE_OVERHEAD_BYTES + max(
        MINIMUM_ETHERNET_PAYLOAD_BYTES, udp_payload_bytes + IPV4_UDP_HEADER_BYTES)


def default_c2dec_path() -> str:
    """Prefer the documented native decoder when it is installed on Windows."""
    native_decoder = Path(r"D:\codec2\bin\c2dec.exe")
    if os.name == "nt" and native_decoder.is_file():
        return str(native_decoder)
    return "c2dec"


def parse_audio_sdp(path: Path) -> Dict[str, int]:
    text = path.read_text(encoding="utf-8")
    media = re.search(r"^m=audio\s+(\d+)\s+RTP/AVP\s+(\d+)\s*$", text, re.MULTILINE)
    if not media:
        raise ValueError("SDP does not contain an audio RTP media line")
    port, payload_type = (int(media.group(1)), int(media.group(2)))
    rtpmap = re.search(
        rf"^a=rtpmap:{payload_type}\s+CODEC2/(\d+)/(\d+)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if not rtpmap:
        raise ValueError("SDP does not describe a CODEC2 RTP payload")
    sample_rate, channels = (int(rtpmap.group(1)), int(rtpmap.group(2)))
    fmtp = re.search(rf"^a=fmtp:{payload_type}\s+(.+)\s*$", text, re.MULTILINE)
    if not fmtp:
        raise ValueError("SDP does not contain fixed Codec2 frame parameters")
    params: Dict[str, int] = {}
    for item in fmtp.group(1).split(";"):
        key, separator, value = item.strip().partition("=")
        if separator:
            try:
                params[key.strip().lower()] = int(value.strip())
            except ValueError as exc:
                raise ValueError(f"invalid Codec2 SDP value: {item}") from exc
    required = ("mode", "frames-per-packet", "bits-per-frame", "bytes-per-frame")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError("SDP missing Codec2 fields: " + ", ".join(missing))
    if sample_rate != 8000 or channels != 1:
        raise ValueError("only 8 kHz mono Codec2 RTP is supported")
    if params["frames-per-packet"] < 1 or params["bytes-per-frame"] < 1:
        raise ValueError("invalid Codec2 packet geometry")
    dtx_enabled = params.get("dtx", 0)
    dtx_keepalive_ms = params.get("dtx-keepalive-ms", 0)
    if dtx_enabled not in (0, 1) or dtx_keepalive_ms < 0:
        raise ValueError("invalid Codec2 DTX SDP parameters")
    return {
        "port": port,
        "payload_type": payload_type,
        "sample_rate": sample_rate,
        "channels": channels,
        "dtx": dtx_enabled,
        "dtx-keepalive-ms": dtx_keepalive_ms,
        **params,
    }


def parse_rtp(datagram: bytes) -> Tuple[int, int, int, bool, bytes]:
    if len(datagram) < 12 or datagram[0] >> 6 != 2:
        raise ValueError("not an RTP version-2 datagram")
    first = datagram[0]
    offset = 12 + (first & 0x0F) * 4
    if offset > len(datagram):
        raise ValueError("truncated RTP CSRC list")
    if first & 0x10:
        if offset + 4 > len(datagram):
            raise ValueError("truncated RTP extension")
        extension_words = struct.unpack_from("!H", datagram, offset + 2)[0]
        offset += 4 + extension_words * 4
    if offset > len(datagram):
        raise ValueError("truncated RTP extension payload")
    end = len(datagram)
    if first & 0x20:
        padding = datagram[-1]
        if padding == 0 or padding > end - offset:
            raise ValueError("invalid RTP padding")
        end -= padding
    payload_type = datagram[1] & 0x7F
    marker = bool(datagram[1] & 0x80)
    sequence, timestamp = struct.unpack_from("!HI", datagram, 2)
    return payload_type, sequence, timestamp, marker, datagram[offset:end]


def decoder_command(c2dec: str, mode: int, wsl_distro: Optional[str]) -> list[str]:
    """Return a decoder command that preserves Windows-side UDP reception.

    With ``wsl_distro`` set, the Codec2 decoder lives in the named WSL
    distribution while this Python process and FFplay remain native Windows
    processes.  The Linux shell expands its own HOME, so this works for any
    WSL user that followed the documented build location.
    """
    if not wsl_distro:
        return [c2dec, str(mode), "-", "-"]
    return [
        "wsl.exe", "-d", wsl_distro, "--", "/bin/bash", "-lc",
        f'exec "$HOME/.local/share/codex-codec2-1.2.0-build/src/c2dec" {mode} - -',
    ]


def ffplay_pcm_command(ffplay: str, sample_rate: int, use_ch_layout: bool) -> list[str]:
    """Build the FFplay command for the decoded 16-bit PCM pipe.

    FFmpeg 9 removed the old ``-ac`` input alias.  Its raw PCM demuxer now
    uses ``-sample_rate`` and ``-ch_layout``.  Older FFplay releases (such as
    the Ubuntu 22.04 package) still require ``-ar`` and ``-ac``.
    """
    command = [
        ffplay, "-hide_banner", "-loglevel", "warning", "-nodisp", "-autoexit",
        "-fflags", "nobuffer", "-flags", "low_delay", "-f", "s16le",
    ]
    if use_ch_layout:
        command += ["-sample_rate", str(sample_rate), "-ch_layout", "mono"]
    else:
        command += ["-ar", str(sample_rate), "-ac", "1"]
    return [*command, "-i", "-"]


def ffplay_uses_ch_layout(ffplay: str) -> bool:
    """Detect the FFplay PCM option spelling without opening an audio device."""
    probe = subprocess.run(
        [ffplay, "-hide_banner", "-h", "demuxer=s16le"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        timeout=5, check=False)
    return "-ch_layout" in (probe.stdout or "")


class Codec2Sink:
    def __init__(self, record_path: Optional[Path], play: bool, mode: int, sample_rate: int,
                 c2dec: str, ffplay: str, wsl_distro: Optional[str] = None) -> None:
        self._record = None
        self._decoder: Optional[subprocess.Popen] = None
        self._player: Optional[subprocess.Popen] = None
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            self._record = record_path.open("wb")
        if play:
            decoder_process_command = decoder_command(c2dec, mode, wsl_distro)
            # c2dec normally buffers stdout when it is piped.  On Linux,
            # stdbuf keeps live speech latency bounded instead of accumulating
            # several kilobytes of decoded PCM first.
            if not wsl_distro and os.name != "nt" and shutil.which("stdbuf"):
                decoder_process_command = ["stdbuf", "-o0", *decoder_process_command]
            try:
                self._decoder = subprocess.Popen(
                    decoder_process_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self._player = subprocess.Popen(
                    ffplay_pcm_command(ffplay, sample_rate, ffplay_uses_ch_layout(ffplay)),
                    stdin=self._decoder.stdout)
                assert self._decoder.stdout is not None
                self._decoder.stdout.close()
                # A bad FFplay option otherwise only becomes visible as a
                # Windows EINVAL while the first RTP packet is written to
                # c2dec.  Fail at startup with the actual child exit status.
                time.sleep(0.10)
                if self._player.poll() is not None:
                    return_code = self._player.returncode
                    self.close()
                    raise RuntimeError(
                        f"ffplay exited during startup (exit code {return_code}); "
                        "check the --ffplay executable and audio device")
            except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
                self.close()
                raise RuntimeError(f"cannot start live audio decoder/player: {exc}") from exc

    def _pipeline_error(self) -> str:
        decoder_code = self._decoder.poll() if self._decoder is not None else None
        player_code = self._player.poll() if self._player is not None else None
        states = []
        if decoder_code is not None:
            states.append(f"c2dec exited ({decoder_code})")
        if player_code is not None:
            states.append(f"ffplay exited ({player_code})")
        return "; ".join(states) if states else "audio decoder/player pipe closed"

    def write(self, encoded: bytes) -> None:
        if self._record is not None:
            self._record.write(encoded)
            self._record.flush()
        if self._decoder is not None and self._decoder.stdin is not None:
            try:
                self._decoder.stdin.write(encoded)
                self._decoder.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(self._pipeline_error()) from exc

    def close(self) -> None:
        if self._record is not None:
            self._record.close()
            self._record = None
        if self._decoder is not None:
            if self._decoder.stdin is not None:
                try:
                    self._decoder.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._decoder.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._decoder.terminate()
                try:
                    self._decoder.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._decoder.kill()
            self._decoder = None
        if self._player is not None:
            self._player.terminate()
            try:
                self._player.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._player.kill()
            self._player = None


def main() -> int:
    parser = argparse.ArgumentParser(description="receive project Codec2/8000 RTP audio")
    parser.add_argument("sdp", type=Path, help="audio-only SDP written by the board sender")
    parser.add_argument("--bind", default="0.0.0.0", help="local address to bind (default: all)")
    parser.add_argument("--record", type=Path, help="save the raw fixed-frame Codec2 stream")
    parser.add_argument("--play", action="store_true", help="decode through c2dec and play through ffplay")
    default_c2dec = default_c2dec_path()
    parser.add_argument("--c2dec", default=default_c2dec,
                        help=f"Codec2 decoder executable (default: {default_c2dec})")
    parser.add_argument("--wsl-c2dec", action="store_true",
                        help="run c2dec from the documented WSL build while UDP stays on Windows")
    parser.add_argument("--wsl-distro", default="Ubuntu-22.04",
                        help="WSL distribution for --wsl-c2dec (default: Ubuntu-22.04)")
    parser.add_argument("--ffplay", default="ffplay", help="FFplay executable (default: ffplay)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds; zero means run until Ctrl+C")
    args = parser.parse_args()
    if not args.record and not args.play:
        parser.error("select at least one of --record or --play")
    try:
        config = parse_audio_sdp(args.sdp)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    expected_payload = config["frames-per-packet"] * config["bytes-per-frame"]
    try:
        sink = Codec2Sink(args.record, args.play, config["mode"], config["sample_rate"],
                          args.c2dec, args.ffplay,
                          args.wsl_distro if args.wsl_c2dec else None)
    except RuntimeError as exc:
        parser.error(str(exc))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((args.bind, config["port"]))
    except OSError as exc:
        sink.close()
        parser.error(f"cannot bind UDP {args.bind}:{config['port']}: {exc}")
    sock.settimeout(0.25)

    received_packets = 0
    received_bytes = 0
    invalid_packets = 0
    lost_packets = 0
    reordered_packets = 0
    last_sequence: Optional[int] = None
    started = time.monotonic()
    last_report = started
    payload_bytes_since_report = 0
    wire_bytes_since_report = 0
    packets_since_report = 0
    talkspurts = 0
    last_packet_time: Optional[float] = None

    def report(now: float) -> None:
        nonlocal last_report, payload_bytes_since_report, wire_bytes_since_report
        nonlocal packets_since_report
        elapsed = max(now - last_report, 1e-9)
        payload_kbps = payload_bytes_since_report * 8.0 / elapsed / 1000.0
        wire_kbps = wire_bytes_since_report * 8.0 / elapsed / 1000.0
        packet_rate = packets_since_report / elapsed
        if payload_bytes_since_report:
            state = "receiving"
        elif (config["dtx"] and last_packet_time is not None and
              now - last_packet_time <= max(2.5, config["dtx-keepalive-ms"] / 500.0)):
            # The sender intentionally emits no Codec2 RTP between speech
            # bursts.  Do not present a healthy DTX interval as a UDP fault.
            state = "dtx-silence"
        else:
            state = "waiting"
        print(
            f"audio codec {payload_kbps:.1f} kbps wire {wire_kbps:.1f} kbps "
            f"RTP {packet_rate:.1f} pps packets {received_packets} "
            f"talkspurts {talkspurts} lost {lost_packets} reorder {reordered_packets} "
            f"invalid {invalid_packets} {state}",
            flush=True)
        last_report = now
        payload_bytes_since_report = 0
        wire_bytes_since_report = 0
        packets_since_report = 0

    dtx_text = (f", DTX on ({config['dtx-keepalive-ms']} ms keepalive)"
                if config["dtx"] else ", DTX off")
    print(
        f"Codec2 RTP listening on UDP {config['port']}: mode {config['mode']}, "
        f"{config['frames-per-packet']} frames/packet, {expected_payload} payload bytes{dtx_text}; "
        "metrics: codec=payload kbps, wire=Ethernet-wire kbps, RTP=packets/s",
        flush=True)
    try:
        while True:
            now = time.monotonic()
            if args.duration > 0 and now - started >= args.duration:
                break
            try:
                datagram, _peer = sock.recvfrom(65535)
            except socket.timeout:
                now = time.monotonic()
                if now - last_report >= 1.0:
                    report(now)
                continue
            try:
                payload_type, sequence, _timestamp, marker, payload = parse_rtp(datagram)
            except ValueError:
                invalid_packets += 1
                continue
            if payload_type != config["payload_type"] or len(payload) != expected_payload:
                invalid_packets += 1
                continue
            if last_sequence is not None:
                delta = (sequence - last_sequence) & 0xFFFF
                if delta == 0:
                    reordered_packets += 1
                    continue
                if delta > 0x8000:
                    reordered_packets += 1
                    continue
                if delta > 1:
                    missing = delta - 1
                    lost_packets += missing
                    # Keep the decoder time base aligned for modest loss runs.
                    # A zero Codec2 frame is a neutral concealment input; it is
                    # preferable to shifting all following fixed-size frames.
                    if missing <= 8:
                        sink.write(bytes(expected_payload * missing))
            sink.write(payload)
            last_sequence = sequence
            if marker:
                talkspurts += 1
            last_packet_time = time.monotonic()
            received_packets += 1
            received_bytes += len(payload)
            payload_bytes_since_report += len(payload)
            wire_bytes_since_report += wire_bytes_for_udp_payload(len(datagram))
            packets_since_report += 1
            now = time.monotonic()
            if now - last_report >= 1.0:
                report(now)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        sink.close()
    print(f"saved/played {received_packets} Codec2 RTP packets ({received_bytes} payload bytes)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

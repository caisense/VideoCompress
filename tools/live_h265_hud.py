#!/usr/bin/env python3
"""Native-PC H.265 RTP viewer with live transport and decoder HUD metrics."""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from rebuild_receiver import (
    REFERENCE_CACHE_TTL_MS,
    REFERENCE_CONTENT_HARD_MAX_AGE_MS,
    REFERENCE_FUTURE_MAX_MS,
    STATE_SYNC_MAX_MS,
    RebuildComposer,
    RebuildReceiver,
    SuperResolver,
)


class RtpStats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: Deque[Tuple[float, int, int]] = collections.deque()
        self.markers: Deque[float] = collections.deque()
        self.p_frames: Deque[float] = collections.deque()
        self.i_frames: Deque[float] = collections.deque()
        self.decoded: Deque[float] = collections.deque()
        self.last_sequence: Optional[int] = None
        self.lost_packets = 0
        self.reordered_packets = 0
        self.duplicate_packets = 0
        self.total_packets = 0
        self.decode_errors = 0
        self.last_idr_time: Optional[float] = None
        self.total_p_frames = 0
        self.total_i_frames = 0
        self._access_unit_has_vcl = False
        self._access_unit_is_i = False
        self.stream_profile: Optional[dict] = None

    @staticmethod
    def _payload_offset(packet: bytes) -> Optional[int]:
        if len(packet) < 12 or packet[0] >> 6 != 2:
            return None
        offset = 12 + 4 * (packet[0] & 0x0F)
        if offset > len(packet):
            return None
        if packet[0] & 0x10:
            if offset + 4 > len(packet):
                return None
            words = int.from_bytes(packet[offset + 2:offset + 4], "big")
            offset += 4 + words * 4
        return offset if offset <= len(packet) else None

    @staticmethod
    def _profile_extension(packet: bytes) -> Optional[dict]:
        if len(packet) < 16 or not packet[0] & 0x10:
            return None
        offset = 12 + 4 * (packet[0] & 0x0F)
        if offset + 12 > len(packet) or packet[offset:offset + 2] != b"RO":
            return None
        words = int.from_bytes(packet[offset + 2:offset + 4], "big")
        payload = packet[offset + 4:offset + 4 + words * 4]
        if len(payload) < 8 or payload[0] != 1:
            return None
        names = {0: "low", 1: "medium", 2: "high", 3: "rebuild"}
        return {
            "name": names.get(payload[1], "unknown"),
            "width": int.from_bytes(payload[2:4], "big"),
            "height": int.from_bytes(payload[4:6], "big"),
            "fps": payload[6],
            "generation": payload[7],
        }

    @staticmethod
    def _wire_bytes(udp_payload_bytes: int) -> int:
        return 38 + max(46, udp_payload_bytes + 28)

    @staticmethod
    def _nal_type(packet: bytes) -> Optional[int]:
        offset = RtpStats._payload_offset(packet)
        if offset is None or offset + 2 > len(packet):
            return None
        nal_type = (packet[offset] >> 1) & 0x3F
        if nal_type == 49 and offset + 3 <= len(packet):  # RFC 7798 FU
            if not packet[offset + 2] & 0x80:
                return None
            nal_type = packet[offset + 2] & 0x3F
        return nal_type

    @staticmethod
    def _is_idr(packet: bytes) -> bool:
        nal_type = RtpStats._nal_type(packet)
        if nal_type is None:
            return False
        return 16 <= nal_type <= 21

    @staticmethod
    def _vcl_type(packet: bytes) -> Optional[int]:
        """Return the HEVC VCL NAL type carried by one RTP packet, if any."""
        nal_type = RtpStats._nal_type(packet)
        if nal_type is None:
            return None
        return nal_type if nal_type <= 31 else None

    def on_packet(self, packet: bytes, now: Optional[float] = None) -> bool:
        if len(packet) < 12 or packet[0] >> 6 != 2:
            return False
        now = time.monotonic() if now is None else now
        sequence = int.from_bytes(packet[2:4], "big")
        profile = self._profile_extension(packet)
        with self.lock:
            profile_changed = False
            if profile is not None:
                if self.stream_profile is None or \
                        profile["generation"] != self.stream_profile["generation"]:
                    profile_changed = self.stream_profile is not None
                    self._access_unit_has_vcl = False
                    self._access_unit_is_i = False
                self.stream_profile = profile
            self.total_packets += 1
            self.samples.append((now, len(packet), self._wire_bytes(len(packet))))
            if packet[1] & 0x80:
                self.markers.append(now)
            if self._is_idr(packet):
                self.last_idr_time = now
            vcl_type = self._vcl_type(packet)
            if vcl_type is not None:
                self._access_unit_has_vcl = True
                self._access_unit_is_i = self._access_unit_is_i or 16 <= vcl_type <= 21
            if packet[1] & 0x80:
                if self._access_unit_has_vcl:
                    if self._access_unit_is_i:
                        self.i_frames.append(now)
                        self.total_i_frames += 1
                    else:
                        self.p_frames.append(now)
                        self.total_p_frames += 1
                self._access_unit_has_vcl = False
                self._access_unit_is_i = False
            if self.last_sequence is not None:
                delta = (sequence - self.last_sequence) & 0xFFFF
                if delta == 0:
                    self.duplicate_packets += 1
                elif delta < 0x8000:
                    self.lost_packets += max(0, delta - 1)
                else:
                    self.reordered_packets += 1
            if self.last_sequence is None or 0 < ((sequence - self.last_sequence) & 0xFFFF) < 0x8000:
                self.last_sequence = sequence
            self._trim(now)
            return profile_changed

    def on_decoded_frame(self) -> None:
        now = time.monotonic()
        with self.lock:
            self.decoded.append(now)
            self._trim(now)

    def on_decode_error(self) -> None:
        with self.lock:
            self.decode_errors += 1

    def _trim(self, now: float) -> None:
        cutoff = now - 1.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.markers and self.markers[0] < cutoff:
            self.markers.popleft()
        while self.p_frames and self.p_frames[0] < cutoff:
            self.p_frames.popleft()
        while self.i_frames and self.i_frames[0] < cutoff:
            self.i_frames.popleft()
        while self.decoded and self.decoded[0] < cutoff:
            self.decoded.popleft()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            self._trim(now)
            packet_count = len(self.samples)
            payload_total = sum(item[1] for item in self.samples)
            return {
                "rx_fps": float(len(self.markers)),
                "p_fps": float(len(self.p_frames)),
                "i_fps": float(len(self.i_frames)),
                "p_frames": self.total_p_frames,
                "i_frames": self.total_i_frames,
                "decode_fps": float(len(self.decoded)),
                "rtp_kbps": payload_total * 8.0 / 1000.0,
                "wire_kbps": sum(item[2] for item in self.samples) * 8.0 / 1000.0,
                "pps": float(packet_count),
                "packet_last_bytes": self.samples[-1][1] if self.samples else 0,
                "packet_avg_bytes": payload_total / packet_count if packet_count else 0.0,
                "packet_max_bytes": max((item[1] for item in self.samples), default=0),
                "lost": self.lost_packets,
                "reordered": self.reordered_packets,
                "duplicates": self.duplicate_packets,
                "packets": self.total_packets,
                "decode_errors": self.decode_errors,
                "idr_age": None if self.last_idr_time is None else now - self.last_idr_time,
                "profile": None if self.stream_profile is None else dict(self.stream_profile),
            }


class ProfileSwitchGate:
    """Buffer the first complete IDR of each RTP profile generation."""

    def __init__(self) -> None:
        self.active_generation: Optional[int] = None
        self.pending_generation: Optional[int] = None
        self.pending_packets = []
        self.pending_has_idr = False

    def feed(self, packet: bytes, profile: Optional[dict]) -> Tuple[Optional[int], list]:
        # Generation -1 keeps compatibility with senders that do not carry
        # the project-specific profile extension.
        generation = -1 if profile is None else profile["generation"]
        if self.active_generation == generation:
            return None, [packet]
        if self.pending_generation != generation:
            self.pending_generation = generation
            self.pending_packets = []
            self.pending_has_idr = False
        self.pending_packets.append(packet)
        self.pending_has_idr = self.pending_has_idr or RtpStats._is_idr(packet)
        if not packet[1] & 0x80:
            return None, []
        if not self.pending_has_idr:
            self.pending_packets = []
            return None, []
        buffered = self.pending_packets
        self.active_generation = generation
        self.pending_generation = None
        self.pending_packets = []
        self.pending_has_idr = False
        return generation, buffered


class HevcRtpDepacketizer:
    """Convert the RFC 7798 packet forms emitted by our sender to Annex-B."""

    START_CODE = b"\x00\x00\x00\x01"

    def __init__(self) -> None:
        self.fu_expected_sequence: Optional[int] = None

    def reset(self) -> None:
        self.fu_expected_sequence = None

    def feed(self, packet: bytes) -> bytes:
        offset = RtpStats._payload_offset(packet)
        if offset is None or offset + 2 > len(packet):
            return b""
        end = len(packet)
        if packet[0] & 0x20:  # RTP padding
            padding = packet[-1]
            if padding == 0 or padding > end - offset:
                return b""
            end -= padding
        payload = packet[offset:end]
        if len(payload) < 2:
            return b""
        nal_type = (payload[0] >> 1) & 0x3F
        if nal_type == 48:  # Aggregation packet
            output = bytearray()
            cursor = 2
            while cursor + 2 <= len(payload):
                size = int.from_bytes(payload[cursor:cursor + 2], "big")
                cursor += 2
                if size == 0 or cursor + size > len(payload):
                    return b""
                output.extend(self.START_CODE)
                output.extend(payload[cursor:cursor + size])
                cursor += size
            return bytes(output) if cursor == len(payload) else b""
        if nal_type != 49:  # A complete NAL unit
            self.fu_expected_sequence = None
            return self.START_CODE + payload
        if len(payload) < 3:
            self.fu_expected_sequence = None
            return b""
        sequence = int.from_bytes(packet[2:4], "big")
        fu_header = payload[2]
        start = bool(fu_header & 0x80)
        end_fragment = bool(fu_header & 0x40)
        if start:
            first_byte = (payload[0] & 0x81) | ((fu_header & 0x3F) << 1)
            self.fu_expected_sequence = (sequence + 1) & 0xFFFF
            if end_fragment:
                self.fu_expected_sequence = None
            return self.START_CODE + bytes((first_byte, payload[1])) + payload[3:]
        if self.fu_expected_sequence != sequence:
            self.fu_expected_sequence = None
            return b""
        self.fu_expected_sequence = None if end_fragment else (sequence + 1) & 0xFFFF
        return payload[3:]


def parse_sdp(path: Path) -> Tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^m=video\s+(\d+)\s+RTP/AVP\s+96\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"SDP has no H.265 RTP/AVP 96 video port: {path}")
    return text, int(match.group(1))


class LatestFrame:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.updated = 0.0
        self.sequence = 0
        self.rtp_timestamp: Optional[int] = None

    def put(self, frame: np.ndarray, rtp_timestamp: Optional[int] = None) -> None:
        with self.lock:
            self.frame = frame
            self.updated = time.monotonic()
            self.sequence += 1
            self.rtp_timestamp = rtp_timestamp

    def get(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.updated)

    def get_versioned(self) -> Tuple[Optional[np.ndarray], float, int, Optional[int]]:
        with self.lock:
            return (None if self.frame is None else self.frame.copy(),
                    self.updated, self.sequence, self.rtp_timestamp)

    def clear(self) -> None:
        with self.lock:
            self.frame = None
            self.updated = 0.0
            self.sequence += 1
            self.rtp_timestamp = None


class DecodedPtsQueue:
    """Bounded FIFO handoff from complete RTP AUs to decoded BMP frames.

    Consuming the newest timestamp associated picture *N* with *N+1* whenever
    FFmpeg had already queued the next AU.  There are no B frames here, so
    decoded output normally follows submission order and consumes the oldest
    pending timestamp.  The bound prevents a decoder stall from creating an
    unbounded queue; the receiver's sliding PTS calibration handles recovery.
    """

    def __init__(self, max_pending: int = 16) -> None:
        if max_pending < 2:
            raise ValueError("decoded PTS queue must hold at least two entries")
        self.lock = threading.Lock()
        self.max_pending = max_pending
        self.items: Deque[int] = collections.deque()
        self.dropped = 0

    def push_rtp_timestamp(self, timestamp: int) -> None:
        with self.lock:
            while len(self.items) >= self.max_pending:
                self.items.popleft()
                self.dropped += 1
            self.items.append(timestamp & 0xFFFFFFFF)

    def pop(self) -> Optional[int]:
        with self.lock:
            if not self.items:
                return None
            return self.items.popleft()

    def snapshot(self) -> dict:
        with self.lock:
            return {"pending": len(self.items), "dropped": self.dropped}

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


class PresentationStats:
    """Honest output provenance for a 12 Hz display fed by a slower decoder."""

    def __init__(self) -> None:
        self.samples: Deque[float] = collections.deque()
        self.total = 0
        self.decoded = 0
        self.held = 0
        self.last_source_sequence: Optional[int] = None
        self.last_provenance = "WAIT"
        self.generation: Optional[int] = None

    def present(self, source_sequence: int, spatial: str, generation: int,
                now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        if self.generation != generation:
            self.total = self.decoded = self.held = 0
            self.last_source_sequence = None
            self.generation = generation
        is_new = source_sequence != self.last_source_sequence
        temporal = "DECODED" if is_new else "HOLD"
        self.total += 1
        if is_new:
            self.decoded += 1
            self.last_source_sequence = source_sequence
        else:
            self.held += 1
        self.samples.append(now)
        while self.samples and self.samples[0] < now - 1.0:
            self.samples.popleft()
        self.last_provenance = f"{temporal}+{spatial}"
        return self.last_provenance

    def snapshot(self) -> dict:
        now = time.monotonic()
        while self.samples and self.samples[0] < now - 1.0:
            self.samples.popleft()
        ratio = 0.0 if self.total == 0 else self.held * 100.0 / self.total
        return {
            "fps": float(len(self.samples)),
            "total": self.total,
            "decoded": self.decoded,
            "held": self.held,
            "held_percent": ratio,
            "provenance": self.last_provenance,
        }


def read_exact(stream, size: int) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def read_bmp_frame(stream) -> Optional[np.ndarray]:
    """Read one self-delimiting BMP image from FFmpeg's image2pipe output.

    Rawvideo has no frame boundary, so reading it with an incorrect configured
    width or height silently joins fragments of different frames into tiled
    images.  BMP carries its own byte length; decoding it here makes the HUD
    safe across sender resolution changes.
    """
    header = read_exact(stream, 14)
    if header is None or header[:2] != b"BM":
        return None
    frame_size = int.from_bytes(header[2:6], "little")
    if frame_size < 54 or frame_size > 64 * 1024 * 1024:
        return None
    payload = read_exact(stream, frame_size - len(header))
    if payload is None:
        return None
    return cv2.imdecode(np.frombuffer(header + payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def postprocess_frame(frame: np.ndarray, rotation: str, denoise: bool) -> np.ndarray:
    """Apply display-only orientation and edge-preserving smoothing."""
    if rotation == "ccw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "cw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if denoise:
        # Bilateral filtering reduces block/ringing noise while retaining ROI
        # edges; this is display-only and does not alter the H.265 stream.
        frame = cv2.bilateralFilter(frame, 5, 24.0, 24.0)
    return frame


def display_dimensions(width: int, height: int, rotation: str, scale: int) -> Tuple[int, int]:
    """Return the OpenCV window size after display rotation."""
    if rotation in ("cw90", "ccw90"):
        width, height = height, width
    return width * scale, height * scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdp", type=Path)
    parser.add_argument("--width", type=int, default=320,
                        help="startup canvas width before the first decoded frame")
    parser.add_argument("--height", type=int, default=180,
                        help="startup canvas height before the first decoded frame")
    parser.add_argument("--scale", type=int, default=0,
                        help="window scale; 0 selects 1x for rebuild and 3x otherwise")
    parser.add_argument("--rotate", choices=("none", "cw90", "ccw90", "180"), default="ccw90",
                        help="display rotation; default is counter-clockwise 90 degrees")
    parser.add_argument("--denoise", choices=("on", "off"), default="on",
                        help="display-only edge-preserving smoothing")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--rebuild-port", type=int, default=0,
                        help="RB/1 UDP port; 0 uses video RTP port + 5")
    parser.add_argument("--rebuild-width", type=int, default=640)
    parser.add_argument("--rebuild-height", type=int, default=360)
    parser.add_argument("--rebuild-fps", type=float, default=12.0)
    parser.add_argument("--rebuild-reference-cache-ttl-ms", type=int,
                        default=REFERENCE_CACHE_TTL_MS)
    parser.add_argument("--rebuild-reference-hard-age-ms", type=int,
                        default=REFERENCE_CONTENT_HARD_MAX_AGE_MS)
    parser.add_argument("--rebuild-reference-future-ms", type=int,
                        default=REFERENCE_FUTURE_MAX_MS)
    parser.add_argument("--rebuild-max-sync-ms", type=int, default=STATE_SYNC_MAX_MS,
                        help="suppress ROI paint when RB/1 state differs from video PTS")
    parser.add_argument("--rebuild-debug-timing", action="store_true",
                        help="emit one structured timing record per composed source frame")
    parser.add_argument("--esrgan", choices=("auto", "off"), default="auto",
                        help="small-ROI Real-ESRGAN; auto falls back to Lanczos4")
    parser.add_argument("--esrgan-model", default=None)
    parser.add_argument("--esrgan-threads", type=int, default=2)
    parser.add_argument("--esrgan-input-side", type=int, default=96,
                        help="fixed square Real-ESRGAN input used for all rebuild crops")
    parser.add_argument("--rebuild-boxes", choices=("on", "off"), default="off",
                        help="diagnostic target rectangles; off avoids display overlays")
    parser.add_argument("--headless", action="store_true", help="print HUD metrics without opening a window")
    parser.add_argument("--duration", type=float, default=0.0, help="optional run duration in seconds")
    args = parser.parse_args()
    if (args.width <= 0 or args.height <= 0 or args.scale < 0 or
            args.rebuild_width <= 0 or args.rebuild_height <= 0 or
            args.rebuild_fps <= 0 or args.rebuild_reference_cache_ttl_ms <= 0 or
            args.rebuild_reference_hard_age_ms <= 0 or
            args.rebuild_reference_future_ms <= 0 or args.rebuild_max_sync_ms <= 0 or
            args.esrgan_threads < 0 or args.esrgan_input_side < 16):
        parser.error("dimensions/rates/ages must be positive and scale/threads non-negative")

    _, listen_port = parse_sdp(args.sdp)
    stats = RtpStats()
    switch_gate = ProfileSwitchGate()
    depacketizer = HevcRtpDepacketizer()
    latest = LatestFrame()
    decoded_pts = DecodedPtsQueue()
    stopping = threading.Event()
    rebuild_port = args.rebuild_port or listen_port + 5
    if rebuild_port < 1 or rebuild_port > 65535 or rebuild_port == listen_port:
        parser.error("rebuild port must be a valid UDP port distinct from video RTP")
    rebuild_receiver = RebuildReceiver(
        rebuild_port, max_sync_ms=args.rebuild_max_sync_ms,
        reference_cache_ttl_ms=args.rebuild_reference_cache_ttl_ms,
        reference_future_max_ms=args.rebuild_reference_future_ms)
    resolver = SuperResolver(
        enabled=False, model_path=args.esrgan_model,
        cpu_threads=args.esrgan_threads, input_side=args.esrgan_input_side)
    rebuild_composer = RebuildComposer(
        output_size=(args.rebuild_width, args.rebuild_height), resolver=resolver,
        reference_cache_ttl_ms=args.rebuild_reference_cache_ttl_ms,
        reference_content_hard_max_age_ms=args.rebuild_reference_hard_age_ms,
        reference_future_max_ms=args.rebuild_reference_future_ms,
        draw_targets=args.rebuild_boxes == "on")
    presentation = PresentationStats()
    try:
        rebuild_receiver.start(stopping)
    except OSError as error:
        parser.error(f"cannot bind RB/1 UDP {rebuild_port}: {error}")
    print(f"RB/1 rebuild companion listening on UDP {rebuild_port}", flush=True)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(("0.0.0.0", listen_port))
    receiver.settimeout(0.2)
    command = [
        args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "1024",
        "-f", "hevc",
        "-i", "pipe:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
        "-f", "image2pipe", "-vcodec", "bmp", "pipe:1",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    decoder_lock = threading.Lock()
    decoder_state = {"process": None, "generation": None, "token": 0}
    decoder_threads = []

    def decode_loop(process: subprocess.Popen, token: int) -> None:
        assert process.stdout is not None
        while not stopping.is_set():
            frame = read_bmp_frame(process.stdout)
            if frame is None:
                break
            with decoder_lock:
                current = token == decoder_state["token"]
            if current:
                latest.put(frame, decoded_pts.pop())
                stats.on_decoded_frame()

    def stderr_loop(process: subprocess.Popen) -> None:
        assert process.stderr is not None
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode(errors="replace").rstrip()
            if not line:
                continue
            print(f"[ffmpeg] {line}")
            if "Error" in line or "Could not find ref" in line or "Invalid" in line:
                stats.on_decode_error()

    def launch_decoder_locked() -> None:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
            creationflags=creationflags,
        )
        decoder_state["token"] += 1
        token = decoder_state["token"]
        decoder_state["process"] = process
        decode_thread = threading.Thread(
            target=decode_loop, args=(process, token), name=f"hevc-decode-{token}", daemon=True)
        error_thread = threading.Thread(
            target=stderr_loop, args=(process,), name=f"ffmpeg-stderr-{token}", daemon=True)
        decoder_threads.extend((decode_thread, error_thread))
        decode_thread.start()
        error_thread.start()

    def stop_decoder_locked() -> None:
        process = decoder_state["process"]
        decoder_state["process"] = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def ensure_decoder(generation: Optional[int]) -> None:
        with decoder_lock:
            process = decoder_state["process"]
            changed = decoder_state["generation"] is not None and \
                generation is not None and generation != decoder_state["generation"]
            dead = process is None or process.poll() is not None
            if changed or dead:
                stop_decoder_locked()
                latest.clear()
                decoded_pts.clear()
                decoder_state["generation"] = generation
                launch_decoder_locked()
            elif decoder_state["generation"] is None and generation is not None:
                decoder_state["generation"] = generation

    def write_decoder(data: bytes) -> None:
        if not data:
            return
        with decoder_lock:
            process = decoder_state["process"]
            if process is None or process.poll() is not None or process.stdin is None:
                return
            try:
                process.stdin.write(data)
            except (BrokenPipeError, OSError):
                pass

    def proxy_loop() -> None:
        while not stopping.is_set():
            try:
                packet, _ = receiver.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            stats.on_packet(packet)
            profile = stats.snapshot()["profile"]
            restart_generation, ready_packets = switch_gate.feed(packet, profile)
            if restart_generation is not None:
                nal_types = sorted({
                    nal_type for nal_type in (RtpStats._nal_type(item) for item in ready_packets)
                    if nal_type is not None
                })
                print(f"Profile generation {restart_generation}: replay IDR RTP NAL types {nal_types}")
                ensure_decoder(restart_generation)
                depacketizer.reset()
            for ready_packet in ready_packets:
                annex_b = depacketizer.feed(ready_packet)
                if annex_b and ready_packet[1] & 0x80:
                    decoded_pts.push_rtp_timestamp(
                        int.from_bytes(ready_packet[4:8], "big"))
                write_decoder(annex_b)

    threads = [
        threading.Thread(target=proxy_loop, name="rtp-proxy", daemon=True),
    ]
    for thread in threads:
        thread.start()

    window = "RK3588 H.265 live HUD (q/Esc to quit)"
    startup_scale = args.scale or 3
    window_size = display_dimensions(args.width, args.height, args.rotate, startup_scale)
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *window_size)
    started_at = time.monotonic()
    last_report = 0.0
    next_rebuild_present = started_at
    current_canvas: Optional[np.ndarray] = None
    current_updated = 0.0
    current_source_sequence: Optional[int] = None
    current_spatial = "BASE-LANCZOS"
    current_mode = "normal"
    try:
        while not stopping.is_set():
            now = time.monotonic()
            sync_ms: Optional[int] = None
            if args.duration > 0 and now - started_at >= args.duration:
                break
            frame, updated, source_sequence, source_rtp_timestamp = latest.get_versioned()
            values = stats.snapshot()
            profile = values["profile"]
            is_rebuild = profile is not None and profile["name"] == "rebuild"
            if current_mode != ("rebuild" if is_rebuild else "normal"):
                current_canvas = None
                current_source_sequence = None
                current_spatial = "BASE-LANCZOS"
                next_rebuild_present = now
                current_mode = "rebuild" if is_rebuild else "normal"
                if is_rebuild and args.esrgan != "off":
                    resolver.enable_async()

            if is_rebuild:
                profile_generation = int(profile["generation"])
                timing_values = rebuild_receiver.snapshot()
                timing_state, timing_generation, _, _ = rebuild_receiver.scene()
                active_track_ids = set()
                advertised_generations = {}
                if timing_state is not None and timing_generation == profile_generation:
                    active_track_ids = {
                        target.track_id for target in timing_state.targets
                    }
                    advertised_generations = {
                        target.track_id: target.reference_generation
                        for target in timing_state.targets
                    }
                rebuild_composer.update_timing(
                    source_rtp_timestamp, timing_values["pts_bias_ms"],
                    profile_generation, active_track_ids,
                    advertised_generations)
                # A complete RB/1 reference reaches this queue on the socket
                # thread.  Prefetch it immediately; it can only affect a
                # later source frame because the current composition is cached.
                for reference in rebuild_receiver.take_completed_references():
                    rebuild_composer.prefetch(reference)
                rebuild_composer.prefetch_pending()

            if is_rebuild and now >= next_rebuild_present:
                interval = 1.0 / args.rebuild_fps
                skipped = max(0, int((now - next_rebuild_present) / interval))
                next_rebuild_present += (skipped + 1) * interval
                if frame is not None:
                    profile_generation = int(profile["generation"])
                    if source_sequence != current_source_sequence:
                        base = frame
                        if args.denoise == "on":
                            base = cv2.bilateralFilter(base, 5, 24.0, 24.0)
                        state, state_generation, references, _, sync_ms = \
                            rebuild_receiver.scene_synced(source_rtp_timestamp)
                        if state_generation != profile_generation:
                            state, references = None, {}
                        receiver_sync = rebuild_receiver.snapshot()
                        rebuilt, spatial = rebuild_composer.render(
                            base, state, profile_generation, references, now,
                            video_rtp_timestamp=source_rtp_timestamp,
                            pts_bias_ms=receiver_sync["pts_bias_ms"],
                            source_sequence=source_sequence)
                        current_canvas = postprocess_frame(rebuilt, args.rotate, False)
                        current_updated = updated
                        current_source_sequence = source_sequence
                        current_spatial = spatial
                        if args.rebuild_debug_timing:
                            debug_values = rebuild_composer.snapshot()
                            bias = receiver_sync["pts_bias_ms"]
                            video_pts_ms = (None if source_rtp_timestamp is None else
                                             int(((source_rtp_timestamp + bias * 90) &
                                                  0xffffffff) / 90))
                            match_text = ("none" if debug_values["match_score"] is None else
                                          f"{debug_values['match_score']:.2f}")
                            print(
                                f"SEQ={source_sequence} "
                                f"VPTS={'none' if video_pts_ms is None else video_pts_ms} "
                                f"SPTS={'none' if receiver_sync['state_pts_ms'] is None else receiver_sync['state_pts_ms']} "
                                f"RPTS={'none' if debug_values['reference_pts_ms'] is None else debug_values['reference_pts_ms']} "
                                f"AGE={'none' if debug_values['reference_content_age_ms'] is None else debug_values['reference_content_age_ms']} "
                                f"GEN={profile_generation} "
                                f"RGEN={'none' if debug_values['reference_generation'] is None else debug_values['reference_generation']} "
                                f"REFREADY={debug_values['reference_ready']} "
                                f"USED={debug_values['refs_used']} "
                                f"MODE={spatial} "
                                f"DROP={debug_values['last_drop_reason']} "
                                f"DROP_REASON={debug_values['last_drop_reason']} "
                                f"MATCH={match_text} "
                                f"DXR={debug_values['geom_dx_ratio']:.2f} "
                                f"DYR={debug_values['geom_dy_ratio']:.2f} "
                                f"AR={debug_values['geom_area_ratio']:.2f} "
                                f"GEOM_DX_RATIO={debug_values['geom_dx_ratio']:.2f} "
                                f"GEOM_DY_RATIO={debug_values['geom_dy_ratio']:.2f} "
                                f"AREA_RATIO={debug_values['geom_area_ratio']:.2f} "
                                f"SR={'HIT' if spatial == 'ROI-ESRGAN' else 'MISS'} "
                                f"SRSTATE={debug_values['sr_state']} "
                                f"SR_STATE={debug_values['sr_state']} "
                                f"SRLOOKUP={debug_values['last_sr_lookup']} "
                                f"SR_LOOKUP={debug_values['last_sr_lookup']} "
                                f"CACHE={debug_values['last_sr_lookup']}",
                                flush=True)
                    presentation.present(source_sequence, current_spatial,
                                         profile_generation, now)
            elif not is_rebuild:
                if frame is not None and current_source_sequence != source_sequence:
                    current_canvas = postprocess_frame(
                        frame, args.rotate, args.denoise == "on")
                    current_updated = updated
                    current_source_sequence = source_sequence

            rebuild_values = rebuild_receiver.snapshot()
            composer_values = rebuild_composer.snapshot()
            presentation_values = presentation.snapshot()
            if args.headless:
                # A headless report can fall between two rebuild presentation
                # ticks (12 fps).  Refresh the synchronized PTS here so the
                # log never hides a live value as ``none`` merely because the
                # reporting wall-clock tick did not render a frame.
                if is_rebuild and sync_ms is None and source_rtp_timestamp is not None:
                    _, _, _, _, sync_ms = rebuild_receiver.scene_synced(
                        source_rtp_timestamp)
                    # scene_synced may learn a new median bias; report the
                    # bias and sync value from the same observation.
                    rebuild_values = rebuild_receiver.snapshot()
                if now - last_report >= 1.0:
                    profile_text = "unknown" if profile is None else (
                        f"{profile['name']}:{profile['width']}x{profile['height']}@"
                        f"{profile['fps']}:g{profile['generation']}"
                    )
                    rebuild_text = ""
                    if is_rebuild:
                        media_wire = values["wire_kbps"] + rebuild_values["wire_kbps"]
                        rebuild_text = (
                            f" output={args.rebuild_width}x{args.rebuild_height}@"
                            f"{args.rebuild_fps:g} frame={presentation_values['provenance']}"
                            f" temporal_hold={presentation_values['held_percent']:.1f}%"
                            f" rb_wire_kbps={rebuild_values['wire_kbps']:.1f}"
                            f" video_rb_wire_kbps={media_wire:.1f} link_cap_kbps=100"
                            f" video_pps={values['pps']:.1f}"
                            f" video_pkt_avg_B={values['packet_avg_bytes']:.0f}"
                            f" rb_pps={rebuild_values['pps']:.1f}"
                            f" rb_pkt_avg_B={rebuild_values['packet_avg_bytes']:.0f}"
                            f" refs={composer_values['refs_used']}/"
                            f"{rebuild_values['active_references']}"
                            f" refready={composer_values['reference_ready']}"
                            f" agedrop={composer_values['age_drops']}"
                            f" futuredrop={composer_values['future_drops']}"
                            f" norefdrop={composer_values['no_reference_drops']}"
                            f" statedrop={composer_values['state_drops']}"
                            f" gendrop={composer_values['generation_drops']}"
                            f" geomdrop={composer_values['registration_drops']}"
                            f" geominvalid={composer_values['geom_invalid_drops']}"
                            f" geomscale={composer_values['geom_scale_drops']}"
                            f" scaledrop={composer_values['geom_scale_drops']}"
                            f" geomoutside={composer_values['geom_outside_drops']}"
                            f" matchdrop={composer_values['content_drops']}"
                            f" rebuild_area={composer_values['rebuild_percent']:.1f}%"
                            f" chroma={composer_values['chroma_mode']}"
                            f" pts_sync_ms={'none' if sync_ms is None else sync_ms}"
                            f" pts_bias_ms={rebuild_values['pts_bias_ms']}"
                            f" ref_pts_age_ms={composer_values['reference_content_age_ms']}"
                            f" gen={profile['generation']}/"
                            f"{composer_values['reference_generation']}"
                            f" sr_run={int(composer_values['sr_running'])}"
                            f" sr_q={composer_values['sr_queue']}"
                            f" sr_done={composer_values['sr_done']}/"
                            f"{composer_values['sr_jobs']}"
                            f" sr_x={composer_values['sr_x']}"
                            f" sr_stale={composer_values['sr_stale']}"
                            f" sr_future_wait={composer_values['sr_future_waits']}"
                            f" sr_invalid={composer_values['sr_invalid_drops']}"
                            f" sr_ms={composer_values['sr_last_ms']:.0f}/"
                            f"{composer_values['sr_p50_ms']:.0f}/"
                            f"{composer_values['sr_p95_ms']:.0f}"
                            f" sr_first={composer_values['new_reference_first_frame_sr_hits']}/"
                            f"{composer_values['new_reference_first_frames']}"
                            f" sync_drops={rebuild_values['sync_drops']}"
                            f" fec_recovered={rebuild_values['parity_recovered']}"
                        )
                    print(
                        f"profile={profile_text} "
                        f"rx_fps={values['rx_fps']:.1f} decode_fps={values['decode_fps']:.1f} "
                        f"p_fps={values['p_fps']:.1f} i_fps={values['i_fps']:.1f} "
                        f"rtp_kbps={values['rtp_kbps']:.1f} wire_kbps={values['wire_kbps']:.1f} "
                        f"lost={values['lost']} reorder={values['reordered']} "
                        f"decode_errors={values['decode_errors']}" + rebuild_text,
                        flush=True,
                    )
                    last_report = now
                time.sleep(0.01)
                continue
            if current_canvas is None:
                blank_width = args.rebuild_width if is_rebuild else args.width
                blank_height = args.rebuild_height if is_rebuild else args.height
                canvas = postprocess_frame(
                    np.zeros((blank_height, blank_width, 3), dtype=np.uint8),
                    args.rotate, False)
            else:
                canvas = current_canvas.copy()
            effective_scale = args.scale or (1 if is_rebuild else 3)
            display_size = (canvas.shape[1] * effective_scale,
                            canvas.shape[0] * effective_scale)
            if not args.headless and display_size != window_size:
                cv2.resizeWindow(window, *display_size)
                window_size = display_size
            if effective_scale != 1:
                canvas = cv2.resize(canvas, display_size, interpolation=cv2.INTER_LANCZOS4)
            if is_rebuild:
                state, _, _, _, sync_ms = rebuild_receiver.scene_synced(
                    source_rtp_timestamp)
                output_width = state.output_width if state is not None else args.rebuild_width
                output_height = state.output_height if state is not None else args.rebuild_height
                output_fps = state.output_fps if state is not None else args.rebuild_fps
                media_wire = values["wire_kbps"] + rebuild_values["wire_kbps"]
                ref_age = composer_values["reference_age"]
                ref_age_text = "none" if ref_age is None else f"{ref_age * 1000.0:.0f}ms"
                ref_pts_age = composer_values["reference_content_age_ms"]
                ref_pts_age_text = "none" if ref_pts_age is None else f"{ref_pts_age:+d}ms"
                sync_text = "none" if sync_ms is None else f"{sync_ms:+d}ms"
                if sync_ms is not None and abs(sync_ms) > args.rebuild_max_sync_ms:
                    sync_text += " DROP"
                lines = [
                    f"RX {values['rx_fps']:.1f} DEC {values['decode_fps']:.1f} "
                    f"OUT {presentation_values['fps']:.1f} fps",
                    f"H265 {values['rtp_kbps']:.1f} RB {rebuild_values['rtp_kbps']:.1f} kbps",
                    f"V+RB WIRE {media_wire:.1f} kbps",
                    "LINK CAP 100 kbps incl audio",
                    f"P/I {values['p_fps']:.1f}/{values['i_fps']:.1f} "
                    f"total {values['p_frames']}/{values['i_frames']}",
                    f"PKT {values['packets']} LOSS {values['lost']} "
                    f"REO {values['reordered']} ERR {values['decode_errors']}",
                    f"PPS V {values['pps']:.1f} RB {rebuild_values['pps']:.1f} "
                    f"S/D/F {rebuild_values['state_pps']:.0f}/"
                    f"{rebuild_values['data_pps']:.0f}/{rebuild_values['parity_pps']:.0f}",
                    f"LEN L/A/M V {values['packet_last_bytes']}/"
                    f"{values['packet_avg_bytes']:.0f}/{values['packet_max_bytes']} B",
                    f"LEN L/A/M RB {rebuild_values['packet_last_bytes']}/"
                    f"{rebuild_values['packet_avg_bytes']:.0f}/"
                    f"{rebuild_values['packet_max_bytes']} B",
                    f"REBUILD Gen {profile['generation']}",
                    f"SRC {profile['width']}x{profile['height']} @{profile['fps']} fps",
                    f"OUT {output_width}x{output_height} @{output_fps} fps",
                    f"FRAME {presentation_values['provenance']}",
                    f"TEMP HOLD {presentation_values['held_percent']:.1f}% "
                    f"({profile['fps']}->{output_fps})",
                            f"REFREADY {composer_values['reference_ready']} USED "
                            f"{composer_values['refs_used']}/"
                            f"{rebuild_values['active_references']} AGE {ref_age_text} "
                            f"PTSAGE {ref_pts_age_text}",
                            f"ROI AREA {composer_values['rebuild_percent']:.1f}%",
                            f"AGEDROP {composer_values['age_drops']} "
                            f"FUTUREDROP {composer_values['future_drops']} "
                            f"NOREFDROP {composer_values['no_reference_drops']}",
                            f"STATEDROP {composer_values['state_drops']} "
                            f"GENDROP {composer_values['generation_drops']} "
                            f"GEOMDROP {composer_values['registration_drops']}",
                            f"SCALEDROP {composer_values['geom_scale_drops']} "
                            f"MATCHDROP {composer_values['content_drops']} "
                            f"DROP_REASON {composer_values['last_drop_reason']}",
                            f"PTS SYNC {sync_text} BIAS {rebuild_values['pts_bias_ms']:+d}ms "
                            f"DROP#{rebuild_values['sync_drops']} "
                            f"TIMINGDROP {composer_values['timing_drops']}",
                    f"CHROMA {composer_values['chroma_mode']} "
                    f"BOX {args.rebuild_boxes.upper()}",
                    f"FEC {rebuild_values['parity_recovered']} "
                    f"INC {rebuild_values['incomplete']} BAD "
                    f"{rebuild_values['invalid'] + rebuild_values['inconsistent']}",
                    f"SR {composer_values['sr_model'][:24]} "
                    f"RUN {int(composer_values['sr_running'])} "
                    f"Q {composer_values['sr_queue']} "
                    f"DONE {composer_values['sr_done']}/{composer_values['sr_jobs']} "
                    f"X {composer_values['sr_x']} "
                    f"STALE {composer_values['sr_stale']} "
                    f"FIRST {composer_values['new_reference_first_frame_sr_hits']}/"
                    f"{composer_values['new_reference_first_frames']} "
                    f"STATE {composer_values['sr_state']}",
                    f"SRWAIT FUT {composer_values['sr_future_waits']} "
                    f"INV {composer_values['sr_invalid_drops']} "
                    f"LAT {composer_values['sr_last_ms']:.0f}/"
                    f"{composer_values['sr_p50_ms']:.0f}/"
                    f"{composer_values['sr_p95_ms']:.0f} ms",
                ]
            else:
                lines = [
                    f"RX {values['rx_fps']:.1f} fps   Decode {values['decode_fps']:.1f} fps   "
                    f"RTP {values['rtp_kbps']:.1f} kbps   Wire {values['wire_kbps']:.1f} kbps",
                    f"P {values['p_fps']:.1f} fps   I {values['i_fps']:.1f} fps   "
                    f"P/I total {values['p_frames']}/{values['i_frames']}",
                    f"Packets {values['packets']}   Lost {values['lost']}   "
                    f"Reorder {values['reordered']}   Decode errors {values['decode_errors']}",
                    f"UDP {values['pps']:.1f} pps   packet L/A/M "
                    f"{values['packet_last_bytes']}/{values['packet_avg_bytes']:.0f}/"
                    f"{values['packet_max_bytes']} B",
                ]
                if profile is not None:
                    lines.append(
                        f"Profile {profile['name']}   {profile['width']}x{profile['height']} @ "
                        f"{profile['fps']} fps   Gen {profile['generation']}"
                    )
            if frame is None:
                lines.append("WAITING FOR COMPLETE IDR - start/restart the board sender now")
            else:
                age_ms = max(0.0, (time.monotonic() - current_updated) * 1000.0)
                idr = "none" if values["idr_age"] is None else f"{values['idr_age']:.1f}s ago"
                lines.append(f"Source age {age_ms:.0f} ms   Last IDR {idr}")
            font_scale = 0.40 if is_rebuild else 0.58
            line_height = 21 if is_rebuild else 26
            for index, line in enumerate(lines):
                cv2.putText(canvas, line, (8 if is_rebuild else 10,
                            20 + index * line_height if is_rebuild else 24 + index * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (80, 255, 80), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1 if is_rebuild else 20) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stopping.set()
        receiver.close()
        rebuild_receiver.close()
        rebuild_composer.close()
        with decoder_lock:
            stop_decoder_locked()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

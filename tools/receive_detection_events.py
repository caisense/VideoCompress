#!/usr/bin/env python3
"""Print ROEV/1 person/car/boat/airplane events received from an RK3588.

The receiver is intentionally independent from the H.265 HUD and the audio
receiver.  It can remain running across rate-profile or video/image mode
changes because the private event channel has a fixed UDP port and no SDP.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional

from detection_event_protocol import FLAG_HEARTBEAT, Packet, mask_names


RECEIVE_POLL_TIMEOUT_SECONDS = 0.25


_CLASS_COUNT_TEXT = (
    lambda count: f"{count}个人",
    lambda count: f"{count}辆车",
    lambda count: f"{count}艘船",
    lambda count: f"{count}架飞机",
)
_CLASS_EXIT_TEXT = ("人员离开", "车辆离开", "船只离开", "飞机离开")


def display_text(packet: Packet) -> str:
    """Return the concise Chinese text for a human-facing event display.

    The sender has already applied ``--event-min-confidence``.  Text mode
    deliberately consumes only the class-state masks and per-class counts;
    confidence remains available in ``--json`` for machine consumers.
    """
    values = []
    for index, count_text in enumerate(_CLASS_COUNT_TEXT):
        bit = 1 << index
        if packet.entered_mask & bit:
            values.append(f"{count_text(packet.counts[index])}出现")
        elif packet.exited_mask & bit:
            values.append(_CLASS_EXIT_TEXT[index])
        elif packet.present_mask & bit:
            values.append(f"当前检测到{count_text(packet.counts[index])}")
    return "；".join(values) if values else "当前未检测到目标"


@dataclass
class Stats:
    packets: int = 0
    lost: int = 0
    reordered: int = 0
    invalid: int = 0
    last_sequence: Optional[int] = None

    def note_packet(self, sequence: int) -> None:
        self.packets += 1
        if self.last_sequence is None:
            self.last_sequence = sequence
            return
        delta = (sequence - self.last_sequence) & 0xFFFFFFFF
        if delta == 0 or delta >= 0x80000000:
            self.reordered += 1
            return
        if delta > 1:
            self.lost += delta - 1
        self.last_sequence = sequence


def print_packet(packet: Packet, as_json: bool) -> None:
    heartbeat = bool(packet.flags & FLAG_HEARTBEAT)
    if as_json:
        print(json.dumps({
            "type": "heartbeat" if heartbeat else "state",
            "sequence": packet.sequence,
            "frame_id": packet.frame_id,
            "pts_ms": packet.pts_ms,
            "present_mask": packet.present_mask,
            "entered_mask": packet.entered_mask,
            "exited_mask": packet.exited_mask,
            "present_classes": list(mask_names(packet.present_mask)),
            "entered_classes": list(mask_names(packet.entered_mask)),
            "exited_classes": list(mask_names(packet.exited_mask)),
            "counts": dict(zip(("person", "car", "boat", "airplane"), packet.counts)),
            "max_confidence_percent": dict(zip(("person", "car", "boat", "airplane"),
                                                 packet.max_confidence_percent)),
        }, ensure_ascii=False), flush=True)
        return
    now = time.strftime("%H:%M:%S")
    print(
        f"[{now}] {'HEARTBEAT' if heartbeat else 'STATE'} seq={packet.sequence} "
        f"frame={packet.frame_id} pts={packet.pts_ms}ms "
        f"{display_text(packet)}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5010, help="local ROEV/1 UDP port (default: 5010)")
    parser.add_argument("--bind", default="0.0.0.0", help="local bind address (default: all interfaces)")
    parser.add_argument("--quiet-heartbeats", action="store_true", help="count but do not print heartbeat packets")
    parser.add_argument("--json", action="store_true", help="print one JSON object per received packet")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="seconds between status lines; 0 disables periodic status")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.stats_interval < 0:
        parser.error("--stats-interval must be non-negative")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((args.bind, args.port))
    receiver.settimeout(RECEIVE_POLL_TIMEOUT_SECONDS)
    stats = Stats()
    last_stats = time.monotonic()
    print(f"ROEV/1 listening on UDP {args.bind}:{args.port}; Ctrl+C stops the receiver", flush=True)
    try:
        while True:
            try:
                datagram, peer = receiver.recvfrom(2048)
            except socket.timeout:
                datagram = b""
                peer = None
            if datagram:
                try:
                    packet = Packet.decode(datagram)
                except ValueError as exc:
                    stats.invalid += 1
                    print(f"invalid ROEV packet from {peer[0]}:{peer[1]}: {exc}", file=sys.stderr,
                          flush=True)
                else:
                    stats.note_packet(packet.sequence)
                    if not (args.quiet_heartbeats and packet.flags & FLAG_HEARTBEAT):
                        print_packet(packet, args.json)
            now = time.monotonic()
            if args.stats_interval and now - last_stats >= args.stats_interval:
                print(f"ROEV stats packets={stats.packets} lost={stats.lost} "
                      f"reorder={stats.reordered} invalid={stats.invalid}", flush=True)
                last_stats = now
    except KeyboardInterrupt:
        print(f"ROEV stopped: packets={stats.packets} lost={stats.lost} "
              f"reorder={stats.reordered} invalid={stats.invalid}", flush=True)
        return 0
    finally:
        receiver.close()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ROEV/1 parser shared by the PC semantic-event receiver and its tests.

ROEV is deliberately a small private UDP companion protocol.  It does not
alter the standards-compliant H.265 RTP payload or require an SDP file.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Tuple


MAGIC = b"ROEV"
VERSION = 1
EVENT_STATE = 1
FLAG_HEARTBEAT = 1 << 0
CLASS_NAMES = ("person", "car", "boat", "airplane")
CLASS_MASK = (1 << len(CLASS_NAMES)) - 1
# magic, version, type, flags, sequence, frame id, PTS ms, three masks,
# reserved, four counts, four maximum confidence percentages, CRC32.
HEADER = struct.Struct("!4sBBHIQIHHHH4B4BI")
CRC_OFFSET = 40


def mask_names(mask: int) -> Tuple[str, ...]:
    """Return stable class names for a ROEV class mask."""
    return tuple(name for index, name in enumerate(CLASS_NAMES) if mask & (1 << index))


def _validate_masks(present: int, entered: int, exited: int) -> None:
    if (present | entered | exited) & ~CLASS_MASK:
        raise ValueError("ROEV contains an unknown class-mask bit")
    if entered & ~present:
        raise ValueError("ROEV entered class is absent from present state")
    if exited & present:
        raise ValueError("ROEV exited class is still present")
    if entered & exited:
        raise ValueError("ROEV class cannot enter and exit in one state")


@dataclass(frozen=True)
class Packet:
    sequence: int
    frame_id: int
    pts_ms: int
    present_mask: int
    entered_mask: int = 0
    exited_mask: int = 0
    counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
    max_confidence_percent: Tuple[int, int, int, int] = (0, 0, 0, 0)
    flags: int = 0
    packet_type: int = EVENT_STATE
    crc32: int = 0

    def _validate(self) -> None:
        if self.packet_type != EVENT_STATE:
            raise ValueError("unsupported ROEV packet type")
        if self.flags & ~FLAG_HEARTBEAT:
            raise ValueError("ROEV contains unknown flags")
        _validate_masks(self.present_mask, self.entered_mask, self.exited_mask)
        if self.flags & FLAG_HEARTBEAT and (self.entered_mask or self.exited_mask):
            raise ValueError("ROEV heartbeat cannot carry edge masks")
        if len(self.counts) != len(CLASS_NAMES) or len(self.max_confidence_percent) != len(CLASS_NAMES):
            raise ValueError("ROEV requires four counts and four confidences")
        for index, (count, confidence) in enumerate(zip(self.counts, self.max_confidence_percent)):
            if not 0 <= count <= 0xFF or not 0 <= confidence <= 100:
                raise ValueError("ROEV count or confidence is out of range")
            if not self.present_mask & (1 << index) and (count or confidence):
                raise ValueError("ROEV has details for an absent class")

    def encode(self) -> bytes:
        self._validate()
        provisional = HEADER.pack(
            MAGIC, VERSION, self.packet_type, self.flags, self.sequence, self.frame_id,
            self.pts_ms, self.present_mask, self.entered_mask, self.exited_mask, 0,
            *self.counts, *self.max_confidence_percent, 0,
        )
        crc32 = zlib.crc32(provisional[:CRC_OFFSET]) & 0xFFFFFFFF
        return provisional[:CRC_OFFSET] + struct.pack("!I", crc32)

    @classmethod
    def decode(cls, datagram: bytes) -> "Packet":
        if len(datagram) != HEADER.size:
            raise ValueError("ROEV datagram must be exactly 44 bytes")
        (magic, version, packet_type, flags, sequence, frame_id, pts_ms,
         present_mask, entered_mask, exited_mask, reserved,
         count0, count1, count2, count3,
         confidence0, confidence1, confidence2, confidence3, crc32) = HEADER.unpack(datagram)
        if magic != MAGIC or version != VERSION or reserved != 0:
            raise ValueError("invalid ROEV magic, version, or reserved field")
        expected_crc = zlib.crc32(datagram[:CRC_OFFSET]) & 0xFFFFFFFF
        if crc32 != expected_crc:
            raise ValueError("ROEV CRC32 mismatch")
        packet = cls(
            sequence=sequence, frame_id=frame_id, pts_ms=pts_ms,
            present_mask=present_mask, entered_mask=entered_mask, exited_mask=exited_mask,
            counts=(count0, count1, count2, count3),
            max_confidence_percent=(confidence0, confidence1, confidence2, confidence3),
            flags=flags, packet_type=packet_type, crc32=crc32,
        )
        packet._validate()
        return packet

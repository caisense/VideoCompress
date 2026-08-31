#!/usr/bin/env python3
"""RB/1 semantic/reference companion protocol for the rebuild video profile."""

from __future__ import annotations

import dataclasses
import struct
import time
import zlib
from typing import Dict, List, Optional, Tuple


MAGIC = b"RB"
VERSION = 1
PROFILE_REBUILD = 3
STATE = 1
PATCH_DATA = 2
PATCH_PARITY = 3
HEARTBEAT = 4
HEADER_BYTES = 28
STATE_HEADER_BYTES = 12
TARGET_BYTES = 16
# The reference crop and the detector box at the instant the crop was taken
# are both carried on every fragment.  Keeping both rectangles is essential:
# the receiver can register the crop against the current box instead of
# assuming that the crop was centred on it (which caused floating/ghosted
# targets when a box moved or was clipped at an image edge).
FRAGMENT_HEADER_BYTES = 40
MAX_TARGETS = 16
MAX_SOURCE_DIMENSION = 8192
MAX_OUTPUT_DIMENSION = 4096
MAX_OUTPUT_PIXELS = 1920 * 1080
MAX_PATCH_DIMENSION = 512
MAX_MASK_DIMENSION = 64
MAX_BLOB_BYTES = 12 * 1024
MAX_CHUNK_BYTES = 1400

_HEADER_PREFIX = struct.Struct("!2sBBBBHIIIHH")
_STATE_HEADER = struct.Struct("!HHHHBBBB")
_TARGET = struct.Struct("!HBBHHHHHBB")
_FRAGMENT = struct.Struct("!IHHHHHHHHHHHHBBBBIHH")


@dataclasses.dataclass(frozen=True)
class Packet:
    packet_type: int
    profile: int
    generation: int
    flags: int
    sequence: int
    frame_id: int
    pts_ms: int
    payload: bytes


@dataclasses.dataclass(frozen=True)
class TargetState:
    track_id: int
    class_id: int
    confidence_percent: int
    left: int
    top: int
    right: int
    bottom: int
    reference_generation: int
    flags: int


@dataclasses.dataclass(frozen=True)
class StateRecord:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_fps: int
    output_fps: int
    flags: int
    targets: Tuple[TargetState, ...]


@dataclasses.dataclass(frozen=True)
class PatchFragment:
    transfer_id: int
    track_id: int
    reference_generation: int
    left: int
    top: int
    right: int
    bottom: int
    reference_left: int
    reference_top: int
    reference_right: int
    reference_bottom: int
    jpeg_width: int
    jpeg_height: int
    mask_width: int
    mask_height: int
    fragment_index: int
    data_fragments: int
    blob_size: int
    mask_rle_bytes: int
    chunk_bytes: int
    data: bytes


@dataclasses.dataclass(frozen=True)
class CompleteReference:
    generation: int
    fragment: PatchFragment
    mask_rle: bytes
    jpeg: bytes
    recovered_with_parity: bool
    received_at: float


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def build_packet(packet: Packet) -> bytes:
    if packet.packet_type not in (STATE, PATCH_DATA, PATCH_PARITY, HEARTBEAT):
        raise ValueError("invalid RB/1 packet type")
    if len(packet.payload) > 0xFFFF:
        raise ValueError("RB/1 payload exceeds uint16 length")
    prefix = _HEADER_PREFIX.pack(
        MAGIC, VERSION, packet.packet_type, packet.profile, packet.generation,
        HEADER_BYTES, packet.sequence, packet.frame_id, packet.pts_ms,
        len(packet.payload), packet.flags,
    )
    return prefix + struct.pack("!I", crc32(prefix + packet.payload)) + packet.payload


def parse_packet(data: bytes) -> Packet:
    if len(data) < HEADER_BYTES:
        raise ValueError("RB/1 datagram is shorter than its header")
    (magic, version, packet_type, profile, generation, header_bytes, sequence,
     frame_id, pts_ms, payload_bytes, flags) = _HEADER_PREFIX.unpack_from(data)
    if magic != MAGIC or version != VERSION or header_bytes != HEADER_BYTES:
        raise ValueError("invalid RB/1 magic, version, or header length")
    if packet_type not in (STATE, PATCH_DATA, PATCH_PARITY, HEARTBEAT):
        raise ValueError("invalid RB/1 packet type")
    if len(data) != HEADER_BYTES + payload_bytes:
        raise ValueError("RB/1 payload length mismatch")
    expected_crc = struct.unpack_from("!I", data, 24)[0]
    payload = data[HEADER_BYTES:]
    if expected_crc != crc32(data[:24] + payload):
        raise ValueError("RB/1 CRC32 mismatch")
    return Packet(packet_type, profile, generation, flags, sequence,
                  frame_id, pts_ms, payload)


def build_state(state: StateRecord) -> bytes:
    if len(state.targets) > 255:
        raise ValueError("too many RB/1 targets")
    payload = bytearray(_STATE_HEADER.pack(
        state.source_width, state.source_height, state.output_width,
        state.output_height, state.source_fps, state.output_fps,
        len(state.targets), state.flags,
    ))
    for target in state.targets:
        payload.extend(_TARGET.pack(
            target.track_id, target.class_id, target.confidence_percent,
            target.left, target.top, target.right, target.bottom,
            target.reference_generation, target.flags, 0,
        ))
    return bytes(payload)


def parse_state(payload: bytes) -> StateRecord:
    if len(payload) < STATE_HEADER_BYTES:
        raise ValueError("RB/1 state is shorter than its header")
    (source_width, source_height, output_width, output_height, source_fps,
     output_fps, count, flags) = _STATE_HEADER.unpack_from(payload)
    if (len(payload) != STATE_HEADER_BYTES + count * TARGET_BYTES or
            count > MAX_TARGETS or source_width == 0 or source_height == 0 or
            source_width > MAX_SOURCE_DIMENSION or
            source_height > MAX_SOURCE_DIMENSION or output_width == 0 or
            output_height == 0 or output_width > MAX_OUTPUT_DIMENSION or
            output_height > MAX_OUTPUT_DIMENSION or
            output_width * output_height > MAX_OUTPUT_PIXELS or
            not 1 <= source_fps <= 60 or not 1 <= output_fps <= 60):
        raise ValueError("RB/1 state target length mismatch")
    targets: List[TargetState] = []
    for index in range(count):
        values = _TARGET.unpack_from(payload, STATE_HEADER_BYTES + index * TARGET_BYTES)
        target = TargetState(*values[:-1])
        if (target.track_id == 0 or target.confidence_percent > 100 or
                target.right <= target.left or target.bottom <= target.top or
                target.right > source_width or target.bottom > source_height):
            raise ValueError("invalid RB/1 target state")
        targets.append(target)
    return StateRecord(source_width, source_height, output_width, output_height,
                       source_fps, output_fps, flags, tuple(targets))


def build_fragment(fragment: PatchFragment) -> bytes:
    if len(fragment.data) > fragment.chunk_bytes:
        raise ValueError("RB/1 fragment data exceeds nominal chunk")
    return _FRAGMENT.pack(
        fragment.transfer_id, fragment.track_id, fragment.reference_generation,
        fragment.left, fragment.top, fragment.right, fragment.bottom,
        fragment.reference_left, fragment.reference_top,
        fragment.reference_right, fragment.reference_bottom,
        fragment.jpeg_width, fragment.jpeg_height, fragment.mask_width,
        fragment.mask_height, fragment.fragment_index, fragment.data_fragments,
        fragment.blob_size, fragment.mask_rle_bytes, fragment.chunk_bytes,
    ) + fragment.data


def parse_fragment(payload: bytes) -> PatchFragment:
    if len(payload) < FRAGMENT_HEADER_BYTES:
        raise ValueError("RB/1 patch is shorter than fragment metadata")
    values = _FRAGMENT.unpack_from(payload)
    fragment = PatchFragment(*values, payload[FRAGMENT_HEADER_BYTES:])
    expected_fragments = ((fragment.blob_size + fragment.chunk_bytes - 1) //
                          fragment.chunk_bytes) if fragment.chunk_bytes else 0
    if (fragment.track_id == 0 or fragment.reference_generation == 0 or
            fragment.data_fragments == 0 or
            fragment.fragment_index > fragment.data_fragments or
            fragment.chunk_bytes == 0 or len(fragment.data) > fragment.chunk_bytes or
            fragment.blob_size == 0 or fragment.mask_rle_bytes > fragment.blob_size or
            fragment.blob_size > MAX_BLOB_BYTES or
            fragment.chunk_bytes > MAX_CHUNK_BYTES or
            fragment.data_fragments != expected_fragments or
            fragment.right <= fragment.left or fragment.bottom <= fragment.top or
            fragment.reference_left < 0 or fragment.reference_top < 0 or
            fragment.reference_right <= fragment.reference_left or
            fragment.reference_bottom <= fragment.reference_top or
            fragment.right > MAX_SOURCE_DIMENSION or
            fragment.bottom > MAX_SOURCE_DIMENSION or
            fragment.reference_right > MAX_SOURCE_DIMENSION or
            fragment.reference_bottom > MAX_SOURCE_DIMENSION or
            fragment.jpeg_width == 0 or
            fragment.jpeg_height == 0 or
            fragment.jpeg_width > MAX_PATCH_DIMENSION or
            fragment.jpeg_height > MAX_PATCH_DIMENSION or fragment.mask_width == 0 or
            fragment.mask_height == 0 or
            fragment.mask_width > MAX_MASK_DIMENSION or
            fragment.mask_height > MAX_MASK_DIMENSION):
        raise ValueError("invalid RB/1 patch fragment metadata")
    return fragment


def decode_mask_rle(data: bytes, width: int, height: int) -> bytes:
    expected = width * height
    output = bytearray()
    if len(data) % 2:
        raise ValueError("RB/1 mask RLE has an odd byte count")
    for cursor in range(0, len(data), 2):
        run, value = data[cursor], data[cursor + 1]
        if run == 0 or value not in (0, 1) or len(output) + run > expected:
            raise ValueError("invalid RB/1 mask RLE run")
        output.extend(bytes((255 if value else 0,)) * run)
    if len(output) != expected:
        raise ValueError("RB/1 mask RLE pixel count mismatch")
    return bytes(output)


class FragmentAssembler:
    """Latest bounded reassembly with recovery of exactly one lost data fragment."""

    @dataclasses.dataclass
    class _Transfer:
        metadata: PatchFragment
        data: Dict[int, bytes]
        parity: Optional[bytes]
        updated: float

    def __init__(self, timeout_seconds: float = 1.5, max_transfers: int = 16) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_transfers = max_transfers
        self.transfers: Dict[Tuple[int, int], FragmentAssembler._Transfer] = {}
        self.completed: Dict[Tuple[int, int], float] = {}
        self.expired = 0
        self.recovered = 0
        self.inconsistent = 0

    @staticmethod
    def _signature(fragment: PatchFragment) -> tuple:
        return (
            fragment.track_id, fragment.reference_generation, fragment.left,
            fragment.top, fragment.right, fragment.bottom,
            fragment.reference_left, fragment.reference_top,
            fragment.reference_right, fragment.reference_bottom,
            fragment.jpeg_width,
            fragment.jpeg_height, fragment.mask_width, fragment.mask_height,
            fragment.data_fragments, fragment.blob_size, fragment.mask_rle_bytes,
            fragment.chunk_bytes,
        )

    def _expire(self, now: float) -> None:
        stale = [key for key, value in self.transfers.items()
                 if now - value.updated > self.timeout_seconds]
        for key in stale:
            del self.transfers[key]
            self.expired += 1
        while len(self.transfers) > self.max_transfers:
            oldest = min(self.transfers, key=lambda key: self.transfers[key].updated)
            del self.transfers[oldest]
            self.expired += 1
        self.completed = {
            key: completed_at for key, completed_at in self.completed.items()
            if now - completed_at <= self.timeout_seconds
        }

    def add(self, generation: int, packet_type: int, fragment: PatchFragment,
            now: Optional[float] = None) -> Optional[CompleteReference]:
        now = time.monotonic() if now is None else now
        self._expire(now)
        key = (generation, fragment.transfer_id)
        # Legacy senders emitted parity after all data.  Once the data already
        # completed a reference, that trailing parity must not create a new
        # phantom incomplete transfer with the same ID.
        if key in self.completed:
            return None
        transfer = self.transfers.get(key)
        if transfer is None:
            transfer = self._Transfer(fragment, {}, None, now)
            self.transfers[key] = transfer
        elif self._signature(transfer.metadata) != self._signature(fragment):
            del self.transfers[key]
            self.inconsistent += 1
            return None
        transfer.updated = now
        if packet_type == PATCH_DATA and fragment.fragment_index < fragment.data_fragments:
            expected_bytes = min(
                fragment.chunk_bytes,
                fragment.blob_size - fragment.fragment_index * fragment.chunk_bytes)
            if len(fragment.data) != expected_bytes:
                self.inconsistent += 1
                del self.transfers[key]
                return None
            previous = transfer.data.get(fragment.fragment_index)
            if previous is not None and previous != fragment.data:
                self.inconsistent += 1
                del self.transfers[key]
                return None
            transfer.data.setdefault(fragment.fragment_index, fragment.data)
        elif packet_type == PATCH_PARITY and fragment.fragment_index == fragment.data_fragments:
            if len(fragment.data) != fragment.chunk_bytes:
                self.inconsistent += 1
                del self.transfers[key]
                return None
            if transfer.parity is not None and transfer.parity != fragment.data:
                self.inconsistent += 1
                del self.transfers[key]
                return None
            transfer.parity = fragment.data
        else:
            return None

        missing = [index for index in range(fragment.data_fragments)
                   if index not in transfer.data]
        recovered = False
        if len(missing) == 1 and transfer.parity is not None:
            restored = bytearray(transfer.parity)
            if len(restored) != fragment.chunk_bytes:
                return None
            for piece in transfer.data.values():
                for index, value in enumerate(piece):
                    restored[index] ^= value
            missing_index = missing[0]
            wanted = min(fragment.chunk_bytes,
                         fragment.blob_size - missing_index * fragment.chunk_bytes)
            if wanted <= 0:
                return None
            transfer.data[missing_index] = bytes(restored[:wanted])
            missing = []
            recovered = True
            self.recovered += 1
        if missing:
            return None
        blob = b"".join(transfer.data[index]
                        for index in range(fragment.data_fragments))
        blob = blob[:fragment.blob_size]
        del self.transfers[key]
        self.completed[key] = now
        if len(blob) != fragment.blob_size:
            return None
        mask_rle = blob[:fragment.mask_rle_bytes]
        jpeg = blob[fragment.mask_rle_bytes:]
        if not jpeg:
            return None
        return CompleteReference(generation, fragment, mask_rle, jpeg, recovered, now)

#!/usr/bin/env python3
"""PC-side RB/1 receiver and bounded 640x360 rebuild compositor."""

from __future__ import annotations

import collections
import dataclasses
import math
import os
import site
import socket
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

import rebuild_protocol as rb


REFERENCE_CACHE_TTL_MS = 1000
REFERENCE_CONTENT_HARD_MAX_AGE_MS = 450
REFERENCE_FUTURE_MAX_MS = 100
STATE_SYNC_MAX_MS = 100
STATE_EXTRAPOLATION_MAX_MS = 200
SR_MODEL_INPUT_SIDE = 96
REGISTRATION_MIN_AREA_RATIO = 0.70
# Head sub-crops must follow the same scale as their parent detector box.  A
# normal 1.2x width/height change is 1.44x area, so 1.40 rejected the explicit
# head mapping case and silently dropped valid local references.
REGISTRATION_MAX_AREA_RATIO = 1.60
REGISTRATION_MIN_SIDE_RATIO = 0.50
REGISTRATION_MAX_SIDE_RATIO = 2.00


def wire_bytes(udp_payload_bytes: int) -> int:
    return 38 + max(46, udp_payload_bytes + 28)


def jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Read JPEG SOF dimensions before OpenCV allocates the decoded image."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("RB/1 reference is not a JPEG")
    cursor = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while cursor + 4 <= len(data):
        if data[cursor] != 0xFF:
            cursor += 1
            continue
        while cursor < len(data) and data[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(data):
            break
        marker = data[cursor]
        cursor += 1
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if cursor + 2 > len(data):
            break
        segment_bytes = int.from_bytes(data[cursor:cursor + 2], "big")
        if segment_bytes < 2 or cursor + segment_bytes > len(data):
            break
        if marker in sof_markers:
            if segment_bytes < 7:
                break
            height = int.from_bytes(data[cursor + 3:cursor + 5], "big")
            width = int.from_bytes(data[cursor + 5:cursor + 7], "big")
            if width <= 0 or height <= 0:
                break
            return width, height
        cursor += segment_bytes
    raise ValueError("RB/1 JPEG has no valid SOF dimensions")


@dataclasses.dataclass(frozen=True)
class VisualReference:
    generation: int
    track_id: int
    reference_generation: int
    crop: Tuple[int, int, int, int]
    # Detector bbox at the instant this crop was captured.  It is distinct
    # from ``crop`` because image-edge clipping makes the crop asymmetric.
    reference_bbox: Tuple[int, int, int, int]
    image: np.ndarray
    mask: np.ndarray
    received_at: float
    recovered_with_parity: bool
    pts_ms: int
    reference_flags: int = 0
    jpeg_width: int = 0
    jpeg_height: int = 0
    jpeg_bytes: int = 0

    @property
    def reference_kind(self) -> str:
        return "HEAD" if self.reference_flags & rb.PACKET_FLAG_HEAD_REFERENCE else "FULL"


@dataclasses.dataclass(frozen=True)
class RegistrationResult:
    crop: Optional[Tuple[int, int, int, int]]
    reason: str
    dx_ratio: float
    dy_ratio: float
    area_ratio: float


@dataclasses.dataclass(frozen=True)
class ContentRegistrationResult:
    """Result of the stable-template identity check before a ROI is painted."""

    position: Optional[Tuple[int, int]]
    score: Optional[float]
    attempted: bool
    reason: str
    visible_ratio: float

    # Keep the small helper source-compatible with the previous tuple return
    # while exposing a named diagnostic contract to the HUD and tests.
    def __iter__(self):
        yield self.position
        yield self.score
        yield self.attempted


@dataclasses.dataclass(frozen=True)
class StateHistoryItem:
    """One received STATE plus transport metadata used for timing diagnosis."""

    generation: int
    pts_ms: int
    state: rb.StateRecord
    arrival_time: float
    sequence: int
    frame_id: int


class RebuildReceiver:
    def __init__(self, port: int = 5009, max_sync_ms: int = STATE_SYNC_MAX_MS,
                 reference_cache_ttl_ms: int = REFERENCE_CACHE_TTL_MS,
                 reference_future_max_ms: int = REFERENCE_FUTURE_MAX_MS) -> None:
        if max_sync_ms <= 0:
            raise ValueError("RB/1 maximum PTS sync error must be positive")
        if reference_cache_ttl_ms <= 0 or reference_future_max_ms <= 0:
            raise ValueError("RB/1 reference cache/future limits must be positive")
        self.port = port
        self.max_sync_ms = max_sync_ms
        self.reference_cache_ttl_ms = reference_cache_ttl_ms
        self.reference_future_max_ms = reference_future_max_ms
        self.lock = threading.Lock()
        self.assembler = rb.FragmentAssembler()
        self.state: Optional[rb.StateRecord] = None
        self.state_generation: Optional[int] = None
        self.state_updated = 0.0
        self.state_pts_ms: Optional[int] = None
        self.last_selected_state_pts_ms: Optional[int] = None
        self.state_sequence: Optional[int] = None
        self.state_history: Deque[StateHistoryItem] = collections.deque(maxlen=32)
        self.references: Dict[int, VisualReference] = {}
        # Keep the short set of generations that may still be named by a
        # PTS-synchronised STATE.  The newest PATCH for a track can arrive
        # before the video frame that advertises it; replacing the sole
        # per-track entry used to hide the previous, still-required crop.
        self.reference_history: Dict[Tuple[int, int], VisualReference] = {}
        # A complete PATCH is local-ready immediately.  This floor is kept
        # separately from STATE flags so a late semantic packet cannot make a
        # valid local reference appear unavailable or move its generation back.
        self.reference_generations: Dict[int, int] = {}
        self.state_reference_generations: Dict[int, int] = {}
        # ``on_datagram`` runs on the RB/1 socket thread while composition
        # runs on the UI thread.  A completed reference must be handed to the
        # compositor immediately, rather than waiting until a future video
        # presentation happens to select a STATE that declares it usable.
        # Otherwise the first Real-ESRGAN job can start at PTSAGE ~= +1000 ms
        # and be correct but too late to display.  Keep a tiny bounded event
        # queue: the compositor only needs the newest reference(s), never a
        # backlog of stale crops.
        self.completed_reference_events: Deque[VisualReference] = collections.deque(maxlen=4)
        self.samples: Deque[Tuple[float, int, int, int]] = collections.deque()
        self.last_sequence: Optional[int] = None
        self.packets = 0
        self.lost = 0
        self.reordered = 0
        self.invalid = 0
        self.state_packets = 0
        self.complete_references = 0
        self.parity_recovered = 0
        self.sync_drops = 0
        self.reference_generation_drops = 0
        self._last_sync_drop_timestamp: Optional[int] = None
        # A decoder/output handoff can introduce a stable presentation offset
        # (for example one 6-fps source frame).  Keep a robust sliding estimate
        # and apply it before the strict 100 ms semantic gate.  The estimate is
        # deliberately learned only from plausible nearest-state observations;
        # a seconds-old state must never move the clock used for registration.
        self.pts_bias_ms = 0
        self.raw_sync_ms: Optional[int] = None
        self.last_state_delta_ms: Optional[int] = None
        self.last_state_arrival_age_ms: Optional[int] = None
        self.last_state_selected_age_ms: Optional[int] = None
        self.last_state_extrapolated = False
        self.last_state_reason = "STATE_NO_HISTORY"
        self.last_state_sequence: Optional[int] = None
        self.last_state_frame_id: Optional[int] = None
        self.pts_bias_samples: Deque[int] = collections.deque(maxlen=31)
        self._last_bias_observation: Optional[Tuple[int, int]] = None
        self.socket: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None

    @staticmethod
    def _newer_u32(value: int, previous: int) -> bool:
        delta = (value - previous) & 0xFFFFFFFF
        return 0 < delta < 0x80000000

    @staticmethod
    def _newer_u8(value: int, previous: int) -> bool:
        delta = (value - previous) & 0xFF
        return 0 < delta < 0x80

    @staticmethod
    def _newer_u16(value: int, previous: int) -> bool:
        delta = (value - previous) & 0xFFFF
        return 0 < delta < 0x8000

    @classmethod
    def _at_least_u16(cls, value: int, previous: int) -> bool:
        return value == previous or cls._newer_u16(value, previous)

    def _prune_references(self, now: float) -> None:
        cutoff = now - self.reference_cache_ttl_ms / 1000.0
        self.references = {
            track_id: reference for track_id, reference in self.references.items()
            if reference.received_at >= cutoff
        }
        self.reference_history = {
            key: reference for key, reference in self.reference_history.items()
            if reference.received_at >= cutoff
        }
        self.completed_reference_events = collections.deque(
            (reference for reference in self.completed_reference_events
             if reference.received_at >= cutoff), maxlen=4)

    def _state_reference_generations_are_valid(self, state: rb.StateRecord) -> bool:
        for target in state.targets:
            previous = self.state_reference_generations.get(target.track_id)
            if target.reference_generation == 0:
                # Zero is only the initial "not advertised yet" placeholder.
                # Once a non-zero generation has been published, a later zero
                # STATE is a semantic rollback even if its packet sequence is
                # newer because it came through a delayed producer queue.
                if previous is not None:
                    return False
                continue
            local = self.reference_generations.get(target.track_id)
            for floor in (previous, local):
                if (floor is not None and target.reference_generation != floor and
                        not self._newer_u16(target.reference_generation, floor)):
                    return False
        return True

    def _record_state_reference_generations(self, state: rb.StateRecord) -> None:
        for target in state.targets:
            if target.reference_generation == 0:
                continue
            previous = self.state_reference_generations.get(target.track_id)
            if (previous is None or target.reference_generation == previous or
                    self._newer_u16(target.reference_generation, previous)):
                self.state_reference_generations[target.track_id] = target.reference_generation

    def on_datagram(self, data: bytes, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        try:
            packet = rb.parse_packet(data)
            if packet.profile != rb.PROFILE_REBUILD:
                raise ValueError("RB/1 packet is not for the rebuild profile")
            state = rb.parse_state(packet.payload) if packet.packet_type == rb.STATE else None
            fragment = rb.parse_fragment(packet.payload) if packet.packet_type in (
                rb.PATCH_DATA, rb.PATCH_PARITY) else None
        except ValueError:
            with self.lock:
                self.invalid += 1
            return

        with self.lock:
            if self.state_generation != packet.generation:
                if (self.state_generation is not None and
                        not self._newer_u8(packet.generation, self.state_generation)):
                    # A delayed packet from the previous profile must never
                    # roll the receiver generation backwards and clear a live scene.
                    self.reordered += 1
                    return
                self.state = None
                self.references.clear()
                self.reference_history.clear()
                self.reference_generations.clear()
                self.state_reference_generations.clear()
                self.completed_reference_events.clear()
                self.state_history.clear()
                self.state_updated = 0.0
                self.state_pts_ms = None
                self.state_sequence = None
                self.last_selected_state_pts_ms = None
                self.last_state_delta_ms = None
                self.last_state_arrival_age_ms = None
                self.last_state_selected_age_ms = None
                self.last_state_extrapolated = False
                self.last_state_reason = "STATE_NO_HISTORY"
                self.last_state_sequence = None
                self.last_state_frame_id = None
                self.state_generation = packet.generation
                # Incomplete fragments from the previous profile generation
                # can never form a valid current reference.
                self.assembler = rb.FragmentAssembler()

        complete = None
        if fragment is not None:
            try:
                complete = self.assembler.add(packet.generation, packet.packet_type,
                                              fragment, now, packet.flags)
            except ValueError:
                with self.lock:
                    self.invalid += 1
                return

        decoded_reference = None
        if complete is not None:
            try:
                mask_bytes = rb.decode_mask_rle(
                    complete.mask_rle, complete.fragment.mask_width,
                    complete.fragment.mask_height)
                mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                    complete.fragment.mask_height, complete.fragment.mask_width).copy()
                jpeg_width, jpeg_height = jpeg_dimensions(complete.jpeg)
                if (jpeg_width != complete.fragment.jpeg_width or
                        jpeg_height != complete.fragment.jpeg_height or
                        jpeg_width > rb.MAX_PATCH_DIMENSION or
                        jpeg_height > rb.MAX_PATCH_DIMENSION):
                    raise ValueError("RB/1 JPEG dimensions disagree with metadata")
                image = cv2.imdecode(np.frombuffer(complete.jpeg, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("RB/1 JPEG decode failed")
                decoded_reference = VisualReference(
                    generation=complete.generation,
                    track_id=complete.fragment.track_id,
                    reference_generation=complete.fragment.reference_generation,
                    crop=(complete.fragment.left, complete.fragment.top,
                          complete.fragment.right, complete.fragment.bottom),
                    reference_bbox=(complete.fragment.reference_left,
                                    complete.fragment.reference_top,
                                    complete.fragment.reference_right,
                                    complete.fragment.reference_bottom),
                    image=image, mask=mask, received_at=complete.received_at,
                    recovered_with_parity=complete.recovered_with_parity,
                    pts_ms=packet.pts_ms,
                    reference_flags=complete.reference_flags,
                    jpeg_width=jpeg_width,
                    jpeg_height=jpeg_height,
                    jpeg_bytes=len(complete.jpeg),
                )
            except (ValueError, cv2.error):
                with self.lock:
                    self.invalid += 1
                return

        with self.lock:
            self.packets += 1
            self.samples.append((now, len(data), wire_bytes(len(data)), packet.packet_type))
            if self.last_sequence is not None:
                delta = (packet.sequence - self.last_sequence) & 0xFFFFFFFF
                if 1 < delta < 0x80000000:
                    self.lost += delta - 1
                elif delta >= 0x80000000:
                    self.reordered += 1
            if self.last_sequence is None or self._newer_u32(packet.sequence,
                                                             self.last_sequence):
                self.last_sequence = packet.sequence
            if state is not None:
                self.state_packets += 1
                # Reference completion emits a second STATE with the same
                # source PTS but a newer RB/1 sequence and its ready flag set.
                # Replace the earlier same-PTS snapshot so nearest-PTS
                # selection cannot keep choosing the old flags=0 record.
                # Conversely, a delayed old UDP STATE must never roll that
                # readiness (or its reference cache) backwards.
                accept_state = (
                    self.state_sequence is None or
                    packet.sequence == self.state_sequence or
                    self._newer_u32(packet.sequence, self.state_sequence))
                if accept_state and not self._state_reference_generations_are_valid(state):
                    # A semantically older STATE is still an old reference
                    # generation, even when its RB/1 packet sequence is newer
                    # because it was emitted by a delayed producer queue.
                    self.reference_generation_drops += 1
                    accept_state = False
                if accept_state:
                    self._record_state_reference_generations(state)
                    self.state = state
                    self.state_updated = now
                    self.state_pts_ms = packet.pts_ms
                    self.state_sequence = packet.sequence
                    state_item = StateHistoryItem(
                        packet.generation, packet.pts_ms, state, now,
                        packet.sequence, packet.frame_id)
                    # A completion STATE can share the source PTS with the
                    # earlier flags=0 STATE, while PATCH fragments for that
                    # reference and later states have already interleaved in
                    # this history.  Replace every same-PTS snapshot so the
                    # nearest-PTS lookup cannot tie-break back to stale
                    # reference-generation metadata.
                    self.state_history = collections.deque(
                        (item for item in self.state_history
                         if not (self._state_item_values(item)[0] == packet.generation and
                                 self._state_item_values(item)[1] == packet.pts_ms)),
                        maxlen=self.state_history.maxlen)
                    self.state_history.append(state_item)
            if decoded_reference is not None:
                existing = self.references.get(decoded_reference.track_id)
                local_generation = self.reference_generations.get(
                    decoded_reference.track_id)
                is_newer_than_local = (
                    local_generation is None or
                    decoded_reference.reference_generation == local_generation or
                    self._newer_u16(decoded_reference.reference_generation,
                                    local_generation))
                is_newer_than_existing = (
                    existing is None or
                    decoded_reference.reference_generation == existing.reference_generation or
                    self._newer_u16(decoded_reference.reference_generation,
                                    existing.reference_generation))
                if (decoded_reference.generation == self.state_generation and
                        is_newer_than_local and is_newer_than_existing):
                    self.references[decoded_reference.track_id] = decoded_reference
                    self.reference_history[(
                        decoded_reference.track_id,
                        decoded_reference.reference_generation)] = decoded_reference
                    self.reference_generations[decoded_reference.track_id] = decoded_reference.reference_generation
                    self.completed_reference_events.append(decoded_reference)
                    self.complete_references += 1
                    if decoded_reference.recovered_with_parity:
                        self.parity_recovered += 1
                elif (local_generation is not None and
                      decoded_reference.reference_generation != local_generation and
                      not self._newer_u16(decoded_reference.reference_generation,
                                          local_generation)):
                    self.reference_generation_drops += 1
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - 1.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        self._prune_references(now)

    def scene(self) -> Tuple[Optional[rb.StateRecord], int,
                             Dict[int, VisualReference], float]:
        with self.lock:
            generation = -1 if self.state_generation is None else self.state_generation
            return self.state, generation, dict(self.references), self.state_updated

    def take_completed_references(self) -> Tuple[VisualReference, ...]:
        """Return newly assembled crops once, for early asynchronous SR.

        The event is already local-ready: the compositor may use its content
        as soon as a valid current scene supplies geometry.  STATE flags are
        deliberately not repeated as a second readiness gate; generation,
        reference-generation, PTS, age, and registration checks still decide
        whether the reference can be blended into a source frame.
        """
        with self.lock:
            events = tuple(self.completed_reference_events)
            self.completed_reference_events.clear()
            return events

    @staticmethod
    def _signed_pts_delta(left: int, right: int) -> int:
        return ((left - right + 0x80000000) & 0xFFFFFFFF) - 0x80000000

    @staticmethod
    def _state_item_values(item) -> Tuple[int, int, rb.StateRecord, float,
                                           Optional[int], Optional[int]]:
        """Read current history entries and legacy test fixtures alike."""
        if isinstance(item, StateHistoryItem):
            return (item.generation, item.pts_ms, item.state, item.arrival_time,
                    item.sequence, item.frame_id)
        generation, pts_ms, state, arrival_time = item
        return generation, pts_ms, state, arrival_time, None, None

    def _observe_pts_bias(self, raw_delta_ms: int, state_pts_ms: int,
                          video_rtp_timestamp: int) -> None:
        """Update the DC offset estimate once per decoded video timestamp."""
        self.raw_sync_ms = raw_delta_ms
        if abs(raw_delta_ms) > 500:
            return
        observation = (video_rtp_timestamp & 0xFFFFFFFF, state_pts_ms)
        if observation == self._last_bias_observation:
            return
        self._last_bias_observation = observation
        self.pts_bias_samples.append(raw_delta_ms)
        # Five samples cover startup jitter while keeping the correction fast
        # enough for a profile switch.  Median rejects a lost/duplicated AU.
        if len(self.pts_bias_samples) >= 5:
            median = int(round(statistics.median(self.pts_bias_samples)))
            self.pts_bias_ms = max(-500, min(500, median))

    def _effective_video_timestamp(self, video_rtp_timestamp: int) -> int:
        return (video_rtp_timestamp + int(round(self.pts_bias_ms * 90.0))) & 0xFFFFFFFF

    @classmethod
    def _extrapolate_state(cls, selected: rb.StateRecord, selected_pts: int,
                           previous: Optional[rb.StateRecord], previous_pts: Optional[int],
                           video_pts: int, max_offset_ms: int
                           ) -> Tuple[Optional[rb.StateRecord], str]:
        """Safely advance a one-source-frame-old STATE to the video PTS.

        Extrapolation is deliberately a narrow fallback for STATE jitter.  It
        is never a substitute for the playout buffer: identity must be stable
        across two observations and the measured motion/scale must be sane.
        """
        if previous is None or previous_pts is None:
            return None, "STATE_EXTRAP_NO_PREVIOUS"
        sample_delta = cls._signed_pts_delta(selected_pts, previous_pts)
        offset = cls._signed_pts_delta(video_pts, selected_pts)
        if sample_delta <= 0 or sample_delta > 2000:
            return None, "STATE_EXTRAP_SAMPLE_INVALID"
        if offset <= 0 or offset > max_offset_ms:
            return None, "STATE_TOO_OLD"
        selected_targets = {target.track_id: target for target in selected.targets}
        previous_targets = {target.track_id: target for target in previous.targets}
        if (not selected_targets or set(selected_targets) != set(previous_targets)):
            return None, "STATE_EXTRAP_TRACK_SWITCH"
        ratio = float(offset) / float(sample_delta)
        predicted = []
        for track_id, target in selected_targets.items():
            old = previous_targets[track_id]
            if old.class_id != target.class_id:
                return None, "STATE_EXTRAP_CLASS_SWITCH"
            old_width = max(1.0, float(old.right - old.left))
            old_height = max(1.0, float(old.bottom - old.top))
            width = max(1.0, float(target.right - target.left))
            height = max(1.0, float(target.bottom - target.top))
            if (width / old_width < 0.5 or width / old_width > 2.0 or
                    height / old_height < 0.5 or height / old_height > 2.0):
                return None, "STATE_EXTRAP_SCALE"
            old_cx = (old.left + old.right) * 0.5
            old_cy = (old.top + old.bottom) * 0.5
            current_cx = (target.left + target.right) * 0.5
            current_cy = (target.top + target.bottom) * 0.5
            if (abs(current_cx - old_cx) > 1.5 * max(old_width, width) or
                    abs(current_cy - old_cy) > 1.5 * max(old_height, height)):
                return None, "STATE_EXTRAP_VELOCITY"
            predicted_cx = current_cx + (current_cx - old_cx) * ratio
            predicted_cy = current_cy + (current_cy - old_cy) * ratio
            projected_width = int(round(width))
            projected_height = int(round(height))
            if (projected_width <= 0 or projected_height <= 0 or
                    projected_width > selected.source_width or
                    projected_height > selected.source_height):
                return None, "STATE_EXTRAP_BOUNDS"
            left = int(round(predicted_cx - projected_width * 0.5))
            top = int(round(predicted_cy - projected_height * 0.5))
            left = max(0, min(selected.source_width - projected_width, left))
            top = max(0, min(selected.source_height - projected_height, top))
            predicted.append(dataclasses.replace(
                target, left=left, top=top,
                right=left + projected_width, bottom=top + projected_height))
        return dataclasses.replace(selected, targets=tuple(predicted)), "STATE_EXTRAP"

    def scene_synced(self, video_rtp_timestamp: Optional[int],
                     frame_arrival_time: Optional[float] = None,
                     now: Optional[float] = None) -> Tuple[
            Optional[rb.StateRecord], int, Dict[int, VisualReference], float,
            Optional[int]]:
        """Select a close semantic state or fail closed to the base video.

        A nearest state is not necessarily a usable state: while a large JPEG
        reference is paced, the newest state can be seconds behind the H.265
        frame.  Painting that old state is the floating-target failure.  Keep
        reporting the signed delta for the HUD.  A separately constrained
        one-frame extrapolation may bridge a small older-state gap; every
        other out-of-window state remains fail-closed.
        """
        selection_now = time.monotonic() if now is None else now
        with self.lock:
            generation = -1 if self.state_generation is None else self.state_generation
            selected_state = None
            selected_updated = self.state_updated
            selected_delta = None
            self.last_selected_state_pts_ms = None
            self.last_state_delta_ms = None
            self.last_state_arrival_age_ms = None
            self.last_state_selected_age_ms = None
            self.last_state_extrapolated = False
            self.last_state_sequence = None
            self.last_state_frame_id = None
            self.last_state_reason = "STATE_NO_HISTORY"
            effective_video_rtp_timestamp = video_rtp_timestamp
            if video_rtp_timestamp is None:
                self.last_state_reason = "STATE_NO_VIDEO_PTS"
            elif not self.state_history:
                self.last_state_reason = "STATE_NO_HISTORY"
            else:
                candidates = [item for item in self.state_history
                              if self._state_item_values(item)[0] == generation]
                if candidates:
                    raw_selected = min(
                        candidates,
                        key=lambda item: abs(self._signed_pts_delta(
                            (self._state_item_values(item)[1] * 90) & 0xFFFFFFFF,
                            video_rtp_timestamp)))
                    _, raw_pts_ms, _, _, _, _ = self._state_item_values(raw_selected)
                    raw_delta_ticks = self._signed_pts_delta(
                        (raw_pts_ms * 90) & 0xFFFFFFFF, video_rtp_timestamp)
                    self._observe_pts_bias(
                        int(round(raw_delta_ticks / 90.0)), raw_pts_ms,
                        video_rtp_timestamp)
                    effective_video_rtp_timestamp = self._effective_video_timestamp(
                        video_rtp_timestamp)
                    # A future STATE beyond the direct gate cannot render this
                    # source frame.  Prefer a one-frame-old history item that
                    # can be safely extrapolated over an equally-near future
                    # item, otherwise packet reordering needlessly causes a
                    # fail-closed BASE frame.
                    candidate_deltas = [
                        (item, int(round(self._signed_pts_delta(
                            (self._state_item_values(item)[1] * 90) & 0xFFFFFFFF,
                            effective_video_rtp_timestamp) / 90.0)))
                        for item in candidates
                    ]
                    direct_candidates = [
                        pair for pair in candidate_deltas
                        if abs(pair[1]) <= self.max_sync_ms
                    ]
                    extrapolation_candidates = [
                        pair for pair in candidate_deltas
                        if -STATE_EXTRAPOLATION_MAX_MS <= pair[1] < -self.max_sync_ms
                    ]
                    if direct_candidates:
                        selected = min(
                            direct_candidates,
                            key=lambda pair: (abs(pair[1]), 0 if pair[1] <= 0 else 1))[0]
                    elif extrapolation_candidates:
                        # Values are negative here, so the largest is closest
                        # to the current video frame and has the least motion
                        # prediction error.
                        selected = max(extrapolation_candidates, key=lambda pair: pair[1])[0]
                    else:
                        selected = min(
                            candidate_deltas,
                            key=lambda pair: (abs(pair[1]), 0 if pair[1] < 0 else 1))[0]
                    (_, selected_pts_ms, selected_state, selected_updated,
                     selected_sequence, selected_frame_id) = self._state_item_values(selected)
                    self.last_selected_state_pts_ms = selected_pts_ms
                    self.last_state_sequence = selected_sequence
                    self.last_state_frame_id = selected_frame_id
                    self.last_state_selected_age_ms = int(round(
                        max(0.0, selection_now - selected_updated) * 1000.0))
                    if frame_arrival_time is not None:
                        self.last_state_arrival_age_ms = int(round(
                            (frame_arrival_time - selected_updated) * 1000.0))
                    delta_ticks = self._signed_pts_delta(
                        (selected_pts_ms * 90) & 0xFFFFFFFF,
                        effective_video_rtp_timestamp)
                    selected_delta = int(round(delta_ticks / 90.0))
                    self.last_state_delta_ms = selected_delta
                    if abs(selected_delta) <= self.max_sync_ms:
                        # Keep the small within-gate motion prediction that
                        # removes a half-frame detector lag, but a real STATE
                        # remains valid when prediction cannot be proven safe.
                        if selected_delta < 0 and selected_state.targets:
                            older = [item for item in candidates
                                     if self._signed_pts_delta(
                                         selected_pts_ms,
                                         self._state_item_values(item)[1]) > 0]
                            previous = None if not older else min(
                                older,
                                key=lambda item: self._signed_pts_delta(
                                    selected_pts_ms, self._state_item_values(item)[1]))
                            predicted, _ = self._extrapolate_state(
                                selected_state, selected_pts_ms,
                                None if previous is None else self._state_item_values(previous)[2],
                                None if previous is None else self._state_item_values(previous)[1],
                                selected_pts_ms - selected_delta,
                                self.max_sync_ms)
                            if predicted is not None:
                                selected_state = predicted
                                self.last_state_extrapolated = True
                                self.last_state_reason = "STATE_EXTRAP"
                            else:
                                self.last_state_reason = "STATE_OK"
                        else:
                            self.last_state_reason = (
                                "STATE_TARGET_EMPTY" if not selected_state.targets else "STATE_OK")
                    elif selected_delta < -self.max_sync_ms and \
                            abs(selected_delta) <= STATE_EXTRAPOLATION_MAX_MS:
                        older = [item for item in candidates
                                 if self._signed_pts_delta(
                                     selected_pts_ms,
                                     self._state_item_values(item)[1]) > 0]
                        previous = None if not older else min(
                            older,
                            key=lambda item: self._signed_pts_delta(
                                selected_pts_ms, self._state_item_values(item)[1]))
                        predicted, reason = self._extrapolate_state(
                            selected_state, selected_pts_ms,
                            None if previous is None else self._state_item_values(previous)[2],
                            None if previous is None else self._state_item_values(previous)[1],
                            selected_pts_ms - selected_delta,
                            STATE_EXTRAPOLATION_MAX_MS)
                        if predicted is not None:
                            selected_state = predicted
                            self.last_state_extrapolated = True
                            self.last_state_reason = reason
                        else:
                            selected_state = None
                            self.last_state_reason = reason
                    else:
                        selected_state = None
                        self.last_state_reason = (
                            "STATE_TOO_OLD" if selected_delta < 0 else "STATE_TOO_FUTURE")
                else:
                    self.last_state_reason = "STATE_GEN_MISMATCH"
            if selected_state is None:
                if (video_rtp_timestamp is not None and selected_delta is not None and
                        self._last_sync_drop_timestamp != video_rtp_timestamp):
                    self.sync_drops += 1
                    self._last_sync_drop_timestamp = video_rtp_timestamp
                return None, generation, {}, selected_updated, selected_delta
            self._last_sync_drop_timestamp = None
            # Resolve the exact reference generation named by the selected
            # STATE.  A newer local PATCH may already exist for the same
            # track, but it belongs to a later source frame and must neither
            # replace nor suppress this frame's crop.
            references = {}
            if selected_state is not None:
                for target in selected_state.targets:
                    if target.reference_generation == 0:
                        continue
                    reference = self.reference_history.get(
                        (target.track_id, target.reference_generation))
                    if reference is not None:
                        references[target.track_id] = reference
            # Reject only future references here.  The compositor owns the
            # separate content hard-age decision so it can expose AGEDROP
            # instead of silently turning an expired local-ready reference
            # into an indistinguishable missing reference.
            references = {
                track_id: reference for track_id, reference in references.items()
                if self._signed_pts_delta(
                    (reference.pts_ms * 90) & 0xFFFFFFFF,
                    effective_video_rtp_timestamp) <= self.reference_future_max_ms * 90
            }
            return (selected_state, generation, references,
                    selected_updated, selected_delta)

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            self._trim(now)
            reference_ages = [now - value.received_at for value in self.references.values()]
            packet_count = len(self.samples)
            payload_total = sum(item[1] for item in self.samples)
            return {
                "rtp_kbps": payload_total * 8.0 / 1000.0,
                "wire_kbps": sum(item[2] for item in self.samples) * 8.0 / 1000.0,
                "pps": float(packet_count),
                "state_pps": float(sum(item[3] == rb.STATE for item in self.samples)),
                "data_pps": float(sum(item[3] == rb.PATCH_DATA for item in self.samples)),
                "parity_pps": float(sum(item[3] == rb.PATCH_PARITY for item in self.samples)),
                "packet_last_bytes": self.samples[-1][1] if self.samples else 0,
                "packet_avg_bytes": payload_total / packet_count if packet_count else 0.0,
                "packet_max_bytes": max((item[1] for item in self.samples), default=0),
                "packets": self.packets,
                "lost": self.lost,
                "reordered": self.reordered,
                "invalid": self.invalid,
                "state_packets": self.state_packets,
                "references": self.complete_references,
                "active_references": len(self.references),
                "reference_age": max(reference_ages) if reference_ages else None,
                "parity_recovered": self.parity_recovered,
                "incomplete": len(self.assembler.transfers),
                "expired": self.assembler.expired,
                "inconsistent": self.assembler.inconsistent,
                "sync_drops": self.sync_drops,
                "reference_generation_drops": self.reference_generation_drops,
                "reference_cache_ttl_ms": self.reference_cache_ttl_ms,
                "reference_future_max_ms": self.reference_future_max_ms,
                "max_sync_ms": self.max_sync_ms,
                "pts_bias_ms": self.pts_bias_ms,
                "pts_bias_samples": len(self.pts_bias_samples),
                "raw_sync_ms": self.raw_sync_ms,
                "state_delta_ms": self.last_state_delta_ms,
                "state_pts_ms": self.last_selected_state_pts_ms,
                "state_age": None if not self.state_updated else now - self.state_updated,
                "state_arrival_age_ms": self.last_state_arrival_age_ms,
                "state_selected_age_ms": self.last_state_selected_age_ms,
                "state_extrapolated": self.last_state_extrapolated,
                "state_reason": self.last_state_reason,
                "state_sequence": self.last_state_sequence,
                "state_frame_id": self.last_state_frame_id,
                "local_reference_generations": dict(self.reference_generations),
                "state_reference_generations": dict(self.state_reference_generations),
            }

    def start(self, stopping: threading.Event) -> None:
        if self.thread is not None:
            return
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        receiver.bind(("0.0.0.0", self.port))
        receiver.settimeout(0.2)
        self.socket = receiver

        def loop() -> None:
            while not stopping.is_set():
                try:
                    data, _ = receiver.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                self.on_datagram(data)

        self.thread = threading.Thread(target=loop, name="rebuild-rx", daemon=True)
        self.thread.start()

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


class SuperResolver:
    """Small-ROI Real-ESRGAN; full-frame work always stays Lanczos4."""

    def __init__(self, enabled: bool = True, model_path: Optional[str] = None,
                 cpu_threads: int = 2, scale: int = 2,
                 input_side: int = SR_MODEL_INPUT_SIDE) -> None:
        if input_side < 16 or input_side % max(1, scale) != 0:
            raise ValueError("SR input side must be >=16 and divisible by scale")
        self.session = None
        self.scale = scale
        self.input_side = input_side
        self.model_name = "Lanczos4"
        self._active_provider: Optional[str] = None
        self._reported_error = False
        self._ready = False
        self.warmup_ms: Optional[float] = None
        self._model_path = model_path
        self._cpu_threads = cpu_threads
        self._load_lock = threading.Lock()
        self._load_started = False
        if enabled:
            self.enable()

    def enable(self) -> None:
        with self._load_lock:
            if self._load_started:
                return
            self._load_started = True
        self._load(self._model_path, self._cpu_threads)

    def enable_async(self) -> None:
        with self._load_lock:
            if self._load_started:
                return
            self._load_started = True
        thread = threading.Thread(
            target=self._load, args=(self._model_path, self._cpu_threads),
            name="rebuild-esrgan-load", daemon=True)
        thread.start()

    @staticmethod
    def _ordered_providers(available):
        """Return the fastest usable ORT provider first, retaining CPU fallback."""
        preferred = ("CUDAExecutionProvider", "DmlExecutionProvider",
                     "CoreMLExecutionProvider", "CPUExecutionProvider")
        return [provider for provider in preferred if provider in available]

    @staticmethod
    def _preload_cuda_dlls(ort, providers) -> None:
        """Make CUDA/cuDNN wheels discoverable before creating a CUDA session.

        onnxruntime-gpu can install the NVIDIA runtime DLLs below site-packages.
        Windows does not search those directories automatically, so ask ORT to
        load them explicitly.  CPU-only and older ORT installations remain
        valid because this is only called when CUDA is advertised.
        """
        if "CUDAExecutionProvider" not in providers:
            return
        SuperResolver._prepare_windows_cuda_dll_path()
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload(directory="")

    @staticmethod
    def _prepare_windows_cuda_dll_path() -> None:
        """Expose pip-installed NVIDIA sublibraries to cuDNN's late loader.

        ORT preloads the top-level CUDA/cuDNN DLLs, but cuDNN 9 loads its
        engine DLLs lazily.  Those engine DLLs live in the ``nvidia/*/bin``
        wheels and Windows' normal search path does not include them.  The
        change is process-local and only applies when a CUDA EP is selected.
        """
        if os.name != "nt":
            return
        try:
            roots = list(site.getsitepackages())
        except AttributeError:
            roots = []
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
        libraries = ("cuda_runtime", "cuda_nvrtc", "cublas", "cudnn",
                     "cufft", "curand", "nvjitlink")
        paths = []
        for root in roots:
            for library in libraries:
                directory = Path(root) / "nvidia" / library / "bin"
                if directory.is_dir():
                    paths.append(str(directory))
        if not paths:
            return
        current = os.environ.get("PATH", "")
        known = {os.path.normcase(os.path.normpath(entry))
                 for entry in current.split(os.pathsep) if entry}
        additions = []
        for directory in paths:
            normalized = os.path.normcase(os.path.normpath(directory))
            if normalized not in known:
                additions.append(directory)
                known.add(normalized)
        if additions:
            os.environ["PATH"] = os.pathsep.join(additions + ([current] if current else []))

    def _refresh_active_provider(self) -> str:
        if self.session is None:
            self._active_provider = None
            return "unknown"
        active = self.session.get_providers()
        provider = active[0] if active else "unknown"
        self._active_provider = provider
        self.model_name = "Real-ESRGAN/" + provider.replace("ExecutionProvider", "")
        return provider

    def _load(self, model_path: Optional[str], cpu_threads: int) -> None:
        if not model_path:
            candidates = (
                Path(__file__).resolve().parents[1] / "model" / "RealESRGAN_x2_dynamic.onnx",
                Path(__file__).resolve().parents[2] /
                "atk_yolov8_seg_cam_v2" / "model" / "RealESRGAN_x2_dynamic.onnx",
                Path(__file__).resolve().parents[2] / "RealESRGAN_x2plus.fp16.onnx",
            )
            model = next((candidate for candidate in candidates if candidate.exists()), None)
        else:
            model = Path(model_path)
        if model is None or not model.exists():
            return
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = self._ordered_providers(available)
            self._preload_cuda_dlls(ort, providers)
            options = ort.SessionOptions()
            if providers == ["CPUExecutionProvider"] and cpu_threads > 0:
                options.intra_op_num_threads = cpu_threads
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(
                str(model), sess_options=options, providers=providers or None)
            provider = self._refresh_active_provider()
            # Keep every rebuild crop on one pre-warmed CUDA shape.  The sender
            # preserves aspect ratio and its JPEG byte cap changes dimensions
            # frequently; letting those dimensions reach ORT made each new
            # shape pay a CUDA setup cost longer than the reference lifetime.
            warmup_started = time.monotonic()
            warmup = np.zeros((self.input_side, self.input_side, 3), dtype=np.uint8)
            self.upscale(warmup, (self.input_side * self.scale,
                                  self.input_side * self.scale))
            if self.session is None:
                return
            self.warmup_ms = (time.monotonic() - warmup_started) * 1000.0
            self._ready = True
            print(f"[Rebuild] loaded ROI model {model} ({provider}); "
                  f"warmup {self.warmup_ms:.0f} ms", flush=True)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"[Rebuild] Real-ESRGAN unavailable ({error}); ROI fallback is Lanczos4",
                  flush=True)
            self.session = None
            self._ready = False

    @property
    def available(self) -> bool:
        return self.session is not None and self._ready

    def upscale(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if self.session is None:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        try:
            source_height, source_width = image.shape[:2]
            if source_height <= 0 or source_width <= 0:
                raise ValueError("empty SR input")
            fit = min(1.0, self.input_side / float(max(source_height, source_width)))
            content_width = max(1, int(round(source_width * fit)))
            content_height = max(1, int(round(source_height * fit)))
            content = image
            if content_width != source_width or content_height != source_height:
                content = cv2.resize(
                    image, (content_width, content_height), interpolation=cv2.INTER_AREA)
            pad_left = (self.input_side - content_width) // 2
            pad_right = self.input_side - content_width - pad_left
            pad_top = (self.input_side - content_height) // 2
            pad_bottom = self.input_side - content_height - pad_top
            image = cv2.copyMakeBorder(
                content, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE)
            model_input = self.session.get_inputs()[0]
            dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
            tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(dtype) / dtype(255.0)
            tensor = tensor.transpose(2, 0, 1)[None, ...]
            provider_before = self._active_provider
            result = self.session.run(
                None, {model_input.name: tensor})[0]
            provider_after = self._refresh_active_provider()
            if provider_before and provider_after != provider_before:
                print(f"[Rebuild] Real-ESRGAN execution provider switched "
                      f"{provider_before} -> {provider_after}", flush=True)
            output = result[0].transpose(1, 2, 0)
            output = cv2.cvtColor(
                np.clip(output * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            output_scale_x = output.shape[1] / float(self.input_side)
            output_scale_y = output.shape[0] / float(self.input_side)
            crop_left = int(round(pad_left * output_scale_x))
            crop_top = int(round(pad_top * output_scale_y))
            crop_right = int(round((pad_left + content_width) * output_scale_x))
            crop_bottom = int(round((pad_top + content_height) * output_scale_y))
            output = output[crop_top:crop_bottom, crop_left:crop_right]
            if output.shape[1] != target_size[0] or output.shape[0] != target_size[1]:
                output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)
            return output
        except Exception as error:  # ORT backends expose provider-specific exceptions.
            if not self._reported_error:
                print(f"[Rebuild] Real-ESRGAN inference failed ({error}); using Lanczos4",
                      flush=True)
                self._reported_error = True
            self.session = None
            self._active_provider = None
            self._ready = False
            self.model_name = "Lanczos4 (SR failed)"
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


class RebuildComposer:
    def __init__(self, output_size: Tuple[int, int] = (640, 360),
                 resolver: Optional[SuperResolver] = None,
                 reference_cache_ttl_ms: int = REFERENCE_CACHE_TTL_MS,
                 reference_content_hard_max_age_ms: int =
                 REFERENCE_CONTENT_HARD_MAX_AGE_MS,
                 reference_future_max_ms: int = REFERENCE_FUTURE_MAX_MS,
                 draw_targets: bool = False,
                 debug_reference_dir: Optional[Path] = None,
                 debug_reference_max_samples: int = 100) -> None:
        if (reference_cache_ttl_ms <= 0 or
                reference_content_hard_max_age_ms <= 0 or
                reference_future_max_ms <= 0 or debug_reference_max_samples < 0):
            raise ValueError("reference cache/content/future limits must be positive")
        self.output_size = output_size
        self.resolver = resolver or SuperResolver(enabled=False)
        self.reference_cache_ttl_ms = reference_cache_ttl_ms
        self.reference_content_hard_max_age_ms = reference_content_hard_max_age_ms
        self.reference_future_max_ms = reference_future_max_ms
        self.draw_targets = draw_targets
        self.debug_reference_dir = (None if debug_reference_dir is None else
                                    Path(debug_reference_dir))
        self.debug_reference_max_samples = debug_reference_max_samples
        self._debug_reference_keys = set()
        self._debug_artifact_keys = set()
        self._debug_reference_error_reported = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebuild-roi-sr")
        self.future: Optional[Future] = None
        self.job_key: Optional[tuple] = None
        self.job_reference: Optional[VisualReference] = None
        self.pending_reference: Optional[VisualReference] = None
        self._known_references: Dict[Tuple[int, int], VisualReference] = {}
        self._latest_video_rtp_timestamp: Optional[int] = None
        self._latest_pts_bias_ms = 0
        self._latest_generation: Optional[int] = None
        self._active_track_ids: Optional[set] = None
        self._advertised_reference_generations: Dict[int, int] = {}
        self.cache: Dict[tuple, np.ndarray] = {}
        self.sr_jobs = 0
        self.sr_done = 0
        self.sr_stale = 0
        self.sr_expired_before_submit = 0
        self.sr_expired_after_compute = 0
        self.sr_future_drops = 0
        self.sr_future_waits = 0
        self.sr_invalid_drops = 0
        self.sr_pending_replaced = 0
        self.sr_cache_hits = 0
        self.sr_cache_misses = 0
        self.last_sr_lookup = "none"
        self.sr_last_ms = 0.0
        self.sr_compute_samples_ms: Deque[float] = collections.deque(maxlen=64)
        self.registration_drops = 0
        self.content_drops = 0
        self.no_reference_drops = 0
        self.state_drops = 0
        self.generation_drops = 0
        self.cache_drops = 0
        self.geom_invalid_drops = 0
        self.geom_scale_drops = 0
        self.geom_outside_drops = 0
        self.content_matches = 0
        self.age_drops = 0
        self.future_drops = 0
        self.timing_drops = 0
        self.reference_ready = 0
        self.last_refs_used = 0
        self.last_ref_age: Optional[float] = None
        self.last_ref_content_age_ms: Optional[int] = None
        self.last_reference_pts_ms: Optional[int] = None
        self.last_reference_generation: Optional[int] = None
        self.last_reference_kind = "NONE"
        self.last_reference_crop_src_width = 0
        self.last_reference_crop_src_height = 0
        self.last_reference_jpeg_width = 0
        self.last_reference_jpeg_height = 0
        self.last_reference_jpeg_quality = -1
        self.last_reference_jpeg_bytes = 0
        self.last_reference_jpeg_scale = 0.0
        self.last_reference_head_pixels_width = 0
        self.last_reference_head_pixels_height = 0
        self.last_state_reference_generation: Optional[int] = None
        self.last_track_id: Optional[int] = None
        self.last_match_score: Optional[float] = None
        self.last_drop_reason = "NONE"
        self.last_content_reason = "CONTENT_NOT_RUN"
        self.last_registration_x: Optional[int] = None
        self.last_registration_y: Optional[int] = None
        self.last_patch_width = 0
        self.last_patch_height = 0
        self.last_mask_width = 0
        self.last_mask_height = 0
        self.last_visible_ratio = 0.0
        self.last_geom_dx_ratio = 0.0
        self.last_geom_dy_ratio = 0.0
        self.last_geom_area_ratio = 0.0
        self.last_rebuild_percent = 0.0
        self.last_spatial = "BASE-LANCZOS"
        self.last_chroma_mode = "COLOR"
        self.last_sr_state = "MISS"
        self.new_reference_first_frames = 0
        self.new_reference_first_frame_sr_hits = 0
        self._seen_reference_frame_keys = set()
        self._smooth_boxes: Dict[tuple, Tuple[float, float, float, float]] = {}
        self._smooth_seen_at: Dict[tuple, float] = {}
        self._handover_references: Dict[Tuple[int, int], Tuple[VisualReference, int]] = {}
        self.sr_handover_frames = 0
        self.last_sr_handover = False
        self._smooth_generation: Optional[int] = None
        self._composed_source_sequence: Optional[int] = None
        self._composed_generation: Optional[int] = None
        self._composed_frame: Optional[np.ndarray] = None
        self._composed_spatial = "BASE-LANCZOS"
        self.composed_frames = 0

    @staticmethod
    def _reference_key(reference: VisualReference) -> tuple:
        return (reference.generation, reference.track_id,
                reference.reference_generation)

    def _debug_stem(self, reference: VisualReference) -> Optional[Path]:
        if (self.debug_reference_dir is None or
                self.debug_reference_max_samples <= 0):
            return None
        key = self._reference_key(reference)
        if key not in self._debug_reference_keys:
            if len(self._debug_reference_keys) >= self.debug_reference_max_samples:
                return None
            try:
                self.debug_reference_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                if not self._debug_reference_error_reported:
                    print(f"[Rebuild] debug reference directory unavailable: {error}",
                          flush=True)
                    self._debug_reference_error_reported = True
                return None
            self._debug_reference_keys.add(key)
        return self.debug_reference_dir / (
            f"track{reference.track_id}_gen{reference.reference_generation}_"
            f"{reference.reference_kind}")

    def _save_decoded_reference_debug(self, reference: VisualReference) -> None:
        self._save_reference_artifact(reference, "decoded", reference.image)
        self._save_reference_artifact(reference, "mask", reference.mask)

    def _save_reference_artifact(self, reference: Optional[VisualReference],
                                 suffix: str, image: np.ndarray) -> None:
        if reference is None or image is None or image.size == 0:
            return
        artifact_key = (self._reference_key(reference), suffix)
        if artifact_key in self._debug_artifact_keys:
            return
        stem = self._debug_stem(reference)
        if stem is not None:
            self._debug_artifact_keys.add(artifact_key)
            cv2.imwrite(str(stem) + f"_{suffix}.png", image)

    def update_timing(self, video_rtp_timestamp: Optional[int],
                      pts_bias_ms: int = 0, generation: Optional[int] = None,
                      active_track_ids: Optional[set] = None,
                      advertised_reference_generations: Optional[Dict[int, int]] = None
                      ) -> None:
        """Publish the immutable clock/scene context used by SR gates.

        The UI thread updates this before polling or submitting work.  A
        worker result is therefore checked against the newest known source
        frame before it can enter the SR cache.
        """
        if (generation is not None and self._latest_generation is not None and
                generation != self._latest_generation):
            # A profile rollover invalidates all reference identities from the
            # previous stream.  The bounded pixel cache is harmless but the
            # newest-reference index must not grow across generations.
            self._known_references.clear()
            self._handover_references.clear()
        self._latest_video_rtp_timestamp = video_rtp_timestamp
        self._latest_pts_bias_ms = pts_bias_ms
        self._latest_generation = generation
        self._active_track_ids = (None if active_track_ids is None else
                                  set(active_track_ids))
        self._advertised_reference_generations = dict(
            advertised_reference_generations or {})

    def _effective_video_timestamp(self) -> Optional[int]:
        if self._latest_video_rtp_timestamp is None:
            return None
        return ((self._latest_video_rtp_timestamp +
                 int(round(self._latest_pts_bias_ms * 90.0))) & 0xFFFFFFFF)

    def _content_age_ms(self, reference: VisualReference) -> Optional[int]:
        effective_video = self._effective_video_timestamp()
        if effective_video is None:
            return None
        age_ticks = RebuildReceiver._signed_pts_delta(
            effective_video, (reference.pts_ms * 90) & 0xFFFFFFFF)
        return int(round(age_ticks / 90.0))

    def _reference_identity_gate_reason(
            self, reference: VisualReference,
            allow_newer_than_advertised: bool = False) -> Optional[str]:
        if (self._latest_generation is not None and
                reference.generation != self._latest_generation):
            return "generation"
        if (self._active_track_ids is not None and
                reference.track_id not in self._active_track_ids):
            return "track"
        known = self._known_references.get(
            (reference.generation, reference.track_id))
        if (known is not None and
                reference.reference_generation != known.reference_generation and
                not RebuildReceiver._newer_u16(reference.reference_generation,
                                               known.reference_generation)):
            return "stale-reference"
        advertised = self._advertised_reference_generations.get(reference.track_id, 0)
        if advertised and reference.reference_generation != advertised:
            if (not allow_newer_than_advertised or
                    not RebuildReceiver._newer_u16(
                        reference.reference_generation, advertised)):
                return "advertised-reference"
        return None

    def _reference_compute_gate_reason(self, reference: VisualReference) -> Optional[str]:
        """Validate an SR candidate without applying the render-time future gate."""
        reason = self._reference_identity_gate_reason(
            reference, allow_newer_than_advertised=True)
        if reason is not None:
            return reason
        age_ms = self._content_age_ms(reference)
        # A missing video clock is not a reason to discard a useful local SR
        # result.  The render gate still refuses to paint until it has a PTS.
        if age_ms is not None and age_ms > self.reference_content_hard_max_age_ms:
            return "expired"
        return None

    def _reference_render_gate_reason(self, reference: VisualReference) -> Optional[str]:
        """Validate a reference for display, including the strict PTS gates."""
        reason = self._reference_identity_gate_reason(reference)
        if reason is not None:
            return reason
        age_ms = self._content_age_ms(reference)
        if age_ms is None:
            return "clock"
        if age_ms > self.reference_content_hard_max_age_ms:
            return "expired"
        if age_ms < -self.reference_future_max_ms:
            return "future"
        return None

    def _record_sr_rejection(self, reason: str, before_submit: bool) -> None:
        if reason == "expired":
            if before_submit:
                self.sr_expired_before_submit += 1
            else:
                self.sr_expired_after_compute += 1
        elif reason == "future":
            self.sr_future_drops += 1
        elif reason != "clock":
            self.sr_invalid_drops += 1

    def _record_drop(self, reason: str) -> None:
        self.last_drop_reason = reason
        if reason == "NOREF":
            self.no_reference_drops += 1
        elif reason == "STATE" or reason.startswith("STATE_"):
            self.state_drops += 1
        elif reason == "GEN":
            self.generation_drops += 1
        elif reason == "CACHE":
            self.cache_drops += 1
        elif reason == "GEOM_INVALID":
            self.geom_invalid_drops += 1
        elif reason == "GEOM_SCALE":
            self.geom_scale_drops += 1
        elif reason == "GEOM_OUTSIDE":
            self.geom_outside_drops += 1
        elif reason.startswith("CONTENT_"):
            self.content_drops += 1

    def _record_first_reference_frame(self, key: tuple) -> bool:
        if key in self._seen_reference_frame_keys:
            return False
        self._seen_reference_frame_keys.add(key)
        if len(self._seen_reference_frame_keys) > 128:
            self._seen_reference_frame_keys.pop()
        self.new_reference_first_frames += 1
        return True

    def _submit(self, reference: VisualReference) -> bool:
        """Submit exactly one model-native job.  Caller owns UI-thread access."""
        key = self._reference_key(reference)
        if not self.resolver.available or key in self.cache or self.future is not None:
            return False
        native_size = (
            max(2, int(round(reference.image.shape[1] * self.resolver.scale))),
            max(2, int(round(reference.image.shape[0] * self.resolver.scale))),
        )
        self.job_key = key
        self.job_reference = reference
        self.future = self.executor.submit(
            self._run_sr, reference.image.copy(), native_size)
        self.sr_jobs += 1
        return True

    def _run_sr(self, image: np.ndarray,
                native_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        started = time.monotonic()
        output = self.resolver.upscale(image, native_size)
        return output, (time.monotonic() - started) * 1000.0

    def _defer_latest(self, reference: VisualReference) -> None:
        """Keep at most one not-yet-submitted crop, preferring the newest."""
        if self.pending_reference is None:
            self.pending_reference = reference
            return
        pending = self.pending_reference
        if self._reference_key(reference) == self._reference_key(pending):
            return
        if reference.received_at >= pending.received_at:
            self.pending_reference = reference
            self.sr_pending_replaced += 1

    def _start_deferred(self) -> None:
        if self.future is not None or self.pending_reference is None:
            return
        reference = self.pending_reference
        key = self._reference_key(reference)
        if key in self.cache:
            self.pending_reference = None
            return
        reason = self._reference_compute_gate_reason(reference)
        if reason is not None:
            self.pending_reference = None
            self._record_sr_rejection(reason, before_submit=True)
            return
        if self._submit(reference):
            self.pending_reference = None

    def prefetch(self, reference: VisualReference) -> None:
        """Start SR as soon as a complete RB/1 crop reaches the PC.

        A reference is still rendered only after the normal safety gates pass.
        Prefetching decouples GPU latency from the final STATE/PTS decision and
        keeps the executor latest-only when two targets refresh together.
        """
        self._poll()
        self._save_decoded_reference_debug(reference)
        known = self._known_references.get((reference.generation, reference.track_id))
        if (known is None or reference.reference_generation == known.reference_generation or
                RebuildReceiver._newer_u16(reference.reference_generation,
                                           known.reference_generation)):
            self._known_references[(reference.generation, reference.track_id)] = reference
        key = self._reference_key(reference)
        if key in self.cache or key == self.job_key:
            return
        if not self.resolver.available:
            # Keep the newest local-ready candidate while model loading or
            # fallback initialization is in progress.  Lanczos remains the
            # immediate display path; the candidate can use SR once ready.
            self._defer_latest(reference)
            return
        reason = self._reference_compute_gate_reason(reference)
        if reason is not None:
            self._record_sr_rejection(reason, before_submit=True)
            return
        if self.future is not None:
            self._defer_latest(reference)
            return
        self._submit(reference)

    def prefetch_pending(self) -> None:
        """Retry a deferred reference after asynchronous model loading/work."""
        self._poll()
        self._start_deferred()

    def _poll(self) -> None:
        if self.future is None or not self.future.done():
            return
        future, key = self.future, self.job_key
        self.future = None
        self.job_key = None
        reference = self.job_reference
        self.job_reference = None
        try:
            image, compute_ms = future.result()
            self.sr_last_ms = compute_ms
            self.sr_compute_samples_ms.append(compute_ms)
        except Exception:
            self.sr_stale += 1
            self._start_deferred()
            return
        # Cache the model-native x2 result by reference identity, not by the
        # current detector-box geometry.  Box jitter changes the paste size on
        # almost every frame; including that size in the key used to launch a
        # new Real-ESRGAN job continuously and produced soft/sharp pulsing.
        # A stale result cannot overwrite a newer reference generation because
        # the generation is part of the key.
        if reference is None:
            self.sr_stale += 1
        else:
            reason = self._reference_compute_gate_reason(reference)
            if reason is not None:
                self.sr_stale += 1
                self._record_sr_rejection(reason, before_submit=False)
            else:
                self.cache[key] = image
                self.sr_done += 1
                self._save_reference_artifact(reference, "sr", image)
                if len(self.cache) > 8:
                    oldest = next(iter(self.cache))
                    if oldest != key:
                        self.cache.pop(oldest, None)
        # A later reference may have arrived while this one was running.  It
        # gets the next (and only) slot instead of waiting for a renderer tick
        # that could occur after its 1 s content-age deadline.
        self._start_deferred()

    def _assets(self, reference: VisualReference, size: Tuple[int, int],
               cache_keys: Optional[set] = None) -> Tuple[np.ndarray,
                                                           np.ndarray,
                                                           bool]:
        self.prefetch(reference)
        key = self._reference_key(reference)
        # A render observes one immutable SR-cache view.  A worker may finish
        # while this method is processing another target, but that result is
        # eligible only for the next source frame.
        native = (self.cache.get(key) if cache_keys is None or key in cache_keys
                  else None)
        enhanced = native is not None and self.resolver.available
        cache_view = self.cache if cache_keys is None else cache_keys
        if enhanced:
            self.sr_cache_hits += 1
        else:
            self.sr_cache_misses += 1
        cached = ",".join(
            f"{cached_key[1]}/{cached_key[2]}" for cached_key in cache_view)
        self.last_sr_lookup = (
            f"{key[1]}/{key[2]}:{'hit' if enhanced else 'miss'}"
            f"[{cached or '-'}]")
        if native is not None:
            patch = cv2.resize(native, size, interpolation=cv2.INTER_LANCZOS4)
        else:
            patch = cv2.resize(reference.image, size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(reference.mask, size, interpolation=cv2.INTER_LINEAR)
        return patch, mask, enhanced

    def _handover_reference(self, current: VisualReference,
                            target: rb.TargetState, generation: int,
                            now: float, cache_keys: set) -> Optional[VisualReference]:
        """Return a still-safe old ESRGAN reference while a new one warms up."""
        previous = self._handover_references.get((generation, target.track_id))
        if previous is None:
            return None
        reference, class_id = previous
        if (class_id != target.class_id or reference.generation != generation or
                reference.track_id != target.track_id or
                reference.reference_generation == current.reference_generation):
            return None
        if (now - reference.received_at > self.reference_cache_ttl_ms / 1000.0 or
                not self.resolver.available or
                self._reference_key(reference) not in cache_keys):
            return None
        content_age_ms = self._content_age_ms(reference)
        if (content_age_ms is None or
                content_age_ms > self.reference_content_hard_max_age_ms or
                content_age_ms < -self.reference_future_max_ms):
            return None
        return reference

    @staticmethod
    def _nearly_grayscale(image: np.ndarray) -> bool:
        """Detect a neutral-chroma base without mistaking dim colour for gray."""
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            return True
        sample = image[::max(1, image.shape[0] // 90),
                       ::max(1, image.shape[1] // 160)].astype(np.int16)
        spread = sample.max(axis=2) - sample.min(axis=2)
        return float(np.percentile(spread, 95)) <= 3.0

    @staticmethod
    def _blend(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
               x: int, y: int, monochrome: bool = False,
               feather_px: Optional[float] = None) -> None:
        height, width = patch.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + width), min(canvas.shape[0], y + height)
        if x1 <= x0 or y1 <= y0:
            return
        patch_roi = patch[y0 - y:y1 - y, x0 - x:x1 - x]
        if monochrome:
            patch_roi = cv2.cvtColor(patch_roi, cv2.COLOR_BGR2GRAY)
            patch_roi = cv2.cvtColor(patch_roi, cv2.COLOR_GRAY2BGR)
        patch_roi = patch_roi.astype(np.float32)
        mask_roi = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        # Use the binary segmentation mask as the ownership decision.  A
        # distance-transform feather keeps the interior at alpha=1 while
        # adapting its radius to the 64x64 mask's output scale.  A fixed
        # four-pixel radius made the quantisation step visible at some target
        # sizes; the old Gaussian*0.84 blend also retained stale base pixels in
        # the object core, producing a visible double image after registration.
        binary = (mask_roi >= 128).astype(np.uint8)
        if not np.any(binary):
            return
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        if feather_px is None:
            feather_px = 4.0
        feather_px = max(2.0, min(10.0, float(feather_px)))
        alpha = np.clip(distance / feather_px, 0.0, 1.0)[:, :, None]
        base = canvas[y0:y1, x0:x1].astype(np.float32)
        selected = alpha[:, :, 0] > 0.5
        if np.any(selected):
            base_mean = base[selected].mean(axis=0)
            patch_mean = patch_roi[selected].mean(axis=0)
            shift = np.clip(base_mean - patch_mean, -20.0, 20.0)
            patch_roi = np.clip(patch_roi + shift, 0.0, 255.0)
        canvas[y0:y1, x0:x1] = (
            patch_roi * alpha + base * (1.0 - alpha)).astype(np.uint8)

    @staticmethod
    def _registration_template(reference: VisualReference,
                               size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create the fixed raw-JPEG template used for content registration.

        This deliberately does not read the SR cache.  The base frame is a
        compressed image domain, so Real-ESRGAN detail must not alter the
        identity decision or its correlation score from one source frame to
        the next.
        """
        image = cv2.resize(reference.image, size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(reference.mask, size, interpolation=cv2.INTER_LINEAR)
        return image, mask

    @staticmethod
    def _content_register(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
                          x: int, y: int, search_radius: int = 8,
                          minimum_score: float = 0.32) -> ContentRegistrationResult:
        """Align a stable, clipped raw template to the current base image.

        Partial image-edge visibility is normal for a crop carrying margin.
        Registration therefore works in the visible template region and only
        fails closed when too little foreground remains or the correlation is
        genuinely unsafe.
        """
        if (patch.ndim != 3 or canvas.ndim != 3 or mask.ndim != 2 or
                patch.shape[:2] != mask.shape[:2]):
            return ContentRegistrationResult(
                None, None, True, "CONTENT_SHAPE", 0.0)
        height, width = patch.shape[:2]
        if width < 4 or height < 4 or canvas.shape[0] < 3 or canvas.shape[1] < 3:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_TEMPLATE_INVALID", 0.0)
        binary = (mask >= 128).astype(np.uint8)
        foreground_total = int(np.count_nonzero(binary))
        if foreground_total < 16:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_TEMPLATE_INVALID", 0.0)
        patch_left = max(0, -x)
        patch_top = max(0, -y)
        patch_right = min(width, canvas.shape[1] - x)
        patch_bottom = min(height, canvas.shape[0] - y)
        if patch_right <= patch_left or patch_bottom <= patch_top:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_PATCH_TOO_LARGE", 0.0)
        visible_foreground = int(np.count_nonzero(
            binary[patch_top:patch_bottom, patch_left:patch_right]))
        visible_ratio = visible_foreground / float(foreground_total)
        if visible_foreground < 16 or visible_ratio < 0.10:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_PATCH_TOO_LARGE", visible_ratio)
        ys, xs = np.where(binary > 0)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        margin = max(1, min(4, int(round(min(width, height) * 0.04))))
        left = max(patch_left, left - margin)
        top = max(patch_top, top - margin)
        right = min(patch_right, right + margin)
        bottom = min(patch_bottom, bottom + margin)
        if right - left < 3 or bottom - top < 3:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_SEARCH_INVALID", visible_ratio)
        template = cv2.cvtColor(patch[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        template = cv2.GaussianBlur(template, (3, 3), 0)
        if float(template.std()) < 2.0:
            return ContentRegistrationResult(
                (x, y), None, False, "CONTENT_LOW_TEXTURE", visible_ratio)

        predicted_x = x + left
        predicted_y = y + top
        max_x = canvas.shape[1] - template.shape[1]
        max_y = canvas.shape[0] - template.shape[0]
        if max_x < 0 or max_y < 0:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_SEARCH_INVALID", visible_ratio)
        candidate_left = max(0, min(max_x, predicted_x - search_radius))
        candidate_top = max(0, min(max_y, predicted_y - search_radius))
        candidate_right = max(0, min(max_x, predicted_x + search_radius))
        candidate_bottom = max(0, min(max_y, predicted_y + search_radius))
        search_right = candidate_right + template.shape[1]
        search_bottom = candidate_bottom + template.shape[0]
        search = cv2.cvtColor(
            canvas[candidate_top:search_bottom, candidate_left:search_right],
            cv2.COLOR_BGR2GRAY)
        search = cv2.GaussianBlur(search, (3, 3), 0)
        if (search.shape[1] < template.shape[1] or
                search.shape[0] < template.shape[0]):
            return ContentRegistrationResult(
                None, None, True, "CONTENT_SEARCH_INVALID", visible_ratio)
        if float(search.std()) < 1.0:
            return ContentRegistrationResult(
                (x, y), None, False, "CONTENT_LOW_TEXTURE", visible_ratio)
        try:
            correlation = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return ContentRegistrationResult(
                None, None, True, "CONTENT_CV_ERROR", visible_ratio)
        _, score, _, location = cv2.minMaxLoc(correlation)
        score = float(score)
        corrected = (candidate_left + int(location[0]) - left,
                     candidate_top + int(location[1]) - top)
        if score < minimum_score:
            return ContentRegistrationResult(
                None, score, True, "CONTENT_LOW_SCORE", visible_ratio)
        return ContentRegistrationResult(
            corrected, score, True, "CONTENT_OK", visible_ratio)

    def _smooth_target(self, target: rb.TargetState, generation: int,
                       now: float) -> rb.TargetState:
        """EMA detector boxes while resetting on a real track jump."""
        key = (generation, target.track_id)
        current = tuple(float(value) for value in
                        (target.left, target.top, target.right, target.bottom))
        previous = self._smooth_boxes.get(key)
        if previous is not None and now - self._smooth_seen_at.get(key, now) > 1.0:
            previous = None
        if previous is None:
            smoothed = current
        else:
            previous_width = max(1.0, previous[2] - previous[0])
            previous_height = max(1.0, previous[3] - previous[1])
            current_cx = (current[0] + current[2]) * 0.5
            current_cy = (current[1] + current[3]) * 0.5
            previous_cx = (previous[0] + previous[2]) * 0.5
            previous_cy = (previous[1] + previous[3]) * 0.5
            current_width = max(1.0, current[2] - current[0])
            current_height = max(1.0, current[3] - current[1])
            if (abs(current_cx - previous_cx) > 0.60 * previous_width or
                    abs(current_cy - previous_cy) > 0.60 * previous_height or
                    current_width / previous_width < 0.5 or
                    current_width / previous_width > 2.0 or
                    current_height / previous_height < 0.5 or
                    current_height / previous_height > 2.0):
                smoothed = current
            else:
                alpha = 0.45
                smoothed = tuple(alpha * now_value + (1.0 - alpha) * old_value
                                 for now_value, old_value in zip(current, previous))
        self._smooth_boxes[key] = smoothed
        self._smooth_seen_at[key] = now
        rounded = tuple(int(round(value)) for value in smoothed)
        left, top, right, bottom = rounded
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        return dataclasses.replace(target, left=left, top=top,
                                   right=right, bottom=bottom)

    @staticmethod
    def _registration_details(
            reference: VisualReference, target: rb.TargetState,
            scale_x: float, scale_y: float,
            output_size: Optional[Tuple[int, int]] = None) -> RegistrationResult:
        """Predict a crop position and report geometry failures separately.

        Detector motion is an input to the reference-to-current mapping, not a
        failure condition.  The content matcher below remains the identity
        check after this geometric prediction.
        """
        try:
            ref_left, ref_top, ref_right, ref_bottom = (
                float(value) for value in reference.reference_bbox)
            cur_left, cur_top, cur_right, cur_bottom = (
                float(value) for value in
                (target.left, target.top, target.right, target.bottom))
            crop_left, crop_top, crop_right, crop_bottom = (
                float(value) for value in reference.crop)
        except (TypeError, ValueError):
            return RegistrationResult(None, "GEOM_INVALID", 0.0, 0.0, 0.0)
        values = (ref_left, ref_top, ref_right, ref_bottom, cur_left, cur_top,
                  cur_right, cur_bottom, crop_left, crop_top, crop_right,
                  crop_bottom, scale_x, scale_y)
        if not all(math.isfinite(value) for value in values):
            return RegistrationResult(None, "GEOM_INVALID", 0.0, 0.0, 0.0)
        ref_width = ref_right - ref_left
        ref_height = ref_bottom - ref_top
        cur_width = cur_right - cur_left
        cur_height = cur_bottom - cur_top
        if min(ref_width, ref_height, cur_width, cur_height) <= 0 or \
                scale_x <= 0 or scale_y <= 0:
            return RegistrationResult(None, "GEOM_INVALID", 0.0, 0.0, 0.0)
        area_ratio = (cur_width * cur_height) / (ref_width * ref_height)
        width_ratio = cur_width / ref_width
        height_ratio = cur_height / ref_height
        dx_ratio = abs((cur_left + cur_right - ref_left - ref_right) * 0.5) / \
            max(ref_width, cur_width)
        dy_ratio = abs((cur_top + cur_bottom - ref_top - ref_bottom) * 0.5) / \
            max(ref_height, cur_height)
        if (width_ratio < REGISTRATION_MIN_SIDE_RATIO or
                width_ratio > REGISTRATION_MAX_SIDE_RATIO or
                height_ratio < REGISTRATION_MIN_SIDE_RATIO or
                height_ratio > REGISTRATION_MAX_SIDE_RATIO or
                area_ratio < REGISTRATION_MIN_AREA_RATIO or
                area_ratio > REGISTRATION_MAX_AREA_RATIO):
            return RegistrationResult(None, "GEOM_SCALE", dx_ratio, dy_ratio, area_ratio)
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            return RegistrationResult(None, "GEOM_INVALID", dx_ratio, dy_ratio, area_ratio)
        isotropic_scale = math.sqrt(area_ratio)
        ref_center_x = (ref_left + ref_right) * 0.5
        ref_center_y = (ref_top + ref_bottom) * 0.5
        cur_center_x = (cur_left + cur_right) * 0.5
        cur_center_y = (cur_top + cur_bottom) * 0.5
        projected = (
            (cur_center_x + (crop_left - ref_center_x) * isotropic_scale) * scale_x,
            (cur_center_y + (crop_top - ref_center_y) * isotropic_scale) * scale_y,
            (cur_center_x + (crop_right - ref_center_x) * isotropic_scale) * scale_x,
            (cur_center_y + (crop_bottom - ref_center_y) * isotropic_scale) * scale_y,
        )
        if not all(math.isfinite(value) for value in projected):
            return RegistrationResult(None, "GEOM_INVALID", dx_ratio, dy_ratio, area_ratio)
        x0, y0, x1, y1 = (int(round(value)) for value in projected)
        if x1 <= x0 or y1 <= y0:
            return RegistrationResult(None, "GEOM_INVALID", dx_ratio, dy_ratio, area_ratio)
        if output_size is not None:
            output_width, output_height = output_size
            if (x1 <= 0 or y1 <= 0 or x0 >= output_width or
                    y0 >= output_height):
                return RegistrationResult(None, "GEOM_OUTSIDE", dx_ratio, dy_ratio, area_ratio)
        return RegistrationResult((x0, y0, x1, y1), "NONE",
                                  dx_ratio, dy_ratio, area_ratio)

    @staticmethod
    def _registered_crop(reference: VisualReference, target: rb.TargetState,
                         scale_x: float, scale_y: float) -> Optional[Tuple[int, int, int, int]]:
        return RebuildComposer._registration_details(
            reference, target, scale_x, scale_y).crop

    def render(self, base: np.ndarray, state: Optional[rb.StateRecord], generation: int,
               references: Dict[int, VisualReference], now: Optional[float] = None,
               video_rtp_timestamp: Optional[int] = None,
               pts_bias_ms: int = 0,
               source_sequence: Optional[int] = None,
               state_reason: str = "STATE_NO_HISTORY") -> Tuple[np.ndarray, str]:
        now = time.monotonic() if now is None else now
        active_track_ids = (set() if state is None else
                            {target.track_id for target in state.targets})
        advertised_generations = ({} if state is None else {
            target.track_id: target.reference_generation for target in state.targets
        })
        self.update_timing(
            video_rtp_timestamp, pts_bias_ms, generation, active_track_ids,
            advertised_generations)
        self._poll()
        # Direct users of RebuildComposer (tests or an alternate UI) may not
        # use RebuildReceiver.take_completed_references().  Start any supplied
        # crop before inspecting PTS so this fallback has the same early-launch
        # property, while render safety remains unchanged below.
        for reference in references.values():
            self.prefetch(reference)
        render_cache_keys = set(self.cache)
        if (source_sequence is not None and
                self._composed_source_sequence == source_sequence and
                self._composed_generation == generation and
                self._composed_frame is not None):
            # PATCH, STATE, and SR callbacks may have changed next-frame state,
            # but an already composed source frame is immutable until its
            # sequence advances.
            return self._composed_frame, self._composed_spatial
        output_width, output_height = self.output_size
        if state is not None and state.output_width > 0 and state.output_height > 0:
            output_width, output_height = state.output_width, state.output_height
        canvas = cv2.resize(base, (output_width, output_height),
                            interpolation=cv2.INTER_LANCZOS4)
        monochrome = self._nearly_grayscale(canvas)
        self.last_chroma_mode = "MONO" if monochrome else "COLOR"
        rebuilt_pixels = np.zeros((output_height, output_width), dtype=np.uint8)
        self.reference_ready = 0
        self.last_refs_used = 0
        self.last_ref_age = None
        self.last_ref_content_age_ms = None
        self.last_reference_pts_ms = None
        self.last_reference_generation = None
        self.last_reference_kind = "NONE"
        self.last_reference_crop_src_width = 0
        self.last_reference_crop_src_height = 0
        self.last_reference_jpeg_width = 0
        self.last_reference_jpeg_height = 0
        self.last_reference_jpeg_quality = -1
        self.last_reference_jpeg_bytes = 0
        self.last_reference_jpeg_scale = 0.0
        self.last_reference_head_pixels_width = 0
        self.last_reference_head_pixels_height = 0
        self.last_state_reference_generation = None
        self.last_track_id = None
        self.last_match_score = None
        self.last_rebuild_percent = 0.0
        self.last_drop_reason = "NONE"
        self.last_content_reason = "CONTENT_NOT_RUN"
        self.last_registration_x = None
        self.last_registration_y = None
        self.last_patch_width = 0
        self.last_patch_height = 0
        self.last_mask_width = 0
        self.last_mask_height = 0
        self.last_visible_ratio = 0.0
        self.last_geom_dx_ratio = 0.0
        self.last_geom_dy_ratio = 0.0
        self.last_geom_area_ratio = 0.0
        self.last_sr_state = "MISS"
        self.last_sr_handover = False
        used_sr = False
        effective_video_rtp_timestamp = None
        if video_rtp_timestamp is not None:
            effective_video_rtp_timestamp = (
                video_rtp_timestamp + int(round(pts_bias_ms * 90.0))) & 0xFFFFFFFF
        if state is not None and state.source_width > 0 and state.source_height > 0:
            scale_x = output_width / float(state.source_width)
            scale_y = output_height / float(state.source_height)
            if self._smooth_generation != generation:
                self._smooth_boxes.clear()
                self._smooth_seen_at.clear()
                self._smooth_generation = generation
            if not state.targets:
                self._record_drop(
                    state_reason if state_reason.startswith("STATE_") else
                    "STATE_TARGET_EMPTY")
            for raw_target in state.targets:
                target = self._smooth_target(raw_target, generation, now)
                self.last_track_id = target.track_id
                self.last_state_reference_generation = target.reference_generation
                reference = references.get(target.track_id)
                if reference is None:
                    self._record_drop(
                        "STATE_TARGET_MISSING" if target.reference_generation else "NOREF")
                    continue
                if (reference.generation != generation or
                        (target.reference_generation and
                         reference.reference_generation != target.reference_generation)):
                    self._record_drop("GEN")
                    continue
                self.reference_ready += 1
                if now - reference.received_at > self.reference_cache_ttl_ms / 1000.0:
                    self._record_drop("CACHE")
                    continue
                self.last_ref_age = max(0.0, now - reference.received_at)
                self.last_reference_pts_ms = reference.pts_ms
                self.last_reference_generation = reference.reference_generation
                self.last_reference_kind = reference.reference_kind
                self.last_reference_crop_src_width = max(
                    0, reference.crop[2] - reference.crop[0])
                self.last_reference_crop_src_height = max(
                    0, reference.crop[3] - reference.crop[1])
                self.last_reference_jpeg_width = reference.jpeg_width or reference.image.shape[1]
                self.last_reference_jpeg_height = reference.jpeg_height or reference.image.shape[0]
                self.last_reference_jpeg_bytes = reference.jpeg_bytes
                crop_width = max(1, self.last_reference_crop_src_width)
                crop_height = max(1, self.last_reference_crop_src_height)
                self.last_reference_jpeg_scale = min(
                    self.last_reference_jpeg_width / float(crop_width),
                    self.last_reference_jpeg_height / float(crop_height))
                if reference.reference_kind == "HEAD":
                    self.last_reference_head_pixels_width = self.last_reference_jpeg_width
                    self.last_reference_head_pixels_height = self.last_reference_jpeg_height
                if effective_video_rtp_timestamp is None:
                    self.timing_drops += 1
                    self._record_drop("STATE_NO_VIDEO_PTS")
                    continue
                content_age_ticks = RebuildReceiver._signed_pts_delta(
                    effective_video_rtp_timestamp,
                    (reference.pts_ms * 90) & 0xFFFFFFFF)
                content_age_ms = int(round(content_age_ticks / 90.0))
                self.last_ref_content_age_ms = content_age_ms
                if content_age_ms > self.reference_content_hard_max_age_ms:
                    self.age_drops += 1
                    self._record_drop("AGE")
                    continue
                if content_age_ms < -self.reference_future_max_ms:
                    self.future_drops += 1
                    self._record_drop("FUTURE")
                    continue
                selected_reference_key = self._reference_key(reference)
                first_reference_frame = self._record_first_reference_frame(
                    selected_reference_key)
                # New reference SR work is already queued above.  If it is
                # not ready on this source boundary, a previous same-track
                # ESRGAN result can bridge the handover only while its own
                # PTS age and geometry remain valid.
                using_handover = False
                if (not (self.resolver.available and
                         selected_reference_key in render_cache_keys)):
                    handover = self._handover_reference(
                        reference, target, generation, now, render_cache_keys)
                    if handover is not None:
                        reference = handover
                        using_handover = True
                        self.last_sr_handover = True
                        self.last_ref_age = max(0.0, now - reference.received_at)
                        self.last_reference_pts_ms = reference.pts_ms
                        self.last_reference_generation = reference.reference_generation
                        handover_age_ticks = RebuildReceiver._signed_pts_delta(
                            effective_video_rtp_timestamp,
                            (reference.pts_ms * 90) & 0xFFFFFFFF)
                        self.last_ref_content_age_ms = int(round(handover_age_ticks / 90.0))
                geometry = self._registration_details(
                    reference, target, scale_x, scale_y,
                    output_size=(output_width, output_height))
                self.last_geom_dx_ratio = geometry.dx_ratio
                self.last_geom_dy_ratio = geometry.dy_ratio
                self.last_geom_area_ratio = geometry.area_ratio
                if geometry.crop is None:
                    self.registration_drops += 1
                    self._record_drop(geometry.reason)
                    continue
                x, y, x1, y1 = geometry.crop
                crop_width = max(2, x1 - x)
                crop_height = max(2, y1 - y)
                registration_patch, registration_mask = self._registration_template(
                    reference, (crop_width, crop_height))
                self.last_patch_height, self.last_patch_width = registration_patch.shape[:2]
                self.last_mask_height, self.last_mask_width = registration_mask.shape[:2]
                registration = self._content_register(
                    canvas, registration_patch, registration_mask, x, y)
                self.last_content_reason = registration.reason
                self.last_visible_ratio = registration.visible_ratio
                if registration.score is not None:
                    self.last_match_score = registration.score if self.last_match_score is None else max(
                        self.last_match_score, registration.score)
                if registration.position is None:
                    self.registration_drops += 1
                    self._record_drop(registration.reason)
                    continue
                if registration.score is not None:
                    self.content_matches += 1
                x, y = registration.position
                self.last_registration_x = x
                self.last_registration_y = y
                patch, mask, enhanced = self._assets(
                    reference, (crop_width, crop_height), render_cache_keys)
                if first_reference_frame and not using_handover:
                    if enhanced:
                        self.new_reference_first_frame_sr_hits += 1
                    self.last_sr_state = "HIT" if enhanced else "MISS"
                feather_px = max(
                    3.0, min(10.0, max(
                        crop_width / float(max(1, reference.mask.shape[1])),
                        crop_height / float(max(1, reference.mask.shape[0])))))
                self._blend(canvas, patch, mask, x, y, monochrome, feather_px)
                x0, y0 = max(0, x), max(0, y)
                x1 = min(output_width, x + crop_width)
                y1 = min(output_height, y + crop_height)
                if x1 > x0 and y1 > y0:
                    self._save_reference_artifact(
                        reference, "final", canvas[y0:y1, x0:x1].copy())
                    mask_roi = mask[y0 - y:y1 - y, x0 - x:x1 - x]
                    rebuilt_pixels[y0:y1, x0:x1] = np.maximum(
                        rebuilt_pixels[y0:y1, x0:x1], (mask_roi > 32).astype(np.uint8))
                used_sr = used_sr or enhanced
                self.last_sr_state = (
                    "HANDOVER" if using_handover else
                    ("HIT" if enhanced else self.last_sr_state))
                self.last_refs_used += 1
                self._handover_references[(generation, target.track_id)] = (
                    reference, target.class_id)
                if using_handover:
                    self.sr_handover_frames += 1
                if self.draw_targets:
                    left = int(round(target.left * scale_x))
                    top = int(round(target.top * scale_y))
                    right = int(round(target.right * scale_x))
                    bottom = int(round(target.bottom * scale_y))
                    cv2.rectangle(canvas, (left, top), (right, bottom), (60, 220, 60), 1)
                    cv2.putText(canvas,
                                f"id{target.track_id} c{target.class_id} "
                                f"{target.confidence_percent}%",
                                (left, max(12, top - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.36, (60, 220, 60), 1, cv2.LINE_AA)
            self._smooth_boxes = {
                key: value for key, value in self._smooth_boxes.items()
                if self._smooth_seen_at.get(key, 0.0) >= now - 1.0
            }
            self._smooth_seen_at = {
                key: seen for key, seen in self._smooth_seen_at.items()
                if key in self._smooth_boxes
            }
        else:
            self._record_drop(
                state_reason if state_reason.startswith("STATE_") else "STATE_INVALID")
        self._handover_references = {
            key: value for key, value in self._handover_references.items()
            if key[0] == generation and
            now - value[0].received_at <= self.reference_cache_ttl_ms / 1000.0
        }
        self.last_spatial = "ROI-ESRGAN" if used_sr else (
            "ROI-LANCZOS" if self.last_refs_used else "BASE-LANCZOS")
        if self.last_refs_used == 0 and self.last_drop_reason == "NONE":
            self._record_drop("NO_TARGET")
        if output_width > 0 and output_height > 0:
            self.last_rebuild_percent = float(np.count_nonzero(rebuilt_pixels)) * 100.0 / (
                output_width * output_height)
        if source_sequence is not None:
            self._composed_source_sequence = source_sequence
            self._composed_generation = generation
            self._composed_frame = canvas
            self._composed_spatial = self.last_spatial
            self.composed_frames += 1
        return canvas, self.last_spatial

    def snapshot(self) -> dict:
        sorted_sr_ms = sorted(self.sr_compute_samples_ms)
        sr_p50_ms = (sorted_sr_ms[max(0, math.ceil(len(sorted_sr_ms) * 0.50) - 1)]
                     if sorted_sr_ms else 0.0)
        sr_p95_ms = (sorted_sr_ms[max(0, math.ceil(len(sorted_sr_ms) * 0.95) - 1)]
                     if sorted_sr_ms else 0.0)
        return {
            "spatial": self.last_spatial,
            "reference_ready": self.reference_ready,
            "refs_used": self.last_refs_used,
            "reference_age": self.last_ref_age,
            "reference_content_age_ms": self.last_ref_content_age_ms,
            "reference_pts_ms": self.last_reference_pts_ms,
            "reference_generation": self.last_reference_generation,
            "reference_kind": self.last_reference_kind,
            "reference_crop_src_width": self.last_reference_crop_src_width,
            "reference_crop_src_height": self.last_reference_crop_src_height,
            "reference_jpeg_width": self.last_reference_jpeg_width,
            "reference_jpeg_height": self.last_reference_jpeg_height,
            "reference_jpeg_quality": self.last_reference_jpeg_quality,
            "reference_jpeg_bytes": self.last_reference_jpeg_bytes,
            "reference_jpeg_scale": self.last_reference_jpeg_scale,
            "reference_head_pixels_width": self.last_reference_head_pixels_width,
            "reference_head_pixels_height": self.last_reference_head_pixels_height,
            "state_reference_generation": self.last_state_reference_generation,
            "track_id": self.last_track_id,
            "rebuild_percent": self.last_rebuild_percent,
            "chroma_mode": self.last_chroma_mode,
            "reference_cache_ttl_ms": self.reference_cache_ttl_ms,
            "reference_content_hard_max_age_ms": self.reference_content_hard_max_age_ms,
            "reference_future_max_ms": self.reference_future_max_ms,
            "age_drops": self.age_drops,
            "future_drops": self.future_drops,
            "timing_drops": self.timing_drops,
            "no_reference_drops": self.no_reference_drops,
            "state_drops": self.state_drops,
            "generation_drops": self.generation_drops,
            "cache_drops": self.cache_drops,
            "geom_invalid_drops": self.geom_invalid_drops,
            "geom_scale_drops": self.geom_scale_drops,
            "scale_drops": self.geom_scale_drops,
            "geom_outside_drops": self.geom_outside_drops,
            "last_drop_reason": self.last_drop_reason,
            "content_reason": self.last_content_reason,
            "registration_x": self.last_registration_x,
            "registration_y": self.last_registration_y,
            "patch_width": self.last_patch_width,
            "patch_height": self.last_patch_height,
            "mask_width": self.last_mask_width,
            "mask_height": self.last_mask_height,
            "visible_ratio": self.last_visible_ratio,
            "geom_dx_ratio": self.last_geom_dx_ratio,
            "geom_dy_ratio": self.last_geom_dy_ratio,
            "geom_area_ratio": self.last_geom_area_ratio,
            "sr_model": self.resolver.model_name,
            "sr_state": self.last_sr_state,
            "sr_pending": self.future is not None,
            "sr_queued": self.pending_reference is not None,
            "sr_running": self.future is not None,
            "sr_queue": 1 if self.pending_reference is not None else 0,
            "sr_jobs": self.sr_jobs,
            "sr_done": self.sr_done,
            "sr_stale": self.sr_stale,
            "sr_x": self.sr_expired_before_submit,
            "sr_expired_after_compute": self.sr_expired_after_compute,
            "sr_future_drops": self.sr_future_drops,
            "sr_future_waits": self.sr_future_waits,
            "sr_invalid_drops": self.sr_invalid_drops,
            "sr_pending_replaced": self.sr_pending_replaced,
            "sr_cache_hits": self.sr_cache_hits,
            "sr_cache_misses": self.sr_cache_misses,
            "last_sr_lookup": self.last_sr_lookup,
            "sr_handover": self.last_sr_handover,
            "sr_handover_frames": self.sr_handover_frames,
            "sr_last_ms": self.sr_last_ms,
            "sr_p50_ms": sr_p50_ms,
            "sr_p95_ms": sr_p95_ms,
            "registration_drops": self.registration_drops,
            "content_drops": self.content_drops,
            "content_matches": self.content_matches,
            "new_reference_first_frames": self.new_reference_first_frames,
            "new_reference_first_frame_sr_hits": self.new_reference_first_frame_sr_hits,
            "new_reference_first_frame_sr_hit_rate": (
                self.new_reference_first_frame_sr_hits /
                float(self.new_reference_first_frames)
                if self.new_reference_first_frames else 0.0),
            "match_score": self.last_match_score,
            "composed_source_sequence": self._composed_source_sequence,
            "composed_frames": self.composed_frames,
        }

    def close(self) -> None:
        if self.future is not None:
            self.future.cancel()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)

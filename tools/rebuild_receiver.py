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


class RebuildReceiver:
    def __init__(self, port: int = 5009, max_sync_ms: int = 100) -> None:
        if max_sync_ms <= 0:
            raise ValueError("RB/1 maximum PTS sync error must be positive")
        self.port = port
        self.max_sync_ms = max_sync_ms
        self.lock = threading.Lock()
        self.assembler = rb.FragmentAssembler()
        self.state: Optional[rb.StateRecord] = None
        self.state_generation: Optional[int] = None
        self.state_updated = 0.0
        self.state_pts_ms: Optional[int] = None
        self.state_sequence: Optional[int] = None
        self.state_history: Deque[Tuple[int, int, rb.StateRecord, float]] = \
            collections.deque(maxlen=32)
        self.references: Dict[int, VisualReference] = {}
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
        self._last_sync_drop_timestamp: Optional[int] = None
        # A decoder/output handoff can introduce a stable presentation offset
        # (for example one 6-fps source frame).  Keep a robust sliding estimate
        # and apply it before the strict 100 ms semantic gate.  The estimate is
        # deliberately learned only from plausible nearest-state observations;
        # a seconds-old state must never move the clock used for registration.
        self.pts_bias_ms = 0
        self.raw_sync_ms: Optional[int] = None
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
                self.completed_reference_events.clear()
                self.state_history.clear()
                self.state_updated = 0.0
                self.state_pts_ms = None
                self.state_sequence = None
                self.state_generation = packet.generation
                # Incomplete fragments from the previous profile generation
                # can never form a valid current reference.
                self.assembler = rb.FragmentAssembler()

        complete = None
        if fragment is not None:
            try:
                complete = self.assembler.add(packet.generation, packet.packet_type,
                                              fragment, now)
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
                if accept_state:
                    self.state = state
                    self.state_updated = now
                    self.state_pts_ms = packet.pts_ms
                    self.state_sequence = packet.sequence
                    state_item = (packet.generation, packet.pts_ms, state, now)
                    if (self.state_history and
                            self.state_history[-1][0] == packet.generation and
                            self.state_history[-1][1] == packet.pts_ms):
                        self.state_history[-1] = state_item
                    else:
                        self.state_history.append(state_item)
                    active = {target.track_id for target in state.targets}
                    self.references = {
                        track_id: reference for track_id, reference in self.references.items()
                        if track_id in active and now - reference.received_at <= 5.0
                    }
            if decoded_reference is not None:
                existing = self.references.get(decoded_reference.track_id)
                if (existing is None or
                        decoded_reference.generation != existing.generation or
                        decoded_reference.reference_generation >=
                        existing.reference_generation):
                    self.references[decoded_reference.track_id] = decoded_reference
                    self.completed_reference_events.append(decoded_reference)
                    self.complete_references += 1
                    if decoded_reference.recovered_with_parity:
                        self.parity_recovered += 1
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - 1.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def scene(self) -> Tuple[Optional[rb.StateRecord], int,
                             Dict[int, VisualReference], float]:
        with self.lock:
            generation = -1 if self.state_generation is None else self.state_generation
            return self.state, generation, dict(self.references), self.state_updated

    def take_completed_references(self) -> Tuple[VisualReference, ...]:
        """Return newly assembled crops once, for early asynchronous SR.

        This deliberately does *not* declare a crop paintable.  The existing
        generation, STATE flag, PTS and registration checks in ``render``
        remain the only authority for blending it into a frame.  The event is
        solely a head start for model-native upscaling while those checks and
        the sender's next STATE are still in flight.
        """
        with self.lock:
            events = tuple(self.completed_reference_events)
            self.completed_reference_events.clear()
            return events

    @staticmethod
    def _signed_pts_delta(left: int, right: int) -> int:
        return ((left - right + 0x80000000) & 0xFFFFFFFF) - 0x80000000

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
                           video_pts: int, max_sync_ms: int) -> rb.StateRecord:
        """Advance boxes a short distance using the last observed velocity.

        STATE is sent at the source rate while the rebuilt display is usually
        faster.  A nearest-state lookup alone leaves a moving target one
        capture interval behind.  Predict only within the already bounded PTS
        window and preserve every non-geometric field; the registration gate
        below still fails closed when the prediction is implausible.
        """
        if previous is None or previous_pts is None:
            return selected
        sample_delta = cls._signed_pts_delta(selected_pts, previous_pts)
        offset = cls._signed_pts_delta(video_pts, selected_pts)
        if sample_delta <= 0 or abs(offset) > max_sync_ms or sample_delta > 2000:
            return selected
        previous_targets = {target.track_id: target for target in previous.targets}
        predicted = []
        for target in selected.targets:
            old = previous_targets.get(target.track_id)
            if old is None or old.class_id != target.class_id:
                predicted.append(target)
                continue
            ratio = float(offset) / float(sample_delta)
            values = []
            for current, prior in zip(
                    (target.left, target.top, target.right, target.bottom),
                    (old.left, old.top, old.right, old.bottom)):
                values.append(int(round(current + (current - prior) * ratio)))
            width = max(1, target.right - target.left)
            height = max(1, target.bottom - target.top)
            left = max(0, min(selected.source_width - width, values[0]))
            top = max(0, min(selected.source_height - height, values[1]))
            predicted.append(dataclasses.replace(
                target, left=left, top=top, right=left + width, bottom=top + height))
        return dataclasses.replace(selected, targets=tuple(predicted))

    def scene_synced(self, video_rtp_timestamp: Optional[int]) -> Tuple[
            Optional[rb.StateRecord], int, Dict[int, VisualReference], float,
            Optional[int]]:
        """Select a close semantic state or fail closed to the base video.

        A nearest state is not necessarily a usable state: while a large JPEG
        reference is paced, the newest state can be seconds behind the H.265
        frame.  Painting that old state is the floating-target failure.  Keep
        reporting the signed delta for the HUD, but never return paintable
        state/references outside the bounded PTS window.
        """
        with self.lock:
            generation = -1 if self.state_generation is None else self.state_generation
            selected_state = None
            selected_updated = self.state_updated
            selected_delta = None
            effective_video_rtp_timestamp = video_rtp_timestamp
            if video_rtp_timestamp is not None and self.state_history:
                candidates = [item for item in self.state_history if item[0] == generation]
                if candidates:
                    raw_selected = min(
                        candidates,
                        key=lambda item: abs(self._signed_pts_delta(
                            (item[1] * 90) & 0xFFFFFFFF, video_rtp_timestamp)))
                    raw_delta_ticks = self._signed_pts_delta(
                        (raw_selected[1] * 90) & 0xFFFFFFFF, video_rtp_timestamp)
                    self._observe_pts_bias(
                        int(round(raw_delta_ticks / 90.0)), raw_selected[1],
                        video_rtp_timestamp)
                    effective_video_rtp_timestamp = self._effective_video_timestamp(
                        video_rtp_timestamp)
                    selected = min(
                        candidates,
                        key=lambda item: abs(self._signed_pts_delta(
                            (item[1] * 90) & 0xFFFFFFFF,
                            effective_video_rtp_timestamp)))
                    selected_state = selected[2]
                    selected_updated = selected[3]
                    delta_ticks = self._signed_pts_delta(
                        (selected[1] * 90) & 0xFFFFFFFF,
                        effective_video_rtp_timestamp)
                    selected_delta = int(round(delta_ticks / 90.0))
                    # Use the immediately older state for a bounded linear
                    # prediction.  This removes the systematic half-frame
                    # lag without allowing an old reference to follow a new
                    # scene indefinitely.
                    older = [item for item in candidates
                             if self._signed_pts_delta(selected[1], item[1]) > 0]
                    previous = None
                    if older:
                        previous = min(
                            older,
                            key=lambda item: self._signed_pts_delta(selected[1], item[1]))
                    selected_state = self._extrapolate_state(
                        selected_state, selected[1],
                        None if previous is None else previous[2],
                        None if previous is None else previous[1],
                        selected[1] - selected_delta,
                        max(self.max_sync_ms * 4, 400))
            if (video_rtp_timestamp is None or selected_delta is None or
                    abs(selected_delta) > self.max_sync_ms):
                if (video_rtp_timestamp is not None and selected_delta is not None and
                        self._last_sync_drop_timestamp != video_rtp_timestamp):
                    self.sync_drops += 1
                    self._last_sync_drop_timestamp = video_rtp_timestamp
                return None, generation, {}, selected_updated, selected_delta
            self._last_sync_drop_timestamp = None
            references = dict(self.references)
            # Do not paint a reference that is materially newer than the
            # decoded base frame. Older references remain useful only while a
            # current, PTS-bounded target state supplies their location.
            references = {
                track_id: reference for track_id, reference in references.items()
                if self._signed_pts_delta(
                    (reference.pts_ms * 90) & 0xFFFFFFFF,
                    effective_video_rtp_timestamp) <= self.max_sync_ms * 90
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
                "max_sync_ms": self.max_sync_ms,
                "pts_bias_ms": self.pts_bias_ms,
                "pts_bias_samples": len(self.pts_bias_samples),
                "raw_sync_ms": self.raw_sync_ms,
                "state_age": None if not self.state_updated else now - self.state_updated,
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
                 cpu_threads: int = 2, scale: int = 2) -> None:
        self.session = None
        self.scale = scale
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
            # CUDA's first dynamic-shape run can include GPU-context and
            # kernel initialization that is far slower than steady-state ROI
            # inference.  Do that work on the asynchronous model-load thread
            # before marking the resolver usable, rather than spending the
            # first received reference's 1 s PTS budget on it.  106x128 is the
            # even-padded form of the typical 105x128 ROI used by this profile.
            warmup_started = time.monotonic()
            warmup = np.zeros((128, 106, 3), dtype=np.uint8)
            self.upscale(warmup, (212, 256))
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
            multiple = max(1, self.scale)
            pad_height = (-source_height) % multiple
            pad_width = (-source_width) % multiple
            if pad_height or pad_width:
                image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width,
                                           cv2.BORDER_REPLICATE)
            input_height, input_width = image.shape[:2]
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
            if pad_height or pad_width:
                crop_height = int(round(output.shape[0] * source_height / input_height))
                crop_width = int(round(output.shape[1] * source_width / input_width))
                output = output[:crop_height, :crop_width]
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
                 reference_max_age: float = 1.0, draw_targets: bool = False) -> None:
        self.output_size = output_size
        self.resolver = resolver or SuperResolver(enabled=False)
        self.reference_max_age = reference_max_age
        self.draw_targets = draw_targets
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebuild-roi-sr")
        self.future: Optional[Future] = None
        self.job_key: Optional[tuple] = None
        self.pending_reference: Optional[VisualReference] = None
        self.cache: Dict[tuple, np.ndarray] = {}
        self.sr_jobs = 0
        self.sr_done = 0
        self.sr_stale = 0
        self.registration_drops = 0
        self.content_drops = 0
        self.content_matches = 0
        self.last_refs_used = 0
        self.last_ref_age: Optional[float] = None
        self.last_ref_content_age_ms: Optional[int] = None
        self.last_match_score: Optional[float] = None
        self.last_rebuild_percent = 0.0
        self.last_spatial = "BASE-LANCZOS"
        self.last_chroma_mode = "COLOR"
        self._smooth_boxes: Dict[tuple, Tuple[float, float, float, float]] = {}
        self._smooth_generation: Optional[int] = None

    @staticmethod
    def _reference_key(reference: VisualReference) -> tuple:
        return (reference.generation, reference.track_id,
                reference.reference_generation)

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
        self.future = self.executor.submit(
            self.resolver.upscale, reference.image.copy(), native_size)
        self.sr_jobs += 1
        return True

    def _defer_latest(self, reference: VisualReference) -> None:
        """Keep at most one not-yet-submitted crop, preferring the newest."""
        if self.pending_reference is None:
            self.pending_reference = reference
            return
        pending = self.pending_reference
        if (reference.generation != pending.generation or
                reference.reference_generation != pending.reference_generation or
                reference.received_at >= pending.received_at):
            self.pending_reference = reference

    def _start_deferred(self) -> None:
        if self.future is not None or self.pending_reference is None:
            return
        reference = self.pending_reference
        key = self._reference_key(reference)
        if key in self.cache:
            self.pending_reference = None
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
        key = self._reference_key(reference)
        if key in self.cache or key == self.job_key:
            return
        if self.future is not None or not self.resolver.available:
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
        try:
            image = future.result()
        except Exception:
            self.sr_stale += 1
            return
        # Cache the model-native x2 result by reference identity, not by the
        # current detector-box geometry.  Box jitter changes the paste size on
        # almost every frame; including that size in the key used to launch a
        # new Real-ESRGAN job continuously and produced soft/sharp pulsing.
        # A stale result cannot overwrite a newer reference generation because
        # the generation is part of the key.
        self.cache[key] = image
        self.sr_done += 1
        if len(self.cache) > 8:
            oldest = next(iter(self.cache))
            if oldest != key:
                self.cache.pop(oldest, None)
        # A later reference may have arrived while this one was running.  It
        # gets the next (and only) slot instead of waiting for a renderer tick
        # that could occur after its 1 s content-age deadline.
        self._start_deferred()

    def _assets(self, reference: VisualReference, size: Tuple[int, int]) -> Tuple[np.ndarray,
                                                                                 np.ndarray,
                                                                                 bool]:
        self.prefetch(reference)
        key = self._reference_key(reference)
        native = self.cache.get(key)
        enhanced = native is not None and self.resolver.available
        if native is not None:
            patch = cv2.resize(native, size, interpolation=cv2.INTER_LANCZOS4)
        else:
            patch = cv2.resize(reference.image, size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(reference.mask, size, interpolation=cv2.INTER_LINEAR)
        return patch, mask, enhanced

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
    def _content_register(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
                          x: int, y: int, search_radius: int = 8,
                          minimum_score: float = 0.32
                          ) -> Tuple[Optional[Tuple[int, int]], Optional[float], bool]:
        """Align a reference patch to the current base within a small window.

        Geometry predicts where a target moved; this correlation step uses the
        decoded content to remove the remaining few-pixel detector jitter.  A
        low-texture target has no reliable signal and is accepted at the
        geometric position (``attempted=False``).  When a textured template
        is available, a low peak is a fail-closed registration drop instead of
        painting a stale coloured patch over unrelated background.
        """
        if patch.ndim != 3 or canvas.ndim != 3 or patch.shape[:2] != mask.shape[:2]:
            return None, None, True
        height, width = patch.shape[:2]
        if width < 4 or height < 4 or width > canvas.shape[1] or height > canvas.shape[0]:
            return None, None, True
        binary = (mask >= 128).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) >= 16:
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            margin = max(1, min(4, int(round(min(width, height) * 0.04))))
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(width, right + margin)
            bottom = min(height, bottom + margin)
        else:
            left, top, right, bottom = 0, 0, width, height
        template = cv2.cvtColor(patch[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        template = cv2.GaussianBlur(template, (3, 3), 0)
        if template.shape[1] < 3 or template.shape[0] < 3:
            return (x, y), None, False
        if float(template.std()) < 2.0:
            return (x, y), None, False

        predicted_x = x + left
        predicted_y = y + top
        max_x = canvas.shape[1] - template.shape[1]
        max_y = canvas.shape[0] - template.shape[0]
        search_x = max(0, min(max_x, predicted_x - search_radius))
        search_y = max(0, min(max_y, predicted_y - search_radius))
        search_right = min(canvas.shape[1], predicted_x + template.shape[1] + search_radius)
        search_bottom = min(canvas.shape[0], predicted_y + template.shape[0] + search_radius)
        search = cv2.cvtColor(
            canvas[search_y:search_bottom, search_x:search_right], cv2.COLOR_BGR2GRAY)
        search = cv2.GaussianBlur(search, (3, 3), 0)
        if (search.shape[1] < template.shape[1] or
                search.shape[0] < template.shape[0] or
                float(search.std()) < 1.0):
            return (x, y), None, False
        try:
            correlation = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return (x, y), None, False
        _, score, _, location = cv2.minMaxLoc(correlation)
        score = float(score)
        corrected = (search_x + int(location[0]) - left,
                     search_y + int(location[1]) - top)
        if score < minimum_score:
            return None, score, True
        return corrected, score, True

    def _smooth_target(self, target: rb.TargetState, generation: int) -> rb.TargetState:
        """EMA detector boxes while resetting on a real track jump."""
        key = (generation, target.track_id, target.reference_generation)
        current = tuple(float(value) for value in
                        (target.left, target.top, target.right, target.bottom))
        previous = self._smooth_boxes.get(key)
        if previous is None:
            smoothed = current
        else:
            previous_width = max(1.0, previous[2] - previous[0])
            previous_height = max(1.0, previous[3] - previous[1])
            current_cx = (current[0] + current[2]) * 0.5
            current_cy = (current[1] + current[3]) * 0.5
            previous_cx = (previous[0] + previous[2]) * 0.5
            previous_cy = (previous[1] + previous[3]) * 0.5
            if (abs(current_cx - previous_cx) > 0.60 * previous_width or
                    abs(current_cy - previous_cy) > 0.60 * previous_height):
                smoothed = current
            else:
                alpha = 0.45
                smoothed = tuple(alpha * now_value + (1.0 - alpha) * old_value
                                 for now_value, old_value in zip(current, previous))
        self._smooth_boxes[key] = smoothed
        rounded = tuple(int(round(value)) for value in smoothed)
        left, top, right, bottom = rounded
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        return dataclasses.replace(target, left=left, top=top,
                                   right=right, bottom=bottom)

    @staticmethod
    def _registered_crop(reference: VisualReference, target: rb.TargetState,
                         scale_x: float, scale_y: float) -> Optional[Tuple[int, int, int, int]]:
        """Map a reference crop through the reference/current detector boxes.

        The crop margin is not symmetric when a subject touches an image edge;
        using only crop width/height and the current box centre therefore
        shifts the old background into the new frame.  The reference crop
        offsets relative to the captured detector-box centre preserve those
        edge offsets while the area-derived scale keeps the geometry isotropic.
        """
        ref_left, ref_top, ref_right, ref_bottom = reference.reference_bbox
        cur_left, cur_top, cur_right, cur_bottom = (
            target.left, target.top, target.right, target.bottom)
        ref_width = ref_right - ref_left
        ref_height = ref_bottom - ref_top
        cur_width = cur_right - cur_left
        cur_height = cur_bottom - cur_top
        if min(ref_width, ref_height, cur_width, cur_height) <= 0:
            return None
        center_dx = abs((cur_left + cur_right) * 0.5 -
                        (ref_left + ref_right) * 0.5)
        center_dy = abs((cur_top + cur_bottom) * 0.5 -
                        (ref_top + ref_bottom) * 0.5)
        area_ratio = (cur_width * cur_height) / float(ref_width * ref_height)
        # A reference older than the PTS gate can still be present in the
        # receiver cache; reject a large motion/scale mismatch before any
        # pixels are painted.
        if (center_dx > 0.25 * max(ref_width, cur_width) or
                center_dy > 0.25 * max(ref_height, cur_height) or
                area_ratio < 0.70 or area_ratio > 1.40):
            return None
        crop_left, crop_top, crop_right, crop_bottom = reference.crop
        # Use one source-space scale derived from area.  Independent x/y
        # scaling lets a one-frame detector aspect-ratio wobble breathe the
        # reference crop and creates a visible edge oscillation.
        isotropic_scale = math.sqrt(area_ratio)
        ref_center_x = (ref_left + ref_right) * 0.5
        ref_center_y = (ref_top + ref_bottom) * 0.5
        cur_center_x = (cur_left + cur_right) * 0.5
        cur_center_y = (cur_top + cur_bottom) * 0.5
        x0 = int(round((cur_center_x + (crop_left - ref_center_x) * isotropic_scale) * scale_x))
        y0 = int(round((cur_center_y + (crop_top - ref_center_y) * isotropic_scale) * scale_y))
        x1 = int(round((cur_center_x + (crop_right - ref_center_x) * isotropic_scale) * scale_x))
        y1 = int(round((cur_center_y + (crop_bottom - ref_center_y) * isotropic_scale) * scale_y))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def render(self, base: np.ndarray, state: Optional[rb.StateRecord], generation: int,
               references: Dict[int, VisualReference], now: Optional[float] = None,
               video_rtp_timestamp: Optional[int] = None,
               pts_bias_ms: int = 0) -> Tuple[np.ndarray, str]:
        now = time.monotonic() if now is None else now
        self._poll()
        # Direct users of RebuildComposer (tests or an alternate UI) may not
        # use RebuildReceiver.take_completed_references().  Start any supplied
        # crop before inspecting target flags/PTS so this fallback has the same
        # early-launch property, while render safety remains unchanged below.
        for reference in references.values():
            self.prefetch(reference)
        output_width, output_height = self.output_size
        if state is not None and state.output_width > 0 and state.output_height > 0:
            output_width, output_height = state.output_width, state.output_height
        canvas = cv2.resize(base, (output_width, output_height),
                            interpolation=cv2.INTER_LANCZOS4)
        monochrome = self._nearly_grayscale(canvas)
        self.last_chroma_mode = "MONO" if monochrome else "COLOR"
        rebuilt_pixels = np.zeros((output_height, output_width), dtype=np.uint8)
        self.last_refs_used = 0
        self.last_ref_age = None
        self.last_ref_content_age_ms = None
        self.last_match_score = None
        self.last_rebuild_percent = 0.0
        self.registration_drops = 0
        self.content_drops = 0
        used_sr = False
        effective_video_rtp_timestamp = None
        if video_rtp_timestamp is not None:
            effective_video_rtp_timestamp = (
                video_rtp_timestamp + int(round(pts_bias_ms * 90.0))) & 0xFFFFFFFF
        active_smooth_keys = set()
        if state is not None and state.source_width > 0 and state.source_height > 0:
            scale_x = output_width / float(state.source_width)
            scale_y = output_height / float(state.source_height)
            if self._smooth_generation != generation:
                self._smooth_boxes.clear()
                self._smooth_generation = generation
            for raw_target in state.targets:
                target = self._smooth_target(raw_target, generation)
                active_smooth_keys.add(
                    (generation, target.track_id, target.reference_generation))
                reference = references.get(target.track_id)
                if (reference is None or not (target.flags & 1) or
                        reference.generation != generation or
                        now - reference.received_at > self.reference_max_age or
                        reference.reference_generation != target.reference_generation):
                    continue
                content_age_ms = None
                if effective_video_rtp_timestamp is not None:
                    content_age_ticks = RebuildReceiver._signed_pts_delta(
                        effective_video_rtp_timestamp,
                        (reference.pts_ms * 90) & 0xFFFFFFFF)
                    content_age_ms = int(round(content_age_ticks / 90.0))
                    # A reference which is newer than the base by more than
                    # the semantic gate, or older than the display age budget,
                    # is unsafe even when its UDP transfer just completed.
                    if (content_age_ms < -100 or
                            content_age_ms > int(round(self.reference_max_age * 1000.0))):
                        continue
                registered = self._registered_crop(reference, target, scale_x, scale_y)
                if registered is None:
                    self.registration_drops += 1
                    continue
                x, y, x1, y1 = registered
                crop_width = max(2, x1 - x)
                crop_height = max(2, y1 - y)
                patch, mask, enhanced = self._assets(
                    reference, (crop_width, crop_height))
                corrected, score, attempted = self._content_register(
                    canvas, patch, mask, x, y)
                if attempted:
                    if score is not None:
                        self.last_match_score = score if self.last_match_score is None else max(
                            self.last_match_score, score)
                    if corrected is None:
                        self.registration_drops += 1
                        self.content_drops += 1
                        continue
                    if score is not None:
                        self.content_matches += 1
                    x, y = corrected
                feather_px = max(
                    3.0, min(10.0, max(
                        crop_width / float(max(1, reference.mask.shape[1])),
                        crop_height / float(max(1, reference.mask.shape[0])))))
                self._blend(canvas, patch, mask, x, y, monochrome, feather_px)
                x0, y0 = max(0, x), max(0, y)
                x1 = min(output_width, x + crop_width)
                y1 = min(output_height, y + crop_height)
                if x1 > x0 and y1 > y0:
                    mask_roi = mask[y0 - y:y1 - y, x0 - x:x1 - x]
                    rebuilt_pixels[y0:y1, x0:x1] = np.maximum(
                        rebuilt_pixels[y0:y1, x0:x1], (mask_roi > 32).astype(np.uint8))
                used_sr = used_sr or enhanced
                self.last_refs_used += 1
                age = now - reference.received_at
                self.last_ref_age = age if self.last_ref_age is None else max(
                    self.last_ref_age, age)
                if content_age_ms is not None:
                    self.last_ref_content_age_ms = (
                        content_age_ms if self.last_ref_content_age_ms is None else
                        max(self.last_ref_content_age_ms, content_age_ms))
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
                if key in active_smooth_keys
            }
        else:
            self._smooth_boxes.clear()
        self.last_spatial = "ROI-ESRGAN" if used_sr else (
            "ROI-LANCZOS" if self.last_refs_used else "BASE-LANCZOS")
        if output_width > 0 and output_height > 0:
            self.last_rebuild_percent = float(np.count_nonzero(rebuilt_pixels)) * 100.0 / (
                output_width * output_height)
        return canvas, self.last_spatial

    def snapshot(self) -> dict:
        return {
            "spatial": self.last_spatial,
            "refs_used": self.last_refs_used,
            "reference_age": self.last_ref_age,
            "reference_content_age_ms": self.last_ref_content_age_ms,
            "rebuild_percent": self.last_rebuild_percent,
            "chroma_mode": self.last_chroma_mode,
            "sr_model": self.resolver.model_name,
            "sr_pending": self.future is not None,
            "sr_queued": self.pending_reference is not None,
            "sr_jobs": self.sr_jobs,
            "sr_done": self.sr_done,
            "sr_stale": self.sr_stale,
            "registration_drops": self.registration_drops,
            "content_drops": self.content_drops,
            "content_matches": self.content_matches,
            "match_score": self.last_match_score,
        }

    def close(self) -> None:
        if self.future is not None:
            self.future.cancel()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)

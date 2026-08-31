#!/usr/bin/env python3

import dataclasses
import importlib.util
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import rebuild_protocol as rb
from rebuild_receiver import (
    RebuildComposer,
    RebuildReceiver,
    SuperResolver,
    VisualReference,
    jpeg_dimensions,
)


def make_fragment(blob: bytes, index: int, count: int, chunk: int,
                  data: bytes) -> rb.PatchFragment:
    return rb.PatchFragment(
        transfer_id=7, track_id=4, reference_generation=2,
        left=100, top=80, right=220, bottom=280,
        reference_left=110, reference_top=90, reference_right=210,
        reference_bottom=270,
        jpeg_width=48, jpeg_height=80, mask_width=4, mask_height=4,
        fragment_index=index, data_fragments=count, blob_size=len(blob),
        mask_rle_bytes=2, chunk_bytes=chunk, data=data,
    )


class RebuildProtocolTests(unittest.TestCase):
    @staticmethod
    def _visual_reference(reference_generation: int = 2,
                          received_at: float = 10.0) -> VisualReference:
        return VisualReference(
            generation=5,
            track_id=4,
            reference_generation=reference_generation,
            crop=(100, 80, 220, 280),
            reference_bbox=(100, 80, 220, 280),
            image=np.full((80, 48, 3), (20, 30, 230), dtype=np.uint8),
            mask=np.full((64, 64), 255, dtype=np.uint8),
            received_at=received_at,
            recovered_with_parity=False,
            pts_ms=1000,
        )

    def test_superresolver_orders_cuda_and_preloads_packaged_dlls(self):
        providers = SuperResolver._ordered_providers((
            "CPUExecutionProvider", "CUDAExecutionProvider", "AzureExecutionProvider"))
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

        class FakeOrt:
            def __init__(self):
                self.directories = []

            def preload_dlls(self, directory=None):
                self.directories.append(directory)

        fake = FakeOrt()
        old_path = os.environ.get("PATH")
        try:
            SuperResolver._preload_cuda_dlls(fake, providers)
            self.assertEqual(fake.directories, [""])

            SuperResolver._preload_cuda_dlls(fake, ["CPUExecutionProvider"])
            self.assertEqual(fake.directories, [""])
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path

    def test_superresolver_refreshes_hud_provider_after_runtime_fallback(self):
        class FakeSession:
            def __init__(self, providers):
                self.providers = providers

            def get_providers(self):
                return self.providers

        resolver = SuperResolver(enabled=False)
        resolver.session = FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(resolver._refresh_active_provider(), "CUDAExecutionProvider")
        self.assertEqual(resolver.model_name, "Real-ESRGAN/CUDA")
        resolver.session.providers = ["CPUExecutionProvider"]
        self.assertEqual(resolver._refresh_active_provider(), "CPUExecutionProvider")
        self.assertEqual(resolver.model_name, "Real-ESRGAN/CPU")

    def test_superresolver_normalizes_dynamic_crops_to_one_model_shape(self):
        class FakeInput:
            name = "input"
            type = "tensor(float)"

        class FakeSession:
            def __init__(self):
                self.shapes = []

            @staticmethod
            def get_inputs():
                return [FakeInput()]

            @staticmethod
            def get_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def run(self, _, feeds):
                tensor = feeds["input"]
                self.shapes.append(tensor.shape)
                return [np.zeros((1, 3, 192, 192), dtype=np.float32)]

        resolver = SuperResolver(enabled=False, input_side=96)
        resolver.session = FakeSession()
        resolver._ready = True
        first = resolver.upscale(np.zeros((41, 43, 3), dtype=np.uint8), (86, 82))
        second = resolver.upscale(np.zeros((48, 53, 3), dtype=np.uint8), (106, 96))
        larger = resolver.upscale(np.zeros((80, 120, 3), dtype=np.uint8), (240, 160))
        self.assertEqual(resolver.session.shapes,
                         [(1, 3, 96, 96), (1, 3, 96, 96), (1, 3, 96, 96)])
        self.assertEqual(first.shape, (82, 86, 3))
        self.assertEqual(second.shape, (96, 106, 3))
        self.assertEqual(larger.shape, (160, 240, 3))

    def test_packet_and_state_roundtrip_with_crc_rejection(self):
        target = rb.TargetState(4, 0, 91, 10, 20, 40, 80, 2, 1)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        packet = rb.Packet(rb.STATE, rb.PROFILE_REBUILD, 3, 0, 10, 22, 1234,
                           rb.build_state(state))
        encoded = rb.build_packet(packet)
        decoded = rb.parse_packet(encoded)
        self.assertEqual(decoded, packet)
        self.assertEqual(rb.parse_state(decoded.payload), state)
        corrupt = bytearray(encoded)
        corrupt[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC32"):
            rb.parse_packet(bytes(corrupt))

    def test_receiver_rejects_old_generation_without_rolling_back(self):
        empty = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, ())
        receiver = RebuildReceiver()
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 10, 1, 1000, rb.build_state(empty))), now=1.0)
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 4, 0, 9, 1, 900, rb.build_state(empty))), now=1.1)
        state, generation, _, _ = receiver.scene()
        self.assertEqual(generation, 5)
        self.assertEqual(state, empty)
        self.assertEqual(receiver.snapshot()["reordered"], 1)

    def test_receiver_reports_packet_rate_and_datagram_lengths(self):
        empty = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, ())
        datagram = rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 10, 1, 1000, rb.build_state(empty)))
        receiver = RebuildReceiver()
        receiver.on_datagram(datagram)
        values = receiver.snapshot()
        self.assertEqual(values["pps"], 1.0)
        self.assertEqual(values["state_pps"], 1.0)
        self.assertEqual(values["data_pps"], 0.0)
        self.assertEqual(values["packet_last_bytes"], len(datagram))
        self.assertEqual(values["packet_avg_bytes"], float(len(datagram)))

    def test_synced_scene_uses_state_generation_when_newer_patch_arrives_early(self):
        receiver = RebuildReceiver()
        advertised = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1),))
        older = self._visual_reference(reference_generation=2, received_at=10.0)
        newer = self._visual_reference(reference_generation=3, received_at=10.1)
        receiver.state_generation = 5
        receiver.state_history.append((5, 1000, advertised, 10.0))
        receiver.references[4] = newer
        receiver.reference_history[(4, 2)] = older
        receiver.reference_history[(4, 3)] = newer

        selected, generation, references, _, delta = receiver.scene_synced(1000 * 90)

        self.assertEqual(generation, 5)
        self.assertEqual(selected, advertised)
        self.assertEqual(delta, 0)
        self.assertEqual(references[4].reference_generation, 2)

    def test_synced_scene_keeps_old_reference_and_rejects_far_future_reference(self):
        receiver = RebuildReceiver()
        advertised = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1),))
        receiver.state_generation = 5
        receiver.state_history.append((5, 1000, advertised, 10.0))

        old_reference = dataclasses.replace(
            self._visual_reference(), pts_ms=833)
        receiver.reference_history[(4, 2)] = old_reference
        _, _, references, _, _ = receiver.scene_synced(1000 * 90)
        self.assertEqual(references[4], old_reference)

        future_reference = dataclasses.replace(
            self._visual_reference(), pts_ms=1101)
        receiver.reference_history[(4, 2)] = future_reference
        _, _, references, _, _ = receiver.scene_synced(1000 * 90)
        self.assertEqual(references, {})

    def test_protocol_bounds_untrusted_dimensions(self):
        oversized = rb.StateRecord(640, 480, 4096, 4096, 6, 12, 1, ())
        with self.assertRaises(ValueError):
            rb.parse_state(rb.build_state(oversized))

    def test_patch_zero_reference_generation_is_not_local_ready(self):
        blob = b"\x10\x01" + bytes(range(8))
        fragment = make_fragment(blob, 0, 1, len(blob), blob)
        invalid = rb.PatchFragment(
            fragment.transfer_id, fragment.track_id, 0,
            fragment.left, fragment.top, fragment.right, fragment.bottom,
            fragment.reference_left, fragment.reference_top,
            fragment.reference_right, fragment.reference_bottom,
            fragment.jpeg_width, fragment.jpeg_height, fragment.mask_width,
            fragment.mask_height, fragment.fragment_index, fragment.data_fragments,
            fragment.blob_size, fragment.mask_rle_bytes, fragment.chunk_bytes,
            fragment.data)
        with self.assertRaisesRegex(ValueError, "invalid RB/1 patch"):
            rb.parse_fragment(rb.build_fragment(invalid))

    def test_xor_parity_recovers_one_missing_fragment(self):
        blob = b"\x10\x01" + bytes(range(61))
        chunk = 20
        count = (len(blob) + chunk - 1) // chunk
        pieces = [blob[index * chunk:(index + 1) * chunk] for index in range(count)]
        parity = bytearray(chunk)
        for piece in pieces:
            for offset, value in enumerate(piece):
                parity[offset] ^= value
        assembler = rb.FragmentAssembler()
        complete = None
        for index, piece in enumerate(pieces):
            if index == 1:
                continue
            complete = assembler.add(3, rb.PATCH_DATA,
                                     make_fragment(blob, index, count, chunk, piece), now=1.0)
        self.assertIsNone(complete)
        complete = assembler.add(
            3, rb.PATCH_PARITY,
            make_fragment(blob, count, count, chunk, bytes(parity)), now=1.1)
        self.assertIsNotNone(complete)
        self.assertTrue(complete.recovered_with_parity)
        self.assertEqual(complete.mask_rle + complete.jpeg, blob)

    def test_late_parity_does_not_create_phantom_incomplete_transfer(self):
        blob = b"\x10\x01" + bytes(range(31))
        chunk = 20
        count = (len(blob) + chunk - 1) // chunk
        pieces = [blob[index * chunk:(index + 1) * chunk] for index in range(count)]
        parity = bytearray(chunk)
        assembler = rb.FragmentAssembler()
        complete = None
        for index, piece in enumerate(pieces):
            for offset, value in enumerate(piece):
                parity[offset] ^= value
            complete = assembler.add(
                3, rb.PATCH_DATA,
                make_fragment(blob, index, count, chunk, piece), now=1.0 + index * 0.1)
        self.assertIsNotNone(complete)
        self.assertFalse(assembler.transfers)
        self.assertIsNone(assembler.add(
            3, rb.PATCH_PARITY,
            make_fragment(blob, count, count, chunk, bytes(parity)), now=1.3))
        self.assertFalse(assembler.transfers)

    def test_receiver_fails_closed_when_state_pts_is_too_old(self):
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        receiver = RebuildReceiver(max_sync_ms=250)
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 1, 10, 1000, rb.build_state(state))), now=2.0)
        selected, generation, references, _, delta_ms = receiver.scene_synced(1400 * 90)
        self.assertIsNone(selected)
        self.assertEqual(generation, 5)
        self.assertEqual(references, {})
        self.assertEqual(delta_ms, -400)
        self.assertEqual(receiver.snapshot()["sync_drops"], 1)
        receiver.scene_synced(1400 * 90)
        self.assertEqual(receiver.snapshot()["sync_drops"], 1)

    def test_receiver_extrapolates_recent_target_motion_within_pts_window(self):
        first = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 180, 200, 2, 1),))
        second = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 110, 80, 190, 200, 2, 1),))
        receiver = RebuildReceiver(max_sync_ms=100)
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 1, 1, 1000, rb.build_state(first))), now=1.0)
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 2, 2, 1167, rb.build_state(second))), now=1.1)
        selected, _, _, _, delta = receiver.scene_synced(1217 * 90)
        self.assertEqual(delta, -50)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.targets[0].left, 113)
        self.assertEqual(selected.targets[0].right, 193)

    def test_receiver_replaces_same_pts_state_when_reference_becomes_ready(self):
        pending = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 0, 0),))
        ready = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1),))
        receiver = RebuildReceiver()
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 5, 0, 10, 1, 1000,
            rb.build_state(pending))), now=1.0)
        # PATCH fragments and a later source frame may interleave before the
        # completion STATE for this source PTS arrives.
        bridge = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, ())
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 5, 0, 11, 2, 1017,
            rb.build_state(bridge))), now=1.05)
        # The sender's completion STATE deliberately shares the source PTS
        # with the first STATE, but has a higher RB/1 sequence and flags=1.
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 5, 0, 12, 1, 1000,
            rb.build_state(ready))), now=1.1)
        selected, _, _, _, delta = receiver.scene_synced(1000 * 90)
        self.assertEqual(delta, 0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.targets[0].reference_generation, 2)
        self.assertEqual(selected.targets[0].flags, 1)
        self.assertEqual(len(receiver.state_history), 2)

        # A reordered copy of the earlier STATE must not clear readiness.
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 5, 0, 10, 1, 1000,
            rb.build_state(pending))), now=1.2)
        selected, _, _, _, _ = receiver.scene_synced(1000 * 90)
        self.assertEqual(selected.targets[0].reference_generation, 2)
        self.assertEqual(selected.targets[0].flags, 1)

    def test_complete_patch_is_local_ready_without_ready_state(self):
        pending = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 0, 0),))
        receiver = RebuildReceiver()
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 1, 10, 1000, rb.build_state(pending))), now=10.0)

        image = np.full((80, 48, 3), (20, 30, 230), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        blob = b"\x10\x01" + encoded.tobytes()
        chunk = 90
        count = (len(blob) + chunk - 1) // chunk
        for index in range(count):
            piece = blob[index * chunk:(index + 1) * chunk]
            fragment = make_fragment(blob, index, count, chunk, piece)
            receiver.on_datagram(rb.build_packet(rb.Packet(
                rb.PATCH_DATA, 3, 5, 0, 2 + index, 10, 1000,
                rb.build_fragment(fragment))), now=10.1 + index * 0.01)

        scene, generation, references, _ = receiver.scene()
        self.assertEqual(generation, 5)
        self.assertEqual(scene.targets[0].flags, 0)
        self.assertEqual(len(references), 1)
        self.assertEqual(receiver.take_completed_references()[0].reference_generation, 2)

        composer = RebuildComposer(resolver=SuperResolver(enabled=False))
        try:
            base = np.full((144, 256, 3), 80, dtype=np.uint8)
            _, spatial = composer.render(
                base, scene, generation, references, now=10.2,
                video_rtp_timestamp=1000 * 90)
            self.assertEqual(spatial, "ROI-LANCZOS")
        finally:
            composer.close()

    def test_late_older_reference_state_cannot_roll_back_local_generation(self):
        current = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1),))
        older = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 1, 1),))
        receiver = RebuildReceiver()
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 10, 10, 1000, rb.build_state(current))), now=1.0)
        # A local-ready reference establishes generation 2 even before the
        # sender's completion STATE is observed.
        reference = self._visual_reference(reference_generation=2, received_at=1.1)
        receiver.references[4] = reference
        receiver.reference_generations[4] = 2
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, 11, 10, 1000, rb.build_state(older))), now=1.2)
        selected, _, _, _, _ = receiver.scene_synced(1000 * 90)
        self.assertEqual(selected.targets[0].reference_generation, 2)
        self.assertEqual(receiver.snapshot()["reference_generation_drops"], 1)

    def test_reference_content_hard_age_is_inclusive_at_450_ms(self):
        reference = self._visual_reference(received_at=10.0)
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 0)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = np.full((144, 256, 3), 80, dtype=np.uint8)
        for age_ms, expected in ((449, "ROI-LANCZOS"),
                                 (450, "ROI-LANCZOS"),
                                 (451, "BASE-LANCZOS")):
            composer = RebuildComposer(resolver=SuperResolver(enabled=False))
            try:
                _, spatial = composer.render(
                    base, state, 5, {4: reference}, now=10.1,
                    video_rtp_timestamp=(1000 + age_ms) * 90)
                self.assertEqual(spatial, expected)
                self.assertEqual(composer.snapshot()["reference_content_age_ms"], age_ms)
                if age_ms == 451:
                    self.assertEqual(composer.snapshot()["age_drops"], 1)
            finally:
                composer.close()

    def test_reference_future_gate_is_inclusive_at_minus_100_ms(self):
        reference = self._visual_reference(received_at=10.0)
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 0)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = np.full((144, 256, 3), 80, dtype=np.uint8)
        for age_ms, expected in ((-100, "ROI-LANCZOS"),
                                 (-101, "BASE-LANCZOS")):
            composer = RebuildComposer(resolver=SuperResolver(enabled=False))
            try:
                _, spatial = composer.render(
                    base, state, 5, {4: reference}, now=10.1,
                    video_rtp_timestamp=(1000 + age_ms) * 90)
                self.assertEqual(spatial, expected)
                self.assertEqual(composer.snapshot()["reference_content_age_ms"], age_ms)
                self.assertEqual(composer.snapshot()["future_drops"],
                                 1 if age_ms == -101 else 0)
            finally:
                composer.close()

    def test_reference_cache_ttl_is_separate_from_content_age(self):
        reference = self._visual_reference(received_at=10.0)
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 0)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = np.full((144, 256, 3), 80, dtype=np.uint8)
        composer = RebuildComposer(resolver=SuperResolver(enabled=False))
        try:
            _, spatial = composer.render(
                base, state, 5, {4: reference}, now=10.5,
                video_rtp_timestamp=1000 * 90)
            self.assertEqual(spatial, "ROI-LANCZOS")
            self.assertEqual(composer.snapshot()["age_drops"], 0)

            _, spatial = composer.render(
                base, state, 5, {4: reference}, now=11.1,
                video_rtp_timestamp=1000 * 90)
            self.assertEqual(spatial, "BASE-LANCZOS")
            self.assertEqual(composer.snapshot()["age_drops"], 0)
            self.assertEqual(composer.snapshot()["reference_cache_ttl_ms"], 1000)
        finally:
            composer.close()

    def test_udp_jitter_and_reorder_never_rolls_generation_back(self):
        empty = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, ())
        receiver = RebuildReceiver()
        current = rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 30, 0, 100, 30, 2000,
            rb.build_state(empty)))
        receiver.on_datagram(current, now=1.000)
        for index, delay_ms in enumerate((0, 20, 50, 100)):
            receiver.on_datagram(rb.build_packet(rb.Packet(
                rb.STATE, rb.PROFILE_REBUILD, 29, 0, 99 - index, 29 - index,
                1999 - delay_ms, rb.build_state(empty))),
                now=1.000 + delay_ms / 1000.0)
        state, generation, _, _ = receiver.scene()
        self.assertEqual(generation, 30)
        self.assertEqual(state, empty)
        self.assertEqual(receiver.snapshot()["reordered"], 4)

    def test_receiver_calibrates_a_stable_decoder_pts_offset(self):
        receiver = RebuildReceiver(max_sync_ms=100)
        for index in range(7):
            pts_ms = 1000 + index * 167
            state = rb.StateRecord(
                640, 480, 640, 360, 6, 12, 1,
                (rb.TargetState(4, 0, 91, 100 + index, 80,
                                180 + index, 200, 2, 1),))
            receiver.on_datagram(rb.build_packet(rb.Packet(
                rb.STATE, 3, 5, 0, index + 1, 10, pts_ms,
                rb.build_state(state))), now=1.0 + index * 0.1)
            selected, _, _, _, delta = receiver.scene_synced((pts_ms + 167) * 90)
            if index >= 4:
                self.assertIsNotNone(selected)
                self.assertEqual(delta, 0)
        values = receiver.snapshot()
        self.assertEqual(values["pts_bias_ms"], -167)
        self.assertGreaterEqual(values["pts_bias_samples"], 5)

    def test_composer_content_registration_corrects_small_geometry_jitter(self):
        patch = np.zeros((20, 24, 3), dtype=np.uint8)
        rng = np.random.default_rng(7)
        patch[:] = rng.integers(0, 255, patch.shape, dtype=np.uint8)
        mask = np.full((20, 24), 255, dtype=np.uint8)
        canvas = np.zeros((70, 90, 3), dtype=np.uint8)
        canvas[28:48, 37:61] = patch
        corrected, score, attempted = RebuildComposer._content_register(
            canvas, patch, mask, 32, 25, search_radius=8)
        self.assertTrue(attempted)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.8)
        self.assertEqual(corrected, (37, 28))

    def test_composer_reuses_native_sr_cache_when_box_size_changes(self):
        class FakeResolver:
            scale = 2
            available = True
            model_name = "Fake"

            def upscale(self, image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        reference = type("Reference", (), {
            "generation": 3, "track_id": 4, "reference_generation": 2,
            "image": np.full((8, 10, 3), 120, dtype=np.uint8),
            "mask": np.full((4, 4), 255, dtype=np.uint8),
        })()
        composer = RebuildComposer(resolver=FakeResolver())
        native = np.full((16, 20, 3), 200, dtype=np.uint8)
        composer.cache[(3, 4, 2)] = native
        first, _, enhanced = composer._assets(reference, (30, 40))
        second, _, enhanced_again = composer._assets(reference, (45, 55))
        self.assertTrue(enhanced)
        self.assertTrue(enhanced_again)
        self.assertEqual(first.shape[:2], (40, 30))
        self.assertEqual(second.shape[:2], (55, 45))
        self.assertEqual(composer.sr_jobs, 0)
        composer.close()

    def test_receiver_and_composer_rebuild_640x360(self):
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        sequence = 1
        receiver = RebuildReceiver()
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, 3, 5, 0, sequence, 10, 1000, rb.build_state(state))), now=2.0)

        image = np.zeros((80, 48, 3), dtype=np.uint8)
        image[:, :] = (20, 30, 230)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.assertEqual(jpeg_dimensions(encoded.tobytes()), (48, 80))
        blob = b"\x10\x01" + encoded.tobytes()
        chunk = 90
        count = (len(blob) + chunk - 1) // chunk
        for index in range(count):
            sequence += 1
            piece = blob[index * chunk:(index + 1) * chunk]
            fragment = make_fragment(blob, index, count, chunk, piece)
            receiver.on_datagram(rb.build_packet(rb.Packet(
                rb.PATCH_DATA, 3, 5, 0, sequence, 10, 1000,
                rb.build_fragment(fragment))), now=2.1 + index * 0.01)
        scene, generation, references, _ = receiver.scene()
        self.assertEqual(generation, 5)
        self.assertEqual(len(references), 1)
        completed_events = receiver.take_completed_references()
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].reference_generation, 2)
        self.assertEqual(receiver.take_completed_references(), ())
        synced, synced_generation, _, _, delta_ms = receiver.scene_synced(1000 * 90)
        self.assertEqual(synced_generation, 5)
        self.assertEqual(synced, scene)
        self.assertEqual(delta_ms, 0)

        composer = RebuildComposer(
            output_size=(640, 360), resolver=SuperResolver(enabled=False),
            draw_targets=False)
        base = np.empty((144, 256, 3), dtype=np.uint8)
        base[:, :] = (60, 80, 100)
        output, spatial = composer.render(
            base, scene, generation, references, now=2.5,
            video_rtp_timestamp=1000 * 90)
        self.assertEqual(output.shape, (360, 640, 3))
        self.assertEqual(spatial, "ROI-LANCZOS")
        self.assertGreater(int(output[:, :, 2].max()), 100)
        self.assertGreater(composer.snapshot()["rebuild_percent"], 0.0)
        self.assertEqual(composer.snapshot()["chroma_mode"], "COLOR")

        gray = np.full((144, 256, 3), 80, dtype=np.uint8)
        mono, spatial = composer.render(
            gray, scene, generation, references, now=2.6,
            video_rtp_timestamp=1000 * 90)
        self.assertEqual(spatial, "ROI-LANCZOS")
        self.assertEqual(composer.snapshot()["chroma_mode"], "MONO")
        self.assertLessEqual(int(np.max(mono.max(axis=2) - mono.min(axis=2))), 1)

        undeclared = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 1, 0),))
        base_only, spatial = composer.render(
            gray, undeclared, generation, references, now=2.7,
            video_rtp_timestamp=1000 * 90)
        self.assertEqual(spatial, "BASE-LANCZOS")
        self.assertTrue(np.array_equal(base_only, cv2.resize(
            gray, (640, 360), interpolation=cv2.INTER_LANCZOS4)))
        composer.close()

    def test_prefetch_finishes_sr_before_a_near_expiry_reference_is_paintable(self):
        class FakeResolver:
            scale = 2
            available = True
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        reference = self._visual_reference()
        composer = RebuildComposer(
            output_size=(640, 360), resolver=FakeResolver())
        try:
            # Arrival triggers SR before the reference is advertised by a
            # PTS-matched STATE.  The candidate is still inside the 450 ms
            # content deadline when it is submitted and completed.
            composer.update_timing(1000 * 90, generation=5, active_track_ids={4})
            composer.prefetch(reference)
            deadline = time.monotonic() + 1.0
            while composer.snapshot()["sr_done"] != 1 and time.monotonic() < deadline:
                composer.prefetch_pending()
                time.sleep(0.005)
            self.assertEqual(composer.snapshot()["sr_done"], 1)
            self.assertFalse(composer.snapshot()["sr_pending"])

            target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1)
            state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
            base = np.zeros((144, 256, 3), dtype=np.uint8)
            _, spatial = composer.render(
                base, state, 5, {4: reference}, now=10.999,
                video_rtp_timestamp=1449 * 90)
            self.assertEqual(spatial, "ROI-ESRGAN")
            self.assertEqual(composer.snapshot()["reference_content_age_ms"], 449)
        finally:
            composer.close()

    def test_prefetch_retains_latest_reference_during_sr_warmup(self):
        class ToggleResolver:
            scale = 2
            available = False
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        resolver = ToggleResolver()
        older = self._visual_reference(reference_generation=2, received_at=10.0)
        newest = self._visual_reference(reference_generation=3, received_at=10.1)
        composer = RebuildComposer(resolver=resolver)
        try:
            composer.update_timing(1000 * 90, generation=5, active_track_ids={4})
            composer.prefetch(older)
            composer.prefetch(newest)
            self.assertEqual(composer.snapshot()["sr_queue"], 1)
            resolver.available = True
            deadline = time.monotonic() + 1.0
            while composer.snapshot()["sr_done"] < 1 and time.monotonic() < deadline:
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            self.assertEqual(values["sr_jobs"], 1)
            self.assertEqual(values["sr_done"], 1)
            self.assertIn((5, 4, 3), composer.cache)
            self.assertNotIn((5, 4, 2), composer.cache)
        finally:
            composer.close()

    def test_future_reference_is_prefetched_but_rendered_only_after_clock_catches_up(self):
        class FakeResolver:
            scale = 2
            available = True
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        reference = self._visual_reference()
        composer = RebuildComposer(resolver=FakeResolver())
        try:
            composer.update_timing(800 * 90, generation=5, active_track_ids={4})
            composer.prefetch(reference)
            values = composer.snapshot()
            self.assertEqual(values["sr_jobs"], 1)
            self.assertEqual(values["sr_future_waits"], 0)
            self.assertEqual(values["sr_future_drops"], 0)

            deadline = time.monotonic() + 1.0
            while composer.snapshot()["sr_done"] != 1 and time.monotonic() < deadline:
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            self.assertEqual(values["sr_jobs"], 1)
            self.assertEqual(values["sr_done"], 1)
            self.assertEqual(values["sr_queue"], 0)
            self.assertIn((5, 4, 2), composer.cache)

            target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1)
            state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
            base = np.full((144, 256, 3), 80, dtype=np.uint8)
            _, future_mode = composer.render(
                base, state, 5, {4: reference}, now=10.0,
                video_rtp_timestamp=800 * 90, source_sequence=1)
            values = composer.snapshot()
            self.assertEqual(future_mode, "BASE-LANCZOS")
            self.assertEqual(values["future_drops"], 1)
            self.assertEqual(values["last_drop_reason"], "FUTURE")
            self.assertIn((5, 4, 2), composer.cache)

            _, ready_mode = composer.render(
                base, state, 5, {4: reference}, now=10.1,
                video_rtp_timestamp=900 * 90, source_sequence=2)
            values = composer.snapshot()
            self.assertEqual(ready_mode, "ROI-ESRGAN")
            self.assertEqual(values["future_drops"], 1)
            self.assertEqual(values["sr_cache_hits"], 1)
        finally:
            composer.close()

    def test_prefetch_keeps_only_the_latest_reference_behind_an_active_job(self):
        class SlowResolver:
            scale = 2
            available = True
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                time.sleep(0.03)
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        first = self._visual_reference(reference_generation=2, received_at=10.0)
        newest = self._visual_reference(reference_generation=3, received_at=10.1)
        composer = RebuildComposer(resolver=SlowResolver())
        try:
            composer.update_timing(1000 * 90, generation=5, active_track_ids={4})
            composer.prefetch(first)
            composer.prefetch(newest)
            self.assertEqual(composer.snapshot()["sr_jobs"], 1)
            self.assertTrue(composer.snapshot()["sr_queued"])
            deadline = time.monotonic() + 1.0
            while (composer.snapshot()["sr_done"] + composer.snapshot()["sr_stale"] < 2 and
                   time.monotonic() < deadline):
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            # The first result becomes stale as soon as the newer local
            # reference arrives; only the latest reference may enter cache.
            self.assertEqual(values["sr_done"], 1)
            self.assertEqual(values["sr_jobs"], 2)
            self.assertEqual(values["sr_stale"], 1)
            self.assertFalse(values["sr_queued"])
            self.assertIn((5, 4, 3), composer.cache)
        finally:
            composer.close()

    def test_sr_expired_pending_candidate_is_rejected_before_submit(self):
        class ControlledResolver:
            scale = 2
            available = True
            model_name = "Fake-SR"

            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self.calls = 0

            def upscale(self, image, target_size):
                self.calls += 1
                self.started.set()
                self.release.wait(1.0)
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        resolver = ControlledResolver()
        first = self._visual_reference(reference_generation=2, received_at=10.0)
        newest = self._visual_reference(reference_generation=3, received_at=10.1)
        composer = RebuildComposer(resolver=resolver)
        try:
            composer.update_timing(1000 * 90, generation=5, active_track_ids={4})
            composer.prefetch(first)
            self.assertTrue(resolver.started.wait(1.0))
            composer.prefetch(newest)
            composer.update_timing(1500 * 90, generation=5, active_track_ids={4})
            resolver.release.set()
            deadline = time.monotonic() + 1.0
            while (composer.snapshot()["sr_stale"] < 1 and
                   time.monotonic() < deadline):
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            self.assertEqual(resolver.calls, 1)
            self.assertEqual(values["sr_jobs"], 1)
            self.assertEqual(values["sr_done"], 0)
            self.assertEqual(values["sr_x"], 1)
            self.assertEqual(values["sr_queue"], 0)
        finally:
            resolver.release.set()
            composer.close()

    def test_sr_result_is_stale_when_reference_generation_changes_during_compute(self):
        class ControlledResolver:
            scale = 2
            available = True
            model_name = "Fake-SR"

            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def upscale(self, image, target_size):
                self.started.set()
                self.release.wait(1.0)
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        resolver = ControlledResolver()
        old = self._visual_reference(reference_generation=20, received_at=10.0)
        new = self._visual_reference(reference_generation=21, received_at=10.1)
        composer = RebuildComposer(resolver=resolver)
        try:
            composer.update_timing(1000 * 90, generation=5, active_track_ids={4})
            composer.prefetch(old)
            self.assertTrue(resolver.started.wait(1.0))
            composer.prefetch(new)
            resolver.release.set()
            deadline = time.monotonic() + 1.0
            while (composer.snapshot()["sr_done"] < 1 and
                   time.monotonic() < deadline):
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            self.assertEqual(values["sr_jobs"], 2)
            self.assertEqual(values["sr_done"], 1)
            self.assertEqual(values["sr_stale"], 1)
            self.assertIn((5, 4, 21), composer.cache)
            self.assertNotIn((5, 4, 20), composer.cache)
        finally:
            resolver.release.set()
            composer.close()

    def test_same_source_frame_reuses_immutable_composed_pixels(self):
        class ManualResolver:
            scale = 2
            available = False
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        resolver = ManualResolver()
        reference = self._visual_reference()
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 0)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = np.full((144, 256, 3), 80, dtype=np.uint8)
        composer = RebuildComposer(resolver=resolver)
        try:
            first, first_mode = composer.render(
                base, state, 5, {4: reference}, now=10.0,
                video_rtp_timestamp=1000 * 90, source_sequence=100)
            resolver.available = True
            composer.cache[(5, 4, 2)] = np.full((160, 96, 3), 240, dtype=np.uint8)
            second, second_mode = composer.render(
                base, state, 5, {4: reference}, now=10.1,
                video_rtp_timestamp=1000 * 90, source_sequence=100)
            self.assertEqual(first_mode, "ROI-LANCZOS")
            self.assertEqual(second_mode, "ROI-LANCZOS")
            self.assertTrue(np.array_equal(first, second))

            third, third_mode = composer.render(
                base, state, 5, {4: reference}, now=10.2,
                video_rtp_timestamp=1000 * 90, source_sequence=101)
            self.assertEqual(third_mode, "ROI-ESRGAN")
            self.assertFalse(np.array_equal(first, third))
            self.assertEqual(composer.snapshot()["composed_frames"], 2)
        finally:
            composer.close()

    def test_registration_preserves_asymmetric_edge_crop_and_rejects_mismatch(self):
        reference = type("Reference", (), {
            "reference_bbox": (50, 40, 150, 140),
            "crop": (0, 20, 180, 160),
        })()
        target = rb.TargetState(4, 0, 91, 65, 55, 175, 165, 2, 1)
        registered = RebuildComposer._registered_crop(reference, target, 1.0, 1.0)
        self.assertEqual(registered, (10, 33, 208, 187))
        moved = rb.TargetState(4, 0, 91, 220, 80, 320, 180, 2, 1)
        self.assertEqual(
            RebuildComposer._registered_crop(reference, moved, 1.0, 1.0),
            (170, 60, 350, 200))
        enlarged = rb.TargetState(4, 0, 91, 100, 80, 260, 240, 2, 1)
        self.assertIsNone(RebuildComposer._registered_crop(reference, enlarged, 1.0, 1.0))

    def test_registration_reports_motion_and_fail_closed_geometry_reasons(self):
        reference = type("Reference", (), {
            "reference_bbox": (50, 40, 150, 140),
            "crop": (0, 20, 180, 160),
        })()
        moved = rb.TargetState(4, 0, 91, 220, 80, 320, 180, 2, 1)
        result = RebuildComposer._registration_details(
            reference, moved, 1.0, 1.0, output_size=(640, 360))
        self.assertEqual(result.reason, "NONE")
        self.assertGreater(result.dx_ratio, 1.0)
        self.assertEqual(result.area_ratio, 1.0)

        invalid = rb.TargetState(4, 0, 91, 100, 80, 90, 200, 2, 1)
        self.assertEqual(
            RebuildComposer._registration_details(reference, invalid, 1.0, 1.0).reason,
            "GEOM_INVALID")
        scaled = rb.TargetState(4, 0, 91, 100, 80, 300, 280, 2, 1)
        self.assertEqual(
            RebuildComposer._registration_details(reference, scaled, 1.0, 1.0).reason,
            "GEOM_SCALE")
        outside = rb.TargetState(4, 0, 91, 800, 80, 900, 180, 2, 1)
        self.assertEqual(
            RebuildComposer._registration_details(
                reference, outside, 1.0, 1.0, output_size=(640, 360)).reason,
            "GEOM_OUTSIDE")

    def test_moving_target_is_rebuilt_without_absolute_center_gate(self):
        class ManualResolver:
            scale = 2
            available = False
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        reference = self._visual_reference()
        target = rb.TargetState(4, 0, 91, 280, 80, 400, 280, 2, 1)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = np.full((144, 256, 3), 80, dtype=np.uint8)
        composer = RebuildComposer(resolver=ManualResolver())
        try:
            _, mode = composer.render(
                base, state, 5, {4: reference}, now=10.0,
                video_rtp_timestamp=1000 * 90, source_sequence=1)
            values = composer.snapshot()
            self.assertEqual(mode, "ROI-LANCZOS")
            self.assertEqual(values["refs_used"], 1)
            self.assertEqual(values["registration_drops"], 0)
            self.assertEqual(values["last_drop_reason"], "NONE")
            self.assertGreater(values["geom_dx_ratio"], 1.0)
        finally:
            composer.close()

    def test_content_mismatch_has_a_distinct_drop_reason(self):
        class ManualResolver:
            scale = 2
            available = False
            model_name = "Fake-SR"

            @staticmethod
            def upscale(image, target_size):
                return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

        random = np.random.RandomState(123)
        reference = dataclasses.replace(
            self._visual_reference(),
            image=random.randint(0, 256, (80, 48, 3), dtype=np.uint8))
        target = rb.TargetState(4, 0, 91, 100, 80, 220, 280, 2, 1)
        state = rb.StateRecord(640, 480, 640, 360, 6, 12, 1, (target,))
        base = random.randint(0, 256, (144, 256, 3), dtype=np.uint8)
        composer = RebuildComposer(resolver=ManualResolver())
        try:
            _, mode = composer.render(
                base, state, 5, {4: reference}, now=10.0,
                video_rtp_timestamp=1000 * 90, source_sequence=1)
            values = composer.snapshot()
            self.assertEqual(mode, "BASE-LANCZOS")
            self.assertEqual(values["last_drop_reason"], "CONTENT")
            self.assertEqual(values["content_drops"], 1)
            self.assertEqual(values["scale_drops"], values["geom_scale_drops"])
        finally:
            composer.close()

    def test_registration_uses_isotropic_area_scale(self):
        reference = type("Reference", (), {
            "reference_bbox": (100, 100, 200, 200),
            "crop": (80, 80, 220, 220),
        })()
        target = rb.TargetState(4, 0, 91, 100, 100, 230, 200, 2, 1)
        registered = RebuildComposer._registered_crop(reference, target, 1.0, 1.0)
        self.assertEqual(registered, (85, 70, 245, 230))

    def test_mask_core_is_fully_replaced_and_edge_is_feathered(self):
        canvas = np.full((40, 40, 3), 100, dtype=np.uint8)
        patch = np.full((20, 20, 3), (20, 40, 240), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:-2, 2:-2] = 255
        RebuildComposer._blend(canvas, patch, mask, 10, 10)
        self.assertEqual(tuple(canvas[20, 20]), (40, 60, 220))
        self.assertLess(int(canvas[13, 20, 0]), 100)
        self.assertLess(int(canvas[13, 20, 2]), 220)


if __name__ == "__main__":
    unittest.main()

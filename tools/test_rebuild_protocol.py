#!/usr/bin/env python3

import importlib.util
import os
import sys
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

    def test_protocol_bounds_untrusted_dimensions(self):
        oversized = rb.StateRecord(640, 480, 4096, 4096, 6, 12, 1, ())
        with self.assertRaises(ValueError):
            rb.parse_state(rb.build_state(oversized))

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
        self.assertEqual(len(receiver.state_history), 1)

        # A reordered copy of the earlier STATE must not clear readiness.
        receiver.on_datagram(rb.build_packet(rb.Packet(
            rb.STATE, rb.PROFILE_REBUILD, 5, 0, 10, 1, 1000,
            rb.build_state(pending))), now=1.2)
        selected, _, _, _, _ = receiver.scene_synced(1000 * 90)
        self.assertEqual(selected.targets[0].reference_generation, 2)
        self.assertEqual(selected.targets[0].flags, 1)

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
        output, spatial = composer.render(base, scene, generation, references, now=2.5)
        self.assertEqual(output.shape, (360, 640, 3))
        self.assertEqual(spatial, "ROI-LANCZOS")
        self.assertGreater(int(output[:, :, 2].max()), 100)
        self.assertGreater(composer.snapshot()["rebuild_percent"], 0.0)
        self.assertEqual(composer.snapshot()["chroma_mode"], "COLOR")

        gray = np.full((144, 256, 3), 80, dtype=np.uint8)
        mono, spatial = composer.render(gray, scene, generation, references, now=2.6)
        self.assertEqual(spatial, "ROI-LANCZOS")
        self.assertEqual(composer.snapshot()["chroma_mode"], "MONO")
        self.assertLessEqual(int(np.max(mono.max(axis=2) - mono.min(axis=2))), 1)

        undeclared = rb.StateRecord(
            640, 480, 640, 360, 6, 12, 1,
            (rb.TargetState(4, 0, 91, 100, 80, 220, 280, 1, 0),))
        base_only, spatial = composer.render(gray, undeclared, generation, references, now=2.7)
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
            output_size=(640, 360), resolver=FakeResolver(), reference_max_age=1.0)
        try:
            # This is the crucial ordering: arrival triggers SR before the
            # reference is advertised by a PTS-matched STATE.  At 999 ms of
            # content age there is no time left to begin a GPU job.
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
                video_rtp_timestamp=1999 * 90)
            self.assertEqual(spatial, "ROI-ESRGAN")
            self.assertEqual(composer.snapshot()["reference_content_age_ms"], 999)
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
            composer.prefetch(first)
            composer.prefetch(newest)
            self.assertEqual(composer.snapshot()["sr_jobs"], 1)
            self.assertTrue(composer.snapshot()["sr_queued"])
            deadline = time.monotonic() + 1.0
            while composer.snapshot()["sr_done"] != 2 and time.monotonic() < deadline:
                composer.prefetch_pending()
                time.sleep(0.005)
            values = composer.snapshot()
            self.assertEqual(values["sr_done"], 2)
            self.assertEqual(values["sr_jobs"], 2)
            self.assertFalse(values["sr_queued"])
            self.assertIn((5, 4, 3), composer.cache)
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
        moved = rb.TargetState(4, 0, 91, 220, 80, 340, 200, 2, 1)
        self.assertIsNone(RebuildComposer._registered_crop(reference, moved, 1.0, 1.0))
        enlarged = rb.TargetState(4, 0, 91, 100, 80, 260, 240, 2, 1)
        self.assertIsNone(RebuildComposer._registered_crop(reference, enlarged, 1.0, 1.0))

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

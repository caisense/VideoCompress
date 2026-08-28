#!/usr/bin/env python3
"""Regression tests for the durable Python snapshot receiver."""

import importlib.util
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("receive_snapshot_udp.py")
SPEC = importlib.util.spec_from_file_location("receive_snapshot_udp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SnapshotReceiverTest(unittest.TestCase):
    def packet(self, packet_type, payload=b"", offset=0):
        self.data = b"high-resolution evidence over a restarted receiver"
        return MODULE.Packet(packet_type, 0x12345678, len(self.data), offset, payload,
                             width=1920, height=1080, class_mask=0x0003,
                             crc32=zlib.crc32(self.data) & 0xFFFFFFFF)

    def test_packet_round_trip_and_restart_resume(self):
        original = self.packet(MODULE.DATA, b"abc", 0)
        self.assertEqual(MODULE.Packet.decode(original.encode()), original)
        with self.assertRaises(ValueError):
            MODULE.Packet.decode(original.encode()[:-1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MODULE.SnapshotStore(root, verbose=False)
            self.assertEqual(first.handle(self.packet(MODULE.START)).packet_type, MODULE.RESUME)
            half = len(self.data) // 2
            reply = first.handle(self.packet(MODULE.DATA, self.data[:half], 0))
            self.assertEqual((reply.packet_type, reply.offset), (MODULE.ACK, half))

            # A new receiver instance derives the durable offset from .part.
            first.close()
            restarted = MODULE.SnapshotStore(root, verbose=False)
            reply = restarted.handle(self.packet(MODULE.START))
            self.assertEqual((reply.packet_type, reply.offset), (MODULE.RESUME, half))
            reply = restarted.handle(self.packet(MODULE.DATA, self.data[half:], half))
            self.assertEqual((reply.packet_type, reply.offset), (MODULE.ACK, len(self.data)))
            reply = restarted.handle(self.packet(MODULE.END, offset=len(self.data)))
            self.assertEqual((reply.packet_type, reply.offset), (MODULE.COMPLETE, len(self.data)))
            files = list(root.glob("*.jpg"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), self.data)

    def test_batched_sync_avoids_a_fsync_per_data_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.SnapshotStore(Path(directory), verbose=False,
                                         sync_every_bytes=16)
            # START establishes an empty durable boundary before measurement.
            store.handle(self.packet(MODULE.START))
            with mock.patch.object(MODULE.os, "fsync", wraps=MODULE.os.fsync) as fsync:
                data_packets = 0
                for offset in range(0, len(self.data), 4):
                    payload = self.data[offset:offset + 4]
                    reply = store.handle(self.packet(MODULE.DATA, payload, offset))
                    self.assertEqual((reply.packet_type, reply.offset),
                                     (MODULE.ACK, offset + len(payload)))
                    data_packets += 1
                fsyncs_before_end = fsync.call_count
                self.assertLess(fsyncs_before_end, data_packets)
                # END always synchronizes the remaining tail before completion.
                complete = store.handle(self.packet(MODULE.END, offset=len(self.data)))
                self.assertEqual((complete.packet_type, complete.offset),
                                 (MODULE.COMPLETE, len(self.data)))
                self.assertEqual(fsync.call_count, fsyncs_before_end + 1)

    def test_start_synchronizes_a_small_unsynced_tail_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.SnapshotStore(Path(directory), verbose=False,
                                         sync_every_bytes=4096)
            store.handle(self.packet(MODULE.START))
            half = len(self.data) // 2
            with mock.patch.object(MODULE.os, "fsync", wraps=MODULE.os.fsync) as fsync:
                store.handle(self.packet(MODULE.DATA, self.data[:half], 0))
                self.assertEqual(fsync.call_count, 0)
                resume = store.handle(self.packet(MODULE.START))
                self.assertEqual((resume.packet_type, resume.offset), (MODULE.RESUME, half))
                self.assertEqual(fsync.call_count, 1)
            store.close()

    def test_gap_and_crc_failure_request_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.SnapshotStore(Path(directory), verbose=False)
            store.handle(self.packet(MODULE.START))
            gap = store.handle(self.packet(MODULE.DATA, b"bad", 3))
            self.assertEqual((gap.packet_type, gap.offset), (MODULE.ACK, 0))
            bad = self.packet(MODULE.DATA, b"x" * len(self.data), 0)
            ack = store.handle(bad)
            self.assertEqual(ack.offset, len(self.data))
            resume = store.handle(self.packet(MODULE.END, offset=len(self.data)))
            self.assertEqual((resume.packet_type, resume.offset), (MODULE.RESUME, 0))

    def test_listener_uses_bounded_receive_timeout(self):
        listener = MODULE.create_listener(0)
        try:
            self.assertEqual(listener.gettimeout(), MODULE.RECEIVE_POLL_TIMEOUT_SECONDS)
        finally:
            listener.close()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the fixed-size ROEV/1 packet parser."""

import unittest

from detection_event_protocol import FLAG_HEARTBEAT, HEADER, Packet, mask_names


class DetectionEventProtocolTest(unittest.TestCase):
    def test_round_trip_and_layout(self) -> None:
        source = Packet(
            sequence=0x01020304,
            frame_id=0x0102030405060708,
            pts_ms=123456,
            present_mask=0b0011,
            entered_mask=0b0011,
            counts=(2, 1, 0, 0),
            max_confidence_percent=(93, 81, 0, 0),
        )
        wire = source.encode()
        self.assertEqual(len(wire), HEADER.size)
        self.assertEqual(wire[:6], b"ROEV\x01\x01")
        parsed = Packet.decode(wire)
        self.assertEqual(parsed.sequence, source.sequence)
        self.assertEqual(parsed.frame_id, source.frame_id)
        self.assertEqual(parsed.pts_ms, source.pts_ms)
        self.assertEqual(parsed.present_mask, source.present_mask)
        self.assertEqual(parsed.entered_mask, source.entered_mask)
        self.assertEqual(parsed.counts, source.counts)
        self.assertEqual(mask_names(parsed.present_mask), ("person", "car"))

    def test_heartbeat_and_crc_rejection(self) -> None:
        source = Packet(
            sequence=9, frame_id=10, pts_ms=11, present_mask=0b0100,
            counts=(0, 0, 1, 0), max_confidence_percent=(0, 0, 74, 0),
            flags=FLAG_HEARTBEAT,
        )
        wire = bytearray(source.encode())
        self.assertEqual(Packet.decode(bytes(wire)).flags, FLAG_HEARTBEAT)
        wire[20] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC32"):
            Packet.decode(bytes(wire))

    def test_shared_cxx_golden_wire_vector(self) -> None:
        # This exact vector is also asserted in cpp/tests/test_detection_event_protocol.cc.
        # Keeping it here verifies that the board C++ producer and PC Python
        # consumer agree on offsets, network byte order and CRC coverage.
        wire = bytes.fromhex(
            "524f455601010000010203040102030405060708112233440005000100000000"
            "020001005b004b00cdb8e559"
        )
        parsed = Packet.decode(wire)
        self.assertEqual(parsed.sequence, 0x01020304)
        self.assertEqual(parsed.frame_id, 0x0102030405060708)
        self.assertEqual(parsed.pts_ms, 0x11223344)
        self.assertEqual(parsed.present_mask, 0b0101)
        self.assertEqual(parsed.entered_mask, 0b0001)
        self.assertEqual(parsed.counts, (2, 0, 1, 0))
        self.assertEqual(parsed.max_confidence_percent, (91, 0, 75, 0))
        self.assertEqual(parsed.encode(), wire)

    def test_invalid_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "entered"):
            Packet(sequence=1, frame_id=1, pts_ms=1, present_mask=0,
                   entered_mask=1).encode()
        with self.assertRaisesRegex(ValueError, "absent"):
            Packet(sequence=1, frame_id=1, pts_ms=1, present_mask=0,
                   counts=(1, 0, 0, 0)).encode()


if __name__ == "__main__":
    unittest.main()

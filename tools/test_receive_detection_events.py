#!/usr/bin/env python3
"""Regression tests for ROEV/1 human-facing receiver wording."""

import contextlib
import io
import json
import unittest

from detection_event_protocol import FLAG_HEARTBEAT, Packet
from receive_detection_events import display_text, print_packet


class DetectionEventDisplayTest(unittest.TestCase):
    def test_entered_class_uses_appearance_wording_without_confidence(self) -> None:
        packet = Packet(
            sequence=1, frame_id=10, pts_ms=100,
            present_mask=0b0011, entered_mask=0b0001,
            counts=(2, 1, 0, 0), max_confidence_percent=(35, 76, 0, 0),
        )
        self.assertEqual(display_text(packet), "2个人出现；当前检测到1辆车")
        self.assertNotIn("35", display_text(packet))
        self.assertNotIn("76", display_text(packet))

    def test_heartbeat_and_exit_use_current_and_left_wording(self) -> None:
        heartbeat = Packet(
            sequence=2, frame_id=11, pts_ms=200,
            present_mask=0b0011, counts=(2, 1, 0, 0),
            max_confidence_percent=(90, 80, 0, 0), flags=FLAG_HEARTBEAT,
        )
        exited = Packet(
            sequence=3, frame_id=12, pts_ms=300,
            present_mask=0b0010, exited_mask=0b0001,
            counts=(0, 1, 0, 0), max_confidence_percent=(0, 81, 0, 0),
        )
        self.assertEqual(display_text(heartbeat), "当前检测到2个人；当前检测到1辆车")
        self.assertEqual(display_text(exited), "人员离开；当前检测到1辆车")

    def test_json_uses_protocol_mask_names(self) -> None:
        packet = Packet(
            sequence=4, frame_id=13, pts_ms=400,
            present_mask=0b0001, entered_mask=0b0001,
            counts=(2, 0, 0, 0), max_confidence_percent=(35, 0, 0, 0),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_packet(packet, as_json=True)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["present_mask"], 0b0001)
        self.assertEqual(payload["entered_mask"], 0b0001)
        self.assertEqual(payload["exited_mask"], 0)
        self.assertEqual(payload["present_classes"], ["person"])
        self.assertNotIn("present", payload)


if __name__ == "__main__":
    unittest.main()

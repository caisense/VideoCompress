#!/usr/bin/env python3
"""Small dependency-free checks for the Codec2 RTP receiver parser."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("receive_codec2_rtp.py")
SPEC = importlib.util.spec_from_file_location("receive_codec2_rtp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RECEIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIVER)


class Codec2RtpReceiverTest(unittest.TestCase):
    def test_wire_accounting_matches_sender_pacer(self) -> None:
        # 14 Codec2 bytes plus the fixed 12-byte RTP header make a 26-byte
        # UDP payload.  On Ethernet it costs 92 bytes, or 9.2 kbps at the
        # 12.5 packet/s cadence of Codec2 1300 with two frames/packet.
        self.assertEqual(RECEIVER.wire_bytes_for_udp_payload(26), 92)
        self.assertEqual(RECEIVER.wire_bytes_for_udp_payload(20), 86)
        with self.assertRaises(ValueError):
            RECEIVER.wire_bytes_for_udp_payload(-1)

    def test_sdp_and_rtp_header_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.sdp"
            path.write_text(
                "v=0\n"
                "m=audio 5006 RTP/AVP 97\n"
                "a=rtpmap:97 CODEC2/8000/1\n"
                "a=fmtp:97 mode=1300;frames-per-packet=2;bits-per-frame=52;bytes-per-frame=7;"
                "dtx=1;dtx-keepalive-ms=1000\n",
                encoding="utf-8")
            config = RECEIVER.parse_audio_sdp(path)
        self.assertEqual(config["port"], 5006)
        self.assertEqual(config["bytes-per-frame"], 7)
        self.assertEqual(config["dtx"], 1)
        self.assertEqual(config["dtx-keepalive-ms"], 1000)

        payload = bytes(range(14))
        # RTP V2, extension flag, marker, dynamic payload type 97, sequence 9,
        # timestamp 640, SSRC 0x524f4932, then a zero-word extension.
        packet = bytes([0x90, 0xE1, 0x00, 0x09, 0x00, 0x00, 0x02, 0x80,
                        0x52, 0x4F, 0x49, 0x32, 0x52, 0x4F, 0x00, 0x00]) + payload
        payload_type, sequence, timestamp, marker, decoded_payload = RECEIVER.parse_rtp(packet)
        self.assertEqual((payload_type, sequence, timestamp, marker), (97, 9, 640, True))
        self.assertEqual(decoded_payload, payload)

    def test_legacy_sdp_defaults_to_dtx_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sdp"
            path.write_text(
                "m=audio 5006 RTP/AVP 97\n"
                "a=rtpmap:97 CODEC2/8000/1\n"
                "a=fmtp:97 mode=1300;frames-per-packet=2;bits-per-frame=52;bytes-per-frame=7\n",
                encoding="utf-8")
            config = RECEIVER.parse_audio_sdp(path)
        self.assertEqual(config["dtx"], 0)
        self.assertEqual(config["dtx-keepalive-ms"], 0)

    def test_rejects_bad_audio_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sdp"
            path.write_text("m=audio 5006 RTP/AVP 97\na=rtpmap:97 CODEC2/8000/1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                RECEIVER.parse_audio_sdp(path)

    def test_wsl_decoder_command_keeps_udp_receiver_native(self) -> None:
        command = RECEIVER.decoder_command("ignored", 1300, "Ubuntu-22.04")
        self.assertEqual(command[:6], ["wsl.exe", "-d", "Ubuntu-22.04", "--", "/bin/bash", "-lc"])
        self.assertIn("$HOME/.local/share/codex-codec2-1.2.0-build/src/c2dec", command[6])
        self.assertTrue(command[6].endswith("1300 - -"))

    def test_ffplay_pcm_options_cover_old_and_new_ffmpeg(self) -> None:
        current = RECEIVER.ffplay_pcm_command("ffplay", 8000, True)
        self.assertIn("-sample_rate", current)
        self.assertIn("-ch_layout", current)
        self.assertNotIn("-ac", current)
        self.assertEqual(current[-2:], ["-i", "-"])

        legacy = RECEIVER.ffplay_pcm_command("ffplay", 8000, False)
        self.assertIn("-ar", legacy)
        self.assertIn("-ac", legacy)
        self.assertNotIn("-ch_layout", legacy)

    def test_default_decoder_is_a_usable_command(self) -> None:
        decoder = RECEIVER.default_c2dec_path()
        self.assertTrue(decoder.endswith("c2dec") or decoder.lower().endswith("c2dec.exe"))


if __name__ == "__main__":
    unittest.main()

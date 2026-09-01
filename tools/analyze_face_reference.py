#!/usr/bin/env python3
"""Summarize a real rebuild person/adaptive capture.

The sender and HUD logs intentionally use cumulative counters.  This tool
converts them into one bounded active-run summary so the post-run HUD tail
cannot inflate drop rates after the board has stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


KEY_VALUE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")


def parse_values(line: str) -> Dict[str, str]:
    return dict(KEY_VALUE.findall(line))


def number(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def integer(row: Dict[str, str], key: str, default: int = 0) -> int:
    return int(round(number(row, key, default)))


def slash_pair(value: str, default: int = 0) -> List[int]:
    parts = value.split("/")
    result: List[int] = []
    for part in parts:
        try:
            result.append(int(float(part)))
        except ValueError:
            result.append(default)
    return result


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "p10": percentile(samples, 0.10),
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "p95": percentile(samples, 0.95),
        "mean": (sum(samples) / len(samples)) if samples else None,
    }


def read_sender(path: Path) -> List[Dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("frame="):
            rows.append(parse_values(line))
    return rows


def sender_duration(rows: Sequence[Dict[str, str]], fps: float) -> float:
    capture = [integer(row, "rebuild_state_capture_us") for row in rows
               if integer(row, "rebuild_state_capture_us") > 0]
    if len(capture) >= 2:
        return max(1.0, (max(capture) - min(capture)) / 1_000_000.0)
    frames = [integer(row, "frame") for row in rows]
    return max(1.0, ((max(frames) - min(frames) + 1) / fps) if frames else 1.0)


def reference_samples(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    previous = 0
    for row in rows:
        total = integer(row, "rebuild_refs")
        delta = total - previous
        if delta > 0:
            samples.extend([row] * delta)
        previous = max(previous, total)
    return samples


def active_hud_rows(path: Path, duration: float, fps: float) -> Dict[str, List[Dict[str, str]]]:
    timing: List[Dict[str, str]] = []
    profiles: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SEQ="):
            timing.append(parse_values(line))
        elif line.startswith("profile=rebuild:"):
            profiles.append(parse_values(line))
    # The viewer continues to print HOLD/BASE lines while the board is gone.
    # Keep only the number of source-frame observations in the sender window.
    timing_limit = max(1, int(round(duration * fps)) + 2)
    profile_limit = max(1, int(math.ceil(duration)) + 2)
    profiles = profiles[:profile_limit]
    nonzero = [index for index, row in enumerate(profiles)
               if number(row, "video_pps") > 0 or number(row, "rb_pps") > 0 or
               number(row, "wire_kbps") > 0]
    if nonzero:
        profiles = profiles[:nonzero[-1] + 1]
    return {"timing": timing[:timing_limit], "profiles": profiles}


def first_last(row_list: Sequence[Dict[str, str]], key: str) -> int:
    return integer(row_list[-1], key) if row_list else 0


def visual_proxy_metrics(sender_dir: Optional[Path], receiver_dir: Optional[Path]) -> Dict[str, object]:
    if sender_dir is None or receiver_dir is None:
        return {"matched": 0, "psnr_db": summary([]), "ssim": summary([])}
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"matched": 0, "psnr_db": summary([]), "ssim": summary([]),
                "unavailable": "opencv/numpy"}

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    matched = 0
    for source_path in sorted(sender_dir.glob("*_source.png")):
        final_path = receiver_dir / (source_path.stem[:-len("_source")] + "_final.png")
        if not final_path.exists():
            continue
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        final = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
        if source is None or final is None or source.size == 0 or final.size == 0:
            continue
        source = cv2.resize(source, (final.shape[1], final.shape[0]),
                            interpolation=cv2.INTER_AREA)
        mse = float(np.mean((source.astype(np.float32) - final.astype(np.float32)) ** 2))
        psnr_values.append(10.0 * math.log10((255.0 * 255.0) / mse)
                           if mse > 0.0 else 99.0)
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float64)
        final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY).astype(np.float64)
        c1, c2 = 6.5025, 58.5225
        mu_a = cv2.GaussianBlur(source_gray, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(final_gray, (11, 11), 1.5)
        sigma_a = cv2.GaussianBlur(source_gray * source_gray, (11, 11), 1.5) - mu_a * mu_a
        sigma_b = cv2.GaussianBlur(final_gray * final_gray, (11, 11), 1.5) - mu_b * mu_b
        sigma_ab = cv2.GaussianBlur(source_gray * final_gray, (11, 11), 1.5) - mu_a * mu_b
        ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2) /
                    ((mu_a * mu_a + mu_b * mu_b + c1) *
                     (sigma_a + sigma_b + c2)))
        ssim_values.append(float(np.mean(ssim_map)))
        matched += 1
    return {"matched": matched, "psnr_db": summary(psnr_values),
            "ssim": summary(ssim_values),
            "definition": "same-key sender source crop vs receiver final crop; proxy, not independent GT"}


def analyze(sender_path: Path, hud_path: Path, fps: float,
            audio_wire_packet_bytes: int, sender_debug: Optional[Path],
            receiver_debug: Optional[Path]) -> Dict[str, object]:
    rows = read_sender(sender_path)
    if not rows:
        raise ValueError(f"no sender frame rows in {sender_path}")
    duration = sender_duration(rows, fps)
    refs = reference_samples(rows)
    hud = active_hud_rows(hud_path, duration, fps)
    timing = hud["timing"]
    profiles = hud["profiles"]
    last_sender = rows[-1]
    last_profile = profiles[-1] if profiles else {}
    active_last = last_profile

    def ref_values(key: str) -> List[float]:
        return [number(row, key) for row in refs if key in row]

    debug_jpeg_sizes = []
    if sender_debug is not None and sender_debug.exists():
        debug_jpeg_sizes = [float(path.stat().st_size)
                            for path in sorted(sender_debug.glob("*_jpeg.jpg"))]
    jpeg_sizes = debug_jpeg_sizes or ref_values("rebuild_ref_jpeg_bytes")

    modes = [row.get("MODE", "UNKNOWN") for row in timing]
    mode_counts = {mode: modes.count(mode) for mode in sorted(set(modes))}
    mode_transitions = sum(1 for left, right in zip(modes, modes[1:]) if left != right)
    sr_hits = sum(1 for row in timing if row.get("SR") == "HIT")
    audio_packets = integer(last_sender, "audio_rtp")
    event_wire = integer(last_sender, "event_wire_bytes")
    h265_wire_samples = [number(row, "wire_kbps") for row in profiles]
    rb_wire_samples = [number(row, "rb_wire_kbps") for row in profiles]
    media_wire_samples = [number(row, "video_rb_wire_kbps") for row in profiles]
    h265_wire = percentile(h265_wire_samples, 0.50) or 0.0
    rb_wire = percentile(rb_wire_samples, 0.50) or 0.0
    media_wire = percentile(media_wire_samples, 0.50) or 0.0
    audio_wire_kbps = audio_packets * audio_wire_packet_bytes * 8.0 / duration / 1000.0
    event_wire_kbps = event_wire * 8.0 / duration / 1000.0
    total_wire = media_wire + audio_wire_kbps + event_wire_kbps
    sr_done_pair = slash_pair(active_last.get("sr_done", "0/0"))
    sr_first_pair = slash_pair(active_last.get("sr_first", "0/0"))
    sr_latency_pair = slash_pair(active_last.get("sr_ms", "0/0/0"))
    ref_delivery = [number(row, "rebuild_ref_delivery_p95_us") / 1000.0
                    for row in rows if number(row, "rebuild_ref_delivery_p95_us") > 0]
    queue_delay = [number(row, "rebuild_ref_q_delay_us") / 1000.0
                   for row in refs if number(row, "rebuild_ref_q_delay_us") >= 0]
    summary_result: Dict[str, object] = {
        "duration_s": duration,
        "sender_frames": len(rows),
        "hud_source_frames": len(timing),
        "reference_transfers": len(refs),
        "reference_transfers_per_s": len(refs) / duration,
        "reference_kind_counts": {
            "FULL": sum(1 for row in refs if row.get("rebuild_ref_kind") == "FULL"),
            "HEAD": sum(1 for row in refs if row.get("rebuild_ref_kind") == "HEAD"),
        },
        "head_fallbacks": first_last(rows, "rebuild_head_fallbacks"),
        "jpeg_bytes": summary(jpeg_sizes),
        "jpeg_quality": summary(ref_values("rebuild_ref_jpeg_q")),
        "head_pixels_width": summary(ref_values("rebuild_ref_head_px_w")),
        "head_pixels_height": summary(ref_values("rebuild_ref_head_px_h")),
        "head_pixels_area": summary(ref_values("rebuild_ref_head_px_area")),
        "head_gain_linear": summary(ref_values("rebuild_ref_gain_linear")),
        "head_gain_area": summary(ref_values("rebuild_ref_gain_area")),
        "crop_source_width": summary(ref_values("rebuild_ref_crop_src_w")),
        "crop_source_height": summary(ref_values("rebuild_ref_crop_src_h")),
        "h265_payload_kbps": sum(number(row, "tx_bytes") for row in rows) * 8.0 / duration / 1000.0,
        "h265_wire_kbps": h265_wire,
        "h265_wire_p95_kbps": percentile(h265_wire_samples, 0.95) or 0.0,
        "rb_wire_kbps": rb_wire,
        "rb_wire_p95_kbps": percentile(rb_wire_samples, 0.95) or 0.0,
        "video_rb_wire_kbps": media_wire,
        "video_rb_wire_p95_kbps": percentile(media_wire_samples, 0.95) or 0.0,
        "video_rb_wire_max_kbps": max(media_wire_samples, default=0.0),
        "audio_wire_kbps": audio_wire_kbps,
        "audio_reserved_kbps": number(last_sender, "audio_reserve_bps") / 1000.0,
        "event_wire_kbps": event_wire_kbps,
        "v_rb_audio_wire_kbps": media_wire + audio_wire_kbps,
        "total_wire_kbps": total_wire,
        "total_wire_p95_kbps": ((percentile(media_wire_samples, 0.95) or 0.0) +
                                audio_wire_kbps + event_wire_kbps),
        "max_total_wire_kbps": (max(media_wire_samples, default=0.0) +
                                 audio_wire_kbps + event_wire_kbps),
        "jpeg_cap_ok": max(jpeg_sizes, default=0.0) <= 800.0,
        "link_cap_kbps": number(last_profile, "link_cap_kbps", 100.0),
        "wire_source": "HUD rolling 1s H265/RB plus exact Codec2 RTP packet geometry from sender counters",
        "age_drop_rate": integer(active_last, "agedrop") / max(1, len(timing)),
        "state_drop_rate": integer(active_last, "statedrop") / max(1, len(timing)),
        "content_drop_rate": integer(active_last, "matchdrop") / max(1, len(timing)),
        "drops": {key: integer(active_last, key) for key in (
            "agedrop", "futuredrop", "norefdrop", "statedrop", "gendrop",
            "geomdrop", "scaledrop", "geomoutside", "matchdrop", "playout_drop")},
        "reference_delivery_p95_ms": summary(ref_delivery),
        "reference_queue_delay_ms": summary(queue_delay),
        "sr": {
            "hit_rate": sr_hits / max(1, len(timing)),
            "done": sr_done_pair[0] if sr_done_pair else 0,
            "jobs": sr_done_pair[1] if len(sr_done_pair) > 1 else 0,
            "stale": integer(active_last, "sr_stale"),
            "latency_ms": {
                "last": sr_latency_pair[0] if sr_latency_pair else 0,
                "p50": sr_latency_pair[1] if len(sr_latency_pair) > 1 else 0,
                "p95": sr_latency_pair[2] if len(sr_latency_pair) > 2 else 0,
            },
            "first_frame_hits": sr_first_pair,
            "model_warmup_line": "CUDAExecutionProvider / RealESRGAN_x2_dynamic.onnx",
        },
        "render_modes": mode_counts,
        "render_mode_percent": {
            mode: count * 100.0 / max(1, len(modes))
            for mode, count in mode_counts.items()
        },
        "mode_transitions": mode_transitions,
        "visual_proxy": visual_proxy_metrics(sender_debug, receiver_debug),
    }
    return summary_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender-log", type=Path, required=True)
    parser.add_argument("--hud-log", type=Path, required=True)
    parser.add_argument("--sender-debug-dir", type=Path, default=None)
    parser.add_argument("--receiver-debug-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--audio-wire-packet-bytes", type=int, default=102)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if args.fps <= 0 or args.audio_wire_packet_bytes <= 0:
        parser.error("fps and audio-wire-packet-bytes must be positive")
    result = analyze(args.sender_log, args.hud_log, args.fps,
                     args.audio_wire_packet_bytes, args.sender_debug_dir,
                     args.receiver_debug_dir)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

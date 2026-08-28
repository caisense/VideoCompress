#!/usr/bin/env python3
"""Evaluate GOAL.md's A/B/C HEVC experiments without non-standard decoding.

The sender emits standard H.265 over RTP/UDP.  Save each received stream as a
raw .h265 file, then run this program on the PC with the original input video
and the segmentation run's 16x16 PGM maps.  It writes both JSON and Markdown.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def run_ffmpeg_decode(path: Path, width: int, height: int, fps: int, hevc: bool) -> List[bytes]:
    command = ["ffmpeg", "-v", "error"]
    if hevc:
        command += ["-f", "hevc"]
    command += ["-i", str(path), "-vf", f"fps={fps},scale={width}:{height}",
                "-an", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    try:
        raw = subprocess.check_output(command, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for A/B/C evaluation") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg cannot decode {path}: {exc.stderr.decode(errors='replace')}") from exc
    frame_size = width * height
    if not raw or len(raw) % frame_size:
        raise RuntimeError(f"decoded data for {path} is not whole {width}x{height} luma frames")
    return [raw[offset:offset + frame_size] for offset in range(0, len(raw), frame_size)]


def load_pgm(path: Path, width: int, height: int) -> bytes:
    with path.open("rb") as source:
        magic = source.readline().strip()
        if magic != b"P5":
            raise RuntimeError(f"{path} is not binary PGM")
        header: List[bytes] = []
        while len(header) < 3:
            line = source.readline()
            if not line:
                raise RuntimeError(f"truncated PGM header: {path}")
            if line.startswith(b"#"):
                continue
            header.extend(line.split())
        image_width, image_height, maximum = (int(value) for value in header[:3])
        if (image_width, image_height, maximum) != (width, height, 255):
            raise RuntimeError(f"unexpected PGM dimensions/range in {path}")
        image = source.read(width * height)
        if len(image) != width * height:
            raise RuntimeError(f"truncated PGM pixels: {path}")
        return image


def numeric_sort_key(path: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)(?=\.pgm$)", Path(path).name)
    return (int(match.group(1)) if match else -1, path)


def load_roi_masks(directory: Path, width: int, height: int, count: int) -> Tuple[List[bytes], List[bytes]]:
    maps = sorted(glob.glob(str(directory / "*.pgm")), key=numeric_sort_key)
    if len(maps) < count:
        raise RuntimeError(
            f"need at least {count} debug ROI PGM files in {directory}, found {len(maps)}; "
            "run the segmentation sender with --debug-roi=on --debug-roi-path='.../roi_{frame_id}.pgm'"
        )
    roi_masks: List[bytes] = []
    edge_masks: List[bytes] = []
    for path in maps[:count]:
        image = load_pgm(Path(path), width, height)
        # roi_debug encodes BACKGROUND=32, HALO=96, CORE=176, EDGE=255.
        roi_masks.append(bytes(1 if pixel > 32 else 0 for pixel in image))
        edge_masks.append(bytes(1 if pixel == 255 else 0 for pixel in image))
    return roi_masks, edge_masks


def psnr(reference: bytes, tested: bytes, mask: Optional[bytes]) -> Optional[float]:
    values = [(a, b) for a, b, enabled in zip(reference, tested, mask or bytes([1]) * len(reference)) if enabled]
    if not values:
        return None
    mse = sum((int(a) - int(b)) ** 2 for a, b in values) / len(values)
    return 99.0 if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)


def block_ssim(reference: bytes, tested: bytes, width: int, height: int, mask: Optional[bytes]) -> Optional[float]:
    c1 = 6.5025
    c2 = 58.5225
    scores: List[float] = []
    for y0 in range(0, height, 8):
        for x0 in range(0, width, 8):
            samples: List[Tuple[int, int]] = []
            for y in range(y0, min(y0 + 8, height)):
                offset = y * width
                for x in range(x0, min(x0 + 8, width)):
                    if mask is None or mask[offset + x]:
                        samples.append((reference[offset + x], tested[offset + x]))
            if len(samples) < 8:
                continue
            left = [float(pair[0]) for pair in samples]
            right = [float(pair[1]) for pair in samples]
            mean_left = statistics.fmean(left)
            mean_right = statistics.fmean(right)
            variance_left = sum((value - mean_left) ** 2 for value in left) / len(left)
            variance_right = sum((value - mean_right) ** 2 for value in right) / len(right)
            covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / len(left)
            scores.append(((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) /
                          ((mean_left ** 2 + mean_right ** 2 + c1) * (variance_left + variance_right + c2)))
    return statistics.fmean(scores) if scores else None


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def quality_metrics(reference: Sequence[bytes], decoded: Sequence[bytes], roi_masks: Sequence[bytes],
                    edge_masks: Sequence[bytes], width: int, height: int) -> Dict[str, Optional[float]]:
    count = min(len(reference), len(decoded), len(roi_masks), len(edge_masks))
    if count == 0:
        raise RuntimeError("no common frames for quality comparison")
    background_masks = [bytes(0 if enabled else 1 for enabled in mask) for mask in roi_masks[:count]]
    return {
        "frames": count,
        "global_psnr_db": average(psnr(reference[i], decoded[i], None) for i in range(count)),
        "global_ssim": average(block_ssim(reference[i], decoded[i], width, height, None) for i in range(count)),
        "roi_psnr_db": average(psnr(reference[i], decoded[i], roi_masks[i]) for i in range(count)),
        "roi_ssim": average(block_ssim(reference[i], decoded[i], width, height, roi_masks[i]) for i in range(count)),
        "background_psnr_db": average(psnr(reference[i], decoded[i], background_masks[i]) for i in range(count)),
        "background_ssim": average(block_ssim(reference[i], decoded[i], width, height, background_masks[i]) for i in range(count)),
        "edge_psnr_db": average(psnr(reference[i], decoded[i], edge_masks[i]) for i in range(count)),
        "edge_ssim": average(block_ssim(reference[i], decoded[i], width, height, edge_masks[i]) for i in range(count)),
    }


def parse_sender_log(path: Optional[Path], fps: int) -> Dict[str, Optional[float]]:
    if path is None:
        return {"average_bitrate_bps": None, "peak_1s_bitrate_bps": None, "average_qp": None,
                "average_e2e_latency_ms": None}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries = []
    for line in lines:
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        if "bytes" in fields:
            entries.append(fields)
    if not entries:
        raise RuntimeError(f"no sender statistics records in {path}")
    encoded_bytes = [int(item["bytes"]) for item in entries]
    qps = [int(item["qp"]) for item in entries if item.get("qp", "-1").lstrip("-").isdigit() and int(item["qp"]) >= 0]
    e2e = [int(item["e2e_us"]) / 1000.0 for item in entries if item.get("e2e_us", "").isdigit()]
    instantaneous = [int(item["inst_bps"]) for item in entries if item.get("inst_bps", "").isdigit()]
    return {
        "average_bitrate_bps": sum(encoded_bytes) * 8.0 * fps / len(encoded_bytes),
        "peak_1s_bitrate_bps": max(instantaneous) if instantaneous else None,
        "average_qp": statistics.fmean(qps) if qps else None,
        "average_e2e_latency_ms": statistics.fmean(e2e) if e2e else None,
    }


def parse_resource_csv(path: Optional[Path]) -> Dict[str, Optional[float]]:
    empty = {"cpu_percent": None, "npu_percent": None, "vpu_percent": None}
    if path is None:
        return empty
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(rows) < 2:
        raise RuntimeError(f"resource CSV has no samples: {path}")
    columns = rows[0].split(",")
    values: Dict[str, List[float]] = {key: [] for key in empty}
    for row in rows[1:]:
        parts = row.split(",")
        for column, value in zip(columns, parts):
            if column in values and value.strip():
                values[column].append(float(value))
    return {key: statistics.fmean(value) if value else None for key, value in values.items()}


def parse_key_value_paths(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"expected MODE=PATH, got {value}")
        mode, path = value.split("=", 1)
        if mode not in ("A", "B", "C"):
            raise RuntimeError(f"mode must be A, B, or C: {value}")
        result[mode] = Path(path)
    return result


def format_value(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: Dict[str, object]) -> None:
    rows = ["# Segmentation-Aware ROI H.265 A/B/C report", "",
            "| Metric | A: normal H.265 | B: bbox ROI | C: segmentation ROI |",
            "|---|---:|---:|---:|"]
    metrics = [
        ("Average bitrate (bps)", "transport", "average_bitrate_bps"),
        ("1 s peak bitrate (bps)", "transport", "peak_1s_bitrate_bps"),
        ("Global PSNR (dB)", "quality", "global_psnr_db"),
        ("Global SSIM", "quality", "global_ssim"),
        ("ROI PSNR (dB)", "quality", "roi_psnr_db"),
        ("ROI SSIM", "quality", "roi_ssim"),
        ("Background PSNR (dB)", "quality", "background_psnr_db"),
        ("Background SSIM", "quality", "background_ssim"),
        ("Target edge PSNR (dB)", "quality", "edge_psnr_db"),
        ("Target edge SSIM", "quality", "edge_ssim"),
        ("Average QP", "transport", "average_qp"),
        ("End-to-end latency (ms)", "transport", "average_e2e_latency_ms"),
        ("RK3588 CPU (%)", "resources", "cpu_percent"),
        ("RK3588 NPU (%)", "resources", "npu_percent"),
        ("RK3588 VPU (%)", "resources", "vpu_percent"),
    ]
    for label, section, metric in metrics:
        values = [report[mode][section][metric] for mode in ("A", "B", "C")]
        rows.append("| " + label + " | " + " | ".join(format_value(value) for value in values) + " |")
    rows.extend(["", "`n/a` means the corresponding sender log or resource CSV was not supplied; it is not a passing measurement."])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="original common input video")
    parser.add_argument("--baseline", type=Path, required=True, help="received A raw .h265")
    parser.add_argument("--bbox", type=Path, required=True, help="received B raw .h265")
    parser.add_argument("--segmentation", type=Path, required=True, help="received C raw .h265")
    parser.add_argument("--roi-maps", type=Path, required=True, help="C debug ROI PGM directory")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--log", action="append", default=[], metavar="MODE=PATH")
    parser.add_argument("--resources", action="append", default=[], metavar="MODE=CSV")
    parser.add_argument("--output", type=Path, default=Path("ab_report.json"))
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise RuntimeError("width, height, and fps must be positive")
    logs = parse_key_value_paths(args.log)
    resources = parse_key_value_paths(args.resources)
    reference = run_ffmpeg_decode(args.reference, args.width, args.height, args.fps, False)
    decoded = {
        "A": run_ffmpeg_decode(args.baseline, args.width, args.height, args.fps, True),
        "B": run_ffmpeg_decode(args.bbox, args.width, args.height, args.fps, True),
        "C": run_ffmpeg_decode(args.segmentation, args.width, args.height, args.fps, True),
    }
    common_frames = min([len(reference)] + [len(value) for value in decoded.values()])
    roi_masks, edge_masks = load_roi_masks(args.roi_maps, args.width, args.height, common_frames)
    report: Dict[str, object] = {}
    for mode in ("A", "B", "C"):
        report[mode] = {
            "quality": quality_metrics(reference[:common_frames], decoded[mode][:common_frames], roi_masks, edge_masks,
                                       args.width, args.height),
            "transport": parse_sender_log(logs.get(mode), args.fps),
            "resources": parse_resource_csv(resources.get(mode)),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Summarize per-source-frame rebuild timing logs."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FIELD_RE = re.compile(r"([A-Z_]+)=([^\s]+)")
NUMERIC_RE = re.compile(r"^-?\d+$")


def percentile(values: Iterable[int], ratio: float) -> Optional[int]:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(len(ordered) * ratio) - 1)]


def parse_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SEQ=" not in line:
            continue
        fields = dict(FIELD_RE.findall(line))
        if "SEQ" in fields and "MODE" in fields:
            rows.append(fields)
    return rows


def integer(fields: Dict[str, str], name: str) -> Optional[int]:
    value = fields.get(name)
    return int(value) if value and NUMERIC_RE.match(value) else None


def run_lengths(rows: List[Dict[str, str]], predicate) -> int:
    longest = current = 0
    previous_seq: Optional[int] = None
    for row in rows:
        sequence = integer(row, "SEQ")
        if sequence is None or (previous_seq is not None and sequence != previous_seq + 1):
            current = 0
        if predicate(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous_seq = sequence
    return longest


def transition_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    wanted = {
        ("ROI-ESRGAN", "BASE-LANCZOS"): "ESRGAN_TO_BASE",
        ("BASE-LANCZOS", "ROI-ESRGAN"): "BASE_TO_ESRGAN",
        ("ROI-ESRGAN", "ROI-LANCZOS"): "ESRGAN_TO_LANCZOS",
        ("ROI-LANCZOS", "ROI-ESRGAN"): "LANCZOS_TO_ESRGAN",
    }
    counts = collections.Counter()
    previous: Optional[str] = None
    for row in rows:
        current = row["MODE"]
        name = wanted.get((previous, current))
        if name:
            counts[name] += 1
        previous = current
    return {name: counts[name] for name in wanted.values()}


def state_delta(row: Dict[str, str]) -> Optional[int]:
    telemetry = integer(row, "STATE_DELTA")
    if telemetry is not None:
        return telemetry
    video_pts = integer(row, "VPTS")
    state_pts = integer(row, "SPTS")
    if video_pts is None or state_pts is None:
        return None
    return state_pts - video_pts


def drop_category(reason: str) -> str:
    if reason.startswith("STATE_") or reason == "STATE":
        return "STATE"
    if reason.startswith("CONTENT_") or reason == "CONTENT":
        return "CONTENT"
    return reason


def first_reference_modes(rows: List[Dict[str, str]]) -> Dict[str, int]:
    modes = collections.Counter()
    seen = set()
    for row in rows:
        rgen = row.get("RGEN")
        reference_pts = row.get("RPTS")
        if not rgen or rgen == "none" or not reference_pts or reference_pts == "none":
            continue
        # RGEN may repeat for distinct tracks in old logs.  RPTS disambiguates
        # those records until TRACK becomes part of every debug line.
        key = (row.get("TRACK", "?"), rgen, reference_pts)
        if key not in seen:
            seen.add(key)
            modes[row["MODE"]] += 1
    return dict(modes)


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    modes = collections.Counter(row["MODE"] for row in rows)
    reasons = [row.get("DROP_REASON", row.get("DROP", "UNKNOWN")) for row in rows]
    drops = collections.Counter(drop_category(reason) for reason in reasons)
    drop_reasons = collections.Counter(reasons)
    ages = [value for row in rows if (value := integer(row, "AGE")) is not None]
    deltas = [abs(value) for row in rows if (value := state_delta(row)) is not None]
    arrival_ages = [value for row in rows
                    if (value := integer(row, "STATE_ARRIVAL_AGE")) is not None]
    rgen_lags = [value for row in rows if (value := integer(row, "RGEN_LAG")) is not None]
    playout_latencies = [value for row in rows
                         if (value := integer(row, "PLAYOUT_MS")) is not None]
    age_drop_rows = [
        row for row in rows
        if row.get("DROP_REASON", row.get("DROP", "UNKNOWN")) == "AGE"
    ]
    mode_rates = {
        name: (count * 100.0 / len(rows) if rows else 0.0)
        for name, count in modes.items()
    }
    return {
        "source_frames": len(rows),
        "modes": dict(modes),
        "mode_rates_percent": mode_rates,
        "drops": dict(drops),
        "drop_reasons": dict(drop_reasons),
        "transitions": transition_counts(rows),
        "longest_base_run": run_lengths(rows, lambda row: row["MODE"] == "BASE-LANCZOS"),
        "longest_state_drop_run": run_lengths(
            rows, lambda row: drop_category(
                row.get("DROP_REASON", row.get("DROP", "UNKNOWN"))) == "STATE"),
        "longest_content_drop_run": run_lengths(
            rows, lambda row: drop_category(
                row.get("DROP_REASON", row.get("DROP", "UNKNOWN"))) == "CONTENT"),
        "reference_age_ms": {
            "p50": percentile(ages, 0.50),
            "p95": percentile(ages, 0.95),
            "max": max(ages) if ages else None,
        },
        "state_abs_delta_ms": {
            "p50": percentile(deltas, 0.50),
            "p95": percentile(deltas, 0.95),
            "max": max(deltas) if deltas else None,
        },
        "state_arrival_age_ms": {
            "p50": percentile(arrival_ages, 0.50),
            "p95": percentile(arrival_ages, 0.95),
            "max": max(arrival_ages) if arrival_ages else None,
        },
        "rgen_lag": {
            "p50": percentile(rgen_lags, 0.50),
            "p95": percentile(rgen_lags, 0.95),
            "max": max(rgen_lags) if rgen_lags else None,
        },
        "playout_latency_ms": {
            "p50": percentile(playout_latencies, 0.50),
            "p95": percentile(playout_latencies, 0.95),
            "max": max(playout_latencies) if playout_latencies else None,
        },
        "state_extrapolated_frames": sum(
            row.get("STATE_EXTRAP") == "1" for row in rows),
        "age_drop_rgen_lag": dict(collections.Counter(
            row.get("RGEN_LAG", "none") for row in age_drop_rows)),
        "age_drop_state_reason": dict(collections.Counter(
            row.get("STATE_REASON", "none") for row in age_drop_rows)),
        "new_reference_first_frame_modes": first_reference_modes(rows),
        "base_without_reason": sum(
            row["MODE"] == "BASE-LANCZOS" and
            row.get("DROP_REASON", row.get("DROP", "NONE")) == "NONE"
            for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="rebuild debug timing log")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    summary = summarize(parse_rows(args.log))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"source_frames={summary['source_frames']}")
    for name in ("modes", "mode_rates_percent", "drops", "drop_reasons", "transitions",
                 "new_reference_first_frame_modes", "age_drop_rgen_lag",
                 "age_drop_state_reason"):
        print(f"{name}={json.dumps(summary[name], ensure_ascii=False, sort_keys=True)}")
    for name in ("longest_base_run", "longest_state_drop_run", "longest_content_drop_run",
                 "base_without_reason", "state_extrapolated_frames"):
        print(f"{name}={summary[name]}")
    for name in ("reference_age_ms", "state_abs_delta_ms", "state_arrival_age_ms",
                 "rgen_lag", "playout_latency_ms"):
        print(f"{name}={json.dumps(summary[name], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()

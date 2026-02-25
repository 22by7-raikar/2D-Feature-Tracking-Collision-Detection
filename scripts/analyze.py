#!/usr/bin/env python3
"""
Analyze keypoint and match statistics produced by 2D_feature_tracking.

Usage
-----
    python3 scripts/analyze.py                        # reads from project root
    python3 scripts/analyze.py --keypoints k.csv --matches m.csv
    python3 scripts/analyze.py --top 5               # show top 5 combinations
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _default_path(name: str) -> Path:
    """Walk up from this script to the project root and find the CSV."""
    here = Path(__file__).resolve().parent
    root = here.parent
    return root / name


# ---------------------------------------------------------------------------
# Keypoint analysis
# ---------------------------------------------------------------------------

def analyse_keypoints(rows: List[Dict[str, str]], top: int) -> None:
    per_detector: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        per_detector[r["DetectorType"]].append(int(r["NumKeypoints"]))

    print("\n=== Keypoint counts per detector (avg over all images) ===")
    ranked = sorted(
        [(det, statistics.mean(counts)) for det, counts in per_detector.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, (det, avg) in enumerate(ranked, 1):
        print(f"  {i:2d}. {det:<12s}: {avg:7.1f} keypoints/image")


# ---------------------------------------------------------------------------
# Match analysis
# ---------------------------------------------------------------------------

def analyse_matches(rows: List[Dict[str, str]], top: int) -> None:
    combos: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        key = f"{r['DetectorType']}/{r['DescriptorType']}"
        combos[key].append(int(r["NumMatches"]))

    ranked = sorted(
        [(k, statistics.mean(v), min(v), max(v)) for k, v in combos.items()],
        key=lambda x: x[1],
        reverse=True,
    )  # type: List[Tuple[str, float, int, int]]

    print(f"\n=== Top {top} combinations by average match count ===")
    header = f"  {'#':>3}  {'Detector/Descriptor':<22}  {'Mean':>6}  {'Min':>5}  {'Max':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (combo, mean, mn, mx) in enumerate(ranked[:top], 1):
        print(f"  {i:3d}  {combo:<22}  {mean:6.1f}  {mn:5d}  {mx:5d}")

    print(f"\n=== Bottom {top} combinations by average match count ===")
    for i, (combo, mean, mn, mx) in enumerate(ranked[-top:], 1):
        print(f"  {i:3d}  {combo:<22}  {mean:6.1f}  {mn:5d}  {mx:5d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse 2D feature tracking results.")
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=None,
        help="Path to keypoint_log.csv (default: <project_root>/keypoint_log.csv)",
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="Path to match_log.csv (default: <project_root>/match_log.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top/bottom combinations to display (default: 10)",
    )
    args = parser.parse_args()

    kpt_path = args.keypoints or _default_path("keypoint_log.csv")
    match_path = args.matches or _default_path("match_log.csv")

    if not kpt_path.exists():
        print(f"[ERROR] Keypoint log not found: {kpt_path}")
        print("  Run the tracker first: cd build && ./2D_feature_tracking")
        raise SystemExit(1)

    if not match_path.exists():
        print(f"[ERROR] Match log not found: {match_path}")
        print("  Run the tracker first: cd build && ./2D_feature_tracking")
        raise SystemExit(1)

    kpt_rows   = load_csv(kpt_path)
    match_rows = load_csv(match_path)

    analyse_keypoints(kpt_rows, args.top)
    analyse_matches(match_rows, args.top)


if __name__ == "__main__":
    main()

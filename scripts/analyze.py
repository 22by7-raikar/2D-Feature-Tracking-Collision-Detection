#!/usr/bin/env python3
"""Summarise results/results.csv written by 2D_feature_tracking.

    python3 scripts/analyze.py                 # print a markdown summary table
    python3 scripts/analyze.py --plot out.png  # also draw speed-vs-matches and TTC charts (needs matplotlib)
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    """Group CSV rows by (detector, descriptor), keeping file order."""
    combos = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            combos[(row["Detector"], row["Descriptor"])].append(row)
    return combos


def summarise(rows):
    matched = [r for r in rows if r["Matches"]]
    ttc = [float(r["TTC"]) for r in matched if r["TTC"]]
    return {
        "keypoints": statistics.mean(int(r["Keypoints"]) for r in rows),
        "matches": statistics.mean(int(r["Matches"]) for r in matched) if matched else 0.0,
        "ms": statistics.mean(float(r["DetectMs"]) + float(r["DescribeMs"]) for r in rows),
        "ttc_median": statistics.median(ttc) if ttc else float("nan"),
        "ttc_min": min(ttc, default=float("nan")),
        "ttc_max": max(ttc, default=float("nan")),
        "ttc": [(int(r["Frame"]), float(r["TTC"]) if r["TTC"] else None) for r in matched],
    }


def print_table(stats):
    print("| Detector / Descriptor | Keypoints in ROI | Matches | ms / frame | TTC median (s) | TTC range (s) |")
    print("|---|---:|---:|---:|---:|---:|")
    for (det, desc), s in sorted(stats.items(), key=lambda kv: -kv[1]["matches"]):
        print(f"| {det} / {desc} | {s['keypoints']:.0f} | {s['matches']:.0f} | {s['ms']:.1f} "
              f"| {s['ttc_median']:.1f} | {s['ttc_min']:.1f} – {s['ttc_max']:.1f} |")


DETECTORS = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
DESCRIPTORS = ["BRISK", "ORB", "AKAZE", "SIFT", "BRIEF", "FREAK"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]  # one per detector
MARKERS = ["o", "s", "^", "D", "v", "P"]                                                # one per descriptor


def plot(stats, highlight, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colour = dict(zip(DETECTORS, COLOURS))
    marker = dict(zip(DESCRIPTORS, MARKERS))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(surface)

    # Left: cost vs. number of matches. Colour = detector, shape = descriptor.
    for (det, desc), s in stats.items():
        ax1.scatter(s["ms"], s["matches"], s=48, color=colour[det], marker=marker[desc],
                    edgecolor=surface, linewidth=1.2, zorder=3)
    for det, desc in highlight:
        s = stats[(det, desc)]
        right = s["ms"] > 30  # put labels of right-hand points on their left
        ax1.annotate(f"{det}/{desc}", (s["ms"], s["matches"]), xytext=(-7 if right else 7, 4),
                     textcoords="offset points", fontsize=8, color=ink, ha="right" if right else "left")
    ax1.set_xscale("log")
    ax1.set_xlabel("detect + describe time per frame (ms, log scale)")
    ax1.set_ylabel("matches per frame pair (mean)")
    ax1.set_title("Speed vs. matches  (up and left is better)", loc="left", fontsize=10, color=ink)
    ax1.grid(color=grid, linewidth=0.6, zorder=0)

    used_det = [d for d in DETECTORS if any(k[0] == d for k in stats)]
    used_desc = [d for d in DESCRIPTORS if any(k[1] == d for k in stats)]
    det_handles = [Line2D([], [], ls="", marker="o", color=colour[d], label=d) for d in used_det]
    desc_handles = [Line2D([], [], ls="", marker=marker[d], color=muted, label=d) for d in used_desc]
    leg = ax1.legend(handles=det_handles, title="detector", frameon=False, fontsize=7.5,
                     title_fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax1.add_artist(leg)
    ax1.legend(handles=desc_handles, title="descriptor", frameon=False, fontsize=7.5,
               title_fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0.0))

    # Right: TTC over the sequence for the highlighted combinations (same detector colours).
    for det, desc in highlight:
        pts = [(f, t) for f, t in stats[(det, desc)]["ttc"] if t is not None]
        ax2.plot(*zip(*pts), color=colour[det], linewidth=2, marker=marker[desc], markersize=5,
                 markeredgecolor=surface, label=f"{det}/{desc}")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("camera TTC (s)")
    ax2.set_title("Time-to-collision per frame", loc="left", fontsize=10, color=ink)
    ax2.grid(color=grid, linewidth=0.6)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, facecolor=surface)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=ROOT / "results" / "results.csv")
    p.add_argument("--plot", type=Path, help="write a PNG summary chart to this path")
    p.add_argument("--highlight", nargs="+", default=["FAST/ORB", "BRISK/BRISK", "AKAZE/AKAZE"],
                   help="combinations to highlight and show in the TTC chart (max 3)")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found - run ./build/2D_feature_tracking first")

    stats = {combo: summarise(rows) for combo, rows in load(args.csv).items()}
    print_table(stats)
    if args.plot:
        highlight = [tuple(h.upper().split("/")) for h in args.highlight[:3]]
        plot(stats, [h for h in highlight if h in stats], args.plot)


if __name__ == "__main__":
    main()

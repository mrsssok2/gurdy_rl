#!/usr/bin/env python3
"""
Plot a path logged by dashboard.py (output/demo/path_<timestamp>.csv) as a
top-down PNG: the path coloured by status, start and end, walk targets and
where Gurdy fell. Needs only matplotlib (no ROS).

    python3 scripts/plot_path.py output/demo/path_20261109_101500.csv [-o out.png]

The PNG goes next to the CSV by default.
"""
import argparse
import csv
import math
import os
import sys

STATUS_COLORS = {"STANDING": "#2e7d32", "WALKING": "#1565c0", "STOPPED": "#ef6c00",
                 "FALLEN": "#c62828"}
RESET_JUMP_M = 0.10  # as dashboard_state: a bigger jump between rows is a reset


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_path(path):
    """Rows of the CSV with x, y, time_s, target_x, target_y as floats (None if empty)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("time_s", "x", "y", "target_x", "target_y", "walk_m"):
                row[key] = _float(row.get(key))
            if row["x"] is not None and row["y"] is not None:
                rows.append(row)
    return rows


def segments(rows):
    """Runs of consecutive rows with one status and no reset jump: [(status, [rows])]."""
    out = []
    for prev, row in zip([None] + rows, rows):
        jump = prev is not None and math.hypot(row["x"] - prev["x"], row["y"] - prev["y"]) > RESET_JUMP_M
        if out and not jump and out[-1][0] == row["status"]:
            out[-1][1].append(row)
        else:
            # Start the new run at the previous point so the line is continuous
            out.append((row["status"], [prev, row] if prev is not None and not jump else [row]))
    return out


def plot_path(rows, png_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    seen = set()
    for status, run in segments(rows):
        ax.plot([r["x"] for r in run], [r["y"] for r in run], "-", lw=2.5,
                color=STATUS_COLORS.get(status, "#555555"),
                label=status.lower() if status not in seen else None)
        seen.add(status)
    ax.plot(rows[0]["x"], rows[0]["y"], "o", color="#2e7d32", ms=12, label="start")
    ax.plot(rows[-1]["x"], rows[-1]["y"], "o", color="#c62828", ms=12, mfc="none", mew=2.5,
            label="end")
    targets = sorted({(r["target_x"], r["target_y"]) for r in rows
                      if r["target_x"] is not None and r["target_y"] is not None})
    if targets:
        ax.plot([t[0] for t in targets], [t[1] for t in targets], "X", color="#6a1b9a", ms=14,
                label="walk target")
    falls = [r for prev, r in zip([None] + rows, rows)
             if r["status"] == "FALLEN" and (prev is None or prev["status"] != "FALLEN")]
    if falls:
        ax.plot([r["x"] for r in falls], [r["y"] for r in falls], "*", color="#c62828", ms=18,
                label="fall")

    xs = [r["x"] for r in rows] + [t[0] for t in targets]
    ys = [r["y"] for r in rows] + [t[1] for t in targets]
    span = max(0.6, max(xs) - min(xs) + 0.3, max(ys) - min(ys) + 0.3)
    cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    duration = ""
    if rows[0]["time_s"] is not None and rows[-1]["time_s"] is not None:
        duration = ", %.0f s" % (rows[-1]["time_s"] - rows[0]["time_s"])
    net = rows[-1]["x"] - rows[0]["x"]
    ax.set_title("%s\nnet x %+.3f m%s" % (title or "Gurdy path", net, duration))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="path CSV written by dashboard.py")
    parser.add_argument("-o", "--output", help="PNG to write (default: next to the CSV)")
    args = parser.parse_args(argv)
    rows = read_path(args.csv)
    if not rows:
        sys.exit("No positions in %s" % args.csv)
    png = args.output or os.path.splitext(args.csv)[0] + ".png"
    plot_path(rows, png, title=os.path.basename(args.csv))
    print("Wrote %s (%d points)" % (png, len(rows)))


if __name__ == "__main__":
    main()

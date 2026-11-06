#!/usr/bin/env python3
"""
find_windows.py — find time windows where the storm is wide from the first
frame to the last, so the sculpture has width across its whole height.

A frame "qualifies" when, at the chosen cloud-top threshold:
    min_cover <= fraction of frame covered <= max_cover
    and no frame edge is covered by more than edge_limit
A window is a continuous run of qualifying frames. Runs are ranked by length,
then by how even the width is (smallest coverage / largest coverage).

Two modes
---------
folder   an extracted day (output of goes_extract_v2.py), at full cadence
         python find_windows.py folder data/2026-09-02_ir

survey   many days, sampled sparsely from the satellite archive, to find
         candidate days before extracting any of them
         python find_windows.py survey --from 2026-06-01 --to 2026-09-30

Thresholds (defaults)
    --threshold 220     cloud-top temperature, kelvin
    --min-cover 0.15    every frame at least 15% covered: width from the start
    --max-cover 0.60    at most 60% covered: avoids a frame-filling block
    --edge-limit 0.35   cut faces allowed, but no edge more than 35% covered
    --min-hours 3       shortest window worth reporting
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Shared measurement and run-finding
# --------------------------------------------------------------------------


def measure(T, threshold):
    """Coverage fraction and worst edge contact of one frame (kelvin array)."""
    m = np.nan_to_num(T, nan=400.0) < threshold
    edge = max(m[0, :].mean(), m[-1, :].mean(), m[:, 0].mean(), m[:, -1].mean())
    return float(m.mean()), float(edge)


def reasons(cover, edge, a):
    """Why frames fail: counts of too little cover, too much, edge contact."""
    low = sum(c < a.min_cover for c in cover)
    high = sum(c > a.max_cover for c in cover)
    edgy = sum(
        (a.min_cover <= c <= a.max_cover) and e > a.edge_limit
        for c, e in zip(cover, edge)
    )
    return f"too little cover {low}, too much {high}, edge contact {edgy}"


def runs(times, cover, edge, a):
    """All maximal runs of qualifying frames, longest first."""
    ok = [
        (a.min_cover <= c <= a.max_cover) and (e <= a.edge_limit)
        for c, e in zip(cover, edge)
    ]
    out, i = [], 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        hours = (times[j] - times[i]).total_seconds() / 3600
        c = cover[i : j + 1]
        out.append(
            {
                "start": times[i],
                "end": times[j],
                "hours": hours,
                "frames": j - i + 1,
                "cover_start": c[0],
                "cover_mean": float(np.mean(c)),
                "cover_min": float(min(c)),
                "cover_max": float(max(c)),
                "evenness": float(min(c) / max(c)) if max(c) > 0 else 0.0,
                "edge_max": float(max(edge[i : j + 1])),
            }
        )
        i = j + 1
    out.sort(key=lambda r: (r["hours"], r["evenness"]), reverse=True)
    return out


def add_common(p):
    p.add_argument("--threshold", type=float, default=220.0)
    p.add_argument("--min-cover", type=float, default=0.15)
    p.add_argument("--max-cover", type=float, default=0.60)
    p.add_argument("--edge-limit", type=float, default=0.35)
    p.add_argument("--min-hours", type=float, default=3.0)


def fmt(r):
    return (
        f"{r['start']:%Y-%m-%d %H:%M}-{r['end']:%H:%M}  {r['hours']:4.1f} h  "
        f"cover {100 * r['cover_start']:3.0f}% at start, "
        f"{100 * r['cover_min']:3.0f}-{100 * r['cover_max']:3.0f}% range, "
        f"evenness {r['evenness']:.2f}, max edge {100 * r['edge_max']:3.0f}%"
    )


# --------------------------------------------------------------------------
# Mode 1: an extracted folder
# --------------------------------------------------------------------------


def run_folder(a):
    folder = Path(a.folder)
    with open(folder / "index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    times, cover, edge = [], [], []
    for r in rows:
        c, e = measure(np.load(folder / r["file"])["cmi"], a.threshold)
        times.append(datetime.fromisoformat(r["scan_local"]))
        cover.append(c)
        edge.append(e)
    found = [r for r in runs(times, cover, edge, a) if r["hours"] >= a.min_hours]
    print(
        f"{len(times)} frames, {times[0]:%H:%M}-{times[-1]:%H:%M}, "
        f"threshold {a.threshold:.0f} K\n"
    )
    if not found:
        best = runs(times, cover, edge, a)[:3]
        print(f"No window of {a.min_hours} h or more meets the criteria.")
        for r in best:
            print("  closest:", fmt(r))
        print(f"\nFrames failing: {reasons(cover, edge, a)} (of {len(cover)})")
        print("\nHourly coverage and worst edge contact:")
        for t, c, e in zip(times, cover, edge):
            if t.minute < 10:
                print(f"  {t:%H:%M}  cover {100 * c:3.0f}%   edge {100 * e:3.0f}%")
        print("\nTry --threshold 230, a lower --min-cover, or a higher --edge-limit.")
    for k, r in enumerate(found[:5], 1):
        print(f"{k}. {fmt(r)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hrs = [(t - times[0]).total_seconds() / 3600 for t in times]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(hrs, 100 * np.array(cover), color="#378ADD", lw=1.6, label="Coverage")
    ax.plot(hrs, 100 * np.array(edge), color="0.4", lw=1, ls="--", label="Worst edge")
    ax.axhspan(100 * a.min_cover, 100 * a.max_cover, color="#9FE1CB", alpha=0.25, lw=0)
    for r in found[:3]:
        x0 = (r["start"] - times[0]).total_seconds() / 3600
        x1 = (r["end"] - times[0]).total_seconds() / 3600
        ax.axvspan(x0, x1, color="#D85A30", alpha=0.18, lw=0)
    step = max(1, len(times) // 12)
    ax.set_xticks(hrs[::step])
    ax.set_xticklabels([t.strftime("%H:%M") for t in times[::step]], fontsize=8)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "windows.png", dpi=130)
    print(
        f"\nPlot written to {folder / 'windows.png'} "
        f"(green band = target coverage, orange = best windows)"
    )


# --------------------------------------------------------------------------
# Mode 2: survey many days from the archive
# --------------------------------------------------------------------------


def run_survey(a):
    try:
        import goes_extract_v2 as ge
    except ImportError:
        import goes_extract as ge
    if not hasattr(ge, "display"):
        raise SystemExit(
            "This needs the updated extraction script (goes_extract_v2.py)."
        )
    ge.BAND = a.band.upper()

    if a.dates:
        days = [date.fromisoformat(d) for d in a.dates]
    else:
        d0, d1 = date.fromisoformat(a.date_from), date.fromisoformat(a.date_to)
        days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]

    fs = ge.connect()
    cache = {}
    slices = {}

    def survey_day(day):
        times, cover, edge = [], [], []
        for t_local in ge.target_times(
            day, time.fromisoformat(a.start), time.fromisoformat(a.end), a.every
        ):
            try:
                hit = ge.nearest_scan(fs, t_local.astimezone(timezone.utc), cache)
                if hit is None:
                    continue
                if "s" not in slices:
                    slices["s"] = ge.crop_slices(ge.open_remote(fs, hit[1]), a.bbox)
                T = ge.read_crop(fs, hit[1], slices["s"])["cmi"]
            except Exception as exc:  # keep the survey going
                print(f"  {day} {t_local:%H:%M}: skipped ({exc.__class__.__name__})")
                continue
            c, e = measure(T, a.threshold)
            times.append(t_local)
            cover.append(c)
            edge.append(e)
        return day, times, cover, edge

    results, frame_rows = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for day, times, cover, edge in pool.map(survey_day, days):
            frame_rows += [
                [t.isoformat(), f"{c:.3f}", f"{e:.3f}"]
                for t, c, e in zip(times, cover, edge)
            ]
            rr = runs(times, cover, edge, a) if times else []
            best = rr[0] if rr else None
            peak = max(cover) if cover else 0.0
            print(
                f"{day}  {len(times):2d} frames  peak {100 * peak:3.0f}%  "
                + (
                    f"best run {best['start']:%H:%M}-{best['end']:%H:%M} "
                    f"({best['hours']:.0f} h)"
                    if best
                    else f"none ({reasons(cover, edge, a)})"
                )
            )
            if best:
                results.append(best)

    results.sort(key=lambda r: (r["hours"], r["evenness"]), reverse=True)
    out = Path(a.out)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "date",
                "start",
                "end",
                "hours",
                "cover_start",
                "cover_min",
                "cover_max",
                "evenness",
                "edge_max",
            ]
        )
        for r in results:
            w.writerow(
                [
                    f"{r['start']:%Y-%m-%d}",
                    f"{r['start']:%H:%M}",
                    f"{r['end']:%H:%M}",
                    f"{r['hours']:.1f}",
                    f"{r['cover_start']:.2f}",
                    f"{r['cover_min']:.2f}",
                    f"{r['cover_max']:.2f}",
                    f"{r['evenness']:.2f}",
                    f"{r['edge_max']:.2f}",
                ]
            )
    with open(out.with_name(out.stem + "_frames.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_local", "cover", "edge"])
        w.writerows(frame_rows)

    print(f"\nTop candidates (windows of {a.min_hours} h or more):")
    shown = [r for r in results if r["hours"] >= a.min_hours][:10]
    for k, r in enumerate(shown, 1):
        print(f"{k:2d}. {fmt(r)}")
    if not shown:
        print("  none; try --threshold 230 or a lower --min-cover")
    print(f"\nAll days written to {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("folder", help="find windows in an extracted day")
    f.add_argument("folder")
    add_common(f)

    s = sub.add_parser("survey", help="survey many days from the archive")
    s.add_argument("--from", dest="date_from", default="2026-06-01")
    s.add_argument("--to", dest="date_to", default="2026-09-30")
    s.add_argument("--dates", nargs="+", help="specific dates instead of a range")
    s.add_argument("--start", default="09:00", help="local time")
    s.add_argument("--end", default="23:00", help="local time")
    s.add_argument("--every", type=int, default=60, help="minutes between samples")
    s.add_argument("--band", default="C13")
    s.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-82.6, 42.3, -76.6, 45.1),
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", default="survey.csv")
    add_common(s)

    a = p.parse_args()
    run_folder(a) if a.mode == "folder" else run_survey(a)


if __name__ == "__main__":
    main()

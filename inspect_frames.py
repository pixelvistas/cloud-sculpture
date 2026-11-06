#!/usr/bin/env python3
"""
inspect_frames.py — checkpoint between Stage 2 and Stage 3.

Reads the frames written by goes_extract_v2.py (infrared, band 13) and
answers two questions before any geometry is built:
  1. Does the storm decay, or leave the frame, before the window ends?
  2. Where, and for how long, does the storm touch the frame edges?

It also shows how much of the frame each candidate cloud-top temperature
threshold would capture, which is the main decision in Stage 3.

Outputs (in the data folder):
  inspect_timeseries.png  cold-cloud fraction and edge contact over time
  inspect_sheet.png       hourly contact sheet with threshold outlines
  inspect_loop.gif        animation of all frames
and prints a short summary.

Usage:
  python inspect_frames.py data/2026-09-02_ir --thresholds 235 220 205
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def load(folder):
    folder = Path(folder)
    with open(folder / "index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    times, temps = [], []
    for r in rows:
        z = np.load(folder / r["file"])
        times.append(datetime.fromisoformat(r["scan_local"]))
        temps.append(z["cmi"].astype(np.float32))
    return times, np.stack(temps)  # (frames, rows, cols), kelvin


def edge_contact(mask):
    """Fraction of each frame edge covered by the mask: N, S, W, E."""
    return {
        "N": mask[:, 0, :].mean(1),
        "S": mask[:, -1, :].mean(1),
        "W": mask[:, :, 0].mean(1),
        "E": mask[:, :, -1].mean(1),
    }


def display(t):
    return np.clip((300.0 - np.nan_to_num(t, nan=300.0)) / 110.0, 0, 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[235, 220, 205],
        help="cloud-top temperatures in kelvin",
    )
    p.add_argument(
        "--edge-limit",
        type=float,
        default=0.10,
        help="edge coverage fraction treated as touching the frame",
    )
    a = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    times, T = load(a.folder)
    out = Path(a.folder)
    hrs = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    labels = [t.strftime("%H:%M") for t in times]
    ths = sorted(a.thresholds, reverse=True)
    colors = ["#5DCAA5", "#378ADD", "#D85A30", "#7F77DD"]

    # --- time series ------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    print(
        f"{len(times)} frames, {labels[0]} to {labels[-1]}, "
        f"frame {T.shape[2]} x {T.shape[1]} px\n"
    )
    for th, c in zip(ths, colors):
        m = T < th
        frac = m.mean(axis=(1, 2))
        ax1.plot(hrs, 100 * frac, color=c, lw=1.6, label=f"< {th:.0f} K")
        i_peak = int(np.argmax(frac))
        print(
            f"< {th:.0f} K ({th - 273.15:.0f} C): peak {100 * frac[i_peak]:.0f}% "
            f"at {labels[i_peak]}; last frame {100 * frac[-1]:.0f}% "
            f"({'decaying' if frac[-1] < 0.5 * frac[i_peak] else 'NOT yet decayed'})"
        )
    ax1.set_ylabel("Frame covered [%]")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    mid = T < ths[len(ths) // 2]
    ec = edge_contact(mid)
    for (side, v), ls in zip(ec.items(), ["-", "--", "-.", ":"]):
        ax2.plot(hrs, 100 * v, ls=ls, color="0.25", lw=1.3, label=side)
    ax2.set_ylabel(f"Edge covered < {ths[len(ths) // 2]:.0f} K [%]")
    ax2.legend(frameon=False, fontsize=9, ncol=4)
    ax2.grid(alpha=0.25)
    step = max(1, len(times) // 12)
    ax2.set_xticks(hrs[::step])
    ax2.set_xticklabels(labels[::step], fontsize=8)
    ax2.set_xlabel("Local time")
    fig.tight_layout()
    fig.savefig(out / "inspect_timeseries.png", dpi=130)
    plt.close(fig)

    print("\nEdge contact at the middle threshold (max % of edge, time):")
    for side, v in ec.items():
        i = int(np.argmax(v))
        print(f"  {side}: {100 * v[i]:.0f}% at {labels[i]}")
    worst = np.max(np.stack(list(ec.values())), axis=0)
    over = np.nonzero(worst > a.edge_limit)[0]
    if len(over) == 0:
        print(f"\nNo edge exceeds {100 * a.edge_limit:.0f}% coverage in any frame.")
    else:
        i = int(over[0])
        side = max(ec, key=lambda s: ec[s][i])
        last_clean = labels[i - 1] if i > 0 else "none"
        print(
            f"\nFirst frame with any edge over {100 * a.edge_limit:.0f}%: "
            f"{labels[i]} ({side} edge). Last clean frame: {last_clean}"
        )

    # --- contact sheet (hourly) with threshold outlines -------------------
    idx = [i for i, t in enumerate(times) if t.minute < 10]
    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    h, w = T.shape[1:]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6 * h / w + 0.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.set_axis_off()
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        ax.imshow(display(T[i]), cmap="gray", vmin=0, vmax=1)
        for th, c in zip(ths, colors):
            ax.contour(T[i] < th, levels=[0.5], colors=[c], linewidths=0.8)
        ax.set_title(labels[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "inspect_sheet.png", dpi=120)
    plt.close(fig)

    # --- animation ---------------------------------------------------------
    frames = [Image.fromarray((display(t) * 255).astype(np.uint8)) for t in T]
    frames[0].save(
        out / "inspect_loop.gif",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    print(
        f"\nWrote inspect_timeseries.png, inspect_sheet.png and "
        f"inspect_loop.gif to {out}"
    )


if __name__ == "__main__":
    main()

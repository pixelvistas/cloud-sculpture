#!/usr/bin/env python3
"""
goes_extract.py — Stage 2: extract one day of GOES-19 visible imagery
(band 2, 0.64 um, 0.5 km) over a fixed frame around Toronto.

Source: NOAA's public GOES-19 bucket on AWS (no account needed)
    s3://noaa-goes19/ABI-L2-CMIPC/<year>/<day-of-year>/<hour UTC>/
    OR_ABI-L2-CMIPC-M6C02_G19_s<YYYYJJJHHMMSSt>_e..._c....nc
CMIPC = Cloud and Moisture Imagery, CONUS sector. For band 2 the variable
CMI is a reflectance factor (roughly 0 for dark ground or water, up to
about 1 for thick cloud). Files are read remotely and only the Toronto
crop is transferred, so each frame costs a small fraction of the full file.

Two modes:
  preview  three frames (11:00, 14:00, 17:00 local) for each candidate day,
           assembled into a contact sheet for visual comparison
  extract  every N minutes between two local times on one day, saved as
           .npz arrays plus .png previews and an index.csv

Usage:
  python goes_extract.py preview --dates 2026-07-14 2026-07-22 2026-08-03
  python goes_extract.py extract --date 2026-07-14 --start 07:00 --end 20:30 \
         --every 15 --out data/2026-07-14
"""

import argparse
import csv
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

LOCAL = ZoneInfo("America/Toronto")
BUCKET = "noaa-goes19"
PRODUCT = "ABI-L2-CMIPC"
BAND = "C02"
# Frame (west, south, east, north) in degrees: Toronto, the western end of
# Lake Ontario, Hamilton and the Niagara shore, and the land north of the city.
DEFAULT_BBOX = (-80.4, 43.0, -78.4, 44.4)

SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_")


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------


def scan_start(name):
    """Scan start time (UTC) encoded in a GOES filename."""
    m = SCAN_RE.search(name)
    if not m:
        return None
    y, doy, hh, mm, ss, tenth = (int(g) for g in m.groups())
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss, milliseconds=100 * tenth
    )


def hour_prefix(t_utc):
    doy = t_utc.timetuple().tm_yday
    return f"{BUCKET}/{PRODUCT}/{t_utc.year}/{doy:03d}/{t_utc.hour:02d}/"


def band_files(fs, t_utc, cache):
    """Band-2 files in the UTC hour containing t_utc, as (start, path)."""
    prefix = hour_prefix(t_utc)
    if prefix not in cache:
        try:
            names = fs.ls(prefix)
        except FileNotFoundError:
            names = []
        cache[prefix] = sorted(
            (scan_start(n), n)
            for n in names
            if f"M6{BAND}_" in n or f"M3{BAND}_" in n or f"M4{BAND}_" in n
        )
    return cache[prefix]


def nearest_scan(fs, t_utc, cache, tolerance_min=6):
    """The scan starting closest to t_utc (searching neighbouring hours)."""
    candidates = []
    for dh in (-1, 0, 1):
        candidates += band_files(fs, t_utc + timedelta(hours=dh), cache)
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs((c[0] - t_utc).total_seconds()))
    if abs((best[0] - t_utc).total_seconds()) > tolerance_min * 60:
        return None
    return best


def target_times(day, start, end, every_min):
    t = datetime.combine(day, start, tzinfo=LOCAL)
    t_end = datetime.combine(day, end, tzinfo=LOCAL)
    out = []
    while t <= t_end:
        out.append(t)
        t += timedelta(minutes=every_min)
    return out


# --------------------------------------------------------------------------
# Projection and cropping
# --------------------------------------------------------------------------


def geos_proj(ds):
    """pyproj geostationary projection built from the file's own metadata."""
    from pyproj import Proj

    p = ds["goes_imager_projection"].attrs
    h = float(p["perspective_point_height"])
    proj = Proj(
        proj="geos",
        h=h,
        lon_0=float(p["longitude_of_projection_origin"]),
        a=float(p["semi_major_axis"]),
        b=float(p["semi_minor_axis"]),
        sweep=str(p["sweep_angle_axis"]),
    )
    return proj, h


def scan_angles(proj, h, lon, lat):
    """Longitude/latitude to fixed-grid scan angles (radians)."""
    x_m, y_m = proj(lon, lat)
    return np.asarray(x_m) / h, np.asarray(y_m) / h


def crop_slices(ds, bbox):
    """Index slices on the fixed grid covering the lon/lat box."""
    proj, h = geos_proj(ds)
    w, s, e, n = bbox
    lons = np.linspace(w, e, 25)
    lats = np.linspace(s, n, 25)
    LON, LAT = np.meshgrid(lons, lats)  # sample the whole edge
    xs, ys = scan_angles(proj, h, LON, LAT)
    x = ds["x"].values
    y = ds["y"].values  # y decreases with row
    ix = np.searchsorted(x, [np.nanmin(xs), np.nanmax(xs)])
    iy0 = np.argmin(np.abs(y - np.nanmax(ys)))
    iy1 = np.argmin(np.abs(y - np.nanmin(ys)))
    return slice(max(ix[0] - 1, 0), ix[1] + 1), slice(iy0, iy1 + 1)


def open_remote(fs, path):
    import xarray as xr

    return xr.open_dataset(fs.open(path, "rb"), engine="h5netcdf")


def read_crop(fs, path, slices):
    ds = open_remote(fs, path)
    sx, sy = slices
    sub = ds[["CMI", "DQF"]].isel(x=sx, y=sy).load()
    out = {
        "cmi": sub["CMI"].values.astype(np.float32),
        "dqf": sub["DQF"].values.astype(np.int16),
        "x": sub["x"].values,
        "y": sub["y"].values,
    }
    ds.close()
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def to_png(cmi, path, gamma=0.6):
    from PIL import Image

    img = np.nan_to_num(np.clip(cmi, 0, 1), nan=0.0) ** gamma
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def connect():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def run_extract(a):
    fs = connect()
    day = date.fromisoformat(a.date)
    out = Path(a.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    cache, slices, rows = {}, None, []
    for t_local in target_times(
        day, time.fromisoformat(a.start), time.fromisoformat(a.end), a.every
    ):
        t_utc = t_local.astimezone(timezone.utc)
        hit = nearest_scan(fs, t_utc, cache)
        if hit is None:
            print(f"{t_local:%H:%M}  no scan found, skipped")
            continue
        scan_t, path = hit
        if slices is None:
            slices = crop_slices(open_remote(fs, path), a.bbox)
        crop = read_crop(fs, path, slices)
        stem = scan_t.astimezone(LOCAL).strftime("%Y%m%d_%H%M%S")
        np.savez_compressed(
            out / f"{stem}.npz", **crop, scan_utc=str(scan_t), source=path
        )
        to_png(crop["cmi"], out / "png" / f"{stem}.png")
        valid = np.isfinite(crop["cmi"]).mean()
        rows.append(
            [
                str(scan_t),
                scan_t.astimezone(LOCAL).isoformat(),
                f"{stem}.npz",
                crop["cmi"].shape[1],
                crop["cmi"].shape[0],
                f"{valid:.3f}",
            ]
        )
        print(
            f"{t_local:%H:%M}  {scan_t.astimezone(LOCAL):%H:%M:%S}  "
            f"{crop['cmi'].shape}  valid={valid:.0%}"
        )
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["scan_utc", "scan_local", "file", "width", "height", "valid_fraction"]
        )
        w.writerows(rows)
    print(f"\n{len(rows)} frames written to {out}")


def run_preview(a):
    from PIL import Image, ImageDraw

    fs = connect()
    cache, slices, tiles = {}, None, []
    for d in a.dates:
        day = date.fromisoformat(d)
        row = []
        for hhmm in a.times:
            t_utc = datetime.combine(
                day, time.fromisoformat(hhmm), tzinfo=LOCAL
            ).astimezone(timezone.utc)
            hit = nearest_scan(fs, t_utc, cache)
            if hit is None:
                row.append(None)
                continue
            if slices is None:
                slices = crop_slices(open_remote(fs, hit[1]), a.bbox)
            cmi = read_crop(fs, hit[1], slices)["cmi"]
            img = (np.nan_to_num(np.clip(cmi, 0, 1)) ** 0.6 * 255).astype(np.uint8)
            row.append(Image.fromarray(img))
            print(f"{d} {hhmm} ok")
        tiles.append((d, row))
    w = max(t.width for _, r in tiles for t in r if t)
    h = max(t.height for _, r in tiles for t in r if t)
    sheet = Image.new("L", (w * len(a.times), (h + 18) * len(tiles)), 0)
    draw = ImageDraw.Draw(sheet)
    for i, (d, row) in enumerate(tiles):
        for j, t in enumerate(row):
            if t:
                sheet.paste(t, (j * w, i * (h + 18) + 18))
            draw.text((j * w + 4, i * (h + 18) + 3), f"{d}  {a.times[j]}", fill=255)
    sheet.save(a.out)
    print(f"Contact sheet written to {a.out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)
    pv = sub.add_parser("preview")
    pv.add_argument("--dates", nargs="+", required=True)
    pv.add_argument("--times", nargs="+", default=["11:00", "14:00", "17:00"])
    pv.add_argument("--out", default="candidates_preview.png")
    ex = sub.add_parser("extract")
    ex.add_argument("--date", required=True)
    ex.add_argument("--start", default="07:00")
    ex.add_argument("--end", default="20:30")
    ex.add_argument("--every", type=int, default=15, help="minutes")
    ex.add_argument("--out", required=True)
    for sp in (pv, ex):
        sp.add_argument(
            "--bbox",
            type=float,
            nargs=4,
            default=DEFAULT_BBOX,
            metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        )
    a = p.parse_args()
    run_preview(a) if a.mode == "preview" else run_extract(a)


if __name__ == "__main__":
    main()

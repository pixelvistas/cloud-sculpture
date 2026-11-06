#!/usr/bin/env python3
"""
build_volume.py — Stage 3: from extracted infrared frames to a sculpture.

Steps
  1. Select frames within a local-time window.
  2. Regrid each frame from the satellite's fixed grid (scan angles) onto a
     regular grid in kilometres centred on the frame, so that pixels are
     square and the storm keeps its true proportions.
  3. Build a field F = threshold - T (positive where the cloud top is colder
     than the threshold), and smooth it lightly in space and time.
  4. Extract the F = 0 surface with marching cubes. The volume is closed with
     a warm border, so wherever cloud meets the frame edge the sculpture gets
     a flat cut face, as a window frame bounds a patch of light.
  5. Scale to millimetres, clean the mesh and write the STL.
  6. Export, for the visualizer, one traced contour set and one texture image
     per frame, plus an index (frames.json).

Infrared brightness temperature needs no solar-angle correction, which is
why that step from the visible-band plan does not appear here.

Usage
  python build_volume.py data/2026-09-02_ir_wide --start 12:00 --end 18:01 \
      --threshold 220 --width 200 --height 80 --out build/sep02

Projection note: the .npz frames store fixed-grid scan angles but not the
projection constants. GOES-East (GOES-19) defaults are used below; the
script prints the geographic extent it reconstructs so it can be checked
against the --bbox used for extraction.
"""

import argparse
import csv
import json
from datetime import datetime, time
from pathlib import Path

import numpy as np

# GOES-R fixed-grid constants for GOES-East.
GOES_H = 35786023.0
GOES_A = 6378137.0
GOES_B = 6356752.31414
GOES_LON0 = -75.0  # fixed-grid origin for GOES-East (satellite itself near 75.2 W)


# --------------------------------------------------------------------------
# Loading and regridding
# --------------------------------------------------------------------------


def load_window(folder, start, end):
    folder = Path(folder)
    with open(folder / "index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    t0, t1 = time.fromisoformat(start), time.fromisoformat(end)
    sel = []
    for r in rows:
        t = datetime.fromisoformat(r["scan_local"])
        if t0 <= t.replace(second=0, microsecond=0).time() <= t1:
            sel.append((t, folder / r["file"]))
    if len(sel) < 3:
        raise SystemExit(f"Only {len(sel)} frames between {start} and {end}.")
    z = np.load(sel[0][1])
    x, y = z["x"], z["y"]
    T = np.stack([np.load(p)["cmi"].astype(np.float32) for _, p in sel])
    return [t for t, _ in sel], x, y, T


def projections(lon0):
    from pyproj import CRS, Transformer

    geos = CRS.from_proj4(
        f"+proj=geos +h={GOES_H} +lon_0={lon0} "
        f"+a={GOES_A} +b={GOES_B} +sweep=x +no_defs"
    )
    return geos


def regrid(x, y, T, grid_km, lon0):
    """Fixed-grid frames -> regular km grid (azimuthal equidistant)."""
    from pyproj import CRS, Transformer
    from scipy.interpolate import RegularGridInterpolator

    geos = projections(lon0)
    to_ll = Transformer.from_crs(geos, "EPSG:4326", always_xy=True)
    X, Y = np.meshgrid(x * GOES_H, y * GOES_H)
    lon, lat = to_ll.transform(X, Y)
    lon_c, lat_c = float(np.nanmean(lon)), float(np.nanmean(lat))
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_c} +lon_0={lon_c} " f"+datum=WGS84 +units=km +no_defs"
    )
    ll_to_km = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    km_to_ll = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    ll_to_geos = Transformer.from_crs("EPSG:4326", geos, always_xy=True)

    ex, ny = ll_to_km.transform(lon, lat)
    # Largest axis-aligned rectangle fully inside the source footprint.
    x0 = max(np.nanmax(ex[:, 0]), np.nanmin(ex))
    x1 = min(np.nanmin(ex[:, -1]), np.nanmax(ex))
    y0 = max(np.nanmax(ny[-1, :]), np.nanmin(ny))
    y1 = min(np.nanmin(ny[0, :]), np.nanmax(ny))
    gx = np.arange(x0, x1, grid_km)
    gy = np.arange(y1, y0, -grid_km)  # north at row 0
    GX, GY = np.meshgrid(gx, gy)
    glon, glat = km_to_ll.transform(GX, GY)
    sx, sy = ll_to_geos.transform(glon, glat)
    pts = np.stack([sy / GOES_H, sx / GOES_H], axis=-1)

    # The source y axis decreases; RegularGridInterpolator needs it increasing.
    order = np.argsort(y)
    out = np.empty((len(T), len(gy), len(gx)), np.float32)
    for i, frame in enumerate(T):
        f = RegularGridInterpolator(
            (y[order], x), frame[order], bounds_error=False, fill_value=np.nan
        )
        out[i] = f(pts)
    info = {
        "center_lonlat": [lon_c, lat_c],
        "extent_km": [float(gx[-1] - gx[0]), float(gy[0] - gy[-1])],
        "lon_range": [float(np.nanmin(glon)), float(np.nanmax(glon))],
        "lat_range": [float(np.nanmin(glat)), float(np.nanmax(glat))],
        "grid_km": grid_km,
    }
    return out, info


# --------------------------------------------------------------------------
# Field, surface, mesh
# --------------------------------------------------------------------------


def build_field(Tk, threshold, sigma_km, sigma_frames, grid_km):
    from scipy.ndimage import gaussian_filter

    T = np.where(np.isfinite(Tk), Tk, 300.0)  # missing = warm (clear)
    F = threshold - T
    return gaussian_filter(
        F, sigma=(sigma_frames, sigma_km / grid_km, sigma_km / grid_km)
    )


def signed_distance(F, grid_km):
    """Per frame: distance (km) to the cloud edge, positive inside the cloud.
    Interpolating this between frames morphs one outline into the next, so
    fast-moving edges give smooth slopes instead of terraces."""
    from scipy.ndimage import distance_transform_edt as edt, gaussian_filter

    out = np.empty_like(F)
    for i, f in enumerate(F):
        inside = f > 0
        d = (edt(inside) - edt(~inside)).astype(np.float32)
        out[i] = gaussian_filter(d, 1.5) * grid_km  # remove pixel stair-steps
    return out


def upsample_time(F, k):
    """Insert k-1 interpolated slices between real frames (smooth terraces)."""
    if k <= 1:
        return F
    from scipy.interpolate import interp1d

    nt = F.shape[0]
    t_new = np.linspace(0, nt - 1, (nt - 1) * k + 1)
    return interp1d(np.arange(nt), F, axis=0, kind="linear")(t_new).astype(np.float32)


def surface(F, width_mm, height_mm, extent_km, min_piece_cm3):
    import trimesh
    from skimage import measure

    nt, ny, nx = F.shape
    pad = np.pad(F, 1, mode="constant", constant_values=-50.0)
    sx = width_mm / (nx - 1)
    sy = sx * (extent_km[1] / (ny - 1)) / (extent_km[0] / (nx - 1))
    sz = height_mm / (nt - 1)
    v, f, _, _ = measure.marching_cubes(pad, level=0.0, spacing=(sz, sy, sx))
    v -= np.array([sz, sy, sx])  # undo the padding offset
    verts = np.column_stack([v[:, 2], (ny - 1) * sy - v[:, 1], v[:, 0]])
    m = trimesh.Trimesh(verts, f[:, ::-1], process=True)
    parts = [
        p
        for p in m.split(only_watertight=False)
        if abs(p.volume) / 1000.0 >= min_piece_cm3
    ]
    if parts:
        m = trimesh.util.concatenate(parts)
    trimesh.smoothing.filter_taubin(m, iterations=10)
    trimesh.repair.fix_normals(m)
    return m, len(parts), (sx, sy, sz)


# --------------------------------------------------------------------------
# Visualizer export
# --------------------------------------------------------------------------


def export_visualizer(out, times, Tk, F, threshold, scale):
    from PIL import Image
    from skimage import measure

    tex = out / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    nt, ny, nx = F.shape
    frames = []
    for i in range(nt):
        img = np.clip((300.0 - np.nan_to_num(Tk[i], nan=300.0)) / 110.0, 0, 1)
        name = f"{times[i]:%H%M}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(tex / name)
        contours = []
        padded = np.pad(F[i], 1, constant_values=-50.0)
        for c in measure.find_contours(padded, 0.0):
            c = c - 1
            u = np.clip(c[:, 1] / (nx - 1), 0, 1)  # 0..1 west -> east
            v = np.clip(c[:, 0] / (ny - 1), 0, 1)  # 0..1 north -> south
            if len(c) >= 8:
                contours.append(np.round(np.column_stack([u, v]), 4).tolist())
        frames.append(
            {
                "time": times[i].isoformat(),
                "label": f"{times[i]:%H:%M}",
                "z": i / (nt - 1),
                "texture": f"textures/{name}",
                "contours": contours,
            }
        )
    meta = {
        "threshold_K": threshold,
        "aspect": (ny - 1) * scale[1] / ((nx - 1) * scale[0]),
        "frames": frames,
    }
    (out / "frames.json").write_text(json.dumps(meta))


def preview(path, mesh, title, box):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = mesh
    if len(m.faces) > 150000:
        try:
            m = m.simplify_quadric_decimation(face_count=150000)
        except Exception:
            pass
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.array([0.4, -0.5, 0.75])
    L /= np.linalg.norm(L)
    shade = 0.3 + 0.7 * np.clip(n @ L, 0, 1)
    col = np.clip(shade[:, None] * np.array([[0.85, 0.88, 0.92]]), 0, 1)
    fig = plt.figure(figsize=(15, 5), facecolor="black")
    bw, bd, bh = box  # full frame, mm
    corners = np.array([[0, 0], [bw, 0], [bw, bd], [0, bd], [0, 0]])
    for k, (el, az) in enumerate([(22, -60), (22, 30), (88, -90)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.set_facecolor("black")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
        for zc in (0, bh):  # frame outline, bottom and top
            ax.plot(corners[:, 0], corners[:, 1], zc, color="0.45", lw=0.6)
        ax.set_xlim(0, bw)
        ax.set_ylim(0, bd)
        ax.set_zlim(0, bh)
        ax.set_box_aspect((bw, bd, bh))
        ax.view_init(elev=el, azim=az)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10, color="white")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder")
    p.add_argument("--start", default="12:00")
    p.add_argument("--end", default="18:00")
    p.add_argument("--threshold", type=float, default=220.0, help="kelvin")
    p.add_argument("--grid-km", type=float, default=2.0)
    p.add_argument("--sigma-km", type=float, default=3.0, help="spatial smoothing")
    p.add_argument("--sigma-frames", type=float, default=0.8, help="temporal smoothing")
    p.add_argument(
        "--upsample",
        type=int,
        default=4,
        help="interpolated slices per frame interval for the mesh",
    )
    p.add_argument(
        "--morph",
        action="store_true",
        help="interpolate edge distance instead of temperature "
        "(experimental; can help when edges jump far between frames)",
    )
    p.add_argument("--width", type=float, default=200.0, help="west-east size [mm]")
    p.add_argument("--height", type=float, default=80.0, help="time axis [mm]")
    p.add_argument("--min-piece", type=float, default=0.2, help="cm^3")
    p.add_argument("--lon0", type=float, default=GOES_LON0)
    p.add_argument("--out", default="build/sculpture")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    times, x, y, T = load_window(a.folder, a.start, a.end)
    print(
        f"{len(times)} frames {times[0]:%H:%M}-{times[-1]:%H:%M}, "
        f"source {T.shape[2]} x {T.shape[1]} px"
    )
    Tk, info = regrid(x, y, T, a.grid_km, a.lon0)
    print(
        f"regridded to {Tk.shape[2]} x {Tk.shape[1]} cells at {a.grid_km} km "
        f"({info['extent_km'][0]:.0f} x {info['extent_km'][1]:.0f} km)"
    )
    print(
        f"reconstructed extent: lon {info['lon_range'][0]:.2f} to "
        f"{info['lon_range'][1]:.2f}, lat {info['lat_range'][0]:.2f} to "
        f"{info['lat_range'][1]:.2f}  (compare with the extraction --bbox)"
    )

    F = build_field(Tk, a.threshold, a.sigma_km, a.sigma_frames, a.grid_km)
    Fm = signed_distance(F, a.grid_km) if a.morph else F
    Fm = upsample_time(Fm, a.upsample)
    if a.morph and a.upsample > 1:
        from scipy.ndimage import gaussian_filter1d

        Fm = gaussian_filter1d(Fm, sigma=0.5 * a.upsample, axis=0)
    mesh, n_parts, scale = surface(
        Fm, a.width, a.height, info["extent_km"], a.min_piece
    )
    stl = out.with_suffix(".stl") if out.suffix else out / "sculpture.stl"
    mesh.export(stl)
    ext = mesh.extents
    print(
        f"mesh: {len(mesh.faces)} faces, {n_parts} piece(s), watertight="
        f"{mesh.is_watertight}, {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm, "
        f"{mesh.volume / 1000:.1f} cm^3"
    )

    export_visualizer(out, times, Tk, F, a.threshold, scale)
    (out / "build_info.json").write_text(
        json.dumps({**info, **vars(a)}, indent=2, default=str)
    )
    preview(
        out / "preview.png",
        mesh,
        f"{times[0]:%Y-%m-%d} {times[0]:%H:%M}-{times[-1]:%H:%M}, "
        f"T < {a.threshold:.0f} K",
        (a.width, a.width * info["extent_km"][1] / info["extent_km"][0], a.height),
    )
    print(f"wrote {stl}, preview.png, frames.json and textures/ to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
slice_layers.py — Stage 5: turn the sculpture STL into laser-cut acrylic
layers, stacked solid like a topographic model.

Each layer is one sheet of acrylic cut to the outline of the cloud volume at
that height. The STL's height (time) is sampled at the middle of each layer,
so you choose the number of layers and the sheet thickness independently of
the model's original height:

    finished height = layers x thickness

Outputs (in --out):
    sheet_01.svg, ...   cut files, pieces nested onto your sheet size.
                        Cut lines are red hairlines; nothing is engraved.
    sheet_01_map.png    the same layout with each piece's ID (layer + letter),
                        for writing IDs on the protective film before cutting
    assembly.png        every layer in order, with the layer below dashed,
                        to show how each piece sits on the one under it
    report.txt          pieces, sheets, cut length, and anything that needs
                        attention (unsupported pieces, slivers, stability)

Usage:
    python slice_layers.py build/sep02_organic/sculpture.stl --layers 12 \
        --thickness 6 --sheet 600 300 --out cut/sep02
"""

import argparse
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------


def slice_mesh(path, layers, width_mm):
    import trimesh
    from shapely.geometry import Polygon, MultiPolygon

    mesh = trimesh.load(path, force="mesh")
    lo, hi = mesh.bounds
    scale = width_mm / (hi[0] - lo[0]) if width_mm else 1.0
    zs = lo[2] + (np.arange(layers) + 0.5) / layers * (hi[2] - lo[2])
    sections = mesh.section_multiplane(
        plane_origin=[0, 0, lo[2]], plane_normal=[0, 0, 1], heights=list(zs - lo[2])
    )
    out = []
    for sec in sections:
        if sec is None:
            out.append(MultiPolygon())
            continue
        # Each closed loop of the section becomes a polygon; combining them by
        # symmetric difference turns loops inside loops into holes (and
        # islands inside holes back into material), without extra libraries.
        # For a horizontal plane the section's 2D frame is world x, y.
        geom = Polygon()
        for ring in sec.discrete:
            if len(ring) >= 4:
                geom = geom.symmetric_difference(
                    Polygon(np.asarray(ring) * scale).buffer(0)
                )
        out.append(geom.buffer(0))
    return out, scale


def as_list(geom):
    if geom.is_empty:
        return []
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def clean(geom, min_feature, min_area, simplify):
    """Drop pieces and slivers too small to cut cleanly."""
    from shapely.ops import unary_union

    r = min_feature / 2.0
    g = geom.buffer(-r).buffer(r) if r > 0 else geom  # morphological opening
    g = g.simplify(simplify, preserve_topology=True) if simplify > 0 else g
    keep = [p for p in as_list(g) if p.area >= min_area]
    return unary_union(keep) if keep else g.buffer(0).__class__()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_support(layers, min_overlap):
    """Pieces whose overlap with the layer below is too small to glue."""
    issues = []
    for k in range(1, len(layers)):
        below = layers[k - 1]
        for j, p in enumerate(as_list(layers[k])):
            ov = p.intersection(below).area / p.area if p.area else 0
            if ov < min_overlap:
                issues.append((k, j, ov))
    return issues


def add_supports(layers, min_overlap, diameter):
    """Clear pillars under pieces that would otherwise float: a disc is added
    in each layer below the piece, down to the first layer with material
    beneath it (or to the base plate)."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    added = 0
    for k in range(1, len(layers)):
        below = layers[k - 1]
        for p in as_list(layers[k]):
            if p.area and p.intersection(below).area / p.area >= min_overlap:
                continue
            free = p.difference(below)
            pt = (free if not free.is_empty else p).representative_point()
            disc = Point(pt.x, pt.y).buffer(diameter / 2, resolution=24)
            for m in range(k - 1, -1, -1):
                if layers[m].contains(pt):
                    break
                layers[m] = unary_union([layers[m], disc])
                added += 1
    return added


def base_plate(layers, margin, radius=6.0):
    """A clear rectangular plate under the whole stack, with rounded corners."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    minx, miny, maxx, maxy = unary_union([g for g in layers if not g.is_empty]).bounds
    r = min(radius, margin)
    return box(
        minx - margin + r, miny - margin + r, maxx + margin - r, maxy + margin - r
    ).buffer(r, resolution=16)


def check_stability(layers, thickness):
    """Is the stack's centre of mass above the footprint of the base layer?"""
    from shapely.geometry import Point

    mass, mx, my = 0.0, 0.0, 0.0
    for g in layers:
        if g.is_empty:
            continue
        a = g.area * thickness
        c = g.centroid
        mass += a
        mx += a * c.x
        my += a * c.y
    if mass == 0 or layers[0].is_empty:
        return None, None
    com = Point(mx / mass, my / mass)
    hull = layers[0].convex_hull
    return hull.contains(com), com.distance(hull.exterior) * (
        1 if hull.contains(com) else -1
    )


# --------------------------------------------------------------------------
# Nesting (simple shelf packing by bounding box)
# --------------------------------------------------------------------------


def nest(pieces, sheet_w, sheet_h, margin, gap):
    """pieces: list of (id, polygon). Returns list of sheets, each a list of
    (id, polygon translated into sheet coordinates)."""
    from shapely import affinity

    prepared = []
    for pid, poly in pieces:
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if w > sheet_w - 2 * margin or h > sheet_h - 2 * margin:
            if h <= sheet_w - 2 * margin and w <= sheet_h - 2 * margin:
                poly = affinity.rotate(poly, 90, origin="centroid")
                minx, miny, maxx, maxy = poly.bounds
                w, h = maxx - minx, maxy - miny
            else:
                raise SystemExit(
                    f"Piece {pid} ({w:.0f} x {h:.0f} mm) does not fit "
                    f"on a {sheet_w:.0f} x {sheet_h:.0f} mm sheet."
                )
        prepared.append((pid, affinity.translate(poly, -minx, -miny), w, h))
    prepared.sort(key=lambda t: t[3], reverse=True)

    sheets, cur = [], []
    x, y, row_h = margin, margin, 0.0
    for pid, poly, w, h in prepared:
        if x + w > sheet_w - margin:  # next row
            x, y, row_h = margin, y + row_h + gap, 0.0
        if y + h > sheet_h - margin:  # next sheet
            sheets.append(cur)
            cur, x, y, row_h = [], margin, margin, 0.0
        cur.append((pid, affinity.translate(poly, x, y)))
        x += w + gap
        row_h = max(row_h, h)
    if cur:
        sheets.append(cur)
    return sheets


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def ring_path(coords):
    pts = list(coords)
    return "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts[:-1]) + " Z"


def poly_path(poly):
    return " ".join(
        [ring_path(poly.exterior.coords)]
        + [ring_path(r.coords) for r in poly.interiors]
    )


def write_svg(path, sheet, sheet_w, sheet_h):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_w}mm" '
        f'height="{sheet_h}mm" viewBox="0 0 {sheet_w} {sheet_h}">',
        f'<rect x="0" y="0" width="{sheet_w}" height="{sheet_h}" fill="none" '
        f'stroke="#0000ff" stroke-width="0.2" id="sheet-outline-do-not-cut"/>',
        '<g id="cut" fill="none" stroke="#ff0000" stroke-width="0.01">',
    ]
    for pid, poly in sheet:
        lines.append(
            f'  <path id="piece-{pid}" fill-rule="evenodd" d="{poly_path(poly)}"/>'
        )
    lines += ["</g>", "</svg>"]
    Path(path).write_text("\n".join(lines))


def plot_polys(ax, geom, **kw):
    for p in as_list(geom):
        x, y = p.exterior.xy
        ax.plot(x, y, **kw)
        for r in p.interiors:
            x, y = r.xy
            ax.plot(x, y, **kw)


def write_map(path, sheet, sheet_w, sheet_h, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10 * sheet_h / sheet_w + 0.4))
    ax.plot(
        [0, sheet_w, sheet_w, 0, 0], [0, 0, sheet_h, sheet_h, 0], color="0.6", lw=0.8
    )
    for pid, poly in sheet:
        plot_polys(ax, poly, color="#D85A30", lw=0.8)
        c = poly.representative_point()
        ax.text(c.x, c.y, pid, ha="center", va="center", fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match SVG orientation (y down)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_assembly(path, layers, ids, has_plate=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(layers)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    allb = [g.bounds for g in layers if not g.is_empty]
    minx = min(b[0] for b in allb)
    miny = min(b[1] for b in allb)
    maxx = max(b[2] for b in allb)
    maxy = max(b[3] for b in allb)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.4, rows * 3.4 * (maxy - miny) / (maxx - minx) + 0.5),
    )
    for ax in np.atleast_1d(axes).ravel():
        ax.set_axis_off()
    for k, (ax, g) in enumerate(zip(np.atleast_1d(axes).ravel(), layers)):
        if k > 0:
            plot_polys(ax, layers[k - 1], color="0.6", lw=0.7, ls="--")
        plot_polys(ax, g, color="#D85A30", lw=1.1)
        for j, p in enumerate(as_list(g)):
            c = p.representative_point()
            ax.text(c.x, c.y, ids[(k, j)], ha="center", va="center", fontsize=7)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        num = k if has_plate else k + 1
        name = "Base plate (layer 0)" if has_plate and k == 0 else f"Layer {num}"
        ax.set_title(name, fontsize=9)
    fig.suptitle(
        "Assembly: orange = this layer, dashed = layer below "
        "(seen from above, north up)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stl")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--thickness", type=float, default=6.0, help="sheet thickness [mm]")
    p.add_argument(
        "--width",
        type=float,
        default=None,
        help="rescale the footprint to this west-east width [mm]",
    )
    p.add_argument(
        "--sheet",
        type=float,
        nargs=2,
        default=(600, 300),
        metavar=("W", "H"),
        help="material sheet size [mm]",
    )
    p.add_argument("--margin", type=float, default=6.0)
    p.add_argument("--gap", type=float, default=4.0, help="space between pieces [mm]")
    p.add_argument(
        "--min-feature",
        type=float,
        default=3.0,
        help="remove necks and slivers narrower than this [mm]",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=40.0,
        help="drop pieces smaller than this [mm^2]",
    )
    p.add_argument(
        "--simplify",
        type=float,
        default=0.15,
        help="outline tolerance [mm]; fewer points, faster cutting",
    )
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.25,
        help="flag pieces resting on less than this fraction of their area",
    )
    p.add_argument(
        "--base",
        choices=["plate", "none"],
        default="plate",
        help="add a clear rectangular base plate under the stack",
    )
    p.add_argument(
        "--base-margin",
        type=float,
        default=10.0,
        help="plate extends this far beyond the sculpture [mm]",
    )
    p.add_argument(
        "--no-supports",
        dest="supports",
        action="store_false",
        help="do not add clear pillars under floating pieces",
    )
    p.add_argument(
        "--support-d", type=float, default=8.0, help="diameter of support pillars [mm]"
    )
    p.add_argument("--out", default="cut/sculpture")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw, scale = slice_mesh(a.stl, a.layers, a.width)
    layers = [clean(g, a.min_feature, a.min_area, a.simplify) for g in raw]
    floating = check_support(layers, a.min_overlap)
    n_supports = add_supports(layers, a.min_overlap, a.support_d) if a.supports else 0
    if a.base == "plate":
        layers = [base_plate(layers, a.base_margin)] + layers

    ids, pieces = {}, []
    for k, g in enumerate(layers):
        for j, poly in enumerate(as_list(g)):
            num = k if a.base == "plate" else k + 1  # plate is layer 0
            pid = f"{num}{chr(ord('a') + j)}" if len(as_list(g)) > 1 else f"{num}"
            ids[(k, j)] = pid
            pieces.append((pid, poly))
    if not pieces:
        raise SystemExit("No layer has any material; check the STL.")

    # SVG's y axis points down; mirror so each piece is drawn as seen from
    # above with north at the top, matching the assembly guide.
    from shapely import affinity

    pieces = [(pid, affinity.scale(poly, 1, -1, origin=(0, 0))) for pid, poly in pieces]
    sheets = nest(pieces, a.sheet[0], a.sheet[1], a.margin, a.gap)
    for s, sheet in enumerate(sheets, 1):
        write_svg(out / f"sheet_{s:02d}.svg", sheet, a.sheet[0], a.sheet[1])
        write_map(
            out / f"sheet_{s:02d}_map.png",
            sheet,
            a.sheet[0],
            a.sheet[1],
            f"Sheet {s}: write each ID on the protective film before cutting",
        )
    write_assembly(out / "assembly.png", layers, ids, a.base == "plate")

    total = len(layers)
    cut_m = (
        sum(
            poly.exterior.length + sum(r.length for r in poly.interiors)
            for _, poly in pieces
        )
        / 1000
    )
    unsupported = check_support(layers, a.min_overlap)
    stable, margin_mm = check_stability(layers, a.thickness)
    empty = [k + 1 for k, g in enumerate(layers) if g.is_empty]

    rep = [
        f"Layers: {a.layers} x {a.thickness} mm"
        + (" + base plate" if a.base == "plate" else "")
        + f" = {total * a.thickness:.0f} mm tall",
        (
            f"Footprint scale: {scale:.3f} x the STL"
            if a.width
            else "Footprint: as in the STL"
        ),
        f"Pieces: {len(pieces)}   Sheets ({a.sheet[0]:.0f} x {a.sheet[1]:.0f} mm): {len(sheets)}",
        f"Total cut length: {cut_m:.2f} m (laser time scales with this)",
        "",
    ]
    if empty:
        rep.append(f"Empty layers (no cloud at that height): {empty}")
    if floating:
        rep.append(
            f"Floating pieces found: {len(floating)}; "
            + (
                f"{n_supports} clear support disc(s) of {a.support_d:.0f} mm added "
                "beneath them (visible in assembly.png)."
                if a.supports
                else "supports are off, so these need pins or removal."
            )
        )
    if unsupported:
        rep.append(
            "Pieces still resting on little material (check by eye; a support "
            "disc may sit under only part of them):"
        )
        rep += [
            f"  {ids[(k, j)]}: {100 * ov:.0f}% resting on the layer below"
            for k, j, ov in unsupported
        ]
    else:
        rep.append(
            f"Every piece rests on at least {100 * a.min_overlap:.0f}% of its area."
        )
    if stable is None:
        rep.append("Stability: base layer is empty; check the STL.")
    elif stable:
        rep.append(
            f"Stability: centre of mass is over the base, "
            f"{margin_mm:.0f} mm inside its outline."
        )
    else:
        rep.append(
            f"Stability: centre of mass is {abs(margin_mm):.0f} mm OUTSIDE the "
            "base; the piece will tip without a wider base or a stand."
        )
    (out / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(
        f"\nWrote {len(sheets)} SVG sheet(s), maps, assembly.png and report.txt to {out}"
    )


if __name__ == "__main__":
    main()

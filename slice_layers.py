#!/usr/bin/env python3
"""
slice_layers.py — Stage 5: turn the sculpture STL into laser-cut acrylic
layers, stacked solid like a topographic model.

The STL is treated as a physical 3D model. It is scaled uniformly to the
finished width (--width; the STL's own size if omitted), then cut into
horizontal sections one sheet thick, from the bottom up. Each section is
cut to the model's outline at its mid-height, so the layer count follows
from the model's height and the sheet thickness:

    layers = model height / thickness (rounded to whole sheets)

--height sets a different finished height (stretching or squashing time
only), and --layers fixes the layer count instead.

Frame-aligned layers
    --every N cuts one layer exactly at every Nth satellite frame instead,
    so each layer is one frame of the visualizer (--every 2 with 10-minute
    frames: one layer per 20 minutes). Frame times and heights are read from
    frames.json and build_info.json next to the STL; --first-frame shifts
    which frame the first layer takes (0 = the window's first frame).

Outputs (in --out):
    sheet_01.svg, ...   cut files, pieces nested onto your sheet size, in
                        three colours, one laser operation each:
                          red   #ff0000  cut through (hairline)
                          blue  #0000ff  score: alignment guides (hairline)
                          black #000000  engrave (fill): piece IDs
                        The grey sheet outline is for reference; switch that
                        layer off in the laser software.
    sheet_01_map.png    the same layout with each piece's ID (layer + letter)
                        and its guides, for checking the sheet before cutting
    assembly.png        every layer in order, with the layer below dashed and
                        the guides its edge must cover in blue
    stack.stl           the layered object exactly as it will be built
    stack_preview.png   that object from three angles
    report.txt          pieces, sheets, cut length, and anything that needs
                        attention (unsupported pieces, slivers, stability)

With --viewer PATH (the visualizer's folder), the stacked object is also
copied to PATH/stacks/<name>/ and listed in PATH/stacks/index.json, so
stack.html shows it next to the visualizer's contour view at true scale.
Run the slicer once per option (different --out and --name) to compare them.

Alignment guides
    On the top face of each piece, the laser scores the part of the next
    layer's outline that falls on that piece, set --etch-inset inside the true
    outline so the line disappears under the upper piece's edge once that
    piece is in place. Assembly is bottom-up from the base plate: set each
    piece dry until its edge just covers every blue guide on the piece below,
    then bond it. A piece that hangs from the layer above
    instead carries the guide for that layer's edge on its own top face, and
    goes on after the piece it hangs from. Each piece's ID is engraved on a
    part of its top face that the next layer covers, so it is hidden once
    built; pieces with no covered area large enough are listed in the report.

Usage:
    python slice_layers.py build/sep02_huge/sculpture.stl --width 240 \
        --thickness 3.175 --sheet 600 300 --out cut/sep02
"""

import argparse
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------


def frame_heights(stl_path, every, first):
    """Heights (STL units) and labels of every Nth frame of the build, from
    the frames.json and build_info.json written next to the STL."""
    import json

    folder = Path(stl_path).parent
    fj, bj = folder / "frames.json", folder / "build_info.json"
    if not fj.exists() or not bj.exists():
        raise SystemExit(f"--every needs frames.json and build_info.json in {folder}")
    frames = json.loads(fj.read_text())["frames"]
    info = json.loads(bj.read_text())
    span = float(info["height"]) - float(
        info.get("cap") or 0.0
    )  # frames end below a cap
    pick = frames[first::every]
    if not pick:
        raise SystemExit(f"--first-frame {first} is beyond the {len(frames)} frames")
    return [f["z"] * span for f in pick], [f["label"] for f in pick], len(frames)


def slice_mesh(path, thickness, width_mm=None, height_mm=None, layers=None, zs=None):
    """Scale the model uniformly to width_mm, optionally set its height, and
    cut it into sections `thickness` thick. Returns the section outlines, the
    footprint scale, and the model's size (w, d, h) in mm after scaling.
    zs, if given, are the exact heights (STL units) to cut at, one per layer."""
    import trimesh
    from shapely.geometry import Polygon, MultiPolygon

    mesh = trimesh.load(path, force="mesh")
    lo, hi = mesh.bounds
    scale = width_mm / (hi[0] - lo[0]) if width_mm else 1.0
    size = (hi - lo) * scale
    if height_mm:
        size[2] = height_mm
    if layers is None:
        layers = max(1, int(round(size[2] / thickness)))
    if zs is None:
        zs = lo[2] + (np.arange(layers) + 0.5) / layers * (hi[2] - lo[2])
    zs = np.clip(np.asarray(zs, float), lo[2] + 1e-6, hi[2] - 1e-6)
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
    return out, scale, tuple(float(v) for v in size)


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


def keep_connected(layers, thickness, keep_share):
    """Keep the storm's main connected body. Pieces are linked when they
    overlap a piece in the layer above or below; each linked group is one
    separate object in the finished stack. The largest group is kept, plus
    any group with at least keep_share of its volume. Returns the filtered
    layers and the volumes (cm^3) of the groups removed."""
    from shapely.ops import unary_union

    nodes = [(k, p) for k, g in enumerate(layers) for p in as_list(g)]
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_layer = {}
    for i, (k, _) in enumerate(nodes):
        by_layer.setdefault(k, []).append(i)
    for k, idx in by_layer.items():
        for i in idx:
            for j in by_layer.get(k + 1, []):
                if nodes[i][1].intersection(nodes[j][1]).area > 1.0:  # > 1 mm^2
                    parent[find(i)] = find(j)
    vol = {}
    for i, (_, p) in enumerate(nodes):
        r = find(i)
        vol[r] = vol.get(r, 0.0) + p.area * thickness
    if not vol:
        return layers, []
    biggest = max(vol.values())
    keep = {r for r, v in vol.items() if v >= keep_share * biggest}
    removed = sorted(
        (v / 1000.0 for r, v in vol.items() if r not in keep), reverse=True
    )
    out = []
    for k in range(len(layers)):
        kept = [nodes[i][1] for i in by_layer.get(k, []) if find(i) in keep]
        out.append(unary_union(kept) if kept else layers[k].__class__())
    return out, removed


def check_support(layers, min_overlap):
    """Pieces glued to too little material, both below and above. A piece that
    overlaps the layer above is held by that layer once glued (the stack is
    assembled upside down, as the figure-sculpture guide recommends), so only
    pieces attached on neither side count as floating."""
    issues = []
    for k in range(1, len(layers)):
        below = layers[k - 1]
        above = layers[k + 1] if k + 1 < len(layers) else None
        for j, p in enumerate(as_list(layers[k])):
            if not p.area:
                continue
            ov = p.intersection(below).area / p.area
            ov_up = p.intersection(above).area / p.area if above is not None else 0
            if ov < min_overlap and ov_up < min_overlap:
                issues.append((k, j, max(ov, ov_up)))
    return issues


def add_supports(layers, min_overlap, diameter, max_pillar):
    """Clear pillars under pieces that would otherwise float: a disc is added
    in each layer below the piece, down to the first layer with material
    beneath it (or to the base plate). A piece that would need a pillar
    taller than max_pillar layers is removed instead, since a long thin
    column of discs is fragile and reads as an artefact."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    added, dropped = 0, []
    for k in range(1, len(layers)):
        below = layers[k - 1]
        above = layers[k + 1] if k + 1 < len(layers) else None
        for p in as_list(layers[k]):
            if p.area and p.intersection(below).area / p.area >= min_overlap:
                continue
            if (
                p.area
                and above is not None
                and p.intersection(above).area / p.area >= min_overlap
            ):
                continue  # hangs from the layer above
            free = p.difference(below)
            pt = (free if not free.is_empty else p).representative_point()
            need = []
            for m in range(k - 1, -1, -1):
                if layers[m].contains(pt):
                    break
                need.append(m)
            if len(need) > max_pillar:
                layers[k] = layers[k].difference(p.buffer(0.01))
                dropped.append((k, p.area, len(need)))
                continue
            disc = Point(pt.x, pt.y).buffer(diameter / 2, resolution=24)
            for m in need:
                layers[m] = unary_union([layers[m], disc])
                added += 1
    return added, dropped


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
# Alignment guides and engraved IDs
# --------------------------------------------------------------------------


def lines_only(geom, min_len):
    """The line parts of a geometry, merged, without fragments under min_len."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    parts = []
    stack = [geom]
    while stack:
        g = stack.pop()
        if g.is_empty:
            continue
        if isinstance(g, LineString):
            parts.append(g)
        elif hasattr(g, "geoms"):
            stack.extend(g.geoms)
    if not parts:
        return MultiLineString()
    merged = linemerge(MultiLineString(parts))
    keep = [
        l
        for l in (merged.geoms if hasattr(merged, "geoms") else [merged])
        if l.length >= min_len
    ]
    return MultiLineString(keep)


def guides_on(piece, above, inset, clear, min_len):
    """Score lines on a piece's top face: the outline of the layer above,
    inset so the kerf leaves it hidden, kept clear of the piece's own edge."""
    if above is None or above.is_empty:
        return lines_only(piece.__class__(), min_len)
    return lines_only(
        above.buffer(-inset).boundary.intersection(piece.buffer(-clear)), min_len
    )


def guide_quality(lines):
    """Total length and how well the lines fix a position in both directions:
    the smaller eigenvalue share of the length-weighted normal tensor, from 0
    (one straight line: the piece can slide along it) to 0.5 (lines facing
    every direction equally)."""
    M = np.zeros((2, 2))
    total = 0.0
    for l in as_list(lines):
        c = np.asarray(l.coords)
        d = np.diff(c, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        ok = L > 0
        n = np.stack([-d[ok, 1], d[ok, 0]], 1) / L[ok, None]
        M += (n * L[ok, None]).T @ n
        total += L.sum()
    if total == 0:
        return 0.0, 0.0
    ev = np.linalg.eigvalsh(M)
    return total, float(ev[0] / ev.sum())


def text_geom(text, height):
    """Bold text as polygons, centred on the origin, `height` mm tall."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely import affinity
    from shapely.geometry import Polygon

    tp = TextPath(
        (0, 0), text, size=10, prop=FontProperties(family="DejaVu Sans", weight="bold")
    )
    g = Polygon()
    for ring in tp.to_polygons():
        if len(ring) >= 3:
            g = g.symmetric_difference(Polygon(ring).buffer(0))
    minx, miny, maxx, maxy = g.bounds
    f = height / (maxy - miny)
    g = affinity.translate(g, -(minx + maxx) / 2, -(miny + maxy) / 2)
    return affinity.scale(g, f, f, origin=(0, 0))


def place_label(region, text, size, step=1.0):
    """Engraved ID at the roomiest spot of `region` where it fits, trying
    smaller sizes down to 2.5 mm. Returns the text polygons or None."""
    import shapely
    from shapely import affinity

    if region.is_empty:
        return None
    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    xs, ys = xs.ravel(), ys.ravel()
    inside = shapely.contains_xy(region, xs, ys)
    xs, ys = xs[inside], ys[inside]
    if not len(xs):
        return None
    room = shapely.distance(region.boundary, shapely.points(xs, ys))
    order = np.argsort(-room)
    for h in sorted({size, max(2.5, 0.75 * size), 2.5}, reverse=True):
        t = text_geom(text, h)
        tminx, tminy, tmaxx, tmaxy = t.bounds
        half = 0.5 * math.hypot(tmaxx - tminx, tmaxy - tminy)
        for i in order[:400]:
            if room[i] < 0.5 * (tmaxy - tminy):
                break
            cand = affinity.translate(t, xs[i], ys[i])
            if room[i] >= half or cand.within(region):
                return cand
    return None


def etch_marks(layers, ids, inset, clear, min_len, label_size):
    """Guides and engraved IDs for every piece, keyed like ids."""
    guides, labels = {}, {}
    for k, g in enumerate(layers):
        above = layers[k + 1] if k + 1 < len(layers) else None
        for j, p in enumerate(as_list(g)):
            gl = guides_on(p, above, inset, clear, min_len)
            guides[(k, j)] = gl
            if above is None or above.is_empty:
                labels[(k, j)] = None  # nothing covers it
                continue
            region = p.buffer(-clear).intersection(above.buffer(-1.5))
            if not gl.is_empty:
                region = region.difference(gl.buffer(1.2))
            labels[(k, j)] = place_label(region, ids[(k, j)], label_size)
    return guides, labels


def alignment_check(layers, ids, guides, inset, clear, min_len, min_overlap):
    """For each piece above the base: what it is aligned to, the guide length
    and how well that guide fixes its position."""
    rows = []
    for k in range(1, len(layers)):
        below = layers[k - 1]
        above = layers[k + 1] if k + 1 < len(layers) else None
        for j, p in enumerate(as_list(layers[k])):
            rests = p.area and p.intersection(below).area / p.area >= min_overlap
            hangs = (
                not rests
                and above is not None
                and p.area
                and p.intersection(above).area / p.area >= min_overlap
            )
            if hangs:
                g = guides[(k, j)]  # parent's edge, on this piece
            else:
                g = lines_only(
                    p.buffer(-inset).boundary.intersection(below.buffer(-clear)),
                    min_len,
                )
            length, spread = guide_quality(g)
            rows.append((k, j, "hangs" if hangs else "rests", length, spread))
    return rows


# --------------------------------------------------------------------------
# Nesting (simple shelf packing by bounding box)
# --------------------------------------------------------------------------


def nest(pieces, sheet_w, sheet_h, margin, gap):
    """pieces: list of (id, polygon, extras), where extras is a list of
    geometries (guides, labels) that move with the piece. Returns a list of
    sheets, each a list of (id, polygon, extras) in sheet coordinates."""
    from shapely import affinity

    def move(geoms, f):
        return [None if e is None else f(e) for e in geoms]

    prepared = []
    for pid, poly, extras in pieces:
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if w > sheet_w - 2 * margin or h > sheet_h - 2 * margin:
            if h <= sheet_w - 2 * margin and w <= sheet_h - 2 * margin:
                c = poly.centroid
                poly = affinity.rotate(poly, 90, origin=c)
                extras = move(extras, lambda e: affinity.rotate(e, 90, origin=c))
                minx, miny, maxx, maxy = poly.bounds
                w, h = maxx - minx, maxy - miny
            else:
                raise SystemExit(
                    f"Piece {pid} ({w:.0f} x {h:.0f} mm) does not fit "
                    f"on a {sheet_w:.0f} x {sheet_h:.0f} mm sheet."
                )
        prepared.append(
            (
                pid,
                affinity.translate(poly, -minx, -miny),
                move(
                    extras, lambda e, dx=-minx, dy=-miny: affinity.translate(e, dx, dy)
                ),
                w,
                h,
            )
        )
    prepared.sort(key=lambda t: t[4], reverse=True)

    sheets, cur = [], []
    x, y, row_h = margin, margin, 0.0
    for pid, poly, extras, w, h in prepared:
        if x + w > sheet_w - margin:  # next row
            x, y, row_h = margin, y + row_h + gap, 0.0
        if y + h > sheet_h - margin:  # next sheet
            sheets.append(cur)
            cur, x, y, row_h = [], margin, margin, 0.0
        cur.append(
            (
                pid,
                affinity.translate(poly, x, y),
                move(extras, lambda e, dx=x, dy=y: affinity.translate(e, dx, dy)),
            )
        )
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


def line_path(lines):
    return " ".join(
        "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in l.coords)
        for l in as_list(lines)
    )


def write_svg(path, sheet, sheet_w, sheet_h):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_w}mm" '
        f'height="{sheet_h}mm" viewBox="0 0 {sheet_w} {sheet_h}">',
        f'<rect x="0" y="0" width="{sheet_w}" height="{sheet_h}" fill="none" '
        f'stroke="#999999" stroke-width="0.2" id="sheet-outline-do-not-cut"/>',
    ]
    # Engrave and score before cutting, so pieces have not shifted yet.
    lines.append('<g id="engrave-ids" fill="#000000" stroke="none">')
    for pid, _, (_, label) in sheet:
        for part in ([] if label is None else as_list(label)):
            lines.append(f'  <path fill-rule="evenodd" d="{poly_path(part)}"/>')
    lines += [
        "</g>",
        '<g id="score-guides" fill="none" stroke="#0000ff" ' 'stroke-width="0.01">',
    ]
    for pid, _, (guide, _) in sheet:
        if guide is not None and not guide.is_empty:
            lines.append(f'  <path id="guide-{pid}" d="{line_path(guide)}"/>')
    lines += ["</g>", '<g id="cut" fill="none" stroke="#ff0000" stroke-width="0.01">']
    for pid, poly, _ in sheet:
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
    for pid, poly, (guide, label) in sheet:
        plot_polys(ax, poly, color="#D85A30", lw=0.8)
        for l in ([] if guide is None else as_list(guide)):
            x, y = l.xy
            ax.plot(x, y, color="#2060D0", lw=0.6)
        if label is not None:
            c = label.centroid
            ax.text(
                c.x, c.y, pid, ha="center", va="center", fontsize=7, fontweight="bold"
            )
        else:
            c = poly.representative_point()
            ax.text(
                c.x,
                c.y,
                pid + "*",
                ha="center",
                va="center",
                fontsize=7,
                color="#B02020",
            )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match SVG orientation (y down)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_assembly(path, layers, ids, has_plate=False, guides=None, times=None):
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
        if guides and k > 0:
            for (kk, _), gl in guides.items():
                if kk == k - 1:
                    for l in as_list(gl):
                        x, y = l.xy
                        ax.plot(x, y, color="#2060D0", lw=1.4)
        for j, p in enumerate(as_list(g)):
            c = p.representative_point()
            ax.text(c.x, c.y, ids[(k, j)], ha="center", va="center", fontsize=7)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        num = k if has_plate else k + 1
        name = "Base plate (layer 0)" if has_plate and k == 0 else f"Layer {num}"
        if times and num >= 1:
            name += f"  ({times[num - 1]})"
        ax.set_title(name, fontsize=9)
    fig.suptitle(
        "Assembly: orange = this layer, dashed = layer below, blue = guides "
        "on the layer below that this layer's edge must cover "
        "(seen from above, north up)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_stack(out, layers, thickness):
    """The layered object as it will be built: an STL to open in any 3D viewer
    and a shaded preview image from three angles."""
    import trimesh

    meshes = []
    for k, g in enumerate(layers):
        for poly in as_list(g):
            m = trimesh.creation.extrude_polygon(poly, thickness)
            m.apply_translation([0, 0, k * thickness])
            meshes.append(m)
    stack = trimesh.util.concatenate(meshes)
    stack.export(out / "stack.stl")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, f = np.asarray(stack.vertices), np.asarray(stack.faces)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.array([0.4, -0.5, 0.75])
    L /= np.linalg.norm(L)
    shade = 0.35 + 0.65 * np.clip(n @ L, 0, 1)
    col = np.clip(shade[:, None] * np.array([[0.86, 0.89, 0.93]]), 0, 1)
    lo, hi = v.min(0), v.max(0)
    fig = plt.figure(figsize=(15, 5), facecolor="black")
    for i, (el, az) in enumerate([(20, -60), (20, 35), (20, 150)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_facecolor("black")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=el, azim=az)
        ax.set_axis_off()
    fig.savefig(
        out / "stack_preview.png", dpi=120, bbox_inches="tight", facecolor="black"
    )
    plt.close(fig)


def export_to_viewer(
    viewer,
    name,
    label,
    stl_path,
    out,
    layers,
    thickness,
    scale,
    times,
    every,
    n_sheets,
    n_pieces,
):
    """Copy the stack and a description of it into the visualizer's folder."""
    import json
    import shutil

    folder = Path(viewer) / "stacks" / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "stack.stl", folder / "stack.stl")
    src = Path(stl_path).parent
    frame = None
    try:
        info = json.loads((src / "build_info.json").read_text())
        aspect = json.loads((src / "frames.json").read_text())["aspect"]
        w = float(info["width"]) * scale
        frame = {
            "width": w,
            "depth": w * aspect,
            "height": (float(info["height"]) - float(info.get("cap") or 0)) * scale,
        }
    except (OSError, KeyError, ValueError):
        pass
    from shapely.ops import unary_union

    b = unary_union([g for g in layers if not g.is_empty]).bounds
    desc = {
        "name": name,
        "label": label or name,
        "layers": len(layers),
        "thickness": thickness,
        "every": every,
        "first": times[0] if times else None,
        "last": times[-1] if times else None,
        "size": [b[2] - b[0], b[3] - b[1], len(layers) * thickness],
        "frame": frame,
        "sheets": n_sheets,
        "pieces": n_pieces,
    }
    (folder / "info.json").write_text(json.dumps(desc, indent=2))
    index_path = Path(viewer) / "stacks" / "index.json"
    try:
        names = json.loads(index_path.read_text())["stacks"]
    except (OSError, ValueError, KeyError):
        names = []
    if name not in names:
        names.append(name)
    index_path.write_text(json.dumps({"stacks": names}, indent=2))
    return folder


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stl")
    p.add_argument(
        "--thickness",
        type=float,
        default=3.175,
        help="sheet thickness [mm]; measure your sheet",
    )
    p.add_argument(
        "--height",
        type=float,
        default=None,
        help="finished height [mm]; default keeps the model's proportions",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="fix the layer count instead of deriving it from the height",
    )
    p.add_argument(
        "--every",
        type=int,
        default=None,
        help="one layer at every Nth satellite frame (2 = every other)",
    )
    p.add_argument(
        "--first-frame",
        type=int,
        default=0,
        help="with --every: frame index for the first layer",
    )
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
    p.add_argument(
        "--keep",
        choices=["main", "all"],
        default="main",
        help="main: keep only the storm's connected body (cells that "
        "never join it are left out); all: keep every piece",
    )
    p.add_argument(
        "--keep-share",
        type=float,
        default=0.15,
        help="with --keep main, also keep separate bodies at least this "
        "share of the main body's volume",
    )
    p.add_argument(
        "--max-pillar",
        type=int,
        default=4,
        help="remove a floating piece rather than support it with a "
        "pillar taller than this many layers",
    )
    p.add_argument(
        "--no-etch",
        dest="etch",
        action="store_false",
        help="cut only: no scored guides or engraved IDs",
    )
    p.add_argument(
        "--etch-inset",
        type=float,
        default=0.2,
        help="guides sit this far inside the next layer's outline, about "
        "half the laser kerf, so they vanish under its edge [mm]",
    )
    p.add_argument(
        "--edge-clear",
        type=float,
        default=1.0,
        help="keep guides and IDs this far from a piece's own edge [mm]",
    )
    p.add_argument(
        "--min-guide",
        type=float,
        default=3.0,
        help="drop guide fragments shorter than this [mm]",
    )
    p.add_argument(
        "--label-size",
        type=float,
        default=4.0,
        help="height of engraved IDs [mm]; shrinks to 2.5 where needed",
    )
    p.add_argument(
        "--viewer",
        default=None,
        help="visualizer folder; also export the stack for stack.html",
    )
    p.add_argument(
        "--name",
        default=None,
        help="with --viewer: short name for this option "
        "(default: the --out folder name)",
    )
    p.add_argument(
        "--label",
        default=None,
        help="with --viewer: caption shown above it in stack.html",
    )
    p.add_argument("--out", default="cut/sculpture")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    times, zs, n_frames = None, None, None
    if a.every:
        zs, times, n_frames = frame_heights(a.stl, a.every, a.first_frame)
    raw, scale, size = slice_mesh(a.stl, a.thickness, a.width, a.height, a.layers, zs)
    fixed_layers = a.layers is not None or a.every is not None
    a.layers = len(raw)
    layers = [clean(g, a.min_feature, a.min_area, a.simplify) for g in raw]
    removed_groups = []
    if a.keep == "main":
        layers, removed_groups = keep_connected(layers, a.thickness, a.keep_share)
    floating = check_support(layers, a.min_overlap)
    n_supports, dropped = (
        add_supports(layers, a.min_overlap, a.support_d, a.max_pillar)
        if a.supports
        else (0, [])
    )
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

    guides, labels = {}, {}
    if a.etch:
        guides, labels = etch_marks(
            layers, ids, a.etch_inset, a.edge_clear, a.min_guide, a.label_size
        )

    # SVG's y axis points down; mirror so each piece is drawn as seen from
    # above with north at the top, matching the assembly guide. Text drawn
    # the right way up in north-up coordinates stays readable after this.
    from shapely import affinity

    flip = lambda e: None if e is None else affinity.scale(e, 1, -1, origin=(0, 0))
    keyed = [(k, j) for k, g in enumerate(layers) for j, _ in enumerate(as_list(g))]
    pieces = [
        (pid, flip(poly), [flip(guides.get(key)), flip(labels.get(key))])
        for key, (pid, poly) in zip(keyed, pieces)
    ]
    sheets = nest(pieces, a.sheet[0], a.sheet[1], a.margin, a.gap)
    for s, sheet in enumerate(sheets, 1):
        write_svg(out / f"sheet_{s:02d}.svg", sheet, a.sheet[0], a.sheet[1])
        write_map(
            out / f"sheet_{s:02d}_map.png",
            sheet,
            a.sheet[0],
            a.sheet[1],
            f"Sheet {s}: blue = scored guides; IDs marked * are not "
            "engraved, so write them on the protective film",
        )
    write_assembly(out / "assembly.png", layers, ids, a.base == "plate", guides, times)
    stack_ok = True
    try:
        write_stack(out, layers, a.thickness)
    except ValueError as exc:  # no triangulation engine installed
        stack_ok = False
        print(f"Skipped stack preview ({exc}); pip install mapbox_earcut to enable it.")
    if a.viewer and stack_ok:
        dest = export_to_viewer(
            a.viewer,
            a.name or out.name,
            a.label,
            a.stl,
            out,
            layers,
            a.thickness,
            scale,
            times,
            a.every,
            len(sheets),
            len(pieces),
        )
        print(f"Exported to {dest} for stack.html")
    elif a.viewer:
        print("Not exported to the viewer: stack.stl could not be built.")

    total = len(layers)
    cut_m = (
        sum(
            poly.exterior.length + sum(r.length for r in poly.interiors)
            for _, poly, _ in pieces
        )
        / 1000
    )
    unsupported = check_support(layers, a.min_overlap)
    stable, margin_mm = check_stability(layers, a.thickness)
    empty = [k + 1 for k, g in enumerate(layers) if g.is_empty]

    built_h = a.layers * a.thickness
    rep = [
        f"Model: {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm (W x D x H)",
        f"Sliced into {a.layers} sections of {a.thickness} mm = {built_h:.1f} mm"
        + (
            ""
            if abs(built_h - size[2]) < 0.05
            else f" ({100 * (built_h / size[2] - 1):+.1f}% vs the model height, "
            + (
                "set by the layer count)"
                if fixed_layers
                else "rounding to whole sheets)"
            )
        ),
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
    if times:
        rep.append(
            f"One layer per {a.every} frame(s), {len(times)} of {n_frames}: "
            f"layer 1 = {times[0]}, layer {len(times)} = {times[-1]}"
        )
        rep.append("  " + "  ".join(f"{k}:{t}" for k, t in enumerate(times, 1)))
    if empty:
        rep.append(f"Empty layers (no cloud at that height): {empty}")
    if n_supports:
        rep.append(
            f"{n_supports} clear support disc(s) of {a.support_d:.0f} mm added "
            "beneath floating pieces (visible in assembly.png)."
        )
    elif floating and not a.supports:
        rep.append(
            f"Floating pieces found: {len(floating)}; supports are off, "
            "so these need pins or removal."
        )
    if removed_groups:
        rep.append(
            f"Left out {len(removed_groups)} separate cell(s) that never join the "
            "main storm (volumes, cm^3): "
            + ", ".join(f"{v:.2f}" for v in removed_groups[:12])
            + (" ..." if len(removed_groups) > 12 else "")
        )
    if dropped:
        rep.append("Removed (would have needed a tall support pillar):")
        rep += [
            f"  a {area:.0f} mm^2 piece in layer {k + 1}, {n} layers above "
            f"any material"
            for k, area, n in dropped
        ]
    if unsupported:
        rep.append("Pieces glued to little material above or below (check by eye):")
        rep += [
            f"  {ids[(k, j)]}: {100 * ov:.0f}% of its area glued"
            for k, j, ov in unsupported
        ]
    else:
        rep.append(
            f"Every piece is glued over at least {100 * a.min_overlap:.0f}% of its "
            "area to the layer above or below."
        )
    if a.etch:
        score_m = sum(g.length for g in guides.values()) / 1000
        rows = alignment_check(
            layers, ids, guides, a.etch_inset, a.edge_clear, a.min_guide, a.min_overlap
        )
        weak = [r for r in rows if r[3] < 20 or r[4] < 0.08]
        hanging = [ids[(k, j)] for k, j, how, _, _ in rows if how == "hangs"]
        unlabelled = [ids[key] for key, lab in labels.items() if lab is None]
        rep += [
            "",
            f"Alignment: {score_m:.2f} m of scored guides, IDs engraved on "
            f"{len(labels) - len(unlabelled)} of {len(labels)} pieces.",
            "Assemble bottom-up from "
            + ("the base plate" if a.base == "plate" else "layer 1")
            + ". Set each piece dry until its "
            "edge just covers every blue guide on the piece below (a line still "
            "showing means the piece is off on that side), then bond it.",
        ]
        if hanging:
            rep.append(
                "Hanging pieces (no piece below; they carry the guide for the "
                "piece above on their own top face, so fit them after it): "
                + ", ".join(hanging)
            )
        if weak:
            rep.append(
                "Weakly guided pieces (short guides, or guides that run in one "
                "direction only); place these against assembly.png and check "
                "from several sides before gluing:"
            )
            rep += [
                f"  {ids[(k, j)]}: {length:.0f} mm of guide, spread {spread:.2f}"
                for k, j, _, length, spread in weak
            ]
        else:
            rep.append(
                "Every piece has at least 20 mm of guide running in more than "
                "one direction."
            )
        if unlabelled:
            rep.append(
                "No hidden area large enough for an engraved ID (write these on "
                "the protective film instead): " + ", ".join(unlabelled)
            )
        rep.append(
            "Test on offcuts first: how visible the engraving is, and which "
            "adhesive bonds two faces without bubbles or crazing."
        )
    else:
        rep.append("No guides etched (--no-etch); align pieces against assembly.png.")
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

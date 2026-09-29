"""Render the README figures into ``docs/img/``.

    pip install matplotlib
    python examples/make_figures.py

Blue is what survives, red is what deoverlap removed, the thin orange outline
is the corridor mask built around kept strokes.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy, union_all
from shapely.geometry import LineString, Point, box
from deoverlap import deoverlap, flatten_geometries

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

KEPT = "#3a9fc4"
REMOVED = "#ff1f4b"
MASK = "#f07040"
GROUP_COLORS = ["#3a9fc4", "#8c5cc7", "#2ca25f", "#e0a100", "#d95f02"]


def _lines(geom):
    for g in flatten_geometries(geom):
        if g.geom_type == "LineString":
            yield g
        elif g.geom_type == "Polygon":
            yield LineString(g.exterior.coords)
            for ring in g.interiors:
                yield LineString(ring.coords)


def _draw(ax, geoms, color, lw, alpha=1.0, zorder=2):
    for geom in geoms:
        for ln in _lines(geom):
            xs, ys = ln.xy
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")


def _mask(ax, mask, lw=0.6):
    merged = union_all(mask)
    for poly in getattr(merged, "geoms", [merged]):
        for ln in _lines(poly):
            xs, ys = ln.xy
            ax.plot(xs, ys, color=MASK, lw=lw, zorder=1)


def _frame(ax, title=None, bounds=None):
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, family="monospace")
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


def _result(ax, result, title, *, mask=True, lw=2.5):
    if mask:
        _mask(ax, result.mask)
    _draw(ax, result.kept, KEPT, lw)
    _draw(ax, result.removed, REMOVED, lw, zorder=3)
    _frame(ax, title)


def _legend(fig, *, mask=True):
    handles = [
        plt.Line2D([], [], color=KEPT, lw=3, label="kept"),
        plt.Line2D([], [], color=REMOVED, lw=3, label="removed"),
    ]
    if mask:
        handles.append(plt.Line2D([], [], color=MASK, lw=1, label="corridor mask"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06),
               ncol=len(handles), frameon=False)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _circle(cx, cy, r):
    return LineString(Point(cx, cy).buffer(r, quad_segs=32).exterior.coords)


def hero():
    geoms = [
        _circle(0, 0, 2.5),
        _circle(0, 3.5, 1.0),
        LineString([(-2.8, 1.0), (3.2, 1.0)]),
        _circle(3.0, -0.2, 0.5),
    ]
    result = deoverlap(geoms, 0.15, prefer="first", angle=90, keep_duplicates=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    _result(ax, result, None)
    _legend(fig)
    _save(fig, "hero.png")


def crop_vs_drop():
    geoms = [
        LineString([(0, 0), (4, 0)]),
        LineString([(1.2, 0.15), (4.8, 0.15)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.8))
    for ax, drop in zip(axes, [None, 0.5]):
        r = deoverlap(geoms, 0.25, prefer="first", drop=drop, keep_duplicates=True)
        _result(ax, r, f"drop={drop}", lw=2)
        ax.set_ylim(-0.4, 0.55)
    _legend(fig)
    _save(fig, "crop_vs_drop.png")


def prefer():
    geoms = [
        LineString([(0, 0), (1.2, 0)]),
        LineString([(0.1, 0.15), (4, 0.15)]),
        LineString([(2.5, 0.0), (3.0, 0.0)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.8))
    for ax, choice in zip(axes, ["longest", "first", "shortest"]):
        r = deoverlap(geoms, 0.25, prefer=choice, keep_duplicates=True)
        _result(ax, r, f'prefer="{choice}"', lw=2)
        ax.set_ylim(-0.4, 0.55)
    _legend(fig)
    _save(fig, "prefer.png")


def angle():
    geoms = [
        LineString([(0, 0), (4, 0)]),
        LineString([(0.5, 0.07), (3.5, 0.07)]),
        LineString([(2, -1.2), (2, 1.2)]),
        LineString([(3.0, -1.2), (3.4, 1.2)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, degrees in zip(axes, [90, 30]):
        r = deoverlap(geoms, 0.15, prefer="first", angle=degrees, keep_duplicates=True)
        _result(ax, r, f"angle={degrees}")
    _legend(fig)
    _save(fig, "angle.png")


def groups():
    """One colour per object: the cut ring is one entry, flattened it is two."""
    cutter = LineString([(1, -1), (1, 3)])
    ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    r = deoverlap([cutter, ring], 0.15, prefer="first", angle=90)
    flat = flatten_geometries(r.kept)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, geoms, title in [
        (axes[0], r.kept, f"result.kept: {len(r.kept)} objects"),
        (axes[1], flat, f"flatten_geometries(result.kept): {len(flat)} objects"),
    ]:
        for i, geom in enumerate(geoms):
            _draw(ax, [geom], GROUP_COLORS[i % len(GROUP_COLORS)], 3)
        _frame(ax, title, bounds=(-0.4, -1.2, 2.4, 3.2))
    _save(fig, "groups.png")


def self_overlap():
    ribbon = LineString([(0, 0), (6, 0), (6.5, 0.4), (6, 0.08), (0, 0.08), (0, 0)])
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.2))
    for ax, flag in zip(axes, [False, True]):
        r = deoverlap([ribbon], 0.1, prefer="first", self_overlap=flag,
                      min_length=0.05, keep_duplicates=True)
        _result(ax, r, f"self_overlap={flag}", lw=2)
        ax.set_ylim(-0.3, 0.6)
    _legend(fig)
    _save(fig, "self_overlap.png")


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _read_svg_paths(path: Path) -> list[LineString]:
    lines = []
    for d in re.findall(r'<path d="([^"]+)"', path.read_text()):
        nums = [float(n) for n in _NUM.findall(d)]
        coords = [(nums[i], -nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def _total(geoms):
    return sum(ln.length for g in geoms for ln in _lines(g))


def _ink(ax, geoms, pen, bounds, color=KEPT, strength=0.5, px=900):
    """Rasterised ``pen``-wide ink with multiply blending.

    Each stroke is one translucent layer; ``n`` overlapping layers transmit
    ``layer ** n`` of the paper, so double-drawn ink reads clearly darker.
    """
    x0, y0, x1, y1 = bounds
    xs = np.linspace(x0, x1, px)
    ys = np.linspace(y1, y0, px)
    gx, gy = np.meshgrid(xs, ys)
    count = np.zeros(gx.shape, dtype=np.int16)
    window = box(*bounds)
    for g in geoms:
        for ln in _lines(g):
            if not ln.intersects(window.buffer(pen)):
                continue
            foot = ln.buffer(pen / 2, quad_segs=6)
            fx0, fy0, fx1, fy1 = foot.bounds
            cols = (xs >= fx0) & (xs <= fx1)
            rows = (ys >= fy0) & (ys <= fy1)
            if not cols.any() or not rows.any():
                continue
            sub = np.ix_(rows, cols)
            count[sub] += contains_xy(foot, gx[sub], gy[sub])
    rgb = np.array(matplotlib.colors.to_rgb(color))
    layer = 1 - strength * (1 - rgb)
    image = layer[None, None, :] ** count[..., None]
    ax.imshow(image, extent=(x0, x1, y0, y1), interpolation="bilinear")
    return int(count.max())


def _box(ax, bounds):
    x0, y0, x1, y1 = bounds
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#555", lw=1, ls="--")


def osm_map(tolerance=0.5, pen=0.5, angle=90, zoom_center=(55, -42), zoom_size=14):
    src = ROOT / "examples" / "macarthur_maze.svg"
    geoms = _read_svg_paths(src)
    r = deoverlap(geoms, tolerance, angle=angle, min_length=2 * pen, keep_duplicates=True)

    before, after = _total(geoms), _total(r.kept)
    saved = 100 * (1 - after / before)
    cx, cy = zoom_center
    half = zoom_size / 2
    zoom = (cx - half, cy - half, cx + half, cy + half)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw(ax, r.kept, KEPT, 0.5)
    _draw(ax, r.removed, REMOVED, 0.9, zorder=3)
    _box(ax, zoom)
    _frame(ax, f"MacArthur Maze, 100 mm card: {len(geoms)} paths, "
               f"{saved:.0f}% of the ink removed")
    _legend(fig, mask=False)
    fig.text(0.5, 0.0, "Map data © OpenStreetMap contributors (ODbL)",
             ha="center", fontsize=8, color="#777")
    _save(fig, "osm_map.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    _ink(axes[0], geoms, pen, zoom)
    _frame(axes[0], f"before, {pen} mm pen", zoom)
    _mask(axes[1], r.mask, lw=0.4)
    _draw(axes[1], r.kept, KEPT, 1.4)
    _draw(axes[1], r.removed, REMOVED, 1.8, zorder=3)
    _frame(axes[1], f"-t {tolerance}mm --angle {angle} -m {2 * pen:g}mm", zoom)
    _ink(axes[2], r.kept, pen, zoom)
    _frame(axes[2], f"after, {pen} mm pen", zoom)
    _legend(fig)
    fig.text(0.5, -0.07, "ink panels multiply like real ink: the darker the "
             "blue, the more passes over the same paper", ha="center",
             fontsize=9, color="#555")
    _save(fig, "osm_zoom.png")
    print(f"map: {len(geoms)} paths, {before:.0f} -> {after:.0f} mm ({saved:.1f}% removed)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4))
    for col, degrees in enumerate([30, angle]):
        ra = deoverlap(geoms, tolerance, angle=degrees, min_length=2 * pen, keep_duplicates=True)
        _mask(axes[0, col], ra.mask, lw=0.4)
        _draw(axes[0, col], ra.kept, KEPT, 1.4)
        _draw(axes[0, col], ra.removed, REMOVED, 1.8, zorder=3)
        crossings = "crossings kept" if degrees < 90 else "crossings cut"
        _frame(axes[0, col], f"--angle {degrees}: {crossings}", zoom)
        _ink(axes[1, col], ra.kept, pen, zoom)
        _frame(axes[1, col], f"after, {pen} mm pen", zoom)
    _legend(fig)
    fig.text(0.5, 0.0, "at 30° a crossing keeps both roads, so the pen passes "
             "twice where they meet", ha="center", fontsize=9, color="#555")
    _save(fig, "osm_angle.png")


def main():
    hero()
    crop_vs_drop()
    prefer()
    angle()
    groups()
    self_overlap()
    osm_map()


if __name__ == "__main__":
    main()

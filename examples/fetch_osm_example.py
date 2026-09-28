"""Fetch a dense road interchange from OpenStreetMap and save it as a plotter SVG.

The output (``examples/macarthur_maze.svg``) is committed, so running this is
only needed to refresh the data or try another place::

    pip install osmnx
    python examples/fetch_osm_example.py --lat 37.8265 --lon -122.2950 --radius 1500

Map data © OpenStreetMap contributors (ODbL).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import osmnx as ox
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import linemerge

HERE = Path(__file__).parent

HIGHWAYS = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "unclassified", "service",
]


def fetch(lat: float, lon: float, radius: float) -> list[LineString]:
    gdf = ox.features_from_point((lat, lon), tags={"highway": HIGHWAYS}, dist=radius)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf = ox.projection.project_gdf(gdf)
    lines: list[LineString] = []
    for geom in gdf.geometry:
        if isinstance(geom, MultiLineString):
            lines.extend(geom.geoms)
        else:
            lines.append(geom)
    return lines


def to_page(lines: list[LineString], width_mm: float, margin_mm: float, radius: float):
    xs = [x for ln in lines for x, _ in ln.coords]
    ys = [y for ln in lines for _, y in ln.coords]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = (width_mm - 2 * margin_mm) / (2 * radius)
    half = width_mm / 2
    frame = box(cx - radius, cy - radius, cx + radius, cy + radius)
    out = []
    for ln in lines:
        clipped = ln.intersection(frame)
        parts = getattr(clipped, "geoms", [clipped])
        for part in parts:
            if part.geom_type != "LineString" or part.length == 0:
                continue
            coords = [(half + (x - cx) * scale, half - (y - cy) * scale) for x, y in part.coords]
            out.append(LineString(coords))
    return out, scale


def write_svg(lines: list[LineString], path: Path, size_mm: float) -> None:
    paths = []
    for ln in lines:
        d = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in ln.coords)
        paths.append(f'    <path d="{d}"/>')
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{size_mm}mm" height="{size_mm}mm" viewBox="0 0 {size_mm} {size_mm}">\n'
        '  <!-- Map data (c) OpenStreetMap contributors, ODbL -->\n'
        '  <g inkscape:groupmode="layer" inkscape:label="1" id="layer1" '
        'fill="none" stroke="black" stroke-width="0.1">\n'
        + "\n".join(paths)
        + "\n  </g>\n</svg>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lat", type=float, default=37.8265)
    ap.add_argument("--lon", type=float, default=-122.2950)
    ap.add_argument("--radius", type=float, default=1500.0, help="metres")
    ap.add_argument("--size", type=float, default=100.0, help="square page side, mm")
    ap.add_argument("--margin", type=float, default=5.0, help="mm")
    ap.add_argument("--out", type=Path, default=HERE / "macarthur_maze.svg")
    args = ap.parse_args()

    lines = fetch(args.lat, args.lon, args.radius)
    merged = linemerge(lines)
    lines = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
    page, scale = to_page(lines, args.size, args.margin, args.radius)
    write_svg(page, args.out, args.size)
    print(f"{len(page)} paths, 1 mm = {1 / scale:.1f} m -> {args.out}")


if __name__ == "__main__":
    main()

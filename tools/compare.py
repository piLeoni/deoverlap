"""Compare the Rust and Python engines on the MacArthur Maze map.

    maturin develop --release
    python tools/compare.py

For each configuration, prints kept length, piece count and timing for both
engines, plus how much kept ink one engine has where the other has none.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from shapely import STRtree
from shapely.geometry import LineString

from deoverlap import deoverlap, flatten_geometries

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"-?\d+(?:\.\d+)?")

CONFIGS = {
    "readme map (parallel 90)": dict(
        tolerance=0.5, keep="longest", parallel_only=True, parallel_angle=90, min_length=1.0
    ),
    "parallel 30": dict(tolerance=0.5, keep="longest", parallel_only=True, parallel_angle=30),
    "plain crop": dict(tolerance=0.5, keep="longest"),
    "segments": dict(tolerance=0.5, keep="longest", parallel_only=True, segments=True, min_length=1.0),
}


def read_paths(svg: Path) -> list[LineString]:
    lines = []
    for d in re.findall(r'<path d="([^"]+)"', svg.read_text()):
        nums = [float(n) for n in NUM.findall(d)]
        coords = [(nums[i], -nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def run(geoms, opts, engine):
    kwargs = {k: v for k, v in opts.items() if k != "tolerance"}
    t = time.perf_counter()
    r = deoverlap(geoms, opts["tolerance"], keep_duplicates=True, engine=engine, **kwargs)
    seconds = time.perf_counter() - t
    kept = [g for k in r.kept for g in flatten_geometries(k) if isinstance(g, LineString)]
    return kept, len(r.wholly_removed), seconds


def only_in(a: list[LineString], b: list[LineString], eps: float) -> float:
    """Length of `a` farther than `eps` from every line of `b`."""
    tree = STRtree(b)
    total = 0.0
    for line in a:
        rest = line
        for i in tree.query(line.buffer(eps)):
            rest = rest.difference(b[i].buffer(eps))
            if rest.is_empty:
                break
        total += rest.length
    return total


def main() -> None:
    geoms = read_paths(ROOT / "examples" / "macarthur_maze.svg")
    print(f"{len(geoms)} paths, {sum(g.length for g in geoms):.0f} mm of ink\n")
    for name, opts in CONFIGS.items():
        py, py_gone, py_s = run(geoms, opts, "python")
        rs, rs_gone, rs_s = run(geoms, opts, "rust")
        eps = opts["tolerance"] * 0.02
        print(f"== {name}")
        for label, kept, gone, s in (("python", py, py_gone, py_s), ("rust", rs, rs_gone, rs_s)):
            length = sum(g.length for g in kept)
            print(f"   {label:7s} {length:8.1f} mm kept, {len(kept):5d} pieces, {gone:4d} gone, {s:6.2f} s")
        print(f"   only python: {only_in(py, rs, eps):6.1f} mm, only rust: {only_in(rs, py, eps):6.1f} mm\n")


if __name__ == "__main__":
    main()

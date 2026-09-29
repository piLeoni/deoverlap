"""Time the engine on the bundled cards, per stage.

    python examples/bench.py

Prints wall time per phase so a change can be judged against a baseline
instead of a hunch.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (keeps the import cost out of the timings)
import numpy as np
from shapely.geometry import LineString
from shapely import get_coordinates

from deoverlap import deoverlap, flatten_geometries
from deoverlap import _core

ROOT = Path(__file__).resolve().parent.parent
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def read_svg_paths(path: Path) -> list[LineString]:
    out = []
    for d in re.findall(r'<path d="([^"]+)"', path.read_text()):
        nums = [float(n) for n in _NUM.findall(d)]
        coords = [(nums[i], -nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(coords) >= 2:
            out.append(LineString(coords))
    return out


def timeit(label, fn, runs=5):
    fn()  # warm up / JIT-free sanity
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    best = min(ts)
    print(f"{label:<28} {best * 1e3:8.1f} ms   (median {np.median(ts) * 1e3:8.1f} ms, n={runs})")
    return best


def bench_maze():
    geoms = read_svg_paths(ROOT / "examples" / "macarthur_maze.svg")
    print(f"card: {len(geoms)} paths, "
          f"{sum(g.length for g in geoms):.0f} mm of ink\n")

    # --- how the geometry crosses into Rust -------------------------------
    coords = [[get_coordinates(p).tolist() for p in flatten_geometries(g)] for g in geoms]

    def round_trip():
        _core.deoverlap(coords, 0.5, prefer="longest", angle=90.0,
                        self_overlap=False, min_length=1.0, drop=None,
                        keep_duplicates=False, mask=[], progress=None)

    def full():
        deoverlap(geoms, 0.5, angle=90, min_length=1.0)

    def full_with_dupes():
        deoverlap(geoms, 0.5, angle=90, min_length=1.0, keep_duplicates=True)

    def with_mask():
        r = deoverlap(geoms, 0.5, angle=90, min_length=1.0)
        r.mask  # force the shapely re-buffer path

    t_rt = timeit("rust only (no shapely)", round_trip)
    t_dupes = timeit("deoverlap(keep_duplicates)", full_with_dupes, runs=3)
    t_full = timeit("deoverlap(angle=90)", full)
    t_mask = timeit("  + result.mask", with_mask, runs=3)

    print(f"\n  of which shapely I/O:   {t_full * 1e3:8.1f} ms "
          f"({100 * (1 - t_rt / t_full):.0f}% of the run)")
    print(f"  of which mask rebuild:  {(t_mask - t_full) * 1e3:8.1f} ms")

    # --- angle=30 keeps the bearing filter on -----------------------------
    timeit("deoverlap(angle=30)", lambda: deoverlap(geoms, 0.5, angle=30, min_length=1.0))

    # --- self-overlap explodes every edge ---------------------------------
    small = geoms[:200]
    timeit("self_overlap, 200 paths", lambda: deoverlap(small, 0.5, angle=90,
                                                       self_overlap=True), runs=3)


def scale_curve():
    """Run time vs path count, to see if lookup is the growing term."""
    geoms = read_svg_paths(ROOT / "examples" / "macarthur_maze.svg")
    print("\nscale:")
    for n in (100, 200, 400, 800, 1473):
        sub = geoms[:n]
        t = timeit(f"  {n} paths", lambda sub=sub: deoverlap(sub, 0.5, angle=90), runs=3)
        print(f"      -> {t / n * 1e6:8.1f} us/path")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    bench_maze()
    scale_curve()

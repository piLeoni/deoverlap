"""Compare the Rust core with the Python engine on the MacArthur Maze map.

    cd core && cargo build --release --example run_json
    python tools/compare.py

For each configuration, prints kept length, piece count and timing for both
engines, plus how much kept ink one engine has where the other has none.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from shapely import STRtree
from shapely.geometry import LineString

CORE = Path(__file__).resolve().parent.parent
ROOT = CORE.parent
sys.path.insert(0, str(ROOT / "src"))
from deoverlap import deoverlap, flatten_geometries  # noqa: E402

BIN = CORE / "target" / "release" / "examples" / "run_json"
NUM = re.compile(r"-?\d+(?:\.\d+)?")

CONFIGS = {
    "readme map (parallel 90)": dict(
        tolerance=0.5, keep="longest", parallel_only=True, parallel_angle=90, min_length=1.0
    ),
    "parallel 30": dict(tolerance=0.5, keep="longest", parallel_only=True, parallel_angle=30),
    "plain crop": dict(tolerance=0.5, keep="longest"),
    "segments": dict(tolerance=0.5, keep="longest", parallel_only=True, segments=True, min_length=1.0),
}


def read_paths(svg: Path) -> list[list[tuple[float, float]]]:
    paths = []
    for d in re.findall(r'<path d="([^"]+)"', svg.read_text()):
        nums = [float(n) for n in NUM.findall(d)]
        coords = [(nums[i], -nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(coords) >= 2:
            paths.append(coords)
    return paths


def run_python(paths, opts):
    geoms = [LineString(p) for p in paths]
    kwargs = {k: v for k, v in opts.items() if k != "tolerance"}
    t = time.perf_counter()
    r = deoverlap(geoms, opts["tolerance"], keep_duplicates=True, **kwargs)
    seconds = time.perf_counter() - t
    kept = [g for k in r.kept for g in flatten_geometries(k) if isinstance(g, LineString)]
    return kept, len(r.wholly_removed), seconds


def run_rust(paths, opts):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"options": {**opts, "keep_duplicates": True}, "paths": paths}, f)
    out = json.loads(subprocess.check_output([str(BIN), f.name]))
    Path(f.name).unlink()
    kept = [LineString(p) for p in out["kept"] if len(p) >= 2]
    return kept, out["wholly_removed"], out["seconds"]


def only_in(a: list[LineString], b: list[LineString], eps: float) -> float:
    """Length of `a` farther than `eps` from every line of `b`."""
    tree = STRtree(b)
    total = 0.0
    for line in a:
        near = [b[i] for i in tree.query(line.buffer(eps))]
        rest = line
        for other in near:
            rest = rest.difference(other.buffer(eps))
            if rest.is_empty:
                break
        total += rest.length
    return total


def main() -> None:
    if not BIN.exists():
        sys.exit("build first: cargo build --release --example run_json")
    paths = read_paths(ROOT / "examples" / "macarthur_maze.svg")
    total = sum(LineString(p).length for p in paths)
    print(f"{len(paths)} paths, {total:.0f} mm of ink\n")
    for name, opts in CONFIGS.items():
        py, py_gone, py_s = run_python(paths, opts)
        rs, rs_gone, rs_s = run_rust(paths, opts)
        py_len = sum(g.length for g in py)
        rs_len = sum(g.length for g in rs)
        eps = opts["tolerance"] * 0.02
        print(f"== {name}")
        print(f"   python: {py_len:8.1f} mm kept, {len(py):5d} pieces, {py_gone:4d} gone, {py_s:6.2f} s")
        print(f"   rust:   {rs_len:8.1f} mm kept, {len(rs):5d} pieces, {rs_gone:4d} gone, {rs_s:6.2f} s")
        print(
            f"   only python: {only_in(py, rs, eps):6.1f} mm, "
            f"only rust: {only_in(rs, py, eps):6.1f} mm\n"
        )


if __name__ == "__main__":
    main()

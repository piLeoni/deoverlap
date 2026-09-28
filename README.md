# Deoverlap

De-overlap Shapely geometries that sit within a tolerance of each other.

Source, examples and issues: [github.com/piLeoni/deoverlap](https://github.com/piLeoni/deoverlap)

The common case is pen-plotter work: two strokes closer than a pen width
visually merge on paper, so only one of them should keep the ink. Unlike
endpoint-only “deduplicate” tools, this library builds a **corridor** around
each kept stroke and crops (or drops) later strokes that fall inside it.

![Tangent circles and a line: the corridor mask (orange) around kept strokes (blue) crops the overlapping arcs (red)](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/hero.png)

In every figure, blue is kept, red is removed and the thin orange outline is
the corridor mask. They are rendered by `examples/make_figures.py`.

```bash
pip install deoverlap
# with the vpype command:
pip install "deoverlap[vpype]"
```

## Quick start

```python
from shapely.geometry import LineString
from deoverlap import deoverlap

geoms = [
    LineString([(0, 0), (2, 0)]),
    LineString([(1, 0.05), (3, 0.05)]),  # parallel, 0.05 away
]
result = deoverlap(geoms, tolerance=0.1)

print(len(result.kept), "surviving geometries")
print("wholly removed:", result.wholly_removed)
```

`tolerance` is a distance in the same units as the geometries: strokes closer
than this to a kept stroke are cut. For a plotter, use the pen width.

The Python function and the vpype command take the same options with the same
defaults; `--self-overlap` on the command line is `self_overlap=True` in Python.

## Prefer — which stroke wins

When two corridors collide, priority is explicit:

| `prefer=` | Behaviour |
|---|---|
| `longest` (default) | The stroke that covers more ground wins |
| `first` | Input order |
| `shortest` | Short marks / detail win |

![The same three strokes under prefer=longest, first and shortest](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/prefer.png)

Original indices are preserved in `result.kept_parts` / `removed_parts` even
when processing order changes.

## Angle — which strokes count as overlapping

Two strokes overlap only where their directions differ by at most `angle`
degrees (default 30). A plotter redrawing a crossing is usually fine; the pain
is parallel near-coincident runs. `angle=90` counts every nearby stroke, so
crossings get cut too.

```python
result = deoverlap(geoms, tolerance=0.1, angle=30)
```

Directions are compared **locally, edge by edge**, so a curving ramp is cropped
only where it actually runs alongside another road, whatever direction its two
ends point in.

Where two strokes meet at an angle θ, they overlap for roughly `pen / sin θ`:
about 2 pen widths at 30° and 4 at 15°. The default of 30° catches parallel
runs and trims shallow merges; 45° or 60° trim steeper merges too.

![angle=90 cuts the crossings too; angle=30 only crops the parallel duplicate](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/angle.png)

## Crop or drop

By default the overlap is cut away and the rest of the stroke is kept. With
`drop=0.5`, a stroke that would lose more than half its length is discarded
whole instead of leaving stubs.

![drop=None keeps the protruding stub, drop=0.5 discards the mostly covered stroke](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/crop_vs_drop.png)

`min_length=` drops pieces shorter than that after cutting (points are never
removed by it).

## Split pieces stay one object

When a ring is cut by a corridor it becomes two arcs. Those arcs are returned
as **one** multipart geometry under the original input index — a logical
stroke, not two anonymous objects:

```python
ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
cutter = LineString([(1, -1), (1, 3)])
result = deoverlap([cutter, ring], 0.15, prefer="first", angle=90)

result.kept_parts[1]  # MultiLineString of both arcs
```

Use `flatten_geometries(result.kept)` for a flat list of primitive pieces.

![A ring cut by a line stays one entry in result.kept](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/groups.png)

## Self-overlap

A thin road outline is often **one** LineString (left kerb → end cap → right
kerb). Normally deoverlap never compares a geometry to itself, so the two sides
stay as a heavy double stroke. With `self_overlap=True` every edge is its own
corridor, so opposite sides can suppress each other. Neighbouring edges never
cut each other, so joints are not nibbled, unless the path folds back on itself
like a hairpin.

![A thin ribbon drawn as one polyline: untouched normally, one side suppressed with self_overlap](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/self_overlap.png)

## Multi-stage with a carried mask

```python
r1 = deoverlap(batch1, 0.1)
r2 = deoverlap(batch2, 0.1, mask=r1.mask)
```

Stage 2 is clipped against everything stage 1 kept.

## Result

`deoverlap(...)` returns a `DeoverlapResult`:

| Field | Meaning |
|---|---|
| `kept` | Surviving geometries, one per input that kept some ink |
| `removed` | Cut pieces, flattened (if `keep_duplicates=True`) |
| `kept_parts` | `{original_index: geometry}` |
| `removed_parts` | `{original_index: geometry}` (if `keep_duplicates=True`) |
| `wholly_removed` | Indices removed entirely |
| `mask` | Corridor polygons (for the next stage) |

## Performance

The engine is written in Rust (on the [`geo`](https://crates.io/crates/geo)
crate) and ships as a compiled extension, so `pip install` needs no Rust
toolchain on the supported platforms. On the MacArthur Maze map below
(1473 paths) a run takes well under a second.

## vpype plugin

Installing `deoverlap[vpype]` registers a `deoverlap` command that is
**layer-safe** (honours `-l`) and uses corridor proximity rather than
endpoint-only matching:

```bash
pip install "deoverlap[vpype]"
vpype read map.svg deoverlap -t 0.1mm -l 1,2,3 write out.svg
```

| Flag | Default | Meaning |
|---|---|---|
| `-t`, `--tolerance` | `0.1mm` | Corridor radius, usually the pen width |
| `--prefer` | `longest` | `longest` \| `first` \| `shortest` |
| `--angle` | `30` | Max direction difference in degrees; `90` cuts crossings too |
| `--self-overlap` | off | Let a path overlap itself |
| `-m`, `--min-length` | `0mm` | Drop pieces shorter than this after cutting |
| `--drop` | off | Discard a stroke when more than this fraction would be cut |
| `-k`, `--keep-duplicates` | off | Keep removed pieces in a separate layer |
| `-p`, `--progress-bar` | off | Display a progress bar |
| `-l`, `--layer` | `all` | Target layer(s) |

`-t`, `-l`, `-k` and `-p` mean the same as in vpype's own commands and the
`deduplicate` plugin; `-m` matches `vpype filter`.

### Walkthrough with the bundled example

`examples/parallel_strokes.svg` is a tiny card with two near-parallel pairs
(0.05 mm apart), one deliberate cross, and a dual-kerb rectangle drawn as a
**single** polyline. No OSM dump required — open it, run the commands, open the
outputs side by side.

```bash
# Inspect path count / drawn length
vpype read examples/parallel_strokes.svg stat

# Crop near-parallel duplicates; leave the perpendicular cross alone
vpype read examples/parallel_strokes.svg \
  deoverlap -t 0.1mm -l 1 \
  write examples/out_parallel.svg

# Same, but also let the dual-kerb ring collapse against itself
vpype read examples/parallel_strokes.svg \
  deoverlap -t 0.12mm --self-overlap -l 1 \
  write examples/out_self_overlap.svg

vpype read examples/out_parallel.svg stat
vpype read examples/out_self_overlap.svg stat
```

What to look for (in the SVG or in `stat`’s drawn length):

1. **Before** — each parallel pair is a double stroke; the ring reads as a heavy
   double outline.
2. **`out_parallel.svg`** — each pair reduced to one survivor; the cross still
   there; the ring still double (one geometry cannot overlap itself).
3. **`out_self_overlap.svg`** — the ring’s opposite sides suppress each other,
   so the dual kerb thins toward a single outline.

### A real map: the MacArthur Maze

`examples/macarthur_maze.svg` is a 3 km square of Oakland's MacArthur Maze
interchange from OpenStreetMap, scaled onto a 100 mm card (1 mm ≈ 33 m). At
that scale dual carriageways, stacked ramps and frontage roads sit a fraction
of a millimetre apart, so a 0.5 mm pen draws the same paper two or three
times. Set the tolerance to the pen width: anything closer than that would
overlap on paper.

```bash
vpype read examples/macarthur_maze.svg stat
vpype read examples/macarthur_maze.svg \
  deoverlap -t 0.5mm --angle 90 -m 1mm \
  write examples/out_maze.svg
vpype read examples/out_maze.svg stat
```

`--angle 90` is the maximum: every nearby stroke counts, whatever its
direction. Side streets stop where the main road's ink begins, and at crossings
the shorter path is split around the longer one. The gap is exactly the other
road's ink width, so on paper the crossing still looks whole, just without the
dark spot of doubled ink.

Cutting leaves fragments; almost every piece under 1 mm on this card is one.
`-m 1mm` (two pen widths) drops them. The path count goes from 1473 to 860 and
the drawn length drops by 35%. With the default `--angle 30` (crossings left
alone) and the same `-m 1mm`, the card loses about 23%.

![The whole card: removed ink in red, the dashed box is the zoom below](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_map.png)

In the zoom, the ink panels are blended like real ink: every pass multiplies
the colour, so the darker the blue, the more times the pen went over the same
spot. Before, the dark bands are doubled carriageways and the dark dots are
junctions, where a round pen tip lands on ink that is already there. After,
the ink is one even layer.

![Zoom at pen width: before, what was removed, after](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_zoom.png)

To try another place (needs `pip install osmnx`):

```bash
python examples/fetch_osm_example.py --lat 51.5074 --lon -0.1278 --radius 1500 \
  --out examples/my_place.svg
```

Map data © OpenStreetMap contributors, available under the
[ODbL](https://www.openstreetmap.org/copyright).

### Different settings per layer

Chain the command to give each layer its own settings:

```bash
vpype read map.svg \
  deoverlap -t 0.1mm -l 3,4 \
  deoverlap -t 0.15mm --self-overlap -l 5 \
  write opt.svg
```

## API

```python
deoverlap(
    geometries,
    tolerance,
    *,
    prefer="longest",      # longest | first | shortest
    angle=30.0,            # degrees, 0–90; 90 cuts crossings too
    self_overlap=False,
    min_length=0.0,
    drop=None,             # fraction 0–1; None always crops
    keep_duplicates=False,
    progress_bar=False,
    mask=None,
) -> DeoverlapResult
```

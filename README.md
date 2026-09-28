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
from deoverlap import deoverlap, KeepPolicy

geoms = [
    LineString([(0, 0), (2, 0)]),
    LineString([(1, 0.05), (3, 0.05)]),  # parallel, 0.05 away
]
result = deoverlap(geoms, tolerance=0.1, keep=KeepPolicy.LONGEST)

print(len(result.kept), "surviving geometries")
print("wholly removed:", result.wholly_removed)
```

`tolerance` is a distance in the same units as the geometries. For a plotter
with a 0.1 mm pen, start around `0.08`–`0.15` (page mm).

## Keep policy — which stroke wins

When two corridors collide, priority is explicit:

| `keep=` | Behaviour |
|---|---|
| `first` (default) | Input order — stable, historical behaviour |
| `longest` | Prefer the stroke that covers more ground |
| `shortest` | Prefer short marks / detail |

![The same three strokes under keep=first, longest and shortest](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/keep_policy.png)

Original indices are preserved in `result.kept_parts` / `removed_parts` even
when processing order changes.

## Crop vs drop

- `mode="crop"` (default) — subtract the overlap, keep the rest.
- `mode="drop"` — if more than `drop_fraction` (default 0.5) of a geometry’s
  length is covered, discard the whole thing instead of leaving stubs.

![crop keeps the protruding stub, drop discards the mostly covered stroke](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/crop_vs_drop.png)

`min_length=` drops lineal fragments shorter than that threshold after clipping
(points are never removed by it).

## Parallel only — keep crossings

By default a corridor punches a hole in *any* later geometry, including a
perpendicular cross. For plotters, overdraw at a cross is usually fine; the
pain is parallel near-coincident runs:

```python
result = deoverlap(
    geoms,
    tolerance=0.1,
    parallel_only=True,
    parallel_angle=30,  # degrees
)
```

Only corridors whose bearing is within `parallel_angle` of the candidate are
used as clip masks. Bearings are compared **locally, edge by edge**, so a
curving ramp is cropped only where it actually runs alongside another road,
whatever direction its two ends point in.

`parallel_angle` is the tangency threshold. Where two strokes meet at an angle
θ, they overlap for roughly `pen / sin θ`: about 2 pen widths at 30° and 4 at
15°. The default of 30° catches parallel runs and trims shallow merges.
Raising it to 45° or 60° trims steeper merges too. Crossings steeper than the
threshold are never cut.

![Without parallel_only the crossings get punched; with it only the parallel duplicate is cropped](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/parallel_only.png)

## Groups — split pieces stay one object

When a ring is cut by a corridor it becomes two arcs. With `group=True`
(default) those arcs are returned as **one** multipart geometry under the
original input index — a logical stroke, not two anonymous objects:

```python
ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
cutter = LineString([(1, -1), (1, 3)])
result = deoverlap([cutter, ring], 0.15, group=True)

result.kept_parts[1]  # MultiLineString of both arcs
```

Set `group=False` for a flat list of primitive pieces (origins still recorded
in `kept_parts`).

![A ring cut by a line: one grouped result versus separate arcs](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/groups.png)

This is the right model for layer-aware pipelines too: treat each input
geometry as a group, run deoverlap per layer, and never let one layer’s
corridor eat another layer’s groups unless you pass a shared `mask`.

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
| `kept` | Drawable geometries (grouped or flat) |
| `removed` | Flattened cut pieces (if `keep_duplicates=True`) |
| `kept_parts` | `{original_index: geometry}` |
| `removed_parts` | `{original_index: geometry}` |
| `wholly_removed` | Indices removed entirely |
| `mask` | Corridor polygons (for the next stage) |

Iterating the result still yields the legacy
`(kept, kept_map, removed, mask)` tuple.

## Performance notes

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
vpype read map.svg deoverlap -t 0.1mm --keep longest -l 1,2,3 write out.svg
```

| Flag | Default | Meaning |
|---|---|---|
| `-t` / `--tolerance` | `0.1mm` | Corridor half-width |
| `--keep` | `longest` | `first` \| `longest` \| `shortest` |
| `--mode` | `crop` | `crop` \| `drop` |
| `--min-length` | `0` | Drop stubs shorter than this |
| `--parallel-only` | on | Do not punch perpendicular crossings |
| `--parallel-angle` | `30` | Local bearing window in degrees |
| `--segments` | off | Self-overlap (explode edges; see below) |
| `--segment-adjacency` | `1` | Keep N neighbours on the same chain |
| `-l` | `all` | Target layer(s) |
| `-k` | off | Keep removed pieces on a new layer |

### Walkthrough with the bundled example

`examples/parallel_strokes.svg` is a tiny card with two near-parallel pairs
(0.05 mm apart), one deliberate cross, and a dual-kerb rectangle drawn as a
**single** polyline. No OSM dump required — open it, run the commands, open the
outputs side by side.

```bash
# Inspect path count / drawn length
vpype read examples/parallel_strokes.svg stat

# Crop near-parallel duplicates; leave the perpendicular cross alone
vpype read examples/parallel_strokes.svg \
  deoverlap -t 0.1mm --keep longest --parallel-only -l 1 \
  write examples/out_parallel.svg

# Same, but also let the dual-kerb ring collapse against itself
vpype read examples/parallel_strokes.svg \
  deoverlap -t 0.12mm --keep longest --parallel-only --segments -l 1 \
  write examples/out_segments.svg

vpype read examples/out_parallel.svg stat
vpype read examples/out_segments.svg stat
```

What to look for (in the SVG or in `stat`’s drawn length):

1. **Before** — each parallel pair is a double stroke; the ring reads as a heavy
   double outline.
2. **`out_parallel.svg`** — each pair reduced to one survivor; the cross still
   there; the ring still double (one geometry cannot self-crop).
3. **`out_segments.svg`** — the ring’s opposite sides suppress each other, so
   the dual kerb thins toward a single outline.

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
  deoverlap -t 0.5mm --keep longest --parallel-angle 90 --min-length 1mm \
  write examples/out_maze.svg
vpype read examples/out_maze.svg stat
```

`--parallel-angle 90` is the maximum: every nearby stroke counts, whatever its
direction. Side streets stop where the main road's ink begins, and at crossings
the shorter path is split around the longer one. The gap is exactly the other
road's ink width, so on paper the crossing still looks whole, just without the
dark spot of doubled ink.

Cutting leaves fragments; almost every piece under 1 mm on this card is one.
`--min-length 1mm` (two pen widths) drops them. The path count goes from 1473
to 860 and the drawn length drops by 35%. For the more conservative default
(`--parallel-angle 30`, crossings left alone), the same card loses about 21%.

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
  deoverlap -t 0.1mm --keep longest --parallel-only -l 3,4 \
  deoverlap -t 0.15mm --keep longest --parallel-only --segments -l 5 \
  write opt.svg
```

### Self-overlap (`--segments`)

A thin road outline is often **one** LineString (left kerb → end cap → right
kerb). Without `--segments`, deoverlap never compares a geometry to itself, so
the two sides stay as a heavy double stroke. With `--segments`, every edge is
its own corridor unit; opposite sides can suppress each other, while immediate
neighbours on the same chain (`--segment-adjacency`, default 1) stay intact so
joints are not nibbled.

![A thin ribbon drawn as one polyline: untouched without segments, one side suppressed with segments](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/segments.png)

```bash
vpype read map.svg deoverlap -t 0.15mm --keep longest --segments -l 1 write out.svg
```

## API

```python
deoverlap(
    geometries,
    tolerance,
    *,
    keep="first",          # first | longest | shortest
    mode="crop",           # crop | drop
    min_length=0.0,
    drop_fraction=0.5,
    parallel_only=False,
    parallel_angle=30.0,
    segments=False,        # self-overlap via edge explosion
    segment_adjacency=1,
    group=True,
    keep_duplicates=False,
    progress_bar=False,
    mask=None,
) -> DeoverlapResult
```

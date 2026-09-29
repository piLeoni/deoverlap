# Deoverlap

De-overlap vector strokes to prevent overdrawing.

Source, examples and issues: [github.com/piLeoni/deoverlap](https://github.com/piLeoni/deoverlap)

Where strokes run on top of each other, deoverlap keeps one and cuts the
others back, so no area gets drawn twice. One **Rust** engine, three front
ends: a **Python** library (Shapely geometries), a **vpype** command for SVG
plotter pipelines, and **Node.js** bindings over flat coordinate buffers.

![Before, removed, and after on a real map at pen width (OpenStreetMap)](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_zoom.png)

A street map drawn with a 0.5 mm pen: before (left), what was cut away
(middle), after (right). Blue is kept, red is removed, orange is the corridor
mask — the same colours in every figure below.

## Quick start

### Python

```bash
pip install deoverlap
```

```python
from shapely.geometry import LineString
from deoverlap import deoverlap

geoms = [
    LineString([(0, 0), (2, 0)]),
    LineString([(1, 0.05), (3, 0.05)]),  # parallel, 0.05 away
]
result = deoverlap(geoms, tolerance=0.1)

print(len(result.kept), "surviving geometries")  # 2: the second is cropped
print("wholly removed:", result.wholly_removed)  # []
```

`tolerance` is a distance in the same units as the geometries: strokes closer
than this to a kept stroke are cut. For a plotter, use the pen width.

### vpype

```bash
pip install "deoverlap[vpype]"
vpype read map.svg deoverlap -t 0.1mm -l 1,2,3 write out.svg
```

Same options as the Python function (`--self-overlap` ↔ `self_overlap=True`);
only the tolerance has a default here, `0.1mm`. Full flag list:
[vpype plugin](#vpype-plugin).

### Node.js

```bash
npm install deoverlap
```

Flat buffers, API and building from source: [`bindings/node/README.md`](https://github.com/piLeoni/deoverlap/blob/main/bindings/node/README.md),
[`docs/WIRE_FORMAT.md`](https://github.com/piLeoni/deoverlap/blob/main/docs/WIRE_FORMAT.md).

## How it works

Geometries are processed one at a time, in priority order. Each kept stroke
gets a corridor of radius `tolerance` around it, and every later stroke loses
the parts that fall inside a corridor. The options below decide which stroke
goes first (`prefer`), which nearby strokes count as overlapping (`angle`),
and what to do with the leftovers of a cut stroke (`drop`, `min_length`).

![Tangent circles and a line: the corridor mask (orange) around kept strokes (blue) crops the overlapping arcs (red)](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/hero.png)

## Prefer — which stroke wins

The processing order decides which of two overlapping strokes is kept whole:

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
degrees (default 30). Parallel runs and shallow merges are cropped; crossings
steeper than that are left alone. `angle=90` counts every nearby stroke, so
crossings get cut too.

```python
result = deoverlap(geoms, tolerance=0.1, angle=90)  # cut crossings too
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

A dropped stroke has no entry in `kept_parts` and its index is listed in
`wholly_removed`. With `keep_duplicates=True`, `removed_parts` holds the
original geometry unchanged.

`min_length=` drops pieces shorter than that after cutting (points are never
removed by it).

## Split pieces stay one object

A ring crossed by another stroke is cut into two arcs. They come back as
**one** `MultiLineString` under the ring's input index, not as two anonymous
pieces:

```python
cutter = LineString([(1, -1), (1, 3)])
ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
result = deoverlap([cutter, ring], 0.15, prefer="first", angle=90)

result.kept_parts[1]  # MultiLineString with both arcs
```

One colour per object. On the left, both halves of the ring are one entry of
`result.kept`; on the right, `flatten_geometries(result.kept)` splits them
into separate single parts, when that is what you need:

![The cut ring is one object in result.kept (same colour), and two after flatten_geometries](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/groups.png)

## Self-overlap

A thin road outline is often **one** LineString (left kerb → end cap → right
kerb). Normally deoverlap never compares a geometry to itself, so the two sides
stay as a heavy double stroke. With `self_overlap=True` every edge is its own
corridor, so opposite sides can suppress each other. Neighbouring edges never
cut each other, so joints are not nibbled, unless the path folds back on itself
like a hairpin.

![A thin ribbon drawn as one polyline: untouched normally, one side suppressed with self_overlap](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/self_overlap.png)

## Multi-stage with a carried mask

To process batches separately (for example, main roads first, then side
streets), pass the previous result as `mask`:

```python
r1 = deoverlap(batch1, 0.1)
r2 = deoverlap(batch2, 0.1, mask=r1)  # or mask=r1.mask
```

Stage 2 is clipped against everything stage 1 kept.

## Result

`deoverlap(...)` returns a `DeoverlapResult`:

| Field | Meaning |
|---|---|
| `kept` | Surviving geometries, one per input that kept something |
| `removed` | Cut pieces, flattened (if `keep_duplicates=True`) |
| `kept_parts` | `{input_index: geometry}` |
| `removed_parts` | `{input_index: geometry}` (if `keep_duplicates=True`) |
| `wholly_removed` | Indices of inputs with nothing kept |
| `mask` | Corridor polygons, for a next stage |

## Performance

The Python wheel is a compiled extension (no Rust toolchain at install time).
The engine builds no polygons: the corridor of a straight edge is a capsule,
and the part of another edge inside it is computed exactly. On the MacArthur
Maze map below (1473 paths) a run takes about 30 ms.

## vpype plugin

`pip install "deoverlap[vpype]"` registers a `deoverlap` command. Each layer
is processed on its own; `-l` picks which ones.

| Flag | Default | Meaning |
|---|---|---|
| `-t`, `--tolerance` | `0.1mm` | Corridor radius, usually the pen width |
| `--prefer` | `longest` | `longest` \| `first` \| `shortest` |
| `--angle` | `30` | Max direction difference in degrees; `90` cuts crossings too |
| `--self-overlap` | off | Let parts of one path cut each other |
| `-m`, `--min-length` | `0mm` | Drop pieces shorter than this after cutting |
| `--drop` | off | Discard a stroke when more than this fraction would be cut |
| `-k`, `--keep-duplicates` | off | Keep removed pieces in a separate layer |
| `-p`, `--progress-bar` | off | Display a progress bar |
| `-l`, `--layer` | `all` | Target layer(s) |

`-t`, `-l`, `-k` and `-p` follow the usual vpype conventions; `-m` matches
`vpype filter`.

Chain the command to give each layer its own settings:

```bash
vpype read map.svg \
  deoverlap -t 0.1mm -l 3,4 \
  deoverlap -t 0.15mm --self-overlap -l 5 \
  write opt.svg
```

### Walkthrough with the bundled example

`examples/parallel_strokes.svg` is a tiny card with two near-parallel pairs
(0.05 mm apart), one deliberate cross, and a dual-kerb rectangle drawn as a
**single** polyline. Run the commands and open the outputs side by side.

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

The whole card after this run; the dashed box is the zoom shown at the
[top of this page](#deoverlap):

![The whole 100 mm card: kept in blue, removed in red; dashed box is the zoom at the top](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_map.png)

`--angle 90` is the maximum: every nearby stroke counts, whatever its
direction. Side streets stop where the main road's ink begins, and at crossings
the shorter path is split around the longer one. The gap is exactly the other
road's ink width, so on paper the crossing still looks whole, just without the
dark spot of doubled ink.

Cutting leaves fragments; almost every piece under 1 mm on this card is one.
`-m 1mm` (two pen widths) drops them. The path count goes from 1473 to 860 and
the drawn length drops by 35%.

With the default `--angle 30` crossings are left alone: parallel runs are
still merged, but both roads are drawn through every junction, and the card
loses about 23% instead of 35%. The same zoom, crossings kept and cut:

![The zoom at --angle 30 and --angle 90: at 30 the crossings keep both roads and leave dark spots, at 90 they are cut](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_angle.png)

Keep crossings when every stroke must stay continuous (e.g. for later
editing); cut them when only the ink on paper matters.

To try another place (needs `pip install osmnx`):

```bash
python examples/fetch_osm_example.py --lat 51.5074 --lon -0.1278 --radius 1500 \
  --out examples/my_place.svg
```

All README figures are rendered from the committed files by
`python examples/make_figures.py` (needs `matplotlib`).

Map data © OpenStreetMap contributors, available under the
[ODbL](https://www.openstreetmap.org/copyright).

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

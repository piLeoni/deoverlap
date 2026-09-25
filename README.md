# Deoverlap

De-overlap Shapely geometries that sit within a tolerance of each other.

The common case is pen-plotter work: two strokes closer than a pen width
visually merge on paper, so only one of them should keep the ink. Unlike
endpoint-only “deduplicate” tools, this library builds a **corridor** around
each kept stroke and crops (or drops) later strokes that fall inside it.

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

Original indices are preserved in `result.kept_parts` / `removed_parts` even
when processing order changes.

## Crop vs drop

- `mode="crop"` (default) — subtract the overlap, keep the rest.
- `mode="drop"` — if more than `drop_fraction` (default 0.5) of a geometry’s
  length is covered, discard the whole thing instead of leaving stubs.

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
used as clip masks.

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

The heavy lifting is GEOS (via Shapely). The Python loop avoids rebuilding the
spatial index on every insert (`tree_rebuild_every`) and periodically dissolves
the mask when bearings are not needed (`mask_union_every`). Do **not** run this
on stipple / dab layers — a dot grid is *meant* to sit closer than a pen width.

## vpype plugin

Installing `deoverlap[vpype]` registers a `deoverlap` command that is
**layer-safe** (honours `-l`) and uses corridor proximity rather than
endpoint-only matching:

```bash
vpype read map.svg deoverlap -t 0.1mm --keep longest -l 1,2,3 write out.svg
```

| Flag | Default | Meaning |
|---|---|---|
| `-t` / `--tolerance` | `0.1mm` | Corridor half-width |
| `--keep` | `longest` | `first` \| `longest` \| `shortest` |
| `--mode` | `crop` | `crop` \| `drop` |
| `--min-length` | `0` | Drop stubs shorter than this |
| `--parallel-only` | on | Do not punch perpendicular crossings |
| `--parallel-angle` | `30` | Bearing window in degrees |
| `-l` | `all` | Target layer(s) |
| `-k` | off | Keep removed pieces on a new layer |

Do **not** run it on stipple / dab layers (park dots): they are meant to sit
closer than a pen width.

### Self-overlap (`--segments`)

A thin road outline is often **one** LineString (left kerb → end cap → right
kerb). Without `--segments`, deoverlap never compares a geometry to itself, so
the two sides stay as a heavy double stroke. With `--segments`, every edge is
its own corridor unit; opposite sides can suppress each other, while immediate
neighbours on the same chain (`--segment-adjacency`, default 1) stay intact so
joints are not nibbled.

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
    tree_rebuild_every=32,
    mask_union_every=64,
) -> DeoverlapResult
```

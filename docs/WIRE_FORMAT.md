# Flat wire format (Node and `_core.deoverlap_flat`)

Language-neutral geometry batches for bindings that do not use Shapely.

## One geometry

| Buffer | Element type | Meaning |
|---|---|---|
| `coords` | `f64` | All vertices: `x0, y0, x1, y1, …` |
| `offsets` | `u32` | Start index of each part in **coords** (f64 slots), ending with `coords.length` |
| `kinds` | `u8` | `0` = polyline (≥2 vertices), `1` = point (1 vertex) |

Implementation and validation: `core/src/flat.rs`.

## Engine mask (multi-stage)

| Field | Meaning |
|---|---|
| `capsules` | `{ a: [x,y], b: [x,y], radius }` corridors from kept strokes |
| `polygons` | `{ exterior: f64[], interiors: f64[][] }` clip regions |

Python: pass `mask=previous_result` to reuse capsules without building Shapely
polygons. Node: pass `options.mask = previousResult.mask`.

The public Python `deoverlap()` API remains Shapely-based and is unchanged for
callers who only use `LineString` / `Polygon`.

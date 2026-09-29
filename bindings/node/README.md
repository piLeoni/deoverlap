# deoverlap

De-overlap vector strokes to prevent overdrawing.

Where strokes run on top of each other, deoverlap keeps one and cuts the
others back, so no area gets drawn twice: doubled roads on a map, duplicated
edges in generated art, outlines drawn twice by a pen plotter. Native Node.js
bindings for a Rust engine, prebuilt for Linux (x64, arm64), macOS (x64,
arm64) and Windows (x64).

![Before, removed, and after on a real map at pen width (OpenStreetMap)](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/osm_zoom.png)

A street map drawn with a 0.5 mm pen: before (left), what was cut away
(middle), after (right). Blue is kept, red is removed, orange is the corridor
mask — the same colours in every figure below.

The same engine ships as a Python library and a vpype command:
[github.com/piLeoni/deoverlap](https://github.com/piLeoni/deoverlap).

## Quick start

```bash
npm install deoverlap
```

```javascript
const { deoverlap } = require("deoverlap");

const line = (x0, y0, x1, y1) => ({ coords: [x0, y0, x1, y1], offsets: [0, 4], kinds: [0] });

const result = deoverlap(
  [line(0, 0, 2, 0), line(1, 0.05, 3, 0.05)], // parallel, 0.05 apart
  0.1 // tolerance
);

console.log(result.kept.length);          // 2: the second line is cropped
console.log(result.kept[1].geometry.coords); // starts after x ≈ 2.09
console.log(result.whollyRemoved);        // []
```

`tolerance` is a distance in the same units as the coordinates: strokes closer
than this to a kept stroke are cut. For a plotter, use the pen width.

Geometries are plain objects of flat number arrays (see
[Geometries](#geometries)); a [GeoJSON converter](#from-and-to-geojson) is a
few lines.

## How it works

Geometries are processed one at a time, in priority order. Each kept stroke
gets a corridor of radius `tolerance` around it, and every later stroke loses
the parts that fall inside a corridor. The options below decide which stroke
goes first (`prefer`), which nearby strokes count as overlapping (`angle`),
and what to do with the leftovers of a cut stroke (`drop`, `minLength`).

![Tangent circles and a line: the corridor mask (orange) around kept strokes (blue) crops the overlapping arcs (red)](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/hero.png)

## prefer — which stroke wins

The processing order decides which of two overlapping strokes is kept whole:

| `prefer` | Behaviour |
|---|---|
| `"longest"` (default) | The stroke that covers more ground wins |
| `"first"` | Input order |
| `"shortest"` | Short marks / detail win |

![The same three strokes under prefer longest, first and shortest](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/prefer.png)

Every result entry carries the input `index`, whatever the processing order.

## angle — which strokes count as overlapping

Two strokes overlap only where their directions differ by at most `angle`
degrees (default 30). Parallel runs and shallow merges are cropped; crossings
steeper than that are left alone. `angle: 90` counts every nearby stroke, so
crossings get cut too.

```javascript
deoverlap(geometries, 0.1, { angle: 90 }); // cut crossings too
```

Directions are compared **locally, edge by edge**, so a curving ramp is cropped
only where it actually runs alongside another road, whatever direction its two
ends point in.

![angle 90 cuts the crossings too; angle 30 only crops the parallel duplicate](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/angle.png)

## drop — crop or discard

By default the overlap is cut away and the rest of the stroke is kept. With
`drop: 0.5`, a stroke that would lose more than half its length is discarded
whole instead of leaving stubs; its index goes into `whollyRemoved`.

![Without drop the protruding stub is kept; drop 0.5 discards the mostly covered stroke](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/crop_vs_drop.png)

`minLength` drops pieces shorter than that after cutting (points are never
removed by it).

## Split pieces stay one object

A ring crossed by another stroke is cut into two arcs. They come back as
**one** multipart geometry under the ring's input index, not as two anonymous
pieces:

```javascript
const cutter = { coords: [1, -1, 1, 3], offsets: [0, 4], kinds: [0] };
const ring = { coords: [0, 0, 2, 0, 2, 2, 0, 2, 0, 0], offsets: [0, 10], kinds: [0] };

const { kept } = deoverlap([cutter, ring], 0.15, { prefer: "first", angle: 90 });
kept[1].index;            // 1
kept[1].geometry.offsets; // [0, 8, 16]: one geometry, two parts
```

One colour per object. On the left, what deoverlap returns: both halves of
the ring are one entry. On the right, the same pieces if every part were a
separate object:

![The cut ring is one object in result.kept (same colour), and two if every part is separate](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/groups.png)

## selfOverlap

A thin road outline is often **one** polyline (left kerb → end cap → right
kerb). Normally deoverlap never compares a geometry to itself, so the two sides
stay as a heavy double stroke. With `selfOverlap: true` every edge is its own
corridor, so opposite sides can suppress each other. Neighbouring edges never
cut each other, so joints are not nibbled, unless the path folds back on itself
like a hairpin.

![A thin ribbon drawn as one polyline: untouched normally, one side suppressed with selfOverlap](https://raw.githubusercontent.com/piLeoni/deoverlap/main/docs/img/self_overlap.png)

## Multi-stage with a carried mask

To process batches separately (for example, main roads first, then side
streets), pass the previous mask:

```javascript
const r1 = deoverlap(mainRoads, 0.1);
const r2 = deoverlap(sideStreets, 0.1, { mask: r1.mask });
```

Stage 2 is clipped against everything stage 1 kept.

## Options

```javascript
deoverlap(geometries, tolerance, {
  prefer: "longest",     // "longest" | "first" | "shortest"
  angle: 30,             // degrees, 0–90; 90 cuts crossings too
  selfOverlap: false,
  minLength: 0,
  drop: undefined,       // fraction 0–1; undefined always crops
  keepDuplicates: false, // fill removed / removedParts
  mask: undefined,       // result.mask of a previous run
});
```

All options are optional; the values above are the defaults.

## Result

| Field | Meaning |
|---|---|
| `kept` | `{ index, geometry }` per input that kept something, in processing order |
| `whollyRemoved` | Indices of inputs with nothing kept |
| `removed` | Cut pieces as separate geometries (with `keepDuplicates`) |
| `removedParts` | `{ index, geometry }` per input, what it lost; no `geometry` if nothing (with `keepDuplicates`) |
| `mask` | `{ capsules, polygons }` corridors, for a next stage |

## Geometries

Each geometry is a plain object of flat arrays:

| Field | Type | Meaning |
|---|---|---|
| `coords` | `number[]` | Every vertex: `[x0, y0, x1, y1, …]` |
| `offsets` | `number[]` | Start of each part in `coords`, plus a final entry `coords.length` |
| `kinds` | `number[]` | Per part: `0` = line, `1` = point |

A two-point line is `offsets: [0, 4]`; a geometry with two parts of four
vertices each is `offsets: [0, 8, 16]`. Full spec:
[WIRE_FORMAT.md](https://github.com/piLeoni/deoverlap/blob/main/docs/WIRE_FORMAT.md).

### From and to GeoJSON

```javascript
function fromGeoJSON(g) {
  const lines = {
    LineString: [g.coordinates],
    MultiLineString: g.coordinates,
    Polygon: g.coordinates,
    MultiPolygon: (g.coordinates || []).flat(),
  }[g.type];
  const points = { Point: [g.coordinates], MultiPoint: g.coordinates }[g.type];
  const coords = [], offsets = [0], kinds = [];
  for (const part of lines || points || []) {
    for (const [x, y] of lines ? part : [part]) coords.push(x, y);
    offsets.push(coords.length);
    kinds.push(lines ? 0 : 1);
  }
  return { coords, offsets, kinds };
}

function toGeoJSON({ coords, offsets, kinds }) {
  const parts = kinds.map((kind, i) => {
    const xy = [];
    for (let j = offsets[i]; j < offsets[i + 1]; j += 2) xy.push([coords[j], coords[j + 1]]);
    return kind === 1 ? { type: "Point", coordinates: xy[0] }
                      : { type: "LineString", coordinates: xy };
  });
  if (parts.length === 1) return parts[0];
  const type = parts[0].type;
  if (parts.some((p) => p.type !== type)) return { type: "GeometryCollection", geometries: parts };
  return { type: `Multi${type}`, coordinates: parts.map((p) => p.coordinates) };
}

const result = deoverlap(features.map((f) => fromGeoJSON(f.geometry)), 0.0001);
const cleaned = result.kept.map(({ index, geometry }) => ({
  ...features[index],
  geometry: toGeoJSON(geometry),
}));
```

Polygons are treated as their outlines, so they come back as lines.

## Performance

The engine builds no polygons: the corridor of a straight edge is a capsule,
and the part of another edge inside it is computed exactly. On the
[MacArthur Maze map](https://github.com/piLeoni/deoverlap#a-real-map-the-macarthur-maze)
(1473 paths) a run takes about 15 ms.

## Building from source

Needs a Rust toolchain.

```bash
git clone https://github.com/piLeoni/deoverlap
cd deoverlap/bindings/node
npm install
npm run build
npm test
```

Map data © OpenStreetMap contributors, available under the
[ODbL](https://www.openstreetmap.org/copyright).

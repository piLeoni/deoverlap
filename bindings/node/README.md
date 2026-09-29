# deoverlap (Node)

De-overlap vector strokes to prevent overdrawing. Where strokes run on top of
each other, one is kept and the others are cut back, so no area gets drawn
twice. Native bindings for the Rust engine of
[deoverlap](https://github.com/piLeoni/deoverlap), which also ships as a
Python library and a vpype command; see its README for figures and a full
explanation of the options.

```bash
npm install deoverlap
```

Prebuilt for Linux (x64, arm64, glibc), macOS (x64, arm64) and Windows (x64).

## Usage

```javascript
const { deoverlap } = require("deoverlap");

const result = deoverlap(
  [
    { coords: [0, 0, 2, 0], offsets: [0, 4], kinds: [0] },
    { coords: [1, 0.05, 3, 0.05], offsets: [0, 4], kinds: [0] },
  ],
  0.1, // tolerance: corridor radius, in the units of the coordinates
  {
    prefer: "longest",     // longest | first | shortest
    angle: 30,             // degrees, 0–90; 90 cuts crossings too
    selfOverlap: false,
    minLength: 0,
    drop: undefined,       // fraction 0–1; undefined always crops
    keepDuplicates: false,
    mask: undefined,       // result.mask of a previous run
  }
);
```

All options are optional; the values above are the defaults.

## Geometries

Each geometry is a plain object of flat buffers:

| Field | Type | Meaning |
|---|---|---|
| `coords` | `number[]` | `[x0, y0, x1, y1, …]` |
| `offsets` | `number[]` | Start of each part in `coords`, plus a final entry `coords.length` |
| `kinds` | `number[]` | Per part: `0` = line, `1` = point |

A multipart geometry has several parts; the two-point line above is
`offsets: [0, 4]`. Convert to and from your geometry library at the edges;
this package has no GeoJSON dependency. Full spec:
[`docs/WIRE_FORMAT.md`](https://github.com/piLeoni/deoverlap/blob/main/docs/WIRE_FORMAT.md).

## Result

| Field | Meaning |
|---|---|
| `kept` | `{ index, geometry }` per input that kept something; pieces of one input stay one geometry |
| `whollyRemoved` | Indices of inputs with nothing kept |
| `removed` | Cut pieces (with `keepDuplicates`) |
| `removedParts` | `{ index, geometry }` per input, what it lost; no `geometry` if nothing (with `keepDuplicates`) |
| `mask` | `{ capsules, polygons }` corridors, for a next stage |

To process batches separately, pass the previous mask:

```javascript
const r1 = deoverlap(batch1, 0.1);
const r2 = deoverlap(batch2, 0.1, { mask: r1.mask });
```

## Building from source

Needs a Rust toolchain.

```bash
git clone https://github.com/piLeoni/deoverlap
cd deoverlap/bindings/node
npm install
npm run build
npm test
```

## Publishing

CI builds one addon per platform (`bindings-<target>` artifacts). Each goes
into its own npm package (`npm/<platform>/`); the main `deoverlap` package
lists them as optional dependencies, so npm installs only the matching one.

```bash
cd bindings/node
npm ci && npm run build          # generates index.js / index.d.ts
gh run download <run-id> -p 'bindings-*' -D artifacts
npm run artifacts                # copies each .node into npm/<platform>/
npm publish                      # publishes npm/* first, then deoverlap
```

After a version bump, `npm run version` updates the `npm/*` packages.

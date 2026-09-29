# deoverlap (Node)

Native Node.js bindings for the Rust `deoverlap-core` engine. This path is
**separate from the Python wheel** (Shapely API); it does not slow down
`pip install deoverlap`.

## Wire format

Each geometry is a plain object:

| Field | Type | Meaning |
|---|---|---|
| `coords` | `number[]` | `[x0, y0, x1, y1, …]` |
| `offsets` | `number[]` | Start of each part in **f64 units**, plus a final entry `coords.length` |
| `kinds` | `number[]` | `0` = line, `1` = point |

Multi-stage runs reuse the engine mask (capsules + polygons), not rebuilt
buffers:

```javascript
const r1 = deoverlap(batch1, 0.1);
const r2 = deoverlap(batch2, 0.1, { mask: r1.mask });
```

## Build

```bash
cd bindings/node
npm install
npm run build
npm test   # node --test test/
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

## API

```javascript
const { deoverlap } = require("deoverlap");

const result = deoverlap(
  [
    { coords: [0, 0, 2, 0], offsets: [0, 4], kinds: [0] },
    { coords: [1, 0.05, 3, 0.05], offsets: [0, 4], kinds: [0] },
  ],
  0.1,
  {
    prefer: "first",       // longest | first | shortest
    angle: 90,             // 0–90
    selfOverlap: false,
    minLength: 0,
    drop: undefined,
    keepDuplicates: false,
    mask: undefined,       // { capsules, polygons } from a prior result
  }
);

// result.kept, result.whollyRemoved, result.mask, result.removed, result.removedParts
```

Convert to/from your JS geometry library at the edges; there is no GeoJSON
dependency in this package.

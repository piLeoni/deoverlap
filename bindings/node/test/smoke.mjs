import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const dir = dirname(fileURLToPath(import.meta.url));
const { deoverlap } = require(join(dir, "..", "index.js"));

test("deoverlap crops a parallel duplicate", () => {
  const result = deoverlap(
    [
      { coords: [0, 0, 2, 0], offsets: [0, 4], kinds: [0] },
      { coords: [1, 0.05, 3, 0.05], offsets: [0, 4], kinds: [0] },
    ],
    0.1,
    { prefer: "first", angle: 90 }
  );
  assert.equal(result.whollyRemoved.length, 0);
  assert.equal(result.kept.length, 2);
  assert.ok(result.mask.capsules.length > 0);
});

test("mask carries to a second stage", () => {
  const r1 = deoverlap([{ coords: [0, 0, 2, 0], offsets: [0, 4], kinds: [0] }], 0.1, {
    prefer: "first",
    angle: 90,
  });
  const r2 = deoverlap([{ coords: [1, 0.05, 3, 0.05], offsets: [0, 4], kinds: [0] }], 0.1, {
    prefer: "first",
    angle: 90,
    keepDuplicates: true,
    mask: r1.mask,
  });
  assert.ok(r2.removed.length > 0);
  assert.ok(r2.kept[0].geometry.coords[0] > 1.5);
});

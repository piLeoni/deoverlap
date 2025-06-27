# Deoverlap

A Python toolkit for resolving overlaps in 2D vector geometries, built on Shapely.

This module provides high-level functions for "de-overlapping" a set of Shapely geometric objects. When multiple geometries are stacked or clustered, it intelligently keeps one geometry while trimming away any subsequent geometries that fall within a specified tolerance.

## Installation

```bash
pip install deoverlap
```

## Quick Start

The primary function is `deoverlap()`. Provide it with a list of Shapely geometries and a tolerance.

```python
from shapely.geometry import LineString
from deoverlap import deoverlap

# Two lines that overlap within a 0.1 tolerance
lines = [
    LineString([(0, 0), (2, 0)]),
    LineString([(1, 0.05), (3, 0.05)])
]

# By default, you get a tuple of (kept_geometries, removed_geometries, mask)
kept, removed, mask = deoverlap(lines, tolerance=0.1, keep_duplicates=True)

print(f"Kept geometries: {len(kept)}")
# Output: Kept geometries: 2

# The first line is fully kept. The second is clipped where it overlaps.
print(kept[0])
# > LINESTRING (0 0, 2 0)
print(kept[1])
# > LINESTRING (2.05... 0.05, 3 0.05)
```

## Features

- **Two Modes:** A fast "flat" mode and a powerful "structured" mode that preserves geometry types.
- **Origin Tracking:** The structured mode can track which original geometry each removed piece came from.
- **Progress Bar:** Built-in `tqdm` integration for long-running tasks.
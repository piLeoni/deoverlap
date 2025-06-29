# Deoverlap

A high-level Python module for de-overlapping a set of Shapely geometric objects. It offers two main modes of operation: a fast "flat" mode that returns simple geometries, and a more powerful "structured" mode that preserves geometry types and can track the origin of removed pieces.

![](https://github.com/piLeoni/deoverlap/raw/main/test_outputs/test_tangent_circles_and_line_s_F_k_T.png)
## Motivation

When working with geospatial data, it's common to encounter datasets where geometries (such as lines representing roads or polygons representing land parcels) overlap one another. For many analysis and visualization tasks, it is desirable to remove these overlaps, ensuring that each point in space is covered by at most one geometry. This process, which we call "de-overlapping," can be complex to implement efficiently.

This module provides a robust and easy-to-use solution for de-overlapping Shapely geometries. It intelligently handles various geometry types and provides fine-grained control over the output, making it a valuable tool for data cleaning and preparation in geospatial workflows.

## Installation

You can install `deoverlap` directly from PyPI using pip:

```bash
pip install deoverlap
```

## How to Use

The core of this module is the `deoverlap` function. It takes an iterable of Shapely geometries and a tolerance value as input and returns the de-overlapped geometries.

### Basic Usage: Flat Mode

The default mode of operation is the "flat" mode (`preserve_types=False`). This mode is optimized for speed and returns a simple list of `LineString` and `Point` geometries. In this mode, any input `Polygon` objects are converted to their boundary `LineString`.

Here's a simple example of de-overlapping two overlapping `LineString`:

```python
from shapely.geometry import LineString
from deoverlap import deoverlap

geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
tolerance = 0.1

kept_geoms, removed_geoms, mask = deoverlap(geoms, tolerance)

print(f"Kept geometries: {len(kept_geoms)}")
print(f"Removed geometries: {len(removed_geoms)}")
```

## Advanced Usage

The `deoverlap` function offers several parameters to control its behavior:

- `preserve_types` (`bool`): When `True`, the function will attempt to preserve the original geometry types (e.g., `Polygon`, `MultiLineString`). This is more computationally intensive but provides a more structured output.
- `keep_duplicates` (`bool`): If `True`, the function will also return the portions of the geometries that were removed due to overlap.
- `track_origins` (`bool`): Only applicable when `preserve_types=True`. If `True`, the output is a dictionary that maps the removed parts to their original index in the input list. This is useful for understanding which original geometries were affected by the de-overlapping process.
- `mask` (`List[Polygon]`, optional): An optional, pre-existing list of polygon masks. If provided, geometries will be de-overlapped against this mask first. This is useful for iterative processing or for ensuring consistency across multiple, separate calls. The returned mask will include these initial polygons.

![](https://github.com/piLeoni/deoverlap/raw/main/test_outputs/test_tangent_circles_and_line_s_T_k_T.png)
### Preserving Geometry Types

Set `preserve_types=True` to maintain the original geometry types in the output. This is particularly useful when working with Polygons or when you need to maintain the distinction between single and multi-part geometries.

**Example:** De-overlapping a `LineString` that intersects another. With `preserve_types=True`, the intersected `LineString` is returned as a `MultiLineString`.

- `preserve_types=False` (default, flat output):
- `preserve_types=True` (structured output):

### Keeping Removed Portions

Set `keep_duplicates=True` to get a list of the geometries (or portions of geometries) that were removed.

**Example:** Visualizing the removed portion of an engulfed `LineString`.

- `keep_duplicates=False` (default):
- `keep_duplicates=True` (showing removed parts):

### Multi-Stage Processing with Mask

The `mask` parameter enables multi-stage processing by allowing you to pass pre-existing mask polygons. This is particularly useful when you need to process geometries in batches while maintaining consistency across all stages.

**Example:** Processing geometries in multiple stages:

```python
from shapely.geometry import LineString, Polygon
from deoverlap import deoverlap

# Stage 1: Process first batch
batch1 = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
tolerance = 0.1

kept1, _, removed1, mask1 = deoverlap(batch1, tolerance, keep_duplicates=True)

# Stage 2: Process second batch using the mask from stage 1
batch2 = [LineString([(2, 0.1), (4, 0.1)]), LineString([(3, 0.15), (5, 0.15)])]

kept2, _, removed2, final_mask = deoverlap(
    batch2, 
    tolerance, 
    keep_duplicates=True,
    mask=mask1  # Use the mask from the previous stage
)

print(f"Stage 1 kept: {len(kept1)}, removed: {len(removed1)}")
print(f"Stage 2 kept: {len(kept2)}, removed: {len(removed2)}")
print(f"Final mask contains {len(final_mask)} polygons")
```

This approach ensures that geometries in the second batch are de-overlapped against both their own batch and the geometries from the first batch, maintaining complete consistency across the entire processing pipeline.

### Tracking the Origin of Removed Parts

For detailed analysis of which input geometries were modified, use `track_origins=True` in conjunction with `preserve_types=True`. This returns a dictionary containing the kept geometries, a map of removed parts to their original indices, a list of indices of geometries that were wholly removed, and the overlap mask.

**Example:**

```python
from shapely.geometry import LineString
from deoverlap import deoverlap

geoms = [
    LineString([(0, 0), (4, 0)]),
    LineString([(1, 0.05), (2, 0.05)]),  # This line will be wholly removed
    LineString([(3, 0.05), (5, 0.05)])
]
tolerance = 0.1

results = deoverlap(
    geoms,
    tolerance,
    preserve_types=True,
    keep_duplicates=True,
    track_origins=True
)

print(f"Kept geometries: {len(results['kept'])}")
print(f"Removed parts map: {results['removed_parts']}")
print(f"Wholly removed indices: {results['wholly_removed_indices']}")
```

This will produce the following output, clearly indicating that the geometry at index 1 was completely removed:

```
Kept geometries: 2
Removed parts map: {0: [<MultiLineString object>], 1: [<LineString object>], 2: [<LineString object>]}
Wholly removed indices: [1]
```

## Working with Polygons and Mixed Geometries

The `deoverlap` function can handle a mix of geometry types, including Polygons. When `preserve_types` is `False`, polygons are treated as their boundary lines. When `preserve_types` is `True`, the function attempts to preserve the `Polygon` objects, clipping them as necessary.

**Example:** De-overlapping a set of tangent circles and a line.

- `preserve_types=False` (flat output):
- `preserve_types=True` (structured output):

## API Reference

### `deoverlap(geometries, tolerance, preserve_types=False, keep_duplicates=False, track_origins=False, progress_bar=False, mask=None)`

De-overlaps a list of geometries with extensive options for output format and origin tracking.

#### Parameters:

- `geometries` (`GeomInput`): An iterable of Shapely geometries.
- `tolerance` (`float`): The buffer distance to consider geometries as overlapping.
- `preserve_types` (`bool`, optional):
  - `False` (Default): Fast mode. Returns a flat list of simple `LineString`s and `Point`s.
  - `True`: Powerful mode. Returns structured geometries (e.g., `MultiLineString`). Slower.
- `keep_duplicates` (`bool`, optional): If `True`, the removed/overlapping portions are returned. Defaults to `False`.
- `track_origins` (`bool`, optional): Only applies when `preserve_types=True`. If `True`, returns a detailed dictionary mapping removed parts to their original index. Defaults to `False`.
- `progress_bar` (`bool`, optional): If `True`, displays a tqdm progress bar. Defaults to `False`.
- `mask` (`List[Polygon]`, optional): An optional, pre-existing list of polygon masks. If provided, geometries will be de-overlapped against this mask first. This is useful for iterative processing or for ensuring consistency across multiple, separate calls. The returned mask will include these initial polygons. Defaults to `None`.

#### Returns:

- The return type depends on the flags:
  - If `preserve_types=False` (flat mode):
    `(kept_geoms, kept_map, removed_geoms, mask)`
    - `kept_geoms`: List of kept (non-overlapping) geometries (LineString, Point)
    - `kept_map`: Always an empty dict in this mode
    - `removed_geoms`: List of removed (overlapping) geometries (if `keep_duplicates=True`, else empty list)
    - `mask`: List of mask polygons used for clipping
  - If `preserve_types=True` and `track_origins=False` (structured mode):
    `(kept_geoms, kept_map, removed_geoms, mask)`
    - `kept_geoms`: List of kept (non-overlapping) structured geometries
    - `kept_map`: Dict mapping input index to list of kept geometries
    - `removed_geoms`: List of removed (overlapping) geometries (if `keep_duplicates=True`, else empty list)
    - `mask`: List of mask polygons used for clipping
  - If `preserve_types=True` and `track_origins=True` (structured mode with origin tracking):
    A dictionary with keys:
    - `"kept"`: List of kept (non-overlapping) structured geometries
    - `"kept_parts"`: Dict mapping input index to list of kept geometries
    - `"removed_parts"`: Dict mapping input index to list of removed geometries (if `keep_duplicates=True`, else empty dict)
    - `"wholly_removed_indices"`: List of indices of geometries that were wholly removed
    - `"mask"`: List of mask polygons used for clipping


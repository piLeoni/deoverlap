"""De-overlap Shapely geometries that sit within a tolerance of each other.

The engine walks geometries in priority order. Each kept piece contributes a
buffered *mask*; later pieces are cropped (or dropped) where they fall inside
that mask. That is the right model for pen plotters: strokes closer than a pen
width visually merge, so only one of them should keep the ink.

``segments=True`` (self-overlap) splits every path into edge segments first, so
two sides of a thin road outline — one continuous LineString — can still
suppress each other. Adjacent segments on the same chain are excluded so joints
are not nibbled.

Pieces that came from the same input stay grouped (``MultiLineString`` etc.)
unless ``group=False``.

The engine itself is written in Rust (``deoverlap._core``); this module only
converts between Shapely geometries and coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, List, Optional, Sequence, Union

from shapely import get_coordinates
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from . import _core

# =============================================================================
#  Public types
# =============================================================================

GeomInput = Union[BaseGeometry, Iterable["GeomInput"]]
FlatGeom = Union[LineString, Point]


class KeepPolicy(str, Enum):
    """Which geometry wins when two corridors collide."""

    FIRST = "first"
    LONGEST = "longest"
    SHORTEST = "shortest"


class ClipMode(str, Enum):
    CROP = "crop"
    DROP = "drop"


@dataclass
class DeoverlapResult:
    kept: List[BaseGeometry] = field(default_factory=list)
    removed: List[BaseGeometry] = field(default_factory=list)
    kept_parts: dict[int, BaseGeometry] = field(default_factory=dict)
    removed_parts: dict[int, BaseGeometry] = field(default_factory=dict)
    wholly_removed: List[int] = field(default_factory=list)
    mask: List[Polygon] = field(default_factory=list)

    def __iter__(self) -> Iterator:
        kept_map = {i: [g] for i, g in self.kept_parts.items()}
        yield self.kept
        yield kept_map
        yield self.removed
        yield self.mask


# =============================================================================
#  Geometry helpers
# =============================================================================

_SNAP_FRACTION = 1e-6


def flatten_geometries(geoms: GeomInput) -> List[FlatGeom]:
    out: List[FlatGeom] = []
    if geoms is None:
        return out

    if isinstance(geoms, (LineString, Point)):
        if not geoms.is_empty:
            out.append(geoms)
    elif isinstance(geoms, (MultiLineString, MultiPoint, GeometryCollection)):
        for g in geoms.geoms:
            out.extend(flatten_geometries(g))
    elif isinstance(geoms, Polygon):
        if not geoms.is_empty:
            out.append(LineString(geoms.exterior.coords))
            for ring in geoms.interiors:
                out.append(LineString(ring.coords))
    elif isinstance(geoms, MultiPolygon):
        for poly in geoms.geoms:
            out.extend(flatten_geometries(poly))
    elif isinstance(geoms, (list, tuple, set)):
        for g in geoms:
            out.extend(flatten_geometries(g))
    elif isinstance(geoms, BaseGeometry):
        pass
    else:
        raise TypeError(f"Unsupported geometry type: {type(geoms)}")
    return out


def _as_list(geometries: GeomInput) -> List[BaseGeometry]:
    if isinstance(geometries, BaseGeometry):
        return [geometries]
    return [g for g in geometries if isinstance(g, BaseGeometry)]


def _length(geom: BaseGeometry) -> float:
    if geom is None or geom.is_empty or isinstance(geom, Point):
        return 0.0
    return float(geom.length)


def _to_coords(geom: Optional[BaseGeometry]) -> list[list[list[float]]]:
    if geom is None or geom.is_empty:
        return []
    return [get_coordinates(p).tolist() for p in flatten_geometries(geom)]


def _from_coords(parts: Sequence[Sequence[Sequence[float]]]) -> List[FlatGeom]:
    return [Point(p[0]) if len(p) == 1 else LineString(p) for p in parts]


def _grouped(
    parts: List[FlatGeom], original: Optional[BaseGeometry] = None, snap: float = 0.0
) -> BaseGeometry:
    """One geometry from its parts.

    An ``original`` polygon whose rings all survived is returned as is.
    """
    if isinstance(original, Polygon) and _length(original) - sum(_length(p) for p in parts) <= snap:
        return original
    if len(parts) == 1:
        return parts[0]
    if all(isinstance(p, LineString) for p in parts):
        return MultiLineString(parts)
    if all(isinstance(p, Point) for p in parts):
        return MultiPoint(parts)
    return GeometryCollection(parts)


# =============================================================================
#  Entry point
# =============================================================================


def deoverlap(
    geometries: GeomInput,
    tolerance: float,
    *,
    keep: Union[KeepPolicy, str] = KeepPolicy.FIRST,
    mode: Union[ClipMode, str] = ClipMode.CROP,
    min_length: float = 0.0,
    drop_fraction: float = 0.5,
    parallel_only: bool = False,
    parallel_angle: float = 30.0,
    segments: bool = False,
    segment_adjacency: int = 1,
    group: bool = True,
    keep_duplicates: bool = False,
    mask: Optional[Sequence[Polygon]] = None,
    preserve_types: Optional[bool] = None,
) -> DeoverlapResult:
    """De-overlap geometries that fall within ``tolerance`` of each other.

    Args:
        segments: If true, explode every path into edge segments and allow
            self-overlap — opposite sides of a thin outline can suppress each
            other. Adjacent segments on the same chain (within
            ``segment_adjacency``) are never clipped against each other.
        segment_adjacency: How many neighbouring segment indices on the same
            chain are exempt from clipping (default 1 = immediate neighbours,
            including wrap-around on closed rings).
    """
    if preserve_types is not None:
        group = bool(preserve_types)

    keep_policy = KeepPolicy(keep)
    clip_mode = ClipMode(mode)
    geoms = _as_list(geometries)

    mask_polys: list[Polygon] = []
    for m in mask or []:
        mask_polys.extend(m.geoms if isinstance(m, MultiPolygon) else [m])

    kept, removed_parts, removed, wholly, mask_out = _core.deoverlap(
        [_to_coords(g) for g in geoms],
        tolerance,
        keep=keep_policy.value,
        mode=clip_mode.value,
        min_length=min_length,
        drop_fraction=drop_fraction,
        parallel_only=parallel_only,
        parallel_angle=parallel_angle,
        segments=segments,
        segment_adjacency=segment_adjacency,
        keep_duplicates=keep_duplicates,
        mask=[
            (get_coordinates(p.exterior).tolist(), [get_coordinates(r).tolist() for r in p.interiors])
            for p in mask_polys
            if not p.is_empty
        ],
    )

    snap = tolerance * _SNAP_FRACTION
    result = DeoverlapResult(wholly_removed=list(wholly))
    for i, parts in kept:
        grouped = _grouped(_from_coords(parts), geoms[i], snap)
        result.kept_parts[i] = grouped
        if group:
            result.kept.append(grouped)
        else:
            result.kept.extend(flatten_geometries(grouped))

    gone = set(wholly)
    for i, parts in enumerate(removed_parts):
        if parts is not None:
            result.removed_parts[i] = geoms[i] if i in gone else _grouped(_from_coords(parts))
    result.removed = _from_coords(removed)
    result.mask = [Polygon(ext, ints) for ext, ints in mask_out]
    return result

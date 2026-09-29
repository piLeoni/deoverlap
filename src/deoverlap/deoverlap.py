"""De-overlap Shapely geometries that sit within a tolerance of each other.

The engine walks geometries in priority order. Each kept piece contributes a
corridor of radius ``tolerance`` to a *mask*; later pieces are cropped (or
dropped) where they fall inside that mask. That is the right model for pen plotters: strokes closer than a pen
width visually merge, so only one of them should keep the ink.

``self_overlap=True`` splits every path into edges first, so the two sides of a
thin road outline — one continuous LineString — can still suppress each other.
Neighbouring edges are excluded so joints are not nibbled.

Pieces that came from the same input stay grouped (``MultiLineString`` etc.).

The engine itself is written in Rust (``deoverlap._core``); this module only
converts between Shapely geometries and coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

# =============================================================================
#  Public types
# =============================================================================

GeomInput = Union[BaseGeometry, Iterable["GeomInput"]]
FlatGeom = Union[LineString, Point]
Prefer = Literal["longest", "first", "shortest"]
Capsule = Tuple[Tuple[float, float], Tuple[float, float], float]
MaskInput = Union[Sequence[Polygon], "DeoverlapResult"]


@dataclass
class DeoverlapResult:
    kept: List[BaseGeometry] = field(default_factory=list)
    removed: List[BaseGeometry] = field(default_factory=list)
    kept_parts: dict[int, BaseGeometry] = field(default_factory=dict)
    removed_parts: dict[int, BaseGeometry] = field(default_factory=dict)
    wholly_removed: List[int] = field(default_factory=list)
    _mask_polygons: List[Polygon] = field(default_factory=list, repr=False)
    _mask_capsules: List[Capsule] = field(default_factory=list, repr=False)
    _mask_shapely: Optional[List[Polygon]] = field(default=None, repr=False)

    @property
    def mask(self) -> List[Polygon]:
        """Corridor polygons of everything kept, plus the mask passed in.

        Built on first access from the engine's capsules; pass ``mask=r1`` or
        ``mask=r1.mask`` to a later run.
        """
        if self._mask_shapely is None:
            self._mask_shapely = list(self._mask_polygons) + [
                _capsule_polygon(c) for c in self._mask_capsules
            ]
        return self._mask_shapely


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


def _capsule_polygon(capsule: Capsule) -> Polygon:
    (ax, ay), (bx, by), radius = capsule
    if ax == bx and ay == by:
        g = Point(ax, ay).buffer(radius)
    else:
        g = LineString([(ax, ay), (bx, by)]).buffer(radius, cap_style="round", join_style="round")
    if isinstance(g, Polygon):
        return g
    return max(g.geoms, key=lambda p: p.area)


def _resolve_mask(
    mask: Optional[MaskInput],
) -> tuple[list[Polygon], list[Capsule], list[tuple[list[list[float]], list[list[list[float]]]]]]:
    """Polygons for Shapely callers, capsules for the engine, polygon rings for ``_core``."""
    if mask is None:
        return [], [], []
    if isinstance(mask, DeoverlapResult):
        polys = list(mask._mask_polygons)
        caps = list(mask._mask_capsules)
        rings = [
            (get_coordinates(p.exterior).tolist(), [get_coordinates(r).tolist() for r in p.interiors])
            for p in polys
            if not p.is_empty
        ]
        return polys, caps, rings
    polys: list[Polygon] = []
    for m in mask:
        polys.extend(m.geoms if isinstance(m, MultiPolygon) else [m])
    rings = [
        (get_coordinates(p.exterior).tolist(), [get_coordinates(r).tolist() for r in p.interiors])
        for p in polys
        if not p.is_empty
    ]
    return polys, [], rings


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
    prefer: Prefer = "longest",
    angle: float = 30.0,
    self_overlap: bool = False,
    min_length: float = 0.0,
    drop: Optional[float] = None,
    keep_duplicates: bool = False,
    progress_bar: bool = False,
    mask: Optional[MaskInput] = None,
) -> DeoverlapResult:
    """De-overlap geometries that fall within ``tolerance`` of each other.

    Args:
        tolerance: Corridor radius: strokes closer than this to a kept
            stroke are cut.
        prefer: Which stroke wins a collision: ``"longest"``, ``"first"``
            (input order) or ``"shortest"``.
        angle: Strokes overlap only where their local bearings differ by at
            most this many degrees (0–90); 90 cuts crossings too.
        self_overlap: Let a path overlap itself, e.g. the two sides of a thin
            outline. Neighbouring edges never cut each other unless the path
            folds back on itself.
        min_length: Drop surviving pieces shorter than this.
        drop: Discard a whole stroke when more than this fraction (0–1) of
            its length would be cut; ``None`` always crops.
        keep_duplicates: Collect the removed pieces in ``removed`` and
            ``removed_parts``.
        progress_bar: Show a tqdm progress bar.
        mask: Corridors from a previous run (``result`` or ``result.mask``)
            that also cut this one.
    """
    geoms = _as_list(geometries)
    _, mask_capsules, mask_rings = _resolve_mask(mask)

    bar = None
    if progress_bar and tqdm is not None:
        desc = "De-overlapping edges" if self_overlap else "De-overlapping"
        bar = tqdm(desc=desc, total=len(geoms))

    def on_progress(done: int, total: int) -> None:
        bar.total = total
        bar.update(done - bar.n)

    try:
        kept, removed_parts, removed, wholly, out_mask = _core.deoverlap(
            [_to_coords(g) for g in geoms],
            tolerance,
            prefer=prefer,
            angle=angle,
            self_overlap=self_overlap,
            min_length=min_length,
            drop=drop,
            keep_duplicates=keep_duplicates,
            mask=mask_rings,
            mask_capsules=mask_capsules,
            progress=on_progress if bar is not None else None,
        )
    finally:
        if bar is not None:
            bar.close()

    snap = tolerance * _SNAP_FRACTION
    out_caps, out_polys = out_mask
    result = DeoverlapResult(
        wholly_removed=list(wholly),
        _mask_polygons=[
            Polygon(exterior, holes=interiors or None)
            for exterior, interiors in out_polys
            if len(exterior) >= 3
        ],
        _mask_capsules=list(out_caps),
    )
    for i, parts in kept:
        grouped = _grouped(_from_coords(parts), geoms[i], snap)
        result.kept_parts[i] = grouped
        result.kept.append(grouped)

    gone = set(wholly)
    for i, parts in enumerate(removed_parts):
        if parts is not None:
            result.removed_parts[i] = geoms[i] if i in gone else _grouped(_from_coords(parts))
    result.removed = _from_coords(removed)
    return result

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, List, Optional, Sequence, Union

from shapely import union_all
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
from shapely.ops import unary_union
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

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


@dataclass(frozen=True)
class _SegId:
    """Identity of one exploded edge, for self-overlap adjacency checks."""

    chain: int
    index: int
    count: int
    closed: bool


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

_ROBUSTNESS_BUFFER = 1e-9


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
    if geom is None or geom.is_empty:
        return 0.0
    if isinstance(geom, Point):
        return 0.0
    try:
        return float(geom.length)
    except Exception:
        return 0.0


def _dedupe_coords(coords: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for c in coords:
        pt = (float(c[0]), float(c[1]))
        if not out or out[-1] != pt:
            out.append(pt)
    return out


def _line_angle(geom: BaseGeometry) -> Optional[float]:
    if isinstance(geom, (Polygon, MultiPolygon, Point, MultiPoint)):
        return None
    if isinstance(geom, LineString) and len(geom.coords) >= 2:
        x0, y0 = geom.coords[0]
        x1, y1 = geom.coords[-1]
        return math.atan2(y1 - y0, x1 - x0) % math.pi
    parts = flatten_geometries(geom)
    line = max(
        (p for p in parts if isinstance(p, LineString) and len(p.coords) >= 2),
        key=lambda p: p.length,
        default=None,
    )
    if line is None:
        return None
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]
    return math.atan2(y1 - y0, x1 - x0) % math.pi


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _seg_adjacent(a: _SegId, b: _SegId, window: int) -> bool:
    """True if ``a`` and ``b`` are neighbours on the same exploded chain."""
    if a.chain != b.chain or window < 0:
        return False
    n = a.count
    d = abs(a.index - b.index)
    if d <= window:
        return True
    if a.closed and n > 2 and (n - d) <= window:
        return True
    return False


def _reassemble(parts: Sequence[BaseGeometry], original: BaseGeometry) -> BaseGeometry:
    clean = [p for p in parts if p is not None and not p.is_empty]
    if not clean:
        return LineString()
    merged = unary_union(clean)
    if (
        original.geom_type == "Polygon"
        and not merged.is_empty
        and merged.equals(original.boundary)
    ):
        return original
    return merged


def _filter_min_length(geom: BaseGeometry, min_length: float) -> BaseGeometry:
    if min_length <= 0 or geom.is_empty:
        return geom
    if isinstance(geom, Point):
        return geom
    if isinstance(geom, LineString):
        return geom if geom.length >= min_length else LineString()
    if isinstance(geom, MultiLineString):
        kept = [g for g in geom.geoms if g.length >= min_length]
        if not kept:
            return MultiLineString()
        if len(kept) == 1:
            return kept[0]
        return MultiLineString(kept)
    if isinstance(geom, GeometryCollection):
        kept = []
        for g in geom.geoms:
            f = _filter_min_length(g, min_length)
            if not f.is_empty:
                kept.append(f)
        return unary_union(kept) if kept else GeometryCollection()
    return geom


def _split_removed(original: BaseGeometry, kept: BaseGeometry) -> BaseGeometry:
    source = original.boundary if isinstance(original, (Polygon, MultiPolygon)) else original
    if source.is_empty:
        return source
    if kept.is_empty:
        return source
    return source.difference(kept)


def _explode_segments(
    geoms: Sequence[BaseGeometry],
) -> tuple[list[LineString], list[int], list[_SegId], list[float]]:
    """Split every lineal geometry into 2-point edges.

    Returns segments, original-index per segment, segment ids, and the parent
    geometry length (for keep-policy sorting).
    """
    segments: list[LineString] = []
    origins: list[int] = []
    seg_ids: list[_SegId] = []
    parent_lengths: list[float] = []
    chain = 0

    for origin, geom in enumerate(geoms):
        parent_len = _length(geom)
        for part in flatten_geometries(geom):
            if isinstance(part, Point) or len(part.coords) < 2:
                continue
            coords = _dedupe_coords(part.coords)
            if len(coords) < 2:
                continue
            closed = len(coords) > 2 and coords[0] == coords[-1]
            if closed:
                coords = coords[:-1]
            if len(coords) < 2:
                continue
            if closed:
                count = len(coords)
                for i in range(count):
                    a, b = coords[i], coords[(i + 1) % count]
                    if a == b:
                        continue
                    segments.append(LineString([a, b]))
                    origins.append(origin)
                    seg_ids.append(_SegId(chain, i, count, True))
                    parent_lengths.append(parent_len)
            else:
                count = len(coords) - 1
                for i in range(count):
                    a, b = coords[i], coords[i + 1]
                    if a == b:
                        continue
                    segments.append(LineString([a, b]))
                    origins.append(origin)
                    seg_ids.append(_SegId(chain, i, count, False))
                    parent_lengths.append(parent_len)
            chain += 1

    return segments, origins, seg_ids, parent_lengths


# =============================================================================
#  Mask index
# =============================================================================


class _MaskIndex:
    """Corridor polygons with a lazily rebuilt STRtree."""

    def __init__(
        self,
        initial: Optional[Sequence[Polygon]] = None,
        *,
        rebuild_every: int = 32,
        union_every: int = 64,
        parallel_only: bool = False,
        segment_adjacency: int = 1,
    ) -> None:
        self.polys: List[BaseGeometry] = list(initial) if initial else []
        self.angles: List[Optional[float]] = [None] * len(self.polys)
        self.seg_ids: List[Optional[_SegId]] = [None] * len(self.polys)
        self.parallel_only = parallel_only
        self.segment_adjacency = max(0, segment_adjacency)
        self.rebuild_every = max(1, rebuild_every)
        self.union_every = max(1, union_every)
        self._since_rebuild = 0
        self._since_union = 0
        self._tree: Optional[STRtree] = None
        # Identity tracking prevents mask dissolve (would lose angle / seg ids).
        self._track_identity = parallel_only or segment_adjacency >= 0
        if self.polys:
            self._tree = STRtree(self.polys)

    def add(
        self,
        geom: BaseGeometry,
        tolerance: float,
        angle: Optional[float],
        seg_id: Optional[_SegId] = None,
    ) -> None:
        if geom.is_empty:
            return
        buf = geom.buffer(tolerance)
        if buf.is_empty:
            return
        self.polys.append(buf)
        self.angles.append(angle if self.parallel_only else None)
        self.seg_ids.append(seg_id)
        self._since_rebuild += 1
        self._since_union += 1
        self._tree = None
        # Only dissolve when we are not tracking per-corridor identity.
        if (
            not self.parallel_only
            and seg_id is None
            and self._since_union >= self.union_every
        ):
            self._consolidate()

    def _consolidate(self) -> None:
        if len(self.polys) <= 1:
            self._since_union = 0
            return
        merged = union_all(self.polys)
        if merged.is_empty:
            self.polys, self.angles, self.seg_ids = [], [], []
        else:
            self.polys = [merged]
            self.angles = [None]
            self.seg_ids = [None]
        self._since_union = 0
        self._since_rebuild = 0
        self._tree = STRtree(self.polys) if self.polys else None

    def _ensure_tree(self) -> Optional[STRtree]:
        if not self.polys:
            return None
        if self._tree is None or self._since_rebuild >= self.rebuild_every:
            self._tree = STRtree(self.polys)
            self._since_rebuild = 0
        return self._tree

    def local_mask(
        self,
        geom: BaseGeometry,
        angle: Optional[float],
        angle_tol_rad: float,
        seg_id: Optional[_SegId] = None,
    ) -> Optional[BaseGeometry]:
        tree = self._ensure_tree()
        if tree is None:
            return None
        nearby = tree.query(geom)
        if getattr(nearby, "size", len(nearby)) == 0:
            return None
        selected: list[BaseGeometry] = []
        for i in (int(j) for j in nearby):
            other_seg = self.seg_ids[i]
            if (
                seg_id is not None
                and other_seg is not None
                and _seg_adjacent(seg_id, other_seg, self.segment_adjacency)
            ):
                continue
            if self.parallel_only and angle is not None:
                other_ang = self.angles[i]
                if other_ang is not None and _angle_diff(angle, other_ang) > angle_tol_rad:
                    continue
            selected.append(self.polys[i])
        if not selected:
            return None
        return union_all(selected)

    def as_list(self) -> List[Polygon]:
        out: List[Polygon] = []
        for p in self.polys:
            if p.is_empty:
                continue
            if isinstance(p, Polygon):
                out.append(p)
            elif isinstance(p, MultiPolygon):
                out.extend(list(p.geoms))
        return out


# =============================================================================
#  Core engine
# =============================================================================


def _priority_order(scores: Sequence[float], keep: KeepPolicy) -> List[int]:
    idxs = list(range(len(scores)))
    if keep is KeepPolicy.FIRST:
        return idxs
    if keep is KeepPolicy.LONGEST:
        return sorted(idxs, key=lambda i: scores[i], reverse=True)
    if keep is KeepPolicy.SHORTEST:
        return sorted(idxs, key=lambda i: scores[i])
    raise ValueError(f"unknown keep policy: {keep!r}")


def _clip_one(
    part: BaseGeometry,
    mask: _MaskIndex,
    *,
    angle: Optional[float],
    angle_tol_rad: float,
    seg_id: Optional[_SegId],
) -> BaseGeometry:
    local = mask.local_mask(part, angle, angle_tol_rad, seg_id=seg_id)
    if local is None:
        return part
    clipped = part.difference(local.buffer(_ROBUSTNESS_BUFFER))
    return clipped if not clipped.is_empty else part.__class__()


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
    progress_bar: bool = False,
    mask: Optional[Sequence[Polygon]] = None,
    tree_rebuild_every: int = 32,
    mask_union_every: int = 64,
    preserve_types: Optional[bool] = None,
    track_origins: bool = False,
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

    keep_policy = KeepPolicy(keep) if not isinstance(keep, KeepPolicy) else keep
    clip_mode = ClipMode(mode) if not isinstance(mode, ClipMode) else mode
    angle_tol_rad = math.radians(parallel_angle)

    geoms = _as_list(geometries)
    index = _MaskIndex(
        mask,
        rebuild_every=tree_rebuild_every,
        union_every=mask_union_every,
        parallel_only=parallel_only,
        segment_adjacency=segment_adjacency if segments else -1,
    )

    result = DeoverlapResult()
    _ = track_origins

    if segments:
        return _deoverlap_segments(
            geoms,
            tolerance,
            keep_policy=keep_policy,
            clip_mode=clip_mode,
            min_length=min_length,
            drop_fraction=drop_fraction,
            parallel_only=parallel_only,
            angle_tol_rad=angle_tol_rad,
            segment_adjacency=segment_adjacency,
            group=group,
            keep_duplicates=keep_duplicates,
            progress_bar=progress_bar,
            index=index,
        )

    order = _priority_order([_length(g) for g in geoms], keep_policy)
    iterable: Iterable[int] = order
    if progress_bar and tqdm is not None:
        iterable = tqdm(order, desc="De-overlapping", total=len(order))

    for i in iterable:
        geom = geoms[i]
        if geom is None or geom.is_empty:
            continue

        parts = flatten_geometries(geom)
        if not parts:
            continue

        angle = _line_angle(geom) if parallel_only else None
        kept_sub = [
            k
            for part in parts
            if not (
                k := _clip_one(
                    part, index, angle=angle, angle_tol_rad=angle_tol_rad, seg_id=None
                )
            ).is_empty
        ]

        if not kept_sub:
            result.wholly_removed.append(i)
            if keep_duplicates:
                result.removed_parts[i] = geom
                result.removed.append(geom)
            continue

        reassembled = _filter_min_length(_reassemble(kept_sub, geom), min_length)
        if reassembled.is_empty:
            result.wholly_removed.append(i)
            if keep_duplicates:
                result.removed_parts[i] = geom
                result.removed.append(geom)
            continue

        if clip_mode is ClipMode.DROP and _length(geom) > 0:
            if _length(reassembled) / _length(geom) < (1.0 - drop_fraction):
                result.wholly_removed.append(i)
                if keep_duplicates:
                    result.removed_parts[i] = geom
                    result.removed.append(geom)
                continue

        removed_portion = _split_removed(geom, reassembled)
        if keep_duplicates and not removed_portion.is_empty:
            result.removed_parts[i] = removed_portion
            result.removed.extend(flatten_geometries(removed_portion))

        result.kept_parts[i] = reassembled
        if group:
            result.kept.append(reassembled)
        else:
            result.kept.extend(flatten_geometries(reassembled))

        index.add(reassembled, tolerance, angle)

    result.mask = index.as_list()
    return result


def _deoverlap_segments(
    geoms: Sequence[BaseGeometry],
    tolerance: float,
    *,
    keep_policy: KeepPolicy,
    clip_mode: ClipMode,
    min_length: float,
    drop_fraction: float,
    parallel_only: bool,
    angle_tol_rad: float,
    segment_adjacency: int,
    group: bool,
    keep_duplicates: bool,
    progress_bar: bool,
    index: _MaskIndex,
) -> DeoverlapResult:
    """Self-overlap path: work on exploded edges, then regroup by origin."""
    segments, origins, seg_ids, parent_lengths = _explode_segments(geoms)
    result = DeoverlapResult()
    if not segments:
        result.mask = index.as_list()
        return result

    index.segment_adjacency = max(0, segment_adjacency)
    order = _priority_order(parent_lengths, keep_policy)

    kept_by_origin: dict[int, list[BaseGeometry]] = {i: [] for i in range(len(geoms))}
    saw_origin = set(origins)

    iterable: Iterable[int] = order
    if progress_bar and tqdm is not None:
        iterable = tqdm(order, desc="De-overlapping segments", total=len(order))

    for si in iterable:
        seg = segments[si]
        origin = origins[si]
        sid = seg_ids[si]
        angle = _line_angle(seg) if parallel_only else None

        kept = _clip_one(
            seg, index, angle=angle, angle_tol_rad=angle_tol_rad, seg_id=sid
        )
        if kept.is_empty:
            if keep_duplicates:
                result.removed.append(seg)
            continue

        if clip_mode is ClipMode.DROP and _length(seg) > 0:
            if _length(kept) / _length(seg) < (1.0 - drop_fraction):
                if keep_duplicates:
                    result.removed.append(seg)
                continue

        kept = _filter_min_length(kept, min_length)
        if kept.is_empty:
            if keep_duplicates:
                result.removed.append(seg)
            continue

        if keep_duplicates and _length(kept) + 1e-12 < _length(seg):
            removed = seg.difference(kept)
            if not removed.is_empty:
                result.removed.extend(flatten_geometries(removed))

        kept_by_origin[origin].append(kept)
        index.add(kept, tolerance, angle, seg_id=sid)

    for origin, pieces in kept_by_origin.items():
        if origin not in saw_origin:
            continue
        if not pieces:
            result.wholly_removed.append(origin)
            if keep_duplicates:
                result.removed_parts[origin] = geoms[origin]
            continue
        reassembled = _filter_min_length(_reassemble(pieces, geoms[origin]), min_length)
        if reassembled.is_empty:
            result.wholly_removed.append(origin)
            if keep_duplicates:
                result.removed_parts[origin] = geoms[origin]
            continue
        result.kept_parts[origin] = reassembled
        if group:
            result.kept.append(reassembled)
        else:
            result.kept.extend(flatten_geometries(reassembled))

        if keep_duplicates:
            removed = _split_removed(geoms[origin], reassembled)
            if not removed.is_empty:
                result.removed_parts[origin] = removed

    result.mask = index.as_list()
    return result

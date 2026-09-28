"""Unit tests for deoverlap 3.x — behaviour, not screenshots.

README figures are rendered separately by ``examples/make_figures.py`` so a
normal pytest run stays fast and dependency-light.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from deoverlap import (
    ClipMode,
    DeoverlapResult,
    KeepPolicy,
    deoverlap,
    flatten_geometries,
)


def test_simple_line_overlap_crops_second():
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
    result = deoverlap(geoms, 0.1, keep_duplicates=True)
    assert isinstance(result, DeoverlapResult)
    assert len(result.kept) == 2
    assert result.kept[0].equals(geoms[0])
    assert result.kept[1].length < geoms[1].length
    assert len(result.removed) > 0


def test_segments_self_overlap_thins_a_narrow_ribbon():
    """Opposite sides of one thin outline suppress each other (self-overlap).

    A long thin rectangle boundary is a *single* LineString; without segments
    mode deoverlap cannot touch it. With segments=True the two long sides are
    separate edges and the later one is cropped/dropped.
    """
    # 0.08 wide ribbon — sides are 0.08 apart.
    ring = LineString(
        [(0, 0), (10, 0), (10, 0.08), (0, 0.08), (0, 0)]
    )
    plain = deoverlap([ring], 0.1, parallel_only=True)
    assert 0 in plain.kept_parts
    # Full perimeter roughly 2*10 + 2*0.08
    assert plain.kept_parts[0].length == pytest.approx(ring.length, rel=0.05)

    selfed = deoverlap(
        [ring],
        0.1,
        segments=True,
        parallel_only=True,
        keep=KeepPolicy.FIRST,
        min_length=0.05,
    )
    assert 0 in selfed.kept_parts
    # One long side (~10) should dominate; the opposite side is suppressed.
    assert selfed.kept_parts[0].length < ring.length * 0.7


def test_segment_adjacency_preserves_joints():
    """Immediate neighbours on a chain are not clipped against each other."""
    # Open polyline with a sharp corner — adjacent edges share a vertex and
    # their buffers overlap, but both must survive.
    elbow = LineString([(0, 0), (2, 0), (2, 2)])
    result = deoverlap([elbow], 0.3, segments=True, parallel_only=False)
    assert 0 in result.kept_parts
    assert result.kept_parts[0].length == pytest.approx(elbow.length, abs=0.05)


def test_fully_engulfed_line_is_removed():
    geoms = [LineString([(0, 0), (3, 0)]), LineString([(1, 0), (2, 0)])]
    result = deoverlap(geoms, 0.2, keep_duplicates=True)
    assert len(result.kept) == 1
    assert result.wholly_removed == [1]
    assert 1 in result.removed_parts


def test_longest_keep_policy_prefers_longer_stroke():
    short = LineString([(0, 0), (1, 0)])
    long = LineString([(0.05, 0), (3, 0)])  # nearly on top of short
    # Input order would keep short first and carve long. LONGEST reverses that.
    result = deoverlap([short, long], 0.1, keep=KeepPolicy.LONGEST)
    assert len(result.kept) == 1 or (
        len(result.kept) == 2 and result.kept_parts[1].length > result.kept_parts.get(0, LineString()).length
    )
    # The long stroke (index 1) must survive in full or nearly so.
    assert 1 in result.kept_parts
    assert result.kept_parts[1].length == pytest.approx(long.length, rel=0.05)
    # The short one is wholly removed or reduced to nothing meaningful.
    assert 0 in result.wholly_removed or 0 not in result.kept_parts


def test_group_keeps_split_ring_as_one_multipart():
    """A closed ring cut by a corridor stays one grouped geometry."""
    ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    cutter = LineString([(1, -1), (1, 3)])  # crosses the ring twice
    result = deoverlap([cutter, ring], 0.15, group=True, keep=KeepPolicy.FIRST)
    assert 1 in result.kept_parts
    grouped = result.kept_parts[1]
    # Cropped ring should be multipart (two arcs) but a single result entry.
    assert isinstance(grouped, (LineString, MultiLineString))
    assert sum(1 for g in result.kept if g is grouped) == 1
    flat = flatten_geometries(grouped)
    assert len(flat) >= 2  # at least two arcs

    flat_result = deoverlap([cutter, ring], 0.15, group=False, keep=KeepPolicy.FIRST)
    # Ungrouped: each arc is its own entry in kept.
    assert len(flat_result.kept) >= 2


def test_cut_ring_joins_across_its_start_point():
    """The arc through the ring's first/last vertex is one piece, not two."""
    ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    cutter = LineString([(1, -1), (1, 3)])
    grouped = deoverlap([cutter, ring], 0.15, group=True, keep=KeepPolicy.FIRST)
    assert len(flatten_geometries(grouped.kept_parts[1])) == 2
    flat = deoverlap([cutter, ring], 0.15, group=False, keep=KeepPolicy.FIRST)
    assert len(flat.kept) == 3  # cutter + two arcs


def test_segments_removes_fold_back_between_neighbours():
    """A hairpin folds onto itself; neighbouring edges must still suppress."""
    spike = LineString([(0, 0), (6, 0), (6.5, 0.4), (6, 0.08), (0, 0.08), (0, 0)])
    result = deoverlap(
        [spike], 0.1, segments=True, parallel_only=True, keep="first",
        min_length=0.05, keep_duplicates=True,
    )
    kept = result.kept_parts[0]
    back = LineString([(6.5, 0.4), (6, 0.08)])
    assert kept.intersection(back.buffer(0.01)).length < 0.1


def test_segments_keeps_ordinary_corners():
    """A 90 degree corner between neighbours is not a fold-back."""
    square = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    result = deoverlap([square], 0.3, segments=True, parallel_only=True)
    assert result.kept_parts[0].length == pytest.approx(square.length, abs=0.05)


def test_parallel_only_preserves_crossing():
    horizontal = LineString([(0, 0), (4, 0)])
    vertical = LineString([(2, -2), (2, 2)])
    # Without parallel_only the vertical line loses a chunk at the cross.
    cropped = deoverlap([horizontal, vertical], 0.3, parallel_only=False)
    crossing = deoverlap(
        [horizontal, vertical],
        0.3,
        parallel_only=True,
        parallel_angle=30,
    )
    assert 1 in cropped.kept_parts and 1 in crossing.kept_parts
    assert crossing.kept_parts[1].length > cropped.kept_parts[1].length
    assert crossing.kept_parts[1].length == pytest.approx(vertical.length, abs=0.05)


def test_parallel_only_uses_local_bearing():
    """A U whose chord is horizontal still has a vertical arm beside the line."""
    wall = LineString([(0, 0), (0, 10)])
    u_turn = LineString([(0.05, 9), (0.05, 1), (3, 1), (3, 9)])
    result = deoverlap([wall, u_turn], 0.1, parallel_only=True, keep="first")
    kept = result.kept_parts[1]
    # The left arm (8 long) is inside the wall's corridor and runs parallel.
    assert kept.length == pytest.approx(u_turn.length - 8, abs=0.3)
    # The rest stays one continuous stroke.
    assert kept.geom_type == "LineString"


def test_min_length_drops_stubs():
    a = LineString([(0, 0), (2, 0)])
    # Runs almost on top of a, but sticks out by 0.05 on the right.
    b = LineString([(0.5, 0.02), (2.05, 0.02)])
    result = deoverlap([a, b], 0.1, min_length=0.1, keep_duplicates=True)
    # The leftover stub (~0.05) should be discarded.
    if 1 in result.kept_parts:
        assert result.kept_parts[1].length >= 0.1
    else:
        assert 1 in result.wholly_removed


def test_drop_mode_discards_mostly_covered_line():
    a = LineString([(0, 0), (3, 0)])
    # Mostly inside a's corridor, but sticks out past the end so crop keeps a stub.
    b = LineString([(0.5, 0.02), (3.5, 0.02)])
    cropped = deoverlap([a, b], 0.1, mode=ClipMode.CROP)
    dropped = deoverlap([a, b], 0.1, mode=ClipMode.DROP, drop_fraction=0.5)
    assert 1 in cropped.kept_parts  # crop keeps the protruding stub
    assert cropped.kept_parts[1].length < b.length
    assert 1 in dropped.wholly_removed


def test_mask_carries_across_stages():
    batch1 = [LineString([(0, 0), (2, 0)])]
    r1 = deoverlap(batch1, 0.1)
    batch2 = [LineString([(1, 0.05), (3, 0.05)])]
    r2 = deoverlap(batch2, 0.1, mask=r1.mask, keep_duplicates=True)
    assert r2.kept[0].length < batch2[0].length
    assert len(r2.removed) > 0


@pytest.mark.parametrize("segments", [False, True])
def test_kept_plus_removed_conserves_length(segments):
    """Removed pieces must not double-count ink that was actually kept."""
    import math

    geoms = []
    for k in range(12):
        a = k * 0.37
        pts = [
            (t * 0.731 + 0.05 * math.sin(t * 1.3 + a), 0.11 * k + 0.043 * math.cos(t + a))
            for t in range(9)
        ]
        geoms.append(LineString(pts))
    result = deoverlap(
        geoms, 0.1, keep="longest", parallel_only=True, segments=segments,
        keep_duplicates=True,
    )
    total_in = sum(g.length for g in geoms)
    kept = sum(g.length for g in result.kept)
    removed = sum(g.length for g in result.removed)
    assert kept < total_in
    assert kept + removed == pytest.approx(total_in, rel=1e-4)


@pytest.mark.parametrize("segments", [False, True])
def test_progress_reaches_the_total(segments):
    from deoverlap import _core

    calls = []
    geoms = [[[[0.0, 0.1 * k], [5.0, 0.1 * k]]] for k in range(300)]
    _core.deoverlap(geoms, 0.1, segments=segments, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (300, 300)
    assert len(calls) <= 102
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)

    plain = deoverlap([LineString(g[0]) for g in geoms], 0.1, segments=segments)
    with_bar = deoverlap([LineString(g[0]) for g in geoms], 0.1, segments=segments, progress_bar=True)
    assert [g.wkt for g in with_bar.kept] == [g.wkt for g in plain.kept]


def test_progress_callback_errors_propagate():
    from deoverlap import _core

    def boom(done, total):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _core.deoverlap([[[[0.0, 0.0], [1.0, 0.0]]]], 0.1, progress=boom)


def test_empty_input():
    result = deoverlap([], 0.1)
    assert result.kept == []
    assert result.removed == []
    assert result.wholly_removed == []


def test_legacy_tuple_unpack():
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0), (3, 0)])]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.1, keep_duplicates=True)
    assert len(kept) >= 1
    assert isinstance(kept_map, dict)
    assert isinstance(mask, list)


def test_polygon_ring_preserved_when_untouched():
    poly = Point(0, 0).buffer(1.0)
    line = LineString([(5, 0), (6, 0)])  # far away
    result = deoverlap([poly, line], 0.1, group=True)
    assert 0 in result.kept_parts
    assert result.kept_parts[0].geom_type == "Polygon"


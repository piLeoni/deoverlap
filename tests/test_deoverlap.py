"""Unit tests for deoverlap 4.x — behaviour, not screenshots.

README figures are rendered separately by ``examples/make_figures.py`` so a
normal pytest run stays fast and dependency-light.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, MultiLineString, Point

from deoverlap import DeoverlapResult, deoverlap, flatten_geometries


def plain(geoms, tolerance, **kwargs):
    """Earlier strokes win and bearings are ignored, unless a test says otherwise."""
    kwargs = {"prefer": "first", "angle": 90, **kwargs}
    return deoverlap(geoms, tolerance, **kwargs)


def test_simple_line_overlap_crops_second():
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
    result = plain(geoms, 0.1, keep_duplicates=True)
    assert isinstance(result, DeoverlapResult)
    assert len(result.kept) == 2
    assert result.kept[0].equals(geoms[0])
    assert result.kept[1].length < geoms[1].length
    assert len(result.removed) > 0


def test_self_overlap_thins_a_narrow_ribbon():
    """Opposite sides of one thin outline suppress each other.

    A long thin rectangle boundary is a *single* LineString; without
    self_overlap deoverlap cannot touch it.
    """
    ring = LineString([(0, 0), (10, 0), (10, 0.08), (0, 0.08), (0, 0)])
    untouched = plain([ring], 0.1, angle=30)
    assert untouched.kept_parts[0].length == pytest.approx(ring.length, rel=0.05)

    thinned = plain([ring], 0.1, angle=30, self_overlap=True, min_length=0.05)
    assert thinned.kept_parts[0].length < ring.length * 0.7


def test_self_overlap_preserves_joints():
    """Neighbouring edges of a path are not cut against each other."""
    elbow = LineString([(0, 0), (2, 0), (2, 2)])
    result = plain([elbow], 0.3, self_overlap=True)
    assert result.kept_parts[0].length == pytest.approx(elbow.length, abs=0.05)


def test_self_overlap_removes_fold_back_between_neighbours():
    """A hairpin folds onto itself; neighbouring edges must still suppress."""
    spike = LineString([(0, 0), (6, 0), (6.5, 0.4), (6, 0.08), (0, 0.08), (0, 0)])
    result = plain([spike], 0.1, angle=30, self_overlap=True, min_length=0.05)
    back = LineString([(6.5, 0.4), (6, 0.08)])
    assert result.kept_parts[0].intersection(back.buffer(0.01)).length < 0.1


def test_self_overlap_keeps_ordinary_corners():
    """A 90 degree corner between neighbours is not a fold-back."""
    square = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    for angle in (30, 90):
        result = plain([square], 0.3, angle=angle, self_overlap=True)
        assert result.kept_parts[0].length == pytest.approx(square.length, abs=0.05)


def test_fully_engulfed_line_is_removed():
    geoms = [LineString([(0, 0), (3, 0)]), LineString([(1, 0), (2, 0)])]
    result = plain(geoms, 0.2, keep_duplicates=True)
    assert len(result.kept) == 1
    assert result.wholly_removed == [1]
    assert 1 in result.removed_parts


def test_prefer_longest_keeps_the_longer_stroke():
    short = LineString([(0, 0), (1, 0)])
    long = LineString([(0.05, 0), (3, 0)])  # nearly on top of short
    result = deoverlap([short, long], 0.1)  # longest is the default
    assert result.kept_parts[1].length == pytest.approx(long.length, rel=0.05)
    assert 0 not in result.kept_parts

    first = deoverlap([short, long], 0.1, prefer="first")
    assert first.kept_parts[0].equals(short)


def test_unknown_prefer_is_rejected():
    with pytest.raises(ValueError, match="prefer"):
        deoverlap([LineString([(0, 0), (1, 0)])], 0.1, prefer="biggest")


def test_split_ring_stays_one_geometry():
    """A closed ring cut by a corridor stays one entry in ``kept``."""
    ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    cutter = LineString([(1, -1), (1, 3)])  # crosses the ring twice
    result = plain([cutter, ring], 0.15)
    assert len(result.kept) == 2
    assert isinstance(result.kept_parts[1], MultiLineString)


def test_cut_ring_joins_across_its_start_point():
    """The arc through the ring's first/last vertex is one piece, not two."""
    ring = LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    cutter = LineString([(1, -1), (1, 3)])
    result = plain([cutter, ring], 0.15)
    assert len(flatten_geometries(result.kept_parts[1])) == 2


def test_narrow_angle_preserves_crossing():
    horizontal = LineString([(0, 0), (4, 0)])
    vertical = LineString([(2, -2), (2, 2)])
    cropped = plain([horizontal, vertical], 0.3)
    crossing = plain([horizontal, vertical], 0.3, angle=30)
    assert crossing.kept_parts[1].length > cropped.kept_parts[1].length
    assert crossing.kept_parts[1].length == pytest.approx(vertical.length, abs=0.05)


def test_angle_uses_local_bearing():
    """A U whose chord is horizontal still has a vertical arm beside the line."""
    wall = LineString([(0, 0), (0, 10)])
    u_turn = LineString([(0.05, 9), (0.05, 1), (3, 1), (3, 9)])
    result = plain([wall, u_turn], 0.1, angle=30)
    kept = result.kept_parts[1]
    # The left arm (8 long) is inside the wall's corridor and runs parallel.
    assert kept.length == pytest.approx(u_turn.length - 8, abs=0.3)
    # The rest stays one continuous stroke.
    assert kept.geom_type == "LineString"


def test_invalid_angle_is_rejected():
    with pytest.raises(ValueError, match="angle"):
        deoverlap([LineString([(0, 0), (1, 0)])], 0.1, angle=120)


def test_min_length_drops_stubs():
    a = LineString([(0, 0), (2, 0)])
    # Runs almost on top of a, but sticks out by 0.05 on the right.
    b = LineString([(0.5, 0.02), (2.05, 0.02)])
    result = plain([a, b], 0.1, min_length=0.1)
    if 1 in result.kept_parts:
        assert result.kept_parts[1].length >= 0.1
    else:
        assert 1 in result.wholly_removed


def test_drop_discards_mostly_covered_line():
    a = LineString([(0, 0), (3, 0)])
    # Mostly inside a's corridor, but sticks out past the end so crop keeps a stub.
    b = LineString([(0.5, 0.02), (3.5, 0.02)])
    cropped = plain([a, b], 0.1)
    dropped = plain([a, b], 0.1, drop=0.5)
    assert cropped.kept_parts[1].length < b.length
    assert 1 in dropped.wholly_removed


def test_invalid_drop_is_rejected():
    with pytest.raises(ValueError, match="drop"):
        deoverlap([LineString([(0, 0), (1, 0)])], 0.1, drop=1.5)


def test_mask_carries_across_stages():
    r1 = plain([LineString([(0, 0), (2, 0)])], 0.1)
    batch2 = [LineString([(1, 0.05), (3, 0.05)])]
    r2 = plain(batch2, 0.1, mask=r1.mask, keep_duplicates=True)
    assert r2.kept[0].length < batch2[0].length
    assert len(r2.removed) > 0


@pytest.mark.parametrize("self_overlap", [False, True])
def test_kept_plus_removed_conserves_length(self_overlap):
    """Removed pieces must not double-count ink that was actually kept."""
    geoms = []
    for k in range(12):
        a = k * 0.37
        pts = [
            (t * 0.731 + 0.05 * math.sin(t * 1.3 + a), 0.11 * k + 0.043 * math.cos(t + a))
            for t in range(9)
        ]
        geoms.append(LineString(pts))
    result = deoverlap(geoms, 0.1, self_overlap=self_overlap, keep_duplicates=True)
    total_in = sum(g.length for g in geoms)
    kept = sum(g.length for g in result.kept)
    removed = sum(g.length for g in result.removed)
    assert kept < total_in
    assert kept + removed == pytest.approx(total_in, rel=1e-4)


@pytest.mark.parametrize("self_overlap", [False, True])
def test_progress_reaches_the_total(self_overlap):
    from deoverlap import _core

    calls = []
    coords = [[[[0.0, 0.1 * k], [5.0, 0.1 * k]]] for k in range(300)]
    _core.deoverlap(coords, 0.1, self_overlap=self_overlap, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (300, 300)
    assert len(calls) <= 102
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)

    geoms = [LineString(c[0]) for c in coords]
    silent = deoverlap(geoms, 0.1, self_overlap=self_overlap)
    with_bar = deoverlap(geoms, 0.1, self_overlap=self_overlap, progress_bar=True)
    assert [g.wkt for g in with_bar.kept] == [g.wkt for g in silent.kept]


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


def test_polygon_ring_preserved_when_untouched():
    poly = Point(0, 0).buffer(1.0)
    line = LineString([(5, 0), (6, 0)])  # far away
    result = deoverlap([poly, line], 0.1)
    assert result.kept_parts[0].geom_type == "Polygon"

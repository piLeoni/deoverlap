import os
import inspect
import pytest
from typing import List, Union, Iterable
import matplotlib.pyplot as plt
from shapely import union_all
from shapely.geometry import (LineString, Point, MultiLineString,
                             Polygon, MultiPolygon, GeometryCollection)
from shapely.geometry.base import BaseGeometry

from deoverlap import deoverlap, flatten_geometries

GeomInput = Union[BaseGeometry, Iterable['GeomInput']]

# =============================================================================
#  Test Visualization Helper
# =============================================================================
def plot_results(input_geoms: GeomInput, kept: list, removed: list, mask: list, filename: str = None, title: str = None, folder: str = 'test_outputs'):
    """
    Generates and saves a visual plot of the deoverlap operation results.
    """
    if filename is None:
        test_name = inspect.stack()[1].function
        try:
            # Get the calling frame's local variables
            frame = inspect.stack()[1][0]
            params = frame.f_locals
            
            # Build a clear parameter string showing the boolean values
            param_parts = []
            if 'preserve_types' in params:
                preserve_val = "Structured" if params['preserve_types'] else "Flat"
                param_parts.append(f"Preserve={preserve_val}")
            if 'keep_duplicates' in params:
                keep_val = "Keep" if params['keep_duplicates'] else "Discard"
                param_parts.append(f"Duplicates={keep_val}")
            if 'track_origins' in params:
                track_val = "Track" if params['track_origins'] else "NoTrack"
                param_parts.append(f"Origins={track_val}")
            
            # Create filename with clear parameter indicators
            if param_parts:
                filename = f"{test_name}_{'_'.join(param_parts)}"
            else:
                filename = test_name
        except (KeyError, IndexError): 
            filename = test_name
    
    if title is None:
        # Create a more descriptive title that clearly shows the parameters
        test_name = inspect.stack()[1].function.replace('_', ' ').title()
        try:
            frame = inspect.stack()[1][0]
            params = frame.f_locals
            
            title_parts = [test_name]
            if 'preserve_types' in params:
                preserve_val = "Structured Mode" if params['preserve_types'] else "Flat Mode"
                title_parts.append(f"({preserve_val}")
            if 'keep_duplicates' in params:
                keep_val = "Keep Duplicates" if params['keep_duplicates'] else "Discard Duplicates"
                title_parts.append(f", {keep_val}")
            if 'track_origins' in params:
                track_val = "Track Origins" if params['track_origins'] else "No Origin Tracking"
                title_parts.append(f", {track_val}")
            
            if len(title_parts) > 1:
                title_parts[-1] = title_parts[-1] + ")"
                title = " ".join(title_parts)
            else:
                title = test_name
        except (KeyError, IndexError):
            title = test_name
    
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8)); labels_used = set()
    def add_to_legend(label):
        if label not in labels_used: labels_used.add(label); return label
        return ""
    if flat_inputs := flatten_geometries(input_geoms):
        all_bounds = [g.bounds for g in flat_inputs if not g.is_empty]
        if all_bounds:
            min_x, min_y, max_x, max_y = (min(b[0] for b in all_bounds), min(b[1] for b in all_bounds), max(b[2] for b in all_bounds), max(b[3] for b in all_bounds))
            width, height = max_x - min_x, max_y - min_y
            max_dim = max(width, height) if max(width, height) > 0 else 1
            padding = max_dim * 0.1; center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            half_size = (max_dim / 2) + padding
            ax.set_xlim(center_x - half_size, center_x + half_size); ax.set_ylim(center_y - half_size, center_y + half_size)
    if mask and not (full_mask := union_all(mask)).is_empty:
        for poly in getattr(full_mask, 'geoms', [full_mask]):
            if isinstance(poly, Polygon):
                ax.plot(*poly.exterior.xy, color='#F29727', linewidth=0.5, label=add_to_legend('Mask Outline'), zorder=1)
                for interior in poly.interiors: ax.plot(*interior.xy, color='#F29727', linewidth=0.5, zorder=1)
    for g in flatten_geometries(input_geoms):
        if g.geom_type == 'LineString': ax.plot(*g.xy, color='#D0E9F2', linewidth=5, alpha=0.6, zorder=2, solid_capstyle='round')
        elif g.geom_type == 'Point': ax.plot(g.x, g.y, 'o', color='#D0E9F2', markersize=15, alpha=0.6, zorder=2)
    for g in flatten_geometries(removed):
        if g.geom_type == 'LineString': ax.plot(*g.xy, color='#F2622E', linewidth=2.5, label=add_to_legend('Removed'), zorder=3, solid_capstyle='round')
        elif g.geom_type == 'Point': ax.plot(g.x, g.y, 'x', color='#F2622E', markersize=10, mew=2.5, label=add_to_legend('Removed'), zorder=4)
    for g in flatten_geometries(kept):
        if g.geom_type == 'LineString': ax.plot(*g.xy, color='#73A2BF', linewidth=2.5, label=add_to_legend('Kept'), zorder=4, solid_capstyle='round')
        elif g.geom_type == 'Point': ax.plot(g.x, g.y, 'o', color='#73A2BF', markersize=8, label=add_to_legend('Kept'), zorder=5)
    if title: plt.title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    if labels_used: ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=150); plt.close(fig)

# =============================================================================
#  Comprehensive Test Suite
# =============================================================================

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_simple_line_overlap(preserve_types, keep_duplicates):
    """Tests basic clipping of one line by another's buffer."""
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
    kept, removed, _ = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types:
        assert len(kept) == 2 and isinstance(kept[1], (LineString, MultiLineString))
    else: assert len(kept) == 2
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_line_fully_engulfed(preserve_types, keep_duplicates):
    """Tests a geometry being completely removed."""
    geoms = [LineString([(0, 0), (3, 0)]), LineString([(1, 0), (2, 0)])]
    kept, removed, _ = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    assert len(kept) == 1
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_complex_intersection(preserve_types, keep_duplicates):
    """Tests a 'plus-sign' intersection where a line is split in the middle."""
    geoms = [LineString([(0, 1), (3, 1)]), LineString([(1.5, 0), (1.5, 2)])]
    kept, removed, _ = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types:
        assert len(kept) == 2 and isinstance(kept[1], MultiLineString)
    else: assert len(kept) == 3 # flat mode produces 3 distinct LineStrings
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_tangent_circles_and_line(preserve_types, keep_duplicates):
    """Tests a complex scenario with tangent polygons and a crossing line."""
    geoms = [ Point(0, 0).buffer(1.5), Point(2.0, 0).buffer(0.5), Point(0, 2.5).buffer(1.0), LineString([(-2, 1), (3, 1)]) ]
    kept, removed, _ = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types:
        assert len(kept) == 4 and isinstance(kept[0], Polygon)
    else: assert len(kept) > 4 # flat mode shatters the result
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_high_volume_points(preserve_types, keep_duplicates):
    """Tests a large grid of points against a clipping polygon."""
    points = [Point(x*0.2, y*0.2) for x in range(10) for y in range(10)]
    remover_poly = Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
    geoms = [remover_poly] + points
    kept, removed, _ = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    kept_points = [g for g in flatten_geometries(kept) if isinstance(g, Point)]
    removed_points = [g for g in flatten_geometries(removed) if isinstance(g, Point)]
    if keep_duplicates: assert len(kept_points) + len(removed_points) == len(points)
    else: assert len(removed) == 0 and len(kept_points) < len(points)

def test_origin_tracking_feature():
    """Tests the special 'track_origins=True' dictionary output."""
    geoms = [ LineString([(0, 0), (4, 0)]), LineString([(1, 0.05), (2, 0.05)]), LineString([(3, 0.05), (5, 0.05)]) ]
    # Call with track_origins=True to get the dictionary.
    results = deoverlap(geoms, 0.1, preserve_types=True, keep_duplicates=True, track_origins=True)
    plot_results(geoms, results["kept"], list(results["removed_parts"].values()), results["mask"])
    assert isinstance(results, dict)
    assert len(results["kept"]) == 2
    assert results["wholly_removed_indices"] == [1]
    assert 1 in results["removed_parts"] and 2 in results["removed_parts"]

@pytest.mark.parametrize("preserve_types", [True, False])
def test_empty_input(preserve_types):
    """Tests that all modes handle empty list input gracefully."""
    output = deoverlap([], 0.1, preserve_types=preserve_types, track_origins=True)
    if preserve_types:
        assert isinstance(output, dict) and not output["kept"] and not output["removed_parts"]
    else:
        assert isinstance(output, tuple) and not output[0] and not output[1]
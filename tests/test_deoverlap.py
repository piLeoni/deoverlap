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
    Generates and saves a plot visualizing the results of the deoverlap process.
    The plot title includes details about the parameters used in the test.
    """
    # --- Generate a unique filename based on test parameters ---
    if filename is None:
        test_name = "unknown_test"
        try:
            stack_frame = inspect.stack()[1]
            test_name = stack_frame.function
            params = stack_frame[0].f_locals
            param_str_list = []
            if 'preserve_types' in params: param_str_list.append(f"s_{str(params['preserve_types'])[0]}")
            if 'keep_duplicates' in params: param_str_list.append(f"k_{str(params['keep_duplicates'])[0]}")
            if 'track_origins' in params: param_str_list.append(f"o_{str(params['track_origins'])[0]}")
            filename = f"{test_name}_{'_'.join(param_str_list)}" if param_str_list else test_name
        except (KeyError, IndexError):
            filename = test_name

    # --- Generate a clean, multi-line title for the plot ---
    final_title_str = ""
    if title:
        final_title_str = title
    else:
        try:
            # Get the clean test name for the main title line
            test_name = inspect.stack()[1].function
            main_title_line = test_name.replace('_', ' ').capitalize()

            # Get parameters for the subtitle lines
            params = inspect.stack()[1][0].f_locals
            param_lines = []
            if 'preserve_types' in params:
                param_lines.append(f"preserve_types: {params.get('preserve_types')}")
            if 'keep_duplicates' in params:
                param_lines.append(f"keep_duplicates: {params.get('keep_duplicates')}")
            if 'track_origins' in params:
                param_lines.append(f"track_origins: {params.get('track_origins', False)}")

            # Combine main title and parameter lines into a single string
            final_title_str = main_title_line
            if param_lines:
                final_title_str += "\n\n" + "\n".join(param_lines)
        except (KeyError, IndexError):
            # Fallback title if inspection fails
            final_title_str = filename.replace('_', ' ').capitalize()

    # --- Plotting setup ---
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    labels_used = set()

    def add_to_legend(label):
        if label not in labels_used:
            labels_used.add(label)
            return label
        return ""

    # --- Set plot bounds ---
    if flat_inputs := flatten_geometries(input_geoms):
        all_bounds = [g.bounds for g in flat_inputs if not g.is_empty]
        if all_bounds:
            min_x, min_y, max_x, max_y = (min(b[0] for b in all_bounds), min(b[1] for b in all_bounds),
                                          max(b[2] for b in all_bounds), max(b[3] for b in all_bounds))
            width, height = max_x - min_x, max_y - min_y
            max_dim = max(width, height) if max(width, height) > 0 else 1
            padding = max_dim * 0.1
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            half_size = (max_dim / 2) + padding
            ax.set_xlim(center_x - half_size, center_x + half_size)
            ax.set_ylim(center_y - half_size, center_y + half_size)

    # --- Draw geometries ---
    if mask and not (full_mask := union_all(mask)).is_empty:
        for poly in getattr(full_mask, 'geoms', [full_mask]):
            if isinstance(poly, Polygon):
                ax.plot(*poly.exterior.xy, color='#F29727', linewidth=0.5, label=add_to_legend('Mask Outline'), zorder=1)
                for interior in poly.interiors:
                    ax.plot(*interior.xy, color='#F29727', linewidth=0.5, zorder=1)

    for g in flatten_geometries(input_geoms):
        if g.geom_type == 'LineString':
            ax.plot(*g.xy, color='#D0E9F2', linewidth=5, alpha=0.6, zorder=2, solid_capstyle='round')
        elif g.geom_type == 'Point':
            ax.plot(g.x, g.y, 'o', color='#D0E9F2', markersize=15, alpha=0.6, zorder=2)

    for g in flatten_geometries(removed):
        if g.geom_type == 'LineString':
            ax.plot(*g.xy, color='#F2622E', linewidth=2.5, label=add_to_legend('Removed'), zorder=3, solid_capstyle='round')
        elif g.geom_type == 'Point':
            ax.plot(g.x, g.y, 'x', color='#F2622E', markersize=10, mew=2.5, label=add_to_legend('Removed'), zorder=4)

    for g in flatten_geometries(kept):
        if g.geom_type == 'LineString':
            ax.plot(*g.xy, color='#73A2BF', linewidth=2.5, label=add_to_legend('Kept'), zorder=4, alpha = 0.75, solid_capstyle='round')
        elif g.geom_type == 'Point':
            ax.plot(g.x, g.y, 'o', color='#73A2BF', markersize=8, label=add_to_legend('Kept'), zorder=5)

    # --- Finalize plot ---
    if final_title_str:
        ax.set_title(final_title_str, fontsize=10, pad=10, loc='center', fontdict={'linespacing': 1.4})

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if labels_used:
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=150)
    plt.close(fig)

# =============================================================================
#  Comprehensive Test Suite
# =============================================================================

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_simple_line_overlap(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
    kept, removed, _ = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types: assert len(kept) == 2 and isinstance(kept[1], (LineString, MultiLineString))
    else: assert len(kept) == 2
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_line_fully_engulfed(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 0), (3, 0)]), LineString([(1, 0), (2, 0)])]
    kept, removed, _ = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    assert len(kept) == 1
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_complex_intersection(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 1), (3, 1)]), LineString([(1.5, 0), (1.5, 2)])]
    kept, removed, _ = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types: assert len(kept) == 2 and isinstance(kept[1], MultiLineString)
    else: assert len(kept) == 3
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_tangent_circles_and_line(preserve_types, keep_duplicates):
    geoms = [ Point(0, 0).buffer(1.5), Point(2.0, 0).buffer(0.5), Point(0, 2.5).buffer(1.0), LineString([(-2, 1), (3, 1)]) ]
    kept, removed, _ = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept, removed, _)
    if preserve_types: assert len(kept) == 4 and isinstance(kept[0], Polygon)
    else: assert len(kept) > 4
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_high_volume_points(preserve_types, keep_duplicates):
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
    geoms = [ LineString([(0, 0), (4, 0)]), LineString([(1, 0.05), (2, 0.05)]), LineString([(3, 0.05), (5, 0.05)]) ]
    results = deoverlap(geoms, 0.1, preserve_types=True, keep_duplicates=True, track_origins=True)
    plot_results(geoms, results["kept"], list(results["removed_parts"].values()), results["mask"])
    assert isinstance(results, dict)
    assert len(results["kept"]) == 2
    assert results["wholly_removed_indices"] == [1]

@pytest.mark.parametrize("preserve_types", [True, False])
def test_empty_input(preserve_types):
    output = deoverlap([], 0.1, preserve_types=preserve_types, track_origins=True)
    if preserve_types:
        assert isinstance(output, dict) and not output["kept"] and not output["removed_parts"]
    else:
        assert isinstance(output, tuple) and not output[0] and not output[1]
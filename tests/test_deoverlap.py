import os
import inspect
import pytest
from typing import Union, Iterable
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from shapely import union_all
from shapely.geometry import (LineString, Point, MultiLineString,
                             Polygon, MultiPolygon, GeometryCollection)
from shapely.geometry.base import BaseGeometry

from deoverlap import deoverlap, flatten_geometries

GeomInput = Union[BaseGeometry, Iterable['GeomInput']]

# =============================================================================
#  Global Color Configuration
# =============================================================================

# Colors for the default "flat" mode plots
FLAT_MODE_COLORS = {
    "kept": "#0A8AB2",
    "removed": "#FF0249",
    "mask": "#F26835",
    "background": "#F2F2F2"
}

# Default color palette for "structured" (preserve_types=True) mode plots.
# This list will be cycled through for each original geometry.
STRUCTURED_COLOR_PALETTE = [
    '#2DA6CC', '#0A8AB2', '#52b69a', '#FFF029', '#C0D99A',
    '#1A1F4A', '#72588C', '#D98299', '#7CA687', '#314022'
]


# =============================================================================
#  Test Visualization Helper (Using Global Colors)
# =============================================================================
def plot_results(input_geoms: GeomInput, kept_items: dict, removed_items: Union[list, dict], mask: list, filename: str = None, title: str = None, folder: str = 'test_outputs', color_palette: list = None):
    """
    Generates and saves a plot visualizing the results of the deoverlap process.
    Uses global color variables for styling.
    """
    # --- Filename and Title Setup (unchanged) ---
    if filename is None:
        test_name = "unknown_test"
        try:
            stack_frame = inspect.stack()[1]
            test_name = stack_frame.function
            if color_palette:
                test_name += "_custom_palette"
            params = stack_frame[0].f_locals
            param_str_list = []
            if 'preserve_types' in params: param_str_list.append(f"s_{str(params['preserve_types'])[0]}")
            if 'keep_duplicates' in params: param_str_list.append(f"k_{str(params['keep_duplicates'])[0]}")
            if 'track_origins' in params: param_str_list.append(f"o_{str(params['track_origins'])[0]}")
            filename = f"{test_name}_{'_'.join(param_str_list)}" if param_str_list else test_name
        except (KeyError, IndexError):
            filename = test_name
    if title is None:
        try:
            test_name = inspect.stack()[1].function
            main_title_line = test_name.replace('_', ' ').capitalize()
            params = inspect.stack()[1][0].f_locals
            param_lines = []
            if 'preserve_types' in params: param_lines.append(f"preserve_types: {params.get('preserve_types')}")
            if 'keep_duplicates' in params: param_lines.append(f"keep_duplicates: {params.get('keep_duplicates')}")
            if 'track_origins' in params: param_lines.append(f"track_origins: {params.get('track_origins', False)}")
            title = main_title_line
            if param_lines:
                title += "\n\n" + "\n".join(param_lines)
        except (KeyError, IndexError):
            title = filename.replace('_', ' ').capitalize()

    # --- Plotting Setup ---
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    labels_used = set()
    params = inspect.stack()[1][0].f_locals
    input_geoms_list = list(input_geoms)
    LEGEND_THRESHOLD = 20
    enable_legend = len(input_geoms_list) <= LEGEND_THRESHOLD
    def add_to_legend(label):
        if enable_legend and label not in labels_used:
            labels_used.add(label)
            return label
        return ""

    # --- Set plot bounds (unchanged) ---
    if flat_inputs := flatten_geometries(input_geoms_list):
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

    # --- Draw geometries using global colors ---
    active_palette = color_palette if color_palette is not None else STRUCTURED_COLOR_PALETTE
    color_cycler = cycle(active_palette)
    structured_colors = {i: next(color_cycler) for i in range(len(input_geoms_list))}

    if mask and not (full_mask := union_all(mask)).is_empty:
        for poly in getattr(full_mask, 'geoms', [full_mask]):
            if isinstance(poly, Polygon):
                ax.plot(*poly.exterior.xy, color=FLAT_MODE_COLORS["mask"], linewidth=0.5, label=add_to_legend('Mask Outline'), zorder=1)
                for interior in poly.interiors:
                    ax.plot(*interior.xy, color=FLAT_MODE_COLORS["mask"], linewidth=0.5, zorder=1)

    for g in flatten_geometries(input_geoms_list):
        if g.geom_type == 'LineString':
            ax.plot(*g.xy, color=FLAT_MODE_COLORS["background"], linewidth=5, alpha=0.6, zorder=2, solid_capstyle='round')
        elif g.geom_type == 'Point':
            ax.plot(g.x, g.y, 'o', color=FLAT_MODE_COLORS["background"], markersize=15, alpha=0.6, zorder=2)

    # --- Plot Kept Items ---
    if params.get('preserve_types'):
        for i, kept_parts in kept_items.items():
            color = structured_colors.get(i)
            for k_geom in flatten_geometries(kept_parts):
                if k_geom.geom_type == 'LineString':
                    ax.plot(*k_geom.xy, color=color, linewidth=2.5, label=add_to_legend(f'Kept from {i}'), zorder=4, solid_capstyle='round')
                elif k_geom.geom_type == 'Point':
                    ax.plot(k_geom.x, k_geom.y, 'o', color=color, markersize=8, label=add_to_legend(f'Kept from {i}'), zorder=5)
    else: # Flat mode
        flat_kept = list(kept_items.values())[0]
        for g in flatten_geometries(flat_kept):
            if g.geom_type == 'LineString':
                ax.plot(*g.xy, color=FLAT_MODE_COLORS["kept"], linewidth=2.5, label=add_to_legend('Kept'), zorder=4, alpha=0.75, solid_capstyle='round')
            elif g.geom_type == 'Point':
                ax.plot(g.x, g.y, 'o', color=FLAT_MODE_COLORS["kept"], markersize=8, label=add_to_legend('Kept'), zorder=5)

    # --- Plot Removed Items ---
    if isinstance(removed_items, dict): # track_origins=True case
        for i, removed_parts_list in removed_items.items():
            color = structured_colors.get(i)
            for r_geom in flatten_geometries(removed_parts_list):
                if r_geom.geom_type == 'LineString':
                    ax.plot(*r_geom.xy, color=color, linewidth=2.0, linestyle='--', label=add_to_legend(f'Removed from {i}'), zorder=3, solid_capstyle='round')
                elif r_geom.geom_type == 'Point':
                    ax.plot(r_geom.x, r_geom.y, 'x', color=color, markersize=10, mew=2.5, label=add_to_legend(f'Removed from {i}'), zorder=4)
    elif isinstance(removed_items, list): # All other cases
        color = FLAT_MODE_COLORS["removed"]
        if params.get('preserve_types'):
            pass
        for g in flatten_geometries(removed_items):
            if g.geom_type == 'LineString':
                ax.plot(*g.xy, color=color, linewidth=2.5, label=add_to_legend('Removed'), zorder=3, solid_capstyle='round')
            elif g.geom_type == 'Point':
                ax.plot(g.x, g.y, 'x', color=color, markersize=10, mew=2.5, label=add_to_legend('Removed'), zorder=4)

    # --- Finalize plot ---
    if title:
        ax.set_title(title, fontsize=10, pad=20, loc='center')
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if labels_used:
        ax.legend(loc='lower right')
    plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=150)
    plt.close(fig)


# =============================================================================
#  Comprehensive Test Suite (Unchanged)
# =============================================================================

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_simple_line_overlap(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 0), (2, 0)]), LineString([(1, 0.05), (3, 0.05)])]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    if preserve_types: assert len(kept) == 2
    else: assert len(kept) == 2
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_line_fully_engulfed(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 0), (3, 0)]), LineString([(1, 0), (2, 0)])]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    assert len(kept) == 1
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_complex_intersection(preserve_types, keep_duplicates):
    geoms = [LineString([(0, 1), (3, 1)]), LineString([(1.5, 0), (1.5, 2)])]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.2, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    if preserve_types: assert len(kept) == 2
    else: assert len(kept) == 3
    if keep_duplicates: assert len(removed) == 1
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_tangent_circles_and_line(preserve_types, keep_duplicates):
    geoms = [ Point(0, 0).buffer(1.5), Point(2.0, 0).buffer(0.5), Point(0, 2.5).buffer(1.0), LineString([(-2, 1), (3, 1)]) ]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    if preserve_types: assert len(kept) == 4
    else: assert len(kept) > 4
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_high_volume_points(preserve_types, keep_duplicates):
    points = [Point(x*0.2, y*0.2) for x in range(10) for y in range(10)]
    remover_poly = Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
    geoms = [remover_poly] + points
    kept, kept_map, removed, mask = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    kept_points = [g for g in flatten_geometries(kept) if isinstance(g, Point)]
    removed_points = [g for g in flatten_geometries(removed) if isinstance(g, Point)]
    if keep_duplicates: assert len(kept_points) + len(removed_points) == len(points)
    else: assert len(removed) == 0 and len(kept_points) < len(points)

def test_origin_tracking_feature():
    geoms = [ LineString([(0, 0), (4, 0)]), LineString([(1, 0.05), (2, 0.05)]), LineString([(3, 0.05), (5, 0.05)]) ]
    results = deoverlap(geoms, 0.1, preserve_types=True, keep_duplicates=True, track_origins=True)
    plot_results(geoms, results["kept_parts"], results["removed_parts"], results["mask"])
    assert isinstance(results, dict)
    assert len(results["kept"]) == 2
    assert results["wholly_removed_indices"] == [1]

@pytest.mark.parametrize("preserve_types", [True, False])
@pytest.mark.parametrize("keep_duplicates", [True, False])
def test_tangent_circles(preserve_types, keep_duplicates):
    geoms = [ Point(0, 1).buffer(1), Point(0, 2).buffer(2), Point(0, 3).buffer(3) ]
    kept, kept_map, removed, mask = deoverlap(geoms, 0.1, preserve_types, keep_duplicates)
    plot_results(geoms, kept_map if preserve_types else {0: kept}, removed, mask)
    if preserve_types: assert len(kept) == 3
    else: assert len(kept) > 3
    if keep_duplicates: assert len(removed) > 0
    else: assert len(removed) == 0

@pytest.mark.parametrize("preserve_types", [True, False])
def test_empty_input(preserve_types):
    output = deoverlap([], 0.1, preserve_types=preserve_types, track_origins=True)
    if preserve_types:
        assert isinstance(output, dict) and not output["kept"] and not output["removed_parts"]
    else:
        assert isinstance(output, tuple) and len(output) == 4
        assert output[0] == []
        assert output[1] == {}
        assert output[2] == []

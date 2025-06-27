"""
This module provides high-level functions for de-overlapping a set of Shapely
geometric objects. It offers two main modes of operation: a fast "flat" mode
that returns simple geometries, and a more powerful "structured" mode that
preserves geometry types and can track the origin of removed pieces.
"""

from typing import List, Union, Iterable, Tuple, Dict, Any
from shapely import union_all
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm
from shapely.geometry import (
    LineString, Point, MultiLineString, MultiPoint,
    Polygon, MultiPolygon, GeometryCollection
)
from shapely.geometry.base import BaseGeometry

# =============================================================================
#  Type Aliases
# =============================================================================

GeomInput = Union[BaseGeometry, Iterable['GeomInput']]
FlatGeomOutput = List[Union[LineString, Point]]

# =============================================================================
#  Helper Function
# =============================================================================

def flatten_geometries(geoms: GeomInput) -> FlatGeomOutput:
    """
    Recursively flattens any geometry input into a flat list of non-empty
    LineStrings and Points. Polygons are converted to their boundary LineStrings.
    """
    out = []
    if geoms is None: return out

    # Use pattern matching for clear, type-based dispatching.
    match geoms:
        case LineString() | Point():
            if not geoms.is_empty: out.append(geoms)
        case MultiLineString() | MultiPoint() | GeometryCollection():
            for g in geoms.geoms: out.extend(flatten_geometries(g))
        case Polygon():
            if not geoms.is_empty:
                out.append(LineString(geoms.exterior.coords))
                for ring in geoms.interiors: out.append(LineString(ring.coords))
        case MultiPolygon():
            for poly in geoms.geoms: out.extend(flatten_geometries(poly))
        case list() | tuple() | set():
            for g in geoms: out.extend(flatten_geometries(g))
        case _:
            # Fallback for any other type that is a valid Shapely geometry
            # but not explicitly listed above. This provides some future-proofing.
            if not isinstance(geoms, BaseGeometry):
                 raise TypeError(f"Unsupported geometry type: {type(geoms)}")
    return out

# =============================================================================
#  Internal Engine Functions
# =============================================================================

def _deoverlap_flat_engine(
    geometries: GeomInput,
    tolerance: float,
    progress_bar: bool,
) -> Tuple[FlatGeomOutput, FlatGeomOutput, list]:
    """
    Internal engine for fast, flat de-overlapping.
    This is the core logic for the simple, high-performance mode. It always
    computes both kept and removed portions.
    """
    mask, flat_geoms, kept_geoms, removed_geoms = [], flatten_geometries(geometries), [], []
    
    # A tiny buffer used to resolve floating-point ambiguities when geometries
    # are perfectly aligned, ensuring consistent clipping.
    ROBUSTNESS_BUFFER = 1e-9

    iterable = tqdm(flat_geoms, desc="De-overlapping (flat)", disable=not progress_bar)

    for geom in iterable:
        kept_portion = geom
        
        # Only perform checks if a mask has been created.
        if mask:
            tree = STRtree(mask)
            nearby_indices = tree.query(geom)

            # Check .size on the NumPy array returned by the query.
            if nearby_indices.size > 0:
                local_mask = union_all([mask[i] for i in nearby_indices])
                # Buffer the mask by a tiny amount for robust clipping.
                kept_portion = geom.difference(local_mask.buffer(ROBUSTNESS_BUFFER))
        
        # Add the kept portion to the results and update the mask for the next iteration.
        if not kept_portion.is_empty:
            kept_geoms.append(kept_portion)
            mask.append(kept_portion.buffer(tolerance))
        
        # The removed portion is simply what's left of the original after the difference.
        if not (removed_portion := geom.difference(kept_portion)).is_empty:
            removed_geoms.append(removed_portion)
            
    # Return flattened lists, as difference operations can create multi-part geometries.
    return flatten_geometries(kept_geoms), flatten_geometries(removed_geoms), mask

def _deoverlap_structured_engine(
    geometries: Iterable[BaseGeometry],
    tolerance: float,
    progress_bar: bool,
) -> Tuple[List[BaseGeometry], Dict[int, List[BaseGeometry]], List[int], List[Polygon]]:
    """
    Internal engine for structure-preserving de-overlapping with origin tracking.
    This is the core logic for the powerful, feature-rich mode.
    """
    kept_results, removed_parts_map, wholly_removed_indices, mask = [], {}, [], []
    ROBUSTNESS_BUFFER = 1e-9

    iterable = tqdm(list(geometries), desc="De-overlapping (structured)", disable=not progress_bar)
    
    # Enumerate to get the original index 'i' for origin tracking.
    for i, geom in enumerate(iterable):
        if geom.is_empty: continue

        # Decompose the current top-level geometry into its constituent parts for clipping.
        parts_to_process = flatten_geometries(geom)
        if not parts_to_process: continue

        kept_sub_parts = []
        if not mask:
            # If the mask is empty, keep all parts of the first geometry.
            kept_sub_parts = parts_to_process
        else:
            # Check each constituent part against the cumulative mask.
            tree = STRtree(mask)
            for part in parts_to_process:
                if (nearby_indices := tree.query(part)).size > 0:
                    local_mask = union_all([mask[i] for i in nearby_indices])
                    if not (kept_part := part.difference(local_mask.buffer(ROBUSTNESS_BUFFER))).is_empty:
                        kept_sub_parts.append(kept_part)
                else: # Part is not near any existing mask, so it's kept.
                    kept_sub_parts.append(part)
        
        # If no sub-parts survived, the entire original geometry was removed.
        if not kept_sub_parts:
            wholly_removed_indices.append(i)
            removed_parts_map[i] = [geom] # The "removed part" is the whole geometry.
            continue
        
        # Reassemble the kept sub-parts into a single, clean geometry (e.g., MultiLineString).
        reassembled_kept_geom = unary_union(kept_sub_parts)
        
        # Check if a Polygon was clipped. If its boundary is intact, preserve the Polygon type.
        # Otherwise, the result is the line-based reassembled geometry.
        final_kept_geom = reassembled_kept_geom
        if geom.geom_type == 'Polygon' and reassembled_kept_geom.equals(geom.boundary):
             final_kept_geom = geom
        
        kept_results.append(final_kept_geom)
        
        # Calculate the removed portion for origin tracking.
        # For Polygons, we must diff against its boundary, not its area, for correct results.
        source_for_diff = geom.boundary if isinstance(geom, (Polygon, MultiPolygon)) else geom
        if not (removed_portion := source_for_diff.difference(reassembled_kept_geom)).is_empty:
             removed_parts_map[i] = [removed_portion]
        
        # Update the master mask with the buffer of the geometry that was actually kept.
        mask.append(reassembled_kept_geom.buffer(tolerance))
        
    return kept_results, removed_parts_map, wholly_removed_indices, mask

# =============================================================================
#  Single Public-Facing Function
# =============================================================================

def deoverlap(
    geometries: GeomInput,
    tolerance: float,
    preserve_types: bool = False,
    keep_duplicates: bool = False,
    track_origins: bool = False,
    progress_bar: bool = False,
) -> Any:
    """
    De-overlaps a list of geometries, with extensive options for output format and origin tracking.

    This function can operate in two primary modes controlled by `preserve_types`.
    A third flag, `track_origins`, provides even more detail in the structured mode.

    Args:
        geometries: An iterable of shapely geometries.
        tolerance: The buffer distance to consider geometries as overlapping.
        preserve_types (bool, optional): 
            - `False` (Default): Fast mode. Returns a flat list of simple LineStrings and Points.
            - `True`: Powerful mode. Returns structured geometries (e.g., MultiLineString). Slower.
        keep_duplicates (bool, optional): 
            - If `True`, the removed/overlapping portions are returned. Defaults to False.
        track_origins (bool, optional): 
            - Only applies when `preserve_types=True`.
            - If `True`, returns a detailed dictionary mapping removed parts to their
              original index, instead of the default tuple. Defaults to False.
        progress_bar (bool, optional): 
            - If `True`, displays a tqdm progress bar. Defaults to False.

    Returns:
        any: The return type depends on the flags:
        - By default, or if `preserve_types=False`: `(kept_geoms, removed_geoms, mask)`
        - If `preserve_types=True` and `track_origins=False`: `(kept_geoms, removed_geoms, mask)`
        - If `preserve_types=True` and `track_origins=True`: A dictionary with keys
          `("kept", "removed_parts", "wholly_removed_indices", "mask")`.
    """
    # --- Mode 1: Fast, Flat Output ---
    if not preserve_types:
        kept, removed, mask = _deoverlap_flat_engine(geometries, tolerance, progress_bar)
        return kept, (removed if keep_duplicates else []), mask
    
    # --- Mode 2: Structure-Preserving Output ---
    kept, removed_map, wholly_removed, mask = _deoverlap_structured_engine(geometries, tolerance, progress_bar)
    
    # Sub-mode: Return the detailed dictionary with origin tracking.
    if track_origins:
        return {
            "kept": kept,
            "removed_parts": (removed_map if keep_duplicates else {}),
            "wholly_removed_indices": wholly_removed,
            "mask": mask
        }
    # Sub-mode: Return a tuple for backward compatibility, but with structured geometries.
    else:
        removed = []
        if keep_duplicates:
            for parts in removed_map.values():
                removed.extend(parts)
        return kept, removed, mask
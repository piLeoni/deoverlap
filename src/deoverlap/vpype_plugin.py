"""vpype plugin: proximity-aware de-overlap for plotter paths.

Replaces coarse endpoint-only ``deduplicate``. Each kept stroke owns a
corridor of width ``tolerance``; later strokes are cropped or dropped where
they fall inside that corridor. Layers are processed independently unless you
pass the same document-level mask yourself (this command does not).
"""

from __future__ import annotations

from typing import List, Union

import click
import numpy as np
import vpype as vp
import vpype_cli
from shapely.geometry import LineString

from deoverlap.deoverlap import ClipMode, KeepPolicy, deoverlap, flatten_geometries


def _lines_from_layer(lines: vp.LineCollection) -> list[LineString]:
    out: list[LineString] = []
    for line in lines:
        if len(line) < 2:
            continue
        coords = [(float(c.real), float(c.imag)) for c in line]
        geom = LineString(coords)
        if not geom.is_empty and geom.length > 0:
            out.append(geom)
    return out


def _layer_from_geoms(geoms) -> vp.LineCollection:
    lc = vp.LineCollection()
    for geom in flatten_geometries(geoms):
        if geom.geom_type != "LineString" or len(geom.coords) < 2:
            continue
        lc.append(np.array([complex(x, y) for x, y in geom.coords], dtype=complex))
    return lc


@click.command(name="deoverlap")
@click.option(
    "-t",
    "--tolerance",
    type=vpype_cli.LengthType(),
    default="0.1mm",
    help="Corridor half-width; strokes closer than this are treated as one ink "
    "(default: 0.1mm ≈ one pen width).",
)
@click.option(
    "--keep",
    type=click.Choice([p.value for p in KeepPolicy], case_sensitive=False),
    default=KeepPolicy.LONGEST.value,
    help="Which stroke wins on collision (default: longest).",
)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in ClipMode], case_sensitive=False),
    default=ClipMode.CROP.value,
    help="crop = subtract overlap; drop = discard mostly-covered strokes.",
)
@click.option(
    "--min-length",
    type=vpype_cli.LengthType(),
    default="0.0mm",
    help="Drop lineal fragments shorter than this after clipping.",
)
@click.option(
    "--drop-fraction",
    type=float,
    default=0.5,
    help="In drop mode, discard a stroke once this fraction of its length is covered.",
)
@click.option(
    "--parallel-only/--no-parallel-only",
    default=True,
    help="Only clip against roughly parallel corridors (keep crossings). Default: on.",
)
@click.option(
    "--parallel-angle",
    type=float,
    default=30.0,
    help="Max local bearing difference in degrees for --parallel-only; raise it "
    "to also trim steeper merges (default: 30).",
)
@click.option(
    "--segments/--no-segments",
    default=False,
    help="Explode paths into edge segments so a thin outline can self-overlap "
    "(opposite sides of one LineString). Default: off.",
)
@click.option(
    "--segment-adjacency",
    type=int,
    default=1,
    help="With --segments, neighbouring segment indices on the same chain are "
    "exempt from clipping (default: 1).",
)
@click.option(
    "-k",
    "--keep-duplicates",
    is_flag=True,
    default=False,
    help="Store removed pieces on a new layer.",
)
@click.option(
    "-p",
    "--progress",
    is_flag=True,
    default=False,
    help="Show a progress bar.",
)
@click.option(
    "-l",
    "--layer",
    type=vpype_cli.LayerType(accept_multiple=True),
    default="all",
    help="Target layer(s) (default: all).",
)
@vpype_cli.global_processor
def deoverlap_cmd(
    document: vp.Document,
    tolerance: float,
    keep: str,
    mode: str,
    min_length: float,
    drop_fraction: float,
    parallel_only: bool,
    parallel_angle: float,
    segments: bool,
    segment_adjacency: int,
    keep_duplicates: bool,
    progress: bool,
    layer: Union[int, List[int]],
) -> vp.Document:
    """Remove near-coincident strokes using corridor de-overlap.

    Unlike endpoint-only deduplicate, this removes (or crops) paths that run
    within TOLERANCE of an earlier kept path — the case that bleeds on a
    plotter. With --segments, opposite sides of a single thin outline can
    suppress each other. Layers are handled one at a time.
    """
    layer_ids = vpype_cli.multiple_to_layer_ids(layer, document)
    new_document = document.empty_copy()
    removed_layer_id = document.free_id() if keep_duplicates else None

    for lines, lid in zip(document.layers_from_ids(layer_ids), layer_ids):
        geoms = _lines_from_layer(lines)
        if not geoms:
            new_document.add(lines, layer_id=lid)
            continue

        result = deoverlap(
            geoms,
            tolerance,
            keep=keep,
            mode=mode,
            min_length=min_length,
            drop_fraction=drop_fraction,
            parallel_only=parallel_only,
            parallel_angle=parallel_angle,
            segments=segments,
            segment_adjacency=segment_adjacency,
            group=True,
            keep_duplicates=keep_duplicates,
            progress_bar=progress,
        )
        new_document.add(_layer_from_geoms(result.kept), layer_id=lid)

        if keep_duplicates and removed_layer_id is not None and result.removed:
            new_document.add(
                _layer_from_geoms(result.removed), layer_id=removed_layer_id
            )

    for lid, lines in document.layers.items():
        if lid not in layer_ids:
            new_document.add(lines, layer_id=lid)

    return new_document


deoverlap_cmd.help_group = "Plugins"

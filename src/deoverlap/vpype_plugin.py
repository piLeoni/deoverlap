"""vpype plugin: proximity-aware de-overlap for plotter paths.

Replaces coarse endpoint-only ``deduplicate``. Each kept stroke owns a
corridor of radius ``tolerance``; later strokes are cropped or dropped where
they fall inside that corridor. Layers are processed independently.
"""

from __future__ import annotations

from typing import List, Optional, Union

import click
import numpy as np
import vpype as vp
import vpype_cli
from shapely.geometry import LineString

from deoverlap.deoverlap import deoverlap, flatten_geometries


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
    help="Corridor radius: strokes closer than this to a kept stroke are cut, "
    "usually the pen width (default: 0.1mm).",
)
@click.option(
    "--prefer",
    type=click.Choice(["longest", "first", "shortest"], case_sensitive=False),
    default="longest",
    help="Which stroke wins a collision; first = input order (default: longest).",
)
@click.option(
    "--angle",
    type=click.FloatRange(0, 90),
    default=30.0,
    help="Strokes overlap only where their directions differ by at most this "
    "many degrees; 90 cuts crossings too (default: 30).",
)
@click.option(
    "--self-overlap",
    is_flag=True,
    default=False,
    help="Let a path overlap itself, e.g. the two sides of a thin outline.",
)
@click.option(
    "-m",
    "--min-length",
    type=vpype_cli.LengthType(),
    default="0mm",
    help="Drop pieces shorter than this after cutting (default: 0mm).",
)
@click.option(
    "--drop",
    type=click.FloatRange(0, 1),
    default=None,
    help="Discard a whole stroke when more than this fraction of it would be "
    "cut (default: always crop).",
)
@click.option(
    "-k",
    "--keep-duplicates",
    is_flag=True,
    default=False,
    help="Keep removed pieces in a separate layer.",
)
@click.option(
    "-p",
    "--progress-bar",
    is_flag=True,
    default=False,
    help="Display a progress bar.",
)
@click.option(
    "-l",
    "--layer",
    type=vpype_cli.LayerType(accept_multiple=True),
    default="all",
    help="Target layer(s) (default: 'all').",
)
@vpype_cli.global_processor
def deoverlap_cmd(
    document: vp.Document,
    tolerance: float,
    prefer: str,
    angle: float,
    self_overlap: bool,
    min_length: float,
    drop: Optional[float],
    keep_duplicates: bool,
    progress_bar: bool,
    layer: Union[int, List[int]],
) -> vp.Document:
    """Remove strokes that run on top of each other.

    Unlike endpoint-only deduplicate, this cuts (or drops) paths that run
    within TOLERANCE of a kept path — the case that bleeds on a plotter.
    Layers are handled one at a time.
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
            prefer=prefer.lower(),
            angle=angle,
            self_overlap=self_overlap,
            min_length=min_length,
            drop=drop,
            keep_duplicates=keep_duplicates,
            progress_bar=progress_bar,
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

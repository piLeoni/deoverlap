"""De-overlap vector strokes to prevent overdrawing."""

__version__ = "4.2.0"

from .deoverlap import (
    DeoverlapResult,
    GeomInput,
    Prefer,
    deoverlap,
    flatten_geometries,
)

__all__ = [
    "DeoverlapResult",
    "GeomInput",
    "Prefer",
    "deoverlap",
    "flatten_geometries",
    "__version__",
]

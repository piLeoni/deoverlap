"""De-overlap Shapely geometries that sit within a tolerance of each other."""

__version__ = "4.1.0"

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

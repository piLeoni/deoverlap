"""De-overlap Shapely geometries that sit within a tolerance of each other."""

__version__ = "3.1.0"

from .deoverlap import (
    ClipMode,
    DeoverlapResult,
    GeomInput,
    KeepPolicy,
    deoverlap,
    flatten_geometries,
)

__all__ = [
    "ClipMode",
    "DeoverlapResult",
    "GeomInput",
    "KeepPolicy",
    "deoverlap",
    "flatten_geometries",
    "__version__",
]

"""Series decomposition, assembly, and schema for COMO paper-mode."""

from .schema import EARecord, RejectedRecord, SeriesDecomposition
from .decomposition import decompose_series
from .assembly import assemble_series_member

__all__ = [
    "EARecord",
    "RejectedRecord",
    "SeriesDecomposition",
    "decompose_series",
    "assemble_series_member",
]

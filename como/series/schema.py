"""Typed data model for a decomposed analog series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem


@dataclass(frozen=True)
class EARecord:
    """One decomposed existing analog.

    site_map: {site_id -> fragment_smiles | None}
        None means H / no substituent at that site.
    """
    input_smiles: str
    canonical_smiles: str
    activity: float
    site_map: dict[int, str | None]
    heavy_atom_count: int


@dataclass(frozen=True)
class RejectedRecord:
    """An EA that was excluded from decomposition with the reason."""
    input_smiles: str
    reason: str  # "invalid_smiles" | "core_no_match" | "fused_ring_mismatch" | "off_exit_vector"


@dataclass
class SeriesDecomposition:
    """All information extracted from a decomposed analog series.

    This object is the shared input consumed by CloseInVAGenerator,
    FreeWilsonVAGenerator, and the scoring protocol.
    """
    core_smiles: str
    core_mol: "Chem.Mol"
    site_list: tuple[int, ...]          # ordered declared substitution sites
    ea_records: tuple[EARecord, ...]
    ea_canonical_set: frozenset[str]
    site_pools: dict[int, frozenset[str]]   # organic fragments only (no None)
    unique_substituents: frozenset[str]     # union of all site_pools
    substitution_probability: float         # global p_sub = filled / (n_EA * n_sites)
    site_substitution_probability: dict[int, float]  # per-site p_sub
    ea_hac_range: tuple[int, int]
    rejected_records: tuple[RejectedRecord, ...]

    # Set only when exit vectors were present in the core
    exit_vector_sites: frozenset[int] = field(default_factory=frozenset)

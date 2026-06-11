"""COMO/DeepCOMO: Compound Optimization Monitor for analog series analysis."""

from rdkit import RDLogger as _RDLogger
_RDLogger.DisableLog('rdApp.*')

from .scoring import ComoAnalysis, score_series, score_with_repeats, c_score, d_score, s_score, p_score, assign_stage
from .report import ComoResult
from .potency import SVRPredictor, PaperSVRPredictor
from .series import SeriesDecomposition, EARecord, RejectedRecord, decompose_series, assemble_series_member

__all__ = [
    "ComoAnalysis",
    "score_series",
    "score_with_repeats",
    "ComoResult",
    "c_score",
    "d_score",
    "s_score",
    "p_score",
    "assign_stage",
    "SVRPredictor",
    "PaperSVRPredictor",
    "SeriesDecomposition",
    "EARecord",
    "RejectedRecord",
    "decompose_series",
    "assemble_series_member",
]

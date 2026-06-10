"""ComoResult dataclass and output writers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .descriptors import DESCRIPTOR_NAMES


@dataclass
class ComoResult:
    # Scores
    C: float
    D: float
    d_mean: float
    S: float
    P: float
    stage: str
    radius_used: float

    # EA info
    n_ea: int
    ea_smiles: list[str]
    ea_activities: np.ndarray

    # SVR CV metrics
    cv_r2: float
    cv_mae: float

    # VA populations
    va_df: pl.DataFrame

    # FW stats
    fw_n_candidates: int
    fw_n_ea_in_nbh: int
    fw_n_sites: int
    fw_n_unique_substituents: int

    # Descriptor info
    descriptor_names: tuple[str, ...] = field(default=DESCRIPTOR_NAMES)
    scaler: object = field(default=None, repr=False)


def build_va_dataframe(
    va_smiles_by_strategy: dict[str, list[str]],
    flat_va_aligned: list[str],
    svr_predictions: np.ndarray,
    ea_activities: np.ndarray,
    membership: np.ndarray,
    fw_predictions: dict[str, float],
    external_predictions: dict[str, float],
) -> pl.DataFrame:
    """Build the VA populations DataFrame.

    A VA appearing in multiple strategies gets one row per strategy.
    """
    # Build lookup from canonical SMILES to SVR prediction index
    pred_lookup: dict[str, float] = {}
    pct_lookup: dict[str, float] = {}
    in_nbh_lookup: dict[str, bool] = {}
    nbh_count_lookup: dict[str, int] = {}

    if svr_predictions.size > 0:
        from .potency import SVRPredictor
        pct = _percentile_rank_static(svr_predictions, ea_activities)
        for i, smi in enumerate(flat_va_aligned):
            pred_lookup[smi] = float(svr_predictions[i]) if not np.isnan(svr_predictions[i]) else float("nan")
            pct_lookup[smi] = float(pct[i]) if not np.isnan(pct[i]) else float("nan")

    if membership.size > 0:
        count_j = membership.sum(axis=1)
        for i, smi in enumerate(flat_va_aligned):
            in_nbh_lookup[smi] = bool(count_j[i] > 0)
            nbh_count_lookup[smi] = int(count_j[i])

    rows = []
    for strategy, smi_list in va_smiles_by_strategy.items():
        for smi in smi_list:
            rows.append({
                "smiles": smi,
                "source_strategy": strategy,
                "pred_pActivity": pred_lookup.get(smi, float("nan")),
                "pActivity_percentile": pct_lookup.get(smi, float("nan")),
                "fw_pred_pActivity": fw_predictions.get(smi),
                "external_pred_pActivity": external_predictions.get(smi),
                "in_nbh": in_nbh_lookup.get(smi, False),
                "nbh_count": nbh_count_lookup.get(smi, 0),
            })

    if not rows:
        return pl.DataFrame(schema={
            "smiles": pl.Utf8,
            "source_strategy": pl.Utf8,
            "pred_pActivity": pl.Float64,
            "pActivity_percentile": pl.Float64,
            "fw_pred_pActivity": pl.Float64,
            "external_pred_pActivity": pl.Float64,
            "in_nbh": pl.Boolean,
            "nbh_count": pl.Int64,
        })

    return pl.DataFrame(rows, schema={
        "smiles": pl.Utf8,
        "source_strategy": pl.Utf8,
        "pred_pActivity": pl.Float64,
        "pActivity_percentile": pl.Float64,
        "fw_pred_pActivity": pl.Float64,
        "external_pred_pActivity": pl.Float64,
        "in_nbh": pl.Boolean,
        "nbh_count": pl.Int64,
    })


def _percentile_rank_static(
    va_predictions: np.ndarray,
    ea_activities: np.ndarray,
) -> np.ndarray:
    result = np.full(len(va_predictions), np.nan)
    for i, pred in enumerate(va_predictions):
        if not np.isnan(pred):
            result[i] = 100.0 * float((ea_activities < pred).mean())
    return result


def write_scores_csv(result: ComoResult, path: Path) -> None:
    """Write scores.csv with metric/value rows."""
    rows = [
        {"metric": "C", "value": f"{result.C:.4f}"},
        {"metric": "D", "value": f"{result.D:.4f}"},
        {"metric": "d_mean", "value": f"{result.d_mean:.4f}"},
        {"metric": "S", "value": f"{result.S:.4f}"},
        {"metric": "P", "value": f"{result.P:.4f}"},
        {"metric": "stage", "value": result.stage},
        {"metric": "radius_used", "value": f"{result.radius_used:.4f}"},
        {"metric": "n_ea", "value": str(result.n_ea)},
        {"metric": "n_va", "value": str(len(result.va_df))},
        {"metric": "cv_r2", "value": f"{result.cv_r2:.4f}"},
        {"metric": "cv_mae", "value": f"{result.cv_mae:.4f}"},
        {"metric": "fw_n_candidates", "value": str(result.fw_n_candidates)},
        {"metric": "fw_n_ea_in_nbh", "value": str(result.fw_n_ea_in_nbh)},
        {"metric": "fw_n_sites", "value": str(result.fw_n_sites)},
        {"metric": "fw_n_unique_substituents", "value": str(result.fw_n_unique_substituents)},
    ]
    pl.DataFrame(rows).write_csv(path)


_STAGE_DESCRIPTIONS = {
    "early": (
        "Early stage: chemical space is largely unexplored (low S) and SAR trends "
        "are unclear (low P). Diverse VA design strategies are recommended to expand "
        "coverage and reveal SAR."
    ),
    "early_mid": (
        "Early-mid stage: some SAR hot spots detected (high P) but chemical space "
        "coverage is still limited (low S). Exploratory design combined with activity "
        "predictions can guide expansion towards more potent regions."
    ),
    "mid": (
        "Mid stage: chemical space is well explored (high S) and SAR is relatively "
        "continuous (low P). Close-in or Free-Wilson VAs are recommended to refine "
        "potency within established SAR."
    ),
    "late": (
        "Late stage: chemical space is well explored (high S) and SAR is highly "
        "discontinuous (high P). The series is chemically saturated; further "
        "optimization may be difficult. Free-Wilson and focused design are recommended."
    ),
}


def write_summary_txt(result: ComoResult, path: Path) -> None:
    """Write human-readable COMO analysis summary."""
    acts = result.ea_activities
    va_df = result.va_df

    lines = [
        "=" * 70,
        "COMO / DeepCOMO Analog Series Analysis",
        "=" * 70,
        "",
        "--- Analog Series Statistics ---",
        f"  EAs (existing analogs):   {result.n_ea}",
        f"  Valid EAs (parsed):       {len(result.ea_smiles)}",
        f"  pActivity range:          {acts.min():.2f} – {acts.max():.2f}",
        f"  pActivity mean ± std:     {acts.mean():.2f} ± {acts.std():.2f}",
        "",
        "--- Chemical Space Parameters ---",
        f"  NBH radius (r):           {result.radius_used:.4f} (normalized 7D space)",
        f"  Descriptors:              {', '.join(result.descriptor_names)}",
        "",
        "--- COMO Scores ---",
        f"  C (coverage):             {result.C:.4f}",
        f"    → Fraction of VAs falling into at least one EA neighborhood.",
        f"      High C = VAs cover well-explored EA chemical space.",
        f"  D (density):              {result.D:.4f}  (d_mean = {result.d_mean:.4f})",
        f"    → Overlap density of EA neighborhoods by VAs.",
        f"      High D = VAs densely map EA-local chemical space.",
        f"  S (saturation):           {result.S:.4f}  [harmonic mean of C and D]",
        f"    → Overall chemical saturation score.",
        f"  P (SAR progression):      {result.P:.4f}",
        f"    → SAR discontinuity in overlapping EA neighborhoods.",
        f"      High P = large potency variations among structurally similar EAs.",
        "",
        "--- Development Stage ---",
        f"  Stage:  {result.stage.upper()}",
        f"  {_STAGE_DESCRIPTIONS.get(result.stage, '')}",
        "",
        "--- Potency Model (SVR, ECFP4, Tanimoto kernel) ---",
        f"  3-fold CV R²:             {result.cv_r2:.4f}",
        f"  3-fold CV MAE:            {result.cv_mae:.4f} pActivity units",
        "",
        "--- Virtual Analog Populations ---",
    ]

    if len(va_df) == 0:
        lines.append("  No VAs generated.")
    else:
        # Summary by strategy
        strategy_counts = (
            va_df.group_by("source_strategy")
            .agg(pl.len().alias("count"))
            .sort("source_strategy")
        )
        for row in strategy_counts.iter_rows(named=True):
            lines.append(f"  {row['source_strategy']:20s}: {row['count']} VAs")

        n_in_nbh = va_df["in_nbh"].sum()
        lines.append(f"  VAs in at least one NBH: {n_in_nbh} ({100*n_in_nbh/len(va_df):.1f}%)")

        # Top 10 VAs by predicted pActivity
        top = (
            va_df.drop_nulls(subset=["pred_pActivity"])
            .sort("pred_pActivity", descending=True)
            .head(10)
        )
        if len(top) > 0:
            lines.append("")
            lines.append("  Top-10 VAs by predicted pActivity:")
            lines.append(f"  {'#':>3}  {'pAct':>6}  {'%ile':>5}  {'strategy':18}  smiles")
            lines.append("  " + "-" * 65)
            for rank, row in enumerate(top.iter_rows(named=True), 1):
                smi = row["smiles"]
                if len(smi) > 40:
                    smi = smi[:37] + "..."
                pct = row.get("pActivity_percentile")
                pct_str = f"{pct:.1f}" if pct is not None and not (isinstance(pct, float) and pct != pct) else "N/A"
                lines.append(
                    f"  {rank:>3}  {row['pred_pActivity']:>6.2f}  {pct_str:>5}  "
                    f"{row['source_strategy']:18}  {smi}"
                )

    lines += [
        "",
        "--- Free-Wilson Analysis ---",
        f"  FW VA candidates:         {result.fw_n_candidates}",
        f"  EAs in FW neighborhoods:  {result.fw_n_ea_in_nbh}",
        f"  Substitution sites:       {result.fw_n_sites}",
        f"  Unique substituents:      {result.fw_n_unique_substituents}",
        "",
        "=" * 70,
    ]

    Path(path).write_text("\n".join(lines))

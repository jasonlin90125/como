"""ComoResult dataclass and output writers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .descriptors import DESCRIPTOR_NAMES

if TYPE_CHECKING:
    from .series.schema import SeriesDecomposition
    from .potency import PaperSVRResult


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

    # FW per-candidate prediction details (smiles -> mean/std/n)
    fw_predictions: dict[str, float] = field(default_factory=dict)
    fw_pred_std: dict[str, float] = field(default_factory=dict)
    fw_pred_n: dict[str, int] = field(default_factory=dict)
    # Retrospective FW EA predictions {smiles: [pred, ...]}
    fw_ea_predictions: dict[str, list[float]] = field(default_factory=dict)

    # Descriptor info
    descriptor_names: tuple[str, ...] = field(default=DESCRIPTOR_NAMES)
    scaler: object = field(default=None, repr=False)

    # Optional: paper-mode extras (populated when paper_mode=True)
    decomp: "SeriesDecomposition | None" = field(default=None, repr=False)
    paper_svr_result: "PaperSVRResult | None" = field(default=None, repr=False)
    run_metadata: dict = field(default_factory=dict)


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


def write_fw_predictions_csv(result: ComoResult, path: Path) -> None:
    """Write fw_predictions.csv with per-entry FW prediction details.

    Columns: smiles, type (fw_va|fw_ea), fw_pred_mean, fw_pred_std, fw_pred_n,
             observed_activity, abs_error
    """
    rows = []

    # FW VA candidates
    act_lookup = dict(zip(result.ea_smiles, result.ea_activities.tolist()))
    va_df = result.va_df
    fw_pred_lookup = {}
    if len(va_df) > 0 and "fw_pred_pActivity" in va_df.columns:
        for row in va_df.iter_rows(named=True):
            smi = row["smiles"]
            val = row.get("fw_pred_pActivity")
            if val is not None and not (isinstance(val, float) and val != val):
                fw_pred_lookup[smi] = val

    # Write FW VA rows
    fw_va_smiles = set()
    if len(result.va_df) > 0:
        fw_rows_df = result.va_df.filter(result.va_df["source_strategy"] == "free_wilson")
        fw_va_smiles = set(fw_rows_df["smiles"].to_list())

    for smi in sorted(fw_va_smiles):
        pred_mean = fw_pred_lookup.get(smi, float("nan"))
        pred_std = result.fw_pred_std.get(smi, 0.0)
        pred_n = result.fw_pred_n.get(smi, 1)
        rows.append({
            "smiles": smi,
            "type": "fw_va",
            "fw_pred_mean": pred_mean,
            "fw_pred_std": pred_std,
            "fw_pred_n": pred_n,
            "observed_activity": float("nan"),
            "abs_error": float("nan"),
        })

    # Write retrospective FW EA rows
    for smi, preds in sorted(result.fw_ea_predictions.items()):
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds)) if len(preds) > 1 else 0.0
        obs = act_lookup.get(smi, float("nan"))
        abs_err = abs(mean_pred - obs) if not (obs != obs) else float("nan")
        rows.append({
            "smiles": smi,
            "type": "fw_ea",
            "fw_pred_mean": mean_pred,
            "fw_pred_std": std_pred,
            "fw_pred_n": len(preds),
            "observed_activity": obs,
            "abs_error": abs_err,
        })

    if not rows:
        pl.DataFrame(schema={
            "smiles": pl.Utf8,
            "type": pl.Utf8,
            "fw_pred_mean": pl.Float64,
            "fw_pred_std": pl.Float64,
            "fw_pred_n": pl.Int64,
            "observed_activity": pl.Float64,
            "abs_error": pl.Float64,
        }).write_csv(path)
        return

    pl.DataFrame(rows, schema={
        "smiles": pl.Utf8,
        "type": pl.Utf8,
        "fw_pred_mean": pl.Float64,
        "fw_pred_std": pl.Float64,
        "fw_pred_n": pl.Int64,
        "observed_activity": pl.Float64,
        "abs_error": pl.Float64,
    }).write_csv(path)


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


def write_decomposition_report(result: "ComoResult", path: Path) -> None:
    """Write decomposition_report.csv with per-EA status and site fragments."""
    decomp = result.decomp
    if decomp is None:
        return

    site_list = list(decomp.site_list)
    rows = []

    for rec in decomp.ea_records:
        row: dict = {
            "input_smiles": rec.input_smiles,
            "canonical_smiles": rec.canonical_smiles,
            "status": "decomposed",
            "rejection_reason": "",
            "activity": rec.activity,
            "heavy_atom_count": rec.heavy_atom_count,
        }
        for s in site_list:
            row[f"site_{s}"] = rec.site_map.get(s) or ""
        rows.append(row)

    for rec in decomp.rejected_records:
        row = {
            "input_smiles": rec.input_smiles,
            "canonical_smiles": "",
            "status": "rejected",
            "rejection_reason": rec.reason,
            "activity": float("nan"),
            "heavy_atom_count": 0,
        }
        for s in site_list:
            row[f"site_{s}"] = ""
        rows.append(row)

    if not rows:
        return

    site_cols = {f"site_{s}": pl.Utf8 for s in site_list}
    schema = {
        "input_smiles": pl.Utf8,
        "canonical_smiles": pl.Utf8,
        "status": pl.Utf8,
        "rejection_reason": pl.Utf8,
        "activity": pl.Float64,
        "heavy_atom_count": pl.Int64,
        **site_cols,
    }
    pl.DataFrame(rows, schema=schema).write_csv(path)


def write_decomposition_summary_json(result: "ComoResult", path: Path) -> None:
    """Write decomposition_summary.json."""
    decomp = result.decomp
    if decomp is None:
        return

    summary = {
        "n_input": len(decomp.ea_records) + len(decomp.rejected_records),
        "n_decomposed": len(decomp.ea_records),
        "n_rejected": len(decomp.rejected_records),
        "site_list": list(decomp.site_list),
        "n_sites": len(decomp.site_list),
        "unique_substituent_count": len(decomp.unique_substituents),
        "substitution_probability_global": decomp.substitution_probability,
        "site_substitution_probability": {
            str(k): v for k, v in decomp.site_substitution_probability.items()
        },
        "ea_hac_range": list(decomp.ea_hac_range),
        "paper_mode": result.run_metadata.get("paper_mode", False),
    }
    Path(path).write_text(json.dumps(summary, indent=2))


def write_fw_candidates_csv(result: "ComoResult", path: Path) -> None:
    """Write fw_candidates.csv with per-candidate FW prediction details."""
    rows = []
    va_df = result.va_df
    if len(va_df) == 0:
        pl.DataFrame(schema={
            "smiles": pl.Utf8,
            "fw_pred_mean": pl.Float64,
            "fw_pred_std": pl.Float64,
            "fw_pred_n": pl.Int64,
        }).write_csv(path)
        return

    fw_df = va_df.filter(va_df["source_strategy"] == "free_wilson")
    for row in fw_df.iter_rows(named=True):
        smi = row["smiles"]
        rows.append({
            "smiles": smi,
            "fw_pred_mean": result.fw_predictions.get(smi, float("nan"))
            if hasattr(result, "fw_predictions") else row.get("fw_pred_pActivity", float("nan")),
            "fw_pred_std": result.fw_pred_std.get(smi, 0.0),
            "fw_pred_n": result.fw_pred_n.get(smi, 1),
        })

    if not rows:
        pl.DataFrame(schema={
            "smiles": pl.Utf8,
            "fw_pred_mean": pl.Float64,
            "fw_pred_std": pl.Float64,
            "fw_pred_n": pl.Int64,
        }).write_csv(path)
        return

    pl.DataFrame(rows, schema={
        "smiles": pl.Utf8,
        "fw_pred_mean": pl.Float64,
        "fw_pred_std": pl.Float64,
        "fw_pred_n": pl.Int64,
    }).write_csv(path)


def write_fw_ea_validation_csv(result: "ComoResult", path: Path) -> None:
    """Write fw_ea_validation.csv with retrospective FW predictions for EAs."""
    act_lookup = dict(zip(result.ea_smiles, result.ea_activities.tolist()))
    rows = []
    for smi, preds in sorted(result.fw_ea_predictions.items()):
        obs = act_lookup.get(smi, float("nan"))
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds)) if len(preds) > 1 else 0.0
        abs_err = abs(mean_pred - obs) if np.isfinite(obs) else float("nan")
        rows.append({
            "smiles": smi,
            "observed_pActivity": obs,
            "fw_pred_mean": mean_pred,
            "fw_pred_std": std_pred,
            "fw_pred_n": len(preds),
            "absolute_error": abs_err,
        })

    schema = {
        "smiles": pl.Utf8,
        "observed_pActivity": pl.Float64,
        "fw_pred_mean": pl.Float64,
        "fw_pred_std": pl.Float64,
        "fw_pred_n": pl.Int64,
        "absolute_error": pl.Float64,
    }
    if not rows:
        pl.DataFrame(schema=schema).write_csv(path)
        return
    pl.DataFrame(rows, schema=schema).write_csv(path)


def write_svr_outputs(result: "ComoResult", output_path: Path) -> None:
    """Write paper SVR output files when paper SVR was used."""
    paper_svr = result.paper_svr_result
    if paper_svr is None:
        return

    meta = result.run_metadata

    # svr_training_summary.csv
    summary_rows = [{
        "svr_mode": "paper",
        "n_ea_train": meta.get("n_ea_train", ""),
        "n_ea_validation": meta.get("n_ea_validation", ""),
        "n_external_loaded": meta.get("n_external_loaded", ""),
        "n_external_valid": meta.get("n_external_valid", ""),
        "n_external_after_as_exclusion": meta.get("n_external_after_as_exclusion", ""),
        "fp_type": "ECFP4_2048",
        "kernel": "tanimoto_precomputed",
        "outer_folds": len(paper_svr.outer_fold_results),
        "inner_folds": meta.get("inner_folds", 3),
        "random_state": meta.get("random_state", 42),
    }]
    pl.DataFrame(summary_rows).write_csv(output_path / "svr_training_summary.csv")

    # svr_outer_folds.csv
    fold_rows = [
        {
            "fold": r.fold,
            "best_C": r.best_C,
            "best_epsilon": r.best_epsilon,
            "outer_r2": r.outer_r2,
            "outer_mae": r.outer_mae,
            "outer_rmse": r.outer_rmse,
            "n_train": r.n_train,
            "n_test": r.n_test,
        }
        for r in paper_svr.outer_fold_results
    ]
    if fold_rows:
        pl.DataFrame(fold_rows, schema={
            "fold": pl.Int64, "best_C": pl.Float64, "best_epsilon": pl.Float64,
            "outer_r2": pl.Float64, "outer_mae": pl.Float64, "outer_rmse": pl.Float64,
            "n_train": pl.Int64, "n_test": pl.Int64,
        }).write_csv(output_path / "svr_outer_folds.csv")

    # svr_external_validation.csv
    val_rows = [
        {
            "smiles": smi,
            "observed_pActivity": obs,
            "predicted_pActivity": pred_mean,
            "prediction_std": pred_std,
            "absolute_error": abs(pred_mean - obs) if np.isfinite(obs) else float("nan"),
        }
        for smi, (obs, pred_mean, pred_std) in sorted(paper_svr.external_validation.items())
    ]
    schema_val = {
        "smiles": pl.Utf8,
        "observed_pActivity": pl.Float64,
        "predicted_pActivity": pl.Float64,
        "prediction_std": pl.Float64,
        "absolute_error": pl.Float64,
    }
    if val_rows:
        pl.DataFrame(val_rows, schema=schema_val).write_csv(
            output_path / "svr_external_validation.csv"
        )
    else:
        pl.DataFrame(schema=schema_val).write_csv(output_path / "svr_external_validation.csv")


def write_run_metadata_json(result: "ComoResult", path: Path) -> None:
    """Write run_metadata.json with full provenance for the pipeline run."""
    import como

    meta = dict(result.run_metadata)
    meta.setdefault("como_version", getattr(como, "__version__", "0.1.0"))

    # Try to get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = ""
    meta.setdefault("git_commit", git_hash)

    meta.setdefault("descriptor_space", "rdkit_7d")
    if result.decomp is not None:
        meta.setdefault("site_list", list(result.decomp.site_list))
        meta.setdefault("n_ea_decomposed", len(result.decomp.ea_records))
        meta.setdefault("n_ea_rejected", len(result.decomp.rejected_records))
        meta.setdefault(
            "core_used_after_exit_vector_stripping", result.decomp.core_smiles
        )

    Path(path).write_text(json.dumps(meta, indent=2, default=str))

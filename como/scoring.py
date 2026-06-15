"""COMO scoring functions (Equations 1–6) and pipeline orchestration."""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .descriptors import compute_descriptors, normalize_descriptors
from .nbh import build_nbh
from .potency import SVRPredictor
from .report import (
    ComoResult,
    build_va_dataframe,
    write_decomposition_report,
    write_decomposition_summary_json,
    write_fw_candidates_csv,
    write_fw_ea_validation_csv,
    write_fw_predictions_csv,
    write_run_metadata_json,
    write_scores_csv,
    write_summary_txt,
    write_svr_outputs,
)

if TYPE_CHECKING:
    from .analogs.base import VAGenerator


# ---------------------------------------------------------------------------
# Score functions — direct implementations of paper Equations 1–6
# ---------------------------------------------------------------------------

def c_score(membership: np.ndarray) -> float:
    """Eq. 1: C = VA_NBH / VA_all."""
    if membership.shape[0] == 0:
        return 0.0
    va_nbh = (membership.sum(axis=1) > 0).sum()
    return float(va_nbh) / float(membership.shape[0])


def d_score(membership: np.ndarray) -> tuple[float, float]:
    """Eqs. 2–3: D = 1 - 1/d_mean, where d_mean is over covered VAs only.

    Returns (D, d_mean). Denominator is only covered VAs, not VA_all.
    """
    if membership.shape[0] == 0:
        return 0.0, 1.0
    count_j = membership.sum(axis=1)  # (n_va,)
    covered = count_j[count_j > 0]
    if len(covered) == 0:
        return 0.0, 1.0
    d_mean = float(covered.mean())
    D = 1.0 - 1.0 / d_mean
    return D, d_mean


def s_score(C: float, D: float) -> float:
    """Eq. 4: S = 2CD / (C + D), the harmonic mean of C and D."""
    if C + D == 0.0:
        return 0.0
    return 2.0 * C * D / (C + D)


def p_score(membership: np.ndarray, ea_activities: np.ndarray) -> float:
    """Eqs. 5–6: SAR progression score based on potency variation in overlapping NBHs.

    Only VAs in >=2 EA neighborhoods contribute.
    ea_activities must be aligned with membership columns.
    P is finite and >= 0, but not bounded above by 1.
    """
    if membership.shape[0] == 0:
        return 0.0

    count_j = membership.sum(axis=1)
    overlapping = np.where(count_j >= 2)[0]
    if len(overlapping) == 0:
        return 0.0

    weighted_num = 0.0
    weighted_den = 0.0
    for j in overlapping:
        ea_idx = np.where(membership[j])[0]
        m_j = len(ea_idx)
        acts = ea_activities[ea_idx]
        pair_sum = sum(abs(a - b) for a, b in combinations(acts.tolist(), 2))
        delta_j = (2.0 / (m_j * (m_j - 1))) * pair_sum
        w_j = 1.0 / m_j
        weighted_num += w_j * delta_j
        weighted_den += w_j

    return weighted_num / weighted_den if weighted_den > 0.0 else 0.0


def assign_stage(
    S: float,
    P: float,
    s_threshold: float = 0.4,
    p_threshold: float = 0.5,
) -> str:
    """Assign development stage from S/P quadrant.

    Returns one of: 'early', 'early_mid', 'mid', 'late'.
    Note: thresholds are a heuristic convenience interpretation, not an
    authoritative DeepCOMO rule.
    """
    high_s = S >= s_threshold
    high_p = P >= p_threshold
    if high_s and not high_p:
        return "mid"
    if high_s and high_p:
        return "late"
    if not high_s and high_p:
        return "early_mid"
    return "early"


# ---------------------------------------------------------------------------
# Core detection helper
# ---------------------------------------------------------------------------

def detect_murcko_core(ea_smiles: list[str]) -> str | None:
    """Return canonical SMILES of the most common Murcko scaffold among EAs."""
    counts: Counter[str] = Counter()
    for smi in ea_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold:
            counts[scaffold] += 1

    if not counts:
        return None

    most_common, freq = counts.most_common(1)[0]
    coverage = freq / len(ea_smiles)
    if coverage < 0.5:
        warnings.warn(
            f"Most common Murcko scaffold covers only {coverage:.0%} of EAs. "
            "Consider providing --core explicitly.",
            stacklevel=3,
        )
    return most_common


# ---------------------------------------------------------------------------
# score_series — one-call public API
# ---------------------------------------------------------------------------

def score_series(
    series_csv: str | Path | pl.DataFrame,
    smiles_col: str = "smiles",
    activity_col: str = "pActivity",
    core: str | None = None,
    va_strategies: list[str] | tuple[str, ...] = ("close_in",),
    va_csv: str | Path | None = None,
    va_csv_activity_col: str | None = None,
    va_n: int = 1000,
    nbh_radius: float | None = None,
    output_dir: str | Path = "results",
    s_threshold: float = 0.4,
    p_threshold: float = 0.5,
    svr_c: float = 10.0,
    svr_epsilon: float = 0.1,
    fragment_lib: str | Path | None = None,
    random_state: int = 42,
    svr_mode: Literal["legacy", "paper", "off"] = "legacy",
    external_actives_csv: str | Path | None = None,
    external_smiles_col: str = "smiles",
    external_activity_col: str = "pActivity",
    ea_train_fraction: float = 0.5,
    outer_folds: int = 3,
    inner_folds: int = 3,
    svr_c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    svr_epsilon_grid: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
) -> ComoResult:
    """Run the full COMO pipeline on an analog series.

    Decomposition is performed once and shared by all VA generators
    and the scoring protocol.

    Parameters
    ----------
    svr_mode:
        "legacy" (default): EA-only 3-fold CV.
        "paper": external actives + 50/50 EA split + nested 3-fold double CV.
            Requires external_actives_csv.
        "off": skip SVR.
    """
    from .analogs.close_in import CloseInVAGenerator
    from .analogs.diverse import DiverseVAGenerator
    from .analogs.free_wilson import FreeWilsonVAGenerator
    from .analogs.csv_plugin import CSVPluginVAGenerator
    from .series.decomposition import decompose_series

    # --- Load EA data ---
    if isinstance(series_csv, pl.DataFrame):
        df_raw = series_csv
    else:
        df_raw = pl.read_csv(series_csv)

    if smiles_col not in df_raw.columns or activity_col not in df_raw.columns:
        raise ValueError(
            f"Data must contain columns {smiles_col!r} and {activity_col!r}. "
            f"Found: {df_raw.columns}"
        )

    n_rows = len(df_raw)
    df = df_raw.drop_nulls(subset=[smiles_col, activity_col])
    ea_smiles = df[smiles_col].to_list()
    ea_activities = np.array(df[activity_col].to_list(), dtype=np.float64)

    if len(ea_smiles) < 3:
        raise ValueError(f"Need at least 3 EAs, got {len(ea_smiles)}.")

    # --- Validate paper SVR requirements ---
    if svr_mode == "paper" and external_actives_csv is None:
        raise ValueError(
            "--svr-mode paper requires --external-actives. "
            "Provide a CSV with external target actives, or use svr_mode='legacy'."
        )

    # --- Warn about auto-radius ---
    if nbh_radius is None:
        import sys
        print(
            "[COMO] Note: using auto NBH radius (adaptive k-NN median). "
            "For benchmark runs, supply an explicit --nbh-radius.",
            file=sys.stderr,
        )

    # --- Input diagnostics ---
    valid_mols = [Chem.MolFromSmiles(s) for s in ea_smiles]
    n_valid = sum(m is not None for m in valid_mols)
    unique_valid = {Chem.MolToSmiles(m) for m in valid_mols if m is not None}
    n_unique = len(unique_valid)

    # --- Auto-detect core ---
    if core is None:
        core = detect_murcko_core(ea_smiles)
        if core:
            print(f"[COMO] Auto-detected core: {core}")
        else:
            warnings.warn(
                "Could not auto-detect Murcko core. "
                "VA strategies requiring a core will be skipped."
            )

    if core is not None:
        core_mol = Chem.MolFromSmiles(core)
        if core_mol is None:
            core_mol = Chem.MolFromSmarts(core)
        if core_mol is not None:
            from rdkit.Chem import ReplaceCore
            from .analogs.close_in import _has_fused_match, _strip_exit_vectors
            match_core, _ = _strip_exit_vectors(core_mol)
            if match_core is None:
                match_core = core_mol
            n_with_core = sum(
                1 for s in unique_valid
                if (m := Chem.MolFromSmiles(s)) is not None
                and ReplaceCore(m, match_core) is not None
                and not _has_fused_match(m, match_core)
            )
        else:
            n_with_core = 0
    else:
        n_with_core = n_unique

    print(
        f"[COMO] Input: {n_rows} rows read  |  "
        f"{n_valid} valid SMILES  |  "
        f"{n_unique} unique  |  "
        f"{n_with_core} contain core"
    )

    # --- Single shared decomposition ---
    decomp = None
    if core is not None:
        decomp = decompose_series(
            core_smiles=core,
            ea_smiles=ea_smiles,
            ea_activities=ea_activities.tolist(),
        )

    # --- Generate VA populations ---
    generators: list = []
    strategy_set = set(va_strategies)

    if "close_in" in strategy_set:
        generators.append(CloseInVAGenerator(random_state=random_state))
    if "diverse" in strategy_set:
        generators.append(DiverseVAGenerator(fragment_lib_path=fragment_lib))
    if "free_wilson" in strategy_set:
        generators.append(FreeWilsonVAGenerator())
    if va_csv is not None:
        generators.append(CSVPluginVAGenerator(csv_path=va_csv, activity_col=va_csv_activity_col))

    va_smiles_by_strategy: dict[str, list[str]] = {}
    for gen in generators:
        if decomp is not None and isinstance(gen, CloseInVAGenerator):
            va_smiles_by_strategy[gen.strategy_name] = \
                gen.generate_from_decomposition(decomp, n=va_n)
        elif decomp is not None and isinstance(gen, FreeWilsonVAGenerator):
            va_smiles_by_strategy[gen.strategy_name] = \
                gen.generate_from_decomposition(decomp)
        else:
            # Diverse, csv_plugin, or no-core fallback
            ea_hac_range = decomp.ea_hac_range if decomp is not None else (0, 100)
            va_smiles_by_strategy[gen.strategy_name] = gen.generate(
                ea_smiles, ea_activities, core, va_n, ea_hac_range
            )

    # --- Build flat deduplicated VA list ---
    ea_set_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    for s in ea_smiles if Chem.MolFromSmiles(s) is not None}
    flat_va: list[str] = []
    seen_va: set[str] = set()
    for smi_list in va_smiles_by_strategy.values():
        for smi in smi_list:
            if smi not in seen_va and smi not in ea_set_canon:
                seen_va.add(smi)
                flat_va.append(smi)

    # --- Compute descriptors ---
    ea_raw, ea_valid_idx = compute_descriptors(ea_smiles)
    ea_acts_aligned = ea_activities[ea_valid_idx]

    if flat_va:
        va_raw, va_valid_idx = compute_descriptors(flat_va)
        flat_va_aligned = [flat_va[i] for i in va_valid_idx]
    else:
        va_raw = np.empty((0, 7), dtype=np.float64)
        va_valid_idx = []
        flat_va_aligned = []

    ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)
    membership, radius_used = build_nbh(ea_norm, va_norm, r=nbh_radius)

    C = c_score(membership)
    D, d_mean = d_score(membership)
    S = s_score(C, D)
    P = p_score(membership, ea_acts_aligned)
    stage = assign_stage(S, P, s_threshold, p_threshold)

    # --- SVR potency prediction ---
    cv_r2 = float("nan")
    cv_mae = float("nan")
    paper_svr_result = None

    if svr_mode == "off":
        va_pred = np.full(len(flat_va_aligned), np.nan)
    elif svr_mode == "paper":
        from .potency import PaperSVRPredictor
        ext_smiles_list, ext_acts_arr, n_ext_loaded, n_ext_valid, n_ext_excl = \
            _load_external_actives(
                external_actives_csv, external_smiles_col, external_activity_col,
                ea_set_canon,
            )
        paper_svr = PaperSVRPredictor(
            c_grid=list(svr_c_grid),
            epsilon_grid=list(svr_epsilon_grid),
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            ea_train_fraction=ea_train_fraction,
            random_state=random_state,
        )
        paper_svr_result = paper_svr.fit_paper_protocol(
            [ea_smiles[i] for i in ea_valid_idx], ea_acts_aligned,
            external_smiles=ext_smiles_list,
            external_activities=ext_acts_arr if ext_acts_arr is not None else None,
        )
        cv_r2 = paper_svr_result.cv_r2_mean
        cv_mae = paper_svr_result.cv_mae_mean
        va_pred, _ = paper_svr.predict_ensemble(flat_va_aligned)
    else:  # legacy
        svr = SVRPredictor(C=svr_c, epsilon=svr_epsilon)
        cv_metrics = svr.fit([ea_smiles[i] for i in ea_valid_idx], ea_acts_aligned)
        cv_r2 = cv_metrics["cv_r2"]
        cv_mae = cv_metrics["cv_mae"]
        va_pred = svr.predict(flat_va_aligned) if flat_va_aligned else np.array([], dtype=np.float64)

    # --- Collect FW and external predictions ---
    fw_predictions: dict[str, float] = {}
    external_predictions: dict[str, float] = {}
    fw_gen = None
    for gen in generators:
        if isinstance(gen, FreeWilsonVAGenerator):
            fw_predictions.update(gen.fw_predictions)
            fw_gen = gen
        elif isinstance(gen, CSVPluginVAGenerator):
            external_predictions.update(gen.external_activities)

    va_df = build_va_dataframe(
        va_smiles_by_strategy=va_smiles_by_strategy,
        flat_va_aligned=flat_va_aligned,
        svr_predictions=va_pred,
        ea_activities=ea_acts_aligned,
        membership=membership,
        fw_predictions=fw_predictions,
        external_predictions=external_predictions,
    )

    fw_n_candidates = len(va_smiles_by_strategy.get("free_wilson", []))
    fw_n_ea_in_nbh = fw_gen.n_ea_in_fw_nbh if fw_gen else 0
    fw_n_sites = fw_gen.n_sites if fw_gen else 0
    fw_n_unique_subs = fw_gen.n_unique_substituents if fw_gen else 0
    fw_pred_std = fw_gen.fw_pred_std if fw_gen else {}
    fw_pred_n_dict = fw_gen.fw_pred_n if fw_gen else {}
    fw_ea_predictions = fw_gen.fw_ea_predictions if fw_gen else {}

    fw_pct = f"{100 * fw_n_ea_in_nbh / n_with_core:.1f}%" if n_with_core > 0 else "N/A"
    print(
        f"[COMO] EAs: {n_with_core}  |  "
        f"EAs in FW NBHs: {fw_n_ea_in_nbh} ({fw_pct})  |  "
        f"FW sites: {fw_n_sites}  |  "
        f"Unique subs: {fw_n_unique_subs}"
    )

    result = ComoResult(
        C=C, D=D, d_mean=d_mean, S=S, P=P,
        stage=stage, radius_used=radius_used,
        n_ea=len(ea_smiles),
        ea_smiles=[ea_smiles[i] for i in ea_valid_idx],
        ea_activities=ea_acts_aligned,
        cv_r2=cv_r2, cv_mae=cv_mae,
        va_df=va_df,
        fw_n_candidates=fw_n_candidates,
        fw_n_ea_in_nbh=fw_n_ea_in_nbh,
        fw_n_sites=fw_n_sites,
        fw_n_unique_substituents=fw_n_unique_subs,
        fw_predictions=fw_predictions,
        fw_pred_std=fw_pred_std,
        fw_pred_n=fw_pred_n_dict,
        fw_ea_predictions=fw_ea_predictions,
        scaler=scaler,
        decomp=decomp,
        paper_svr_result=paper_svr_result,
        run_metadata={
            "random_state": random_state,
            "core_input": core,
            "va_strategies": list(va_strategies),
            "va_n": va_n,
            "nbh_radius": radius_used,
            "nbh_radius_mode": "explicit" if nbh_radius is not None else "auto",
            "svr_mode": svr_mode,
            "s_threshold": s_threshold,
            "p_threshold": p_threshold,
            "n_ea_input": n_rows,
        },
    )

    # --- Write outputs ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_scores_csv(result, output_path / "scores.csv")
    result.va_df.write_csv(output_path / "va_populations.csv")
    write_summary_txt(result, output_path / "summary.txt")
    write_fw_predictions_csv(result, output_path / "fw_predictions.csv")
    write_fw_candidates_csv(result, output_path / "fw_candidates.csv")
    write_fw_ea_validation_csv(result, output_path / "fw_ea_validation.csv")
    write_run_metadata_json(result, output_path / "run_metadata.json")
    if result.decomp is not None:
        write_decomposition_report(result, output_path / "decomposition_report.csv")
        write_decomposition_summary_json(result, output_path / "decomposition_summary.json")
    if result.paper_svr_result is not None:
        write_svr_outputs(result, output_path)

    print(
        f"[COMO] C={result.C:.3f}  D={result.D:.3f}  S={result.S:.3f}  "
        f"P={result.P:.3f}  stage={result.stage}"
    )
    print(f"[COMO] Results written to {output_path}/")
    return result


def _load_external_actives(
    csv_path,
    smiles_col: str,
    activity_col: str,
    ea_set_canon: set[str],
) -> tuple[list[str], np.ndarray | None, int, int, int]:
    """Load and filter external actives CSV.

    Returns (smiles, activities, n_loaded, n_valid, n_excluded_as_ea).
    """
    if csv_path is None:
        return [], None, 0, 0, 0

    df = pl.read_csv(csv_path)
    if smiles_col not in df.columns or activity_col not in df.columns:
        raise ValueError(
            f"External actives CSV must have columns {smiles_col!r} and {activity_col!r}. "
            f"Found: {df.columns}"
        )
    df = df.drop_nulls(subset=[smiles_col, activity_col])
    n_loaded = len(df)

    ext_smiles_raw = df[smiles_col].to_list()
    ext_acts_raw = np.array(df[activity_col].to_list(), dtype=np.float64)

    valid_smiles: list[str] = []
    valid_acts: list[float] = []
    n_invalid = 0
    n_excluded = 0

    for smi, act in zip(ext_smiles_raw, ext_acts_raw):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        if not np.isfinite(act):
            n_invalid += 1
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in ea_set_canon:
            n_excluded += 1
            continue
        valid_smiles.append(canon)
        valid_acts.append(float(act))

    n_valid = len(valid_smiles)
    acts_arr = np.array(valid_acts, dtype=np.float64) if valid_smiles else None
    return valid_smiles, acts_arr, n_loaded, n_valid, n_excluded


# ---------------------------------------------------------------------------
# score_with_repeats — paper-aligned diagnostic scoring protocol
# ---------------------------------------------------------------------------

def score_with_repeats(
    series_csv: str | Path | pl.DataFrame,
    smiles_col: str = "smiles",
    activity_col: str = "pActivity",
    core: str | None = None,
    n_va: int = 1000,
    n_repeats: int = 10,
    nbh_radius: float | None = None,
    random_state: int = 42,
    s_threshold: float = 0.4,
    p_threshold: float = 0.5,
) -> dict:
    """Run the COMO diagnostic scoring protocol with multiple random repeats.

    Generates n_va close-in VAs per repeat using H-aware random sampling and
    computes C, D, S, and P for each. Reports mean and standard deviation.

    Parameters
    ----------
    random_state:
        Base seed. Each repeat uses random_state + repeat_index.

    Returns
    -------
    dict with keys:
        'repeats': list of dicts, one per repeat (C, D, S, P, stage, ...)
        'C_mean', 'C_std', 'D_mean', 'D_std', 'S_mean', 'S_std',
        'P_mean', 'P_std'
        'settings': dict of all settings used
    """
    from .analogs.close_in import CloseInVAGenerator
    from .series.decomposition import decompose_series

    # --- Load data ---
    if isinstance(series_csv, pl.DataFrame):
        df_raw = series_csv
    else:
        df_raw = pl.read_csv(series_csv)

    df = df_raw.drop_nulls(subset=[smiles_col, activity_col])
    ea_smiles = df[smiles_col].to_list()
    ea_activities = np.array(df[activity_col].to_list(), dtype=np.float64)

    if len(ea_smiles) < 3:
        raise ValueError(f"Need at least 3 EAs, got {len(ea_smiles)}.")

    # Auto-detect core if not provided
    if core is None:
        core = detect_murcko_core(ea_smiles)
        if core:
            print(f"[COMO] Auto-detected core: {core}")

    # --- Shared decomposition ---
    decomp = None
    if core is not None:
        decomp = decompose_series(
            core_smiles=core,
            ea_smiles=ea_smiles,
            ea_activities=ea_activities.tolist(),
        )

    # Compute EA descriptors (same for all repeats)
    ea_raw, ea_valid_idx = compute_descriptors(ea_smiles)
    ea_activities_aligned = ea_activities[ea_valid_idx]

    # Fallback HAC range when no decomp
    ea_hacs = [
        m.GetNumHeavyAtoms()
        for s in ea_smiles
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    ea_hac_range_fallback = (min(ea_hacs) - 3, max(ea_hacs) + 3) if ea_hacs else (0, 100)

    repeat_results = []

    for rep in range(n_repeats):
        seed = random_state + rep
        gen = CloseInVAGenerator(random_state=seed)

        if decomp is not None:
            va_smiles = gen.generate_from_decomposition(decomp, n=n_va)
        else:
            va_smiles = gen.generate(
                ea_smiles, ea_activities, core, n_va, ea_hac_range_fallback
            )

        if not va_smiles:
            repeat_results.append({
                "repeat": rep, "seed": seed,
                "C": 0.0, "D": 0.0, "S": 0.0, "P": 0.0,
                "stage": "early", "n_va": 0,
                "n_ea": len(ea_smiles), "nbh_radius": float("nan"),
                "d_mean": 1.0, "n_covered_va": 0, "n_overlap_va": 0,
            })
            continue

        va_raw, va_valid_idx = compute_descriptors(va_smiles)
        ea_norm, va_norm, _ = normalize_descriptors(ea_raw, va_raw)

        membership, radius = build_nbh(ea_norm, va_norm, r=nbh_radius)

        C = c_score(membership)
        D, d_mean = d_score(membership)
        S = s_score(C, D)
        P = p_score(membership, ea_activities_aligned)
        stage = assign_stage(S, P, s_threshold, p_threshold)

        count_j = membership.sum(axis=1)
        n_covered = int((count_j > 0).sum())
        n_overlap = int((count_j >= 2).sum())

        repeat_results.append({
            "repeat": rep,
            "seed": seed,
            "C": C,
            "D": D,
            "S": S,
            "P": P,
            "stage": stage,
            "n_va": len(va_smiles),
            "n_ea": len(ea_smiles),
            "nbh_radius": radius,
            "d_mean": d_mean,
            "n_covered_va": n_covered,
            "n_overlap_va": n_overlap,
        })
        print(
            f"[COMO] Repeat {rep + 1:2d}/{n_repeats}  "
            f"C={C:.3f}  D={D:.3f}  S={S:.3f}  P={P:.3f}  "
            f"stage={stage}  n_va={len(va_smiles)}"
        )

    Cs = [r["C"] for r in repeat_results]
    Ds = [r["D"] for r in repeat_results]
    Ss = [r["S"] for r in repeat_results]
    Ps = [r["P"] for r in repeat_results]

    summary = {
        "repeats": repeat_results,
        "C_mean": float(np.mean(Cs)),
        "C_std": float(np.std(Cs)),
        "D_mean": float(np.mean(Ds)),
        "D_std": float(np.std(Ds)),
        "S_mean": float(np.mean(Ss)),
        "S_std": float(np.std(Ss)),
        "P_mean": float(np.mean(Ps)),
        "P_std": float(np.std(Ps)),
        "settings": {
            "n_va": n_va,
            "n_repeats": n_repeats,
            "nbh_radius": nbh_radius,
            "random_state": random_state,
            "s_threshold": s_threshold,
            "p_threshold": p_threshold,
        },
    }
    print(
        f"[COMO] Summary  "
        f"C={summary['C_mean']:.3f}±{summary['C_std']:.3f}  "
        f"D={summary['D_mean']:.3f}±{summary['D_std']:.3f}  "
        f"S={summary['S_mean']:.3f}±{summary['S_std']:.3f}  "
        f"P={summary['P_mean']:.3f}±{summary['P_std']:.3f}"
    )
    return summary

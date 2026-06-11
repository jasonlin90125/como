"""COMO scoring functions (Equations 1–6) and pipeline orchestration."""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .descriptors import compute_descriptors, normalize_descriptors
from .nbh import build_nbh
from .potency import SVRPredictor
from .report import ComoResult, build_va_dataframe, write_fw_predictions_csv, write_scores_csv, write_summary_txt

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
# ComoAnalysis — orchestrating class
# ---------------------------------------------------------------------------

class ComoAnalysis:
    """Orchestrates the full COMO pipeline.

    Holds state (scaler, SVR model, fitted NBH radius) for reuse.
    """

    def __init__(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        va_generators: list["VAGenerator"],
        nbh_radius: float | None = None,
        s_threshold: float = 0.4,
        p_threshold: float = 0.5,
        svr_c: float = 10.0,
        svr_epsilon: float = 0.1,
    ) -> None:
        self.ea_smiles = ea_smiles
        self.ea_activities = np.asarray(ea_activities, dtype=np.float64)
        self.va_generators = va_generators
        self.nbh_radius = nbh_radius
        self.s_threshold = s_threshold
        self.p_threshold = p_threshold
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon

    def run(self) -> ComoResult:
        """Execute the complete COMO pipeline and return a ComoResult."""
        ea_smiles = self.ea_smiles
        ea_activities = self.ea_activities

        # --- Compute EA heavy atom count range for VA size filtering ---
        ea_hacs = []
        for smi in ea_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ea_hacs.append(mol.GetNumHeavyAtoms())
        if ea_hacs:
            ea_hac_range = (min(ea_hacs) - 3, max(ea_hacs) + 3)
        else:
            ea_hac_range = (0, 100)

        # --- Detect core (may already be known via score_series) ---
        core_smiles = getattr(self, "_core_smiles", None)

        # --- Generate VA populations ---
        va_smiles_by_strategy: dict[str, list[str]] = {}
        for gen in self.va_generators:
            va_smiles_by_strategy[gen.strategy_name] = gen.generate(
                ea_smiles, ea_activities, core_smiles,
                getattr(self, "_va_n", 1000), ea_hac_range
            )

        # --- Build flat deduplicated VA list ---
        ea_set = set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ea_smiles
                     if Chem.MolFromSmiles(s) is not None)
        flat_va: list[str] = []
        seen: set[str] = set()
        for smi_list in va_smiles_by_strategy.values():
            for smi in smi_list:
                if smi not in seen and smi not in ea_set:
                    seen.add(smi)
                    flat_va.append(smi)

        # --- Compute descriptors ---
        ea_raw, ea_valid_idx = compute_descriptors(ea_smiles)
        ea_activities_aligned = ea_activities[ea_valid_idx]

        if flat_va:
            va_raw, va_valid_idx = compute_descriptors(flat_va)
            flat_va_aligned = [flat_va[i] for i in va_valid_idx]
        else:
            va_raw = np.empty((0, 7), dtype=np.float64)
            va_valid_idx = []
            flat_va_aligned = []

        # --- Normalize (fit on combined EA+VA population) ---
        ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)

        # --- Build NBH membership matrix ---
        membership, radius_used = build_nbh(ea_norm, va_norm, r=self.nbh_radius)

        # --- Compute scores ---
        C = c_score(membership)
        D, d_mean = d_score(membership)
        S = s_score(C, D)
        P = p_score(membership, ea_activities_aligned)
        stage = assign_stage(S, P, self.s_threshold, self.p_threshold)

        # --- SVR potency prediction ---
        svr = SVRPredictor(C=self.svr_c, epsilon=self.svr_epsilon)
        cv_metrics = svr.fit(
            [ea_smiles[i] for i in ea_valid_idx], ea_activities_aligned
        )
        if flat_va_aligned:
            va_pred = svr.predict(flat_va_aligned)
            va_pct = svr.percentile_rank(va_pred, ea_activities_aligned)
        else:
            va_pred = np.array([], dtype=np.float64)
            va_pct = np.array([], dtype=np.float64)

        # --- Collect FW and external predictions ---
        from .analogs.free_wilson import FreeWilsonVAGenerator
        from .analogs.csv_plugin import CSVPluginVAGenerator

        fw_predictions: dict[str, float] = {}
        external_predictions: dict[str, float] = {}
        for gen in self.va_generators:
            if isinstance(gen, FreeWilsonVAGenerator):
                fw_predictions.update(gen.fw_predictions)
            elif isinstance(gen, CSVPluginVAGenerator):
                external_predictions.update(gen.external_activities)

        # --- Build per-VA DataFrame ---
        va_df = build_va_dataframe(
            va_smiles_by_strategy=va_smiles_by_strategy,
            flat_va_aligned=flat_va_aligned,
            svr_predictions=va_pred,
            ea_activities=ea_activities_aligned,
            membership=membership,
            fw_predictions=fw_predictions,
            external_predictions=external_predictions,
        )

        # --- FW stats ---
        fw_n_candidates = sum(
            len(v) for k, v in va_smiles_by_strategy.items() if k == "free_wilson"
        )
        fw_gen = next(
            (g for g in self.va_generators if isinstance(g, FreeWilsonVAGenerator)),
            None,
        )
        fw_n_ea_in_nbh = fw_gen.n_ea_in_fw_nbh if fw_gen is not None else 0
        fw_n_sites = fw_gen.n_sites if fw_gen is not None else 0
        fw_n_unique_substituents = fw_gen.n_unique_substituents if fw_gen is not None else 0
        fw_pred_std = fw_gen.fw_pred_std if fw_gen is not None else {}
        fw_pred_n = fw_gen.fw_pred_n if fw_gen is not None else {}
        fw_ea_predictions = fw_gen.fw_ea_predictions if fw_gen is not None else {}

        n_ea_total = getattr(self, "_n_ea_with_core", len(ea_smiles))
        fw_pct = f"{100 * fw_n_ea_in_nbh / n_ea_total:.1f}%" if n_ea_total > 0 else "N/A"
        print(
            f"[COMO] EAs: {n_ea_total}  |  "
            f"EAs in FW NBHs: {fw_n_ea_in_nbh} ({fw_pct})  |  "
            f"FW sites: {fw_n_sites}  |  "
            f"Unique subs: {fw_n_unique_substituents}"
        )

        result = ComoResult(
            C=C,
            D=D,
            d_mean=d_mean,
            S=S,
            P=P,
            stage=stage,
            radius_used=radius_used,
            n_ea=len(ea_smiles),
            ea_smiles=[ea_smiles[i] for i in ea_valid_idx],
            ea_activities=ea_activities_aligned,
            cv_r2=cv_metrics["cv_r2"],
            cv_mae=cv_metrics["cv_mae"],
            va_df=va_df,
            fw_n_candidates=fw_n_candidates,
            fw_n_ea_in_nbh=fw_n_ea_in_nbh,
            fw_n_sites=fw_n_sites,
            fw_n_unique_substituents=fw_n_unique_substituents,
            fw_pred_std=fw_pred_std,
            fw_pred_n=fw_pred_n,
            fw_ea_predictions=fw_ea_predictions,
            scaler=scaler,
        )
        return result


# ---------------------------------------------------------------------------
# score_series — one-call public convenience API
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
    paper_mode: bool = False,
    random_state: int = 42,
) -> ComoResult:
    """Run the full COMO pipeline on an analog series.

    series_csv: path to a CSV file, or a Polars DataFrame already in memory.
    The data must have at minimum columns for SMILES and pActivity.
    Writes scores.csv, va_populations.csv, and summary.txt to output_dir.
    """
    from .analogs.close_in import CloseInVAGenerator
    from .analogs.diverse import DiverseVAGenerator
    from .analogs.free_wilson import FreeWilsonVAGenerator
    from .analogs.csv_plugin import CSVPluginVAGenerator

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

    # --- Input diagnostics ---
    # Count from df (non-null smiles + activity) so all numbers reflect the
    # same population that actually enters the pipeline.
    raw_smiles = df[smiles_col].to_list()
    valid_mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    n_valid = sum(m is not None for m in valid_mols)
    unique_valid = {Chem.MolToSmiles(m) for m in valid_mols if m is not None}
    n_unique = len(unique_valid)

    # --- Auto-detect core ---
    if core is None:
        core = detect_murcko_core(ea_smiles)
        if core:
            print(f"[COMO] Auto-detected core: {core}")
        else:
            warnings.warn("Could not auto-detect Murcko core. VA strategies requiring a core will be skipped.")

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

    # --- Build generators ---
    generators: list = []
    strategy_set = set(va_strategies)

    if "close_in" in strategy_set:
        generators.append(CloseInVAGenerator(paper_mode=paper_mode, random_state=random_state))
    if "diverse" in strategy_set:
        generators.append(DiverseVAGenerator(fragment_lib_path=fragment_lib))
    if "free_wilson" in strategy_set:
        generators.append(FreeWilsonVAGenerator())
    if va_csv is not None:
        generators.append(CSVPluginVAGenerator(csv_path=va_csv, activity_col=va_csv_activity_col))

    # --- Set up and run analysis ---
    analysis = ComoAnalysis(
        ea_smiles=ea_smiles,
        ea_activities=ea_activities,
        va_generators=generators,
        nbh_radius=nbh_radius,
        s_threshold=s_threshold,
        p_threshold=p_threshold,
        svr_c=svr_c,
        svr_epsilon=svr_epsilon,
    )
    analysis._core_smiles = core
    analysis._va_n = va_n
    analysis._n_ea_with_core = n_with_core

    result = analysis.run()

    # --- Write outputs ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_scores_csv(result, output_path / "scores.csv")
    result.va_df.write_csv(output_path / "va_populations.csv")
    write_summary_txt(result, output_path / "summary.txt")
    write_fw_predictions_csv(result, output_path / "fw_predictions.csv")

    print(
        f"[COMO] C={result.C:.3f}  D={result.D:.3f}  S={result.S:.3f}  "
        f"P={result.P:.3f}  stage={result.stage}"
    )
    print(f"[COMO] Results written to {output_path}/")
    return result


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
    paper_mode: bool = True,
) -> dict:
    """Run the COMO diagnostic scoring protocol with multiple random repeats.

    Generates n_va close-in VAs per repeat (paper-mode random sampling) and
    computes C, D, S, P for each.  Reports mean and std over n_repeats.

    Parameters
    ----------
    paper_mode:
        When True (default), uses CloseInVAGenerator(paper_mode=True) with
        H-aware random sampling.  When False, uses legacy deterministic mode
        (all repeats will be identical).
    random_state:
        Base seed.  Each repeat uses random_state + repeat_index.

    Returns
    -------
    dict with keys:
        'repeats': list of dicts, one per repeat (C, D, S, P, stage)
        'C_mean', 'C_std', 'D_mean', 'D_std', 'S_mean', 'S_std',
        'P_mean', 'P_std'
        'settings': dict of all settings used
    """
    from .analogs.close_in import CloseInVAGenerator

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

    # Compute EA HAC range for VA size filter
    ea_hacs = [
        m.GetNumHeavyAtoms()
        for s in ea_smiles
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    ea_hac_range = (min(ea_hacs) - 3, max(ea_hacs) + 3) if ea_hacs else (0, 100)

    # Compute EA descriptors (same for all repeats)
    ea_raw, ea_valid_idx = compute_descriptors(ea_smiles)
    ea_activities_aligned = ea_activities[ea_valid_idx]

    repeat_results = []

    for rep in range(n_repeats):
        seed = random_state + rep
        gen = CloseInVAGenerator(
            paper_mode=paper_mode,
            random_state=seed,
        )
        va_smiles = gen.generate(ea_smiles, ea_activities, core, n_va, ea_hac_range)

        if not va_smiles:
            repeat_results.append({"repeat": rep, "C": 0.0, "D": 0.0, "S": 0.0, "P": 0.0, "stage": "early", "n_va": 0})
            continue

        va_raw, va_valid_idx = compute_descriptors(va_smiles)
        _, va_norm, _ = normalize_descriptors(ea_raw, va_raw)
        ea_norm, _, _ = normalize_descriptors(ea_raw, va_raw)

        membership, _ = build_nbh(ea_norm, va_norm, r=nbh_radius)

        C = c_score(membership)
        D, _ = d_score(membership)
        S = s_score(C, D)
        P = p_score(membership, ea_activities_aligned)
        stage = assign_stage(S, P, s_threshold, p_threshold)

        repeat_results.append({
            "repeat": rep,
            "C": C,
            "D": D,
            "S": S,
            "P": P,
            "stage": stage,
            "n_va": len(va_smiles),
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
            "paper_mode": paper_mode,
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

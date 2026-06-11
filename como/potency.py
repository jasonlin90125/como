"""Potency prediction: SVR with ECFP4 fingerprints and Tanimoto kernel.

Provides two predictors:
  SVRPredictor       — legacy EA-only 3-fold CV (convenience baseline)
  PaperSVRPredictor  — paper-aligned: external actives + 50/50 EA split +
                       nested 3-fold double CV
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVR

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


def _smiles_to_fp(smi: str):
    """Return an RDKit ExplicitBitVect or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _morgan_gen.GetFingerprint(mol)


def _tanimoto_matrix(fps_a: list, fps_b: list) -> np.ndarray:
    """Compute Tanimoto similarity matrix of shape (len(fps_a), len(fps_b))."""
    mat = np.zeros((len(fps_a), len(fps_b)), dtype=np.float64)
    for i, fp in enumerate(fps_a):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_b)
        mat[i] = sims
    return mat


class SVRPredictor:
    """Global SVR model using ECFP4 fingerprints and a precomputed Tanimoto kernel.

    Mirrors the approach described in the DeepCOMO paper (ref [33]).
    """

    def __init__(self, C: float = 10.0, epsilon: float = 0.1) -> None:
        self.C = C
        self.epsilon = epsilon
        self._model: SVR | None = None
        self._train_fps: list = []

    def fit(self, smiles: list[str], activities: np.ndarray) -> dict[str, float]:
        """Train SVR on all provided SMILES and run 3-fold CV.

        Returns:
            dict with 'cv_r2' and 'cv_mae'
        """
        activities = np.asarray(activities, dtype=np.float64)
        fps = [_smiles_to_fp(s) for s in smiles]
        valid_mask = [fp is not None for fp in fps]
        fps_valid = [fp for fp in fps if fp is not None]
        acts_valid = activities[valid_mask]

        if len(fps_valid) < 4:
            warnings.warn("Too few valid molecules for SVR training (<4). Skipping CV.")
            self._train_fps = fps_valid
            if fps_valid:
                K = _tanimoto_matrix(fps_valid, fps_valid)
                self._model = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
                self._model.fit(K, acts_valid)
            return {"cv_r2": float("nan"), "cv_mae": float("nan")}

        # 3-fold cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        r2_scores: list[float] = []
        mae_scores: list[float] = []

        for train_idx, test_idx in kf.split(fps_valid):
            fps_train = [fps_valid[i] for i in train_idx]
            fps_test = [fps_valid[i] for i in test_idx]
            acts_train = acts_valid[train_idx]
            acts_test = acts_valid[test_idx]

            K_train = _tanimoto_matrix(fps_train, fps_train)
            K_test = _tanimoto_matrix(fps_test, fps_train)

            svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
            svr.fit(K_train, acts_train)
            preds = svr.predict(K_test)

            ss_res = np.sum((acts_test - preds) ** 2)
            ss_tot = np.sum((acts_test - acts_test.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(acts_test - preds)))
            r2_scores.append(r2)
            mae_scores.append(mae)

        # Retrain on all data
        K_full = _tanimoto_matrix(fps_valid, fps_valid)
        self._model = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        self._model.fit(K_full, acts_valid)
        self._train_fps = fps_valid

        return {
            "cv_r2": float(np.nanmean(r2_scores)),
            "cv_mae": float(np.nanmean(mae_scores)),
        }

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Predict pActivity for a list of SMILES.

        Invalid SMILES receive np.nan. Columns of K_test align with training order.
        """
        if self._model is None or not self._train_fps:
            return np.full(len(smiles), np.nan)

        fps = [_smiles_to_fp(s) for s in smiles]
        preds = np.full(len(smiles), np.nan)

        valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
        fps_valid = [fps[i] for i in valid_idx]

        if not fps_valid:
            return preds

        K_test = _tanimoto_matrix(fps_valid, self._train_fps)
        pred_valid = self._model.predict(K_test)
        for i, idx in enumerate(valid_idx):
            preds[idx] = pred_valid[i]

        return preds

    def percentile_rank(
        self,
        va_predictions: np.ndarray,
        ea_activities: np.ndarray,
    ) -> np.ndarray:
        """Return % of EA activities below each VA prediction (0–100)."""
        result = np.full(len(va_predictions), np.nan)
        for i, pred in enumerate(va_predictions):
            if not np.isnan(pred):
                result[i] = 100.0 * float((ea_activities < pred).mean())
        return result


# ---------------------------------------------------------------------------
# PaperSVRPredictor — paper-aligned nested CV + external actives
# ---------------------------------------------------------------------------

@dataclass
class OuterFoldResult:
    fold: int
    best_C: float
    best_epsilon: float
    outer_r2: float
    outer_mae: float
    outer_rmse: float
    n_train: int
    n_test: int


@dataclass
class PaperSVRResult:
    """Full output from PaperSVRPredictor.fit_paper_protocol."""
    outer_fold_results: list[OuterFoldResult]
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    cv_mae_std: float
    # External validation EA predictions {canonical_smiles: (observed, pred_mean, pred_std)}
    external_validation: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Models from each outer fold (for ensemble prediction)
    _models: list[SVR] = field(default_factory=list, repr=False)
    _train_fps_per_fold: list[list] = field(default_factory=list, repr=False)


class PaperSVRPredictor:
    """Paper-aligned SVR potency predictor.

    Protocol:
    1. Split EAs 50/50 into train half and external validation half
       (optional stratification by potency bins).
    2. Combine external target actives + EA train half into training pool.
    3. Run 3-fold double nested CV on training pool:
       - Outer fold: choose best hyperparameters via inner 3-fold CV.
       - Keep the 3 outer models for ensemble prediction.
    4. Evaluate the 3-model ensemble on external validation EAs.

    Parameters
    ----------
    c_grid, epsilon_grid:
        Hyperparameter search grids.
    outer_folds, inner_folds:
        Number of folds for outer/inner CV.
    ea_train_fraction:
        Fraction of EAs in the training pool (paper: 0.5).
    random_state:
        Seed for all splits.
    allow_ea_only:
        When True, fit using EA data only even if no external actives are
        provided. When False (default), raise if no external actives given.
    stratify:
        Bin EAs by potency for stratified splitting.
    """

    def __init__(
        self,
        c_grid: list[float] | None = None,
        epsilon_grid: list[float] | None = None,
        outer_folds: int = 3,
        inner_folds: int = 3,
        ea_train_fraction: float = 0.5,
        random_state: int = 42,
        allow_ea_only: bool = False,
        stratify: bool = True,
    ) -> None:
        self.c_grid = c_grid or [0.1, 1.0, 10.0, 100.0]
        self.epsilon_grid = epsilon_grid or [0.01, 0.1, 0.5]
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.ea_train_fraction = ea_train_fraction
        self.random_state = random_state
        self.allow_ea_only = allow_ea_only
        self.stratify = stratify
        self._result: PaperSVRResult | None = None

    def fit_paper_protocol(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        external_smiles: list[str] | None = None,
        external_activities: np.ndarray | None = None,
    ) -> PaperSVRResult:
        """Run paper-aligned fitting protocol.

        Parameters
        ----------
        ea_smiles, ea_activities:
            The EA series.
        external_smiles, external_activities:
            Non-EA target actives from ChEMBL (or similar).  Required unless
            allow_ea_only=True.
        """
        if (external_smiles is None or len(external_smiles) == 0) and not self.allow_ea_only:
            raise ValueError(
                "PaperSVRPredictor requires external target actives unless "
                "allow_ea_only=True is set explicitly."
            )

        ea_activities = np.asarray(ea_activities, dtype=np.float64)

        # --- Compute valid EA fps ---
        ea_fps = [_smiles_to_fp(s) for s in ea_smiles]
        ea_valid_mask = np.array([fp is not None for fp in ea_fps])
        ea_fps_valid = [fp for fp in ea_fps if fp is not None]
        ea_acts_valid = ea_activities[ea_valid_mask]
        ea_smiles_valid = [ea_smiles[i] for i, m in enumerate(ea_valid_mask) if m]

        n_ea = len(ea_fps_valid)
        if n_ea < 4:
            raise ValueError(f"Need at least 4 valid EAs, got {n_ea}.")

        # --- Split EAs 50/50 ---
        rng = np.random.default_rng(self.random_state)
        n_train_ea = max(1, int(round(n_ea * self.ea_train_fraction)))

        if self.stratify and n_ea >= 6:
            n_bins = min(4, n_ea // 2)
            bins = np.quantile(ea_acts_valid, np.linspace(0, 1, n_bins + 1))
            bin_labels = np.digitize(ea_acts_valid, bins[1:-1])
            # Stratified split: pick proportional counts from each bin
            train_idx: list[int] = []
            val_idx: list[int] = []
            for b in np.unique(bin_labels):
                b_idx = np.where(bin_labels == b)[0]
                rng.shuffle(b_idx)
                n_b_train = max(1, round(len(b_idx) * self.ea_train_fraction))
                train_idx.extend(b_idx[:n_b_train].tolist())
                val_idx.extend(b_idx[n_b_train:].tolist())
        else:
            perm = rng.permutation(n_ea)
            train_idx = perm[:n_train_ea].tolist()
            val_idx = perm[n_train_ea:].tolist()

        ea_train_fps = [ea_fps_valid[i] for i in train_idx]
        ea_train_acts = ea_acts_valid[train_idx]
        ea_val_fps = [ea_fps_valid[i] for i in val_idx]
        ea_val_acts = ea_acts_valid[val_idx]
        ea_val_smiles = [ea_smiles_valid[i] for i in val_idx]

        # --- Build training pool = external actives + EA train half ---
        pool_fps: list = list(ea_train_fps)
        pool_acts: list = list(ea_train_acts)

        if external_smiles:
            ext_acts = np.asarray(external_activities or [], dtype=np.float64)
            for smi, act in zip(external_smiles, ext_acts):
                fp = _smiles_to_fp(smi)
                if fp is not None:
                    pool_fps.append(fp)
                    pool_acts.append(float(act))

        pool_acts_arr = np.array(pool_acts)
        n_pool = len(pool_fps)

        if n_pool < self.outer_folds:
            warnings.warn(
                f"Training pool ({n_pool} molecules) smaller than outer_folds "
                f"({self.outer_folds}). Reducing folds."
            )
            self.outer_folds = max(2, n_pool)

        # --- Nested 3-fold double CV ---
        outer_kf = KFold(n_splits=self.outer_folds, shuffle=True,
                         random_state=self.random_state)
        outer_results: list[OuterFoldResult] = []
        outer_models: list[SVR] = []
        outer_train_fps_list: list[list] = []

        for fold_idx, (outer_tr, outer_te) in enumerate(outer_kf.split(pool_fps)):
            fps_outer_tr = [pool_fps[i] for i in outer_tr]
            acts_outer_tr = pool_acts_arr[outer_tr]
            fps_outer_te = [pool_fps[i] for i in outer_te]
            acts_outer_te = pool_acts_arr[outer_te]

            if len(fps_outer_tr) < self.inner_folds:
                # Skip inner CV, use defaults
                best_C, best_eps = self.c_grid[-1], self.epsilon_grid[0]
            else:
                # Inner CV to find best hyperparameters
                inner_kf = KFold(n_splits=self.inner_folds, shuffle=True,
                                  random_state=self.random_state + fold_idx)
                best_C, best_eps = self._inner_cv(
                    fps_outer_tr, acts_outer_tr, inner_kf
                )

            K_tr = _tanimoto_matrix(fps_outer_tr, fps_outer_tr)
            K_te = _tanimoto_matrix(fps_outer_te, fps_outer_tr)
            model = SVR(kernel="precomputed", C=best_C, epsilon=best_eps)
            model.fit(K_tr, acts_outer_tr)
            preds_te = model.predict(K_te)

            ss_res = np.sum((acts_outer_te - preds_te) ** 2)
            ss_tot = np.sum((acts_outer_te - acts_outer_te.mean()) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(acts_outer_te - preds_te)))
            rmse = float(np.sqrt(np.mean((acts_outer_te - preds_te) ** 2)))

            outer_results.append(OuterFoldResult(
                fold=fold_idx,
                best_C=best_C,
                best_epsilon=best_eps,
                outer_r2=r2,
                outer_mae=mae,
                outer_rmse=rmse,
                n_train=len(fps_outer_tr),
                n_test=len(fps_outer_te),
            ))
            outer_models.append(model)
            outer_train_fps_list.append(fps_outer_tr)

        r2_vals = [r.outer_r2 for r in outer_results if not np.isnan(r.outer_r2)]
        mae_vals = [r.outer_mae for r in outer_results]

        # --- External validation ---
        ext_val: dict[str, tuple[float, float, float]] = {}
        if ea_val_fps:
            preds_per_fold = []
            for model, tr_fps in zip(outer_models, outer_train_fps_list):
                K_val = _tanimoto_matrix(ea_val_fps, tr_fps)
                preds_per_fold.append(model.predict(K_val))
            preds_arr = np.stack(preds_per_fold, axis=0)  # (n_folds, n_val)
            pred_means = preds_arr.mean(axis=0)
            pred_stds = preds_arr.std(axis=0)
            for i, (smi, obs, pm, ps) in enumerate(zip(
                ea_val_smiles, ea_val_acts, pred_means, pred_stds
            )):
                canon = _canon(smi)
                if canon:
                    ext_val[canon] = (float(obs), float(pm), float(ps))

        result = PaperSVRResult(
            outer_fold_results=outer_results,
            cv_r2_mean=float(np.mean(r2_vals)) if r2_vals else float("nan"),
            cv_r2_std=float(np.std(r2_vals)) if len(r2_vals) > 1 else 0.0,
            cv_mae_mean=float(np.mean(mae_vals)) if mae_vals else float("nan"),
            cv_mae_std=float(np.std(mae_vals)) if len(mae_vals) > 1 else 0.0,
            external_validation=ext_val,
            _models=outer_models,
            _train_fps_per_fold=outer_train_fps_list,
        )
        self._result = result
        return result

    def predict_ensemble(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the ensemble of outer-fold models.

        Returns (pred_mean, pred_std) arrays aligned with smiles.
        """
        if self._result is None or not self._result._models:
            nan = np.full(len(smiles), np.nan)
            return nan, nan

        fps = [_smiles_to_fp(s) for s in smiles]
        valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
        fps_valid = [fps[i] for i in valid_idx]

        preds_per_fold = []
        for model, tr_fps in zip(
            self._result._models, self._result._train_fps_per_fold
        ):
            K = _tanimoto_matrix(fps_valid, tr_fps)
            preds_per_fold.append(model.predict(K))

        preds_arr = np.stack(preds_per_fold, axis=0)
        means = preds_arr.mean(axis=0)
        stds = preds_arr.std(axis=0)

        out_mean = np.full(len(smiles), np.nan)
        out_std = np.full(len(smiles), np.nan)
        for i, idx in enumerate(valid_idx):
            out_mean[idx] = means[i]
            out_std[idx] = stds[i]
        return out_mean, out_std

    def _inner_cv(
        self,
        fps: list,
        acts: np.ndarray,
        inner_kf: KFold,
    ) -> tuple[float, float]:
        """Grid search over c_grid × epsilon_grid, return best (C, epsilon)."""
        best_C, best_eps, best_mae = self.c_grid[0], self.epsilon_grid[0], float("inf")
        for C in self.c_grid:
            for eps in self.epsilon_grid:
                maes = []
                for tr_idx, val_idx in inner_kf.split(fps):
                    fps_tr = [fps[i] for i in tr_idx]
                    fps_val = [fps[i] for i in val_idx]
                    K_tr = _tanimoto_matrix(fps_tr, fps_tr)
                    K_val = _tanimoto_matrix(fps_val, fps_tr)
                    m = SVR(kernel="precomputed", C=C, epsilon=eps)
                    m.fit(K_tr, acts[tr_idx])
                    preds = m.predict(K_val)
                    maes.append(float(np.mean(np.abs(acts[val_idx] - preds))))
                mean_mae = float(np.mean(maes))
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_C, best_eps = C, eps
        return best_C, best_eps


def _canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None

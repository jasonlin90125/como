"""Potency prediction: SVR with ECFP4 fingerprints and Tanimoto kernel."""

from __future__ import annotations

import warnings

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import KFold
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

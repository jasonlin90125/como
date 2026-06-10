"""Unit tests for SVRPredictor."""

import numpy as np
import pytest
from como.potency import SVRPredictor, _tanimoto_matrix, _smiles_to_fp


SMILES = [
    "c1ccccc1",
    "CC(=O)O",
    "c1ccncc1",
    "c1ccc(F)cc1",
    "CCO",
    "c1ccc(Cl)cc1",
]
ACTS = np.array([5.0, 6.0, 5.5, 6.5, 4.5, 6.2])


def test_smiles_to_fp_valid():
    fp = _smiles_to_fp("c1ccccc1")
    assert fp is not None


def test_smiles_to_fp_invalid():
    fp = _smiles_to_fp("NOT_VALID")
    assert fp is None


def test_tanimoto_diagonal_is_one():
    fps = [_smiles_to_fp(s) for s in SMILES[:4]]
    mat = _tanimoto_matrix(fps, fps)
    assert mat.shape == (4, 4)
    assert np.allclose(np.diag(mat), 1.0)


def test_tanimoto_symmetric():
    fps = [_smiles_to_fp(s) for s in SMILES[:4]]
    mat = _tanimoto_matrix(fps, fps)
    assert np.allclose(mat, mat.T)


def test_tanimoto_range():
    fps = [_smiles_to_fp(s) for s in SMILES]
    mat = _tanimoto_matrix(fps, fps)
    assert mat.min() >= 0.0
    assert mat.max() <= 1.0


def test_svr_fit_returns_cv_metrics():
    svr = SVRPredictor()
    metrics = svr.fit(SMILES, ACTS)
    assert "cv_r2" in metrics
    assert "cv_mae" in metrics
    assert isinstance(metrics["cv_r2"], float)
    assert isinstance(metrics["cv_mae"], float)


def test_svr_predict_shape():
    svr = SVRPredictor()
    svr.fit(SMILES, ACTS)
    preds = svr.predict(SMILES[:3])
    assert preds.shape == (3,)


def test_svr_nan_for_invalid():
    svr = SVRPredictor()
    svr.fit(SMILES, ACTS)
    preds = svr.predict(["NOT_VALID", "c1ccccc1"])
    assert np.isnan(preds[0])
    assert not np.isnan(preds[1])


def test_percentile_rank_range():
    svr = SVRPredictor()
    svr.fit(SMILES, ACTS)
    va_preds = svr.predict(SMILES)
    pct = svr.percentile_rank(va_preds, ACTS)
    valid = pct[~np.isnan(pct)]
    assert np.all(valid >= 0)
    assert np.all(valid <= 100)


def test_percentile_rank_at_median():
    svr = SVRPredictor()
    svr.fit(SMILES, ACTS)
    # A prediction exactly at the EA median should have ≈50th percentile
    median = float(np.median(ACTS))
    pct = svr.percentile_rank(np.array([median]), ACTS)
    assert pct[0] == pytest.approx(50.0, abs=20.0)  # loose tolerance

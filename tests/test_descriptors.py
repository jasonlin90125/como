import numpy as np
import pytest
from como.descriptors import compute_descriptors, normalize_descriptors, DESCRIPTOR_NAMES


GEFITINIB_LIKE = "COc1cc2ncnc(Nc3ccc(F)cc3Cl)c2cc1OCCCN1CCOCC1"
SIMPLE_SMILES = ["c1ccccc1", "CC(=O)O", "c1ccncc1"]


def test_descriptor_names_count():
    assert len(DESCRIPTOR_NAMES) == 7


def test_compute_seven_descriptors():
    mat, valid = compute_descriptors([GEFITINIB_LIKE])
    assert len(valid) == 1
    assert mat.shape == (1, 7)
    # All values should be finite and positive (or zero)
    assert np.all(np.isfinite(mat))
    assert mat[0, 0] > 100  # MW > 100 for gefitinib-like


def test_all_descriptors_in_range():
    mat, valid = compute_descriptors(SIMPLE_SMILES)
    assert len(valid) == 3
    mw, logp, tpsa, hbd, hba, rotb, rings = mat.T
    assert np.all(mw > 0)
    assert np.all(tpsa >= 0)
    assert np.all(hbd >= 0)
    assert np.all(hba >= 0)
    assert np.all(rotb >= 0)
    assert np.all(rings >= 0)


def test_invalid_smiles_dropped():
    smiles = ["c1ccccc1", "NOT_A_SMILES", "CC(=O)O"]
    mat, valid = compute_descriptors(smiles)
    assert valid == [0, 2]
    assert mat.shape == (2, 7)


def test_all_invalid_returns_empty():
    mat, valid = compute_descriptors(["INVALID1", "INVALID2"])
    assert valid == []
    assert mat.shape == (0, 7)


def test_normalization_combined_mean_zero():
    ea_raw, _ = compute_descriptors(SIMPLE_SMILES)
    va_raw, _ = compute_descriptors(["c1ccc(F)cc1", "CCCO"])
    ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)
    combined = np.vstack([ea_norm, va_norm])
    # Combined mean should be ≈ 0 and std ≈ 1
    assert np.allclose(combined.mean(axis=0), 0, atol=1e-10)


def test_normalization_splits_correctly():
    ea_raw, _ = compute_descriptors(SIMPLE_SMILES)
    va_smiles = ["c1ccc(F)cc1", "CCCO", "c1ccncc1"]
    va_raw, _ = compute_descriptors(va_smiles)
    ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)
    assert ea_norm.shape[0] == len(SIMPLE_SMILES)
    assert va_norm.shape[0] == len(va_smiles)


def test_normalization_empty_va():
    ea_raw, _ = compute_descriptors(SIMPLE_SMILES)
    va_raw = np.empty((0, 7))
    ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)
    assert va_norm.shape[0] == 0
    assert ea_norm.shape[0] == len(SIMPLE_SMILES)

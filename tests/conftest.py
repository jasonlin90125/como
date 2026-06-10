"""Pytest fixtures for COMO tests."""

import csv
from pathlib import Path

import numpy as np
import pytest


# Quinazoline core with 2 substitution sites (R1 = 6-position, R2 = 4-anilino)
# These are simplified quinazoline analogs representative of EGFR inhibitors
_EGFR_EAS = [
    # (smiles, pIC50)
    ("COc1cc2ncnc(Nc3ccccc3)c2cc1OC", 7.2),
    ("COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OC", 7.8),
    ("COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC", 7.5),
    ("CCOc1cc2ncnc(Nc3ccccc3)c2cc1OCC", 6.9),
    ("CCOc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCC", 7.4),
    ("CCOc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OCC", 7.1),
    ("Cc1cc2ncnc(Nc3ccccc3)c2cc1C", 6.5),
    ("Cc1cc2ncnc(Nc3ccc(F)cc3)c2cc1C", 7.0),
    ("Cc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1C", 6.7),
]


@pytest.fixture
def synthetic_egfr_series():
    smiles = [row[0] for row in _EGFR_EAS]
    activities = np.array([row[1] for row in _EGFR_EAS])
    core = "c1cc2ncncc2cc1"  # quinazoline scaffold
    return {"smiles": smiles, "activities": activities, "core": core}


# Minimal series: 5 EAs on a simple phenylacetic acid scaffold
_MINIMAL_EAS = [
    ("c1ccc(CC(=O)O)cc1", 5.5),
    ("c1ccc(CC(=O)O)c(F)c1", 6.0),
    ("c1ccc(CC(=O)O)c(Cl)c1", 6.3),
    ("c1ccc(CC(=O)O)c(C)c1", 5.8),
    ("c1ccc(CC(=O)O)c(OC)c1", 6.1),
]


@pytest.fixture
def minimal_series():
    smiles = [row[0] for row in _MINIMAL_EAS]
    activities = np.array([row[1] for row in _MINIMAL_EAS])
    return {"smiles": smiles, "activities": activities}


@pytest.fixture
def synthetic_egfr_csv(tmp_path) -> Path:
    """Write synthetic EGFR series to a CSV file and return the path."""
    csv_path = tmp_path / "synthetic_egfr.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "pActivity"])
        writer.writerows(_EGFR_EAS)
    return csv_path


@pytest.fixture
def minimal_csv(tmp_path) -> Path:
    csv_path = tmp_path / "minimal.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "pActivity"])
        writer.writerows(_MINIMAL_EAS)
    return csv_path

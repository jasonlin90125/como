"""Tests for score_with_repeats() diagnostic scoring protocol."""

import numpy as np
import pytest

from como.scoring import score_with_repeats


_EGFR_EAS = [
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
def egfr_csv(tmp_path):
    import csv
    p = tmp_path / "egfr.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "pActivity"])
        writer.writerows(_EGFR_EAS)
    return p


class TestScoreWithRepeats:
    def test_returns_dict_with_required_keys(self, egfr_csv):
        result = score_with_repeats(
            egfr_csv, n_va=50, n_repeats=3, random_state=42
        )
        for key in ("repeats", "C_mean", "C_std", "D_mean", "D_std",
                    "S_mean", "S_std", "P_mean", "P_std", "settings"):
            assert key in result, f"Missing key: {key}"

    def test_repeat_count(self, egfr_csv):
        result = score_with_repeats(
            egfr_csv, n_va=30, n_repeats=5, random_state=42
        )
        assert len(result["repeats"]) == 5

    def test_scores_in_range(self, egfr_csv):
        result = score_with_repeats(
            egfr_csv, n_va=50, n_repeats=3, random_state=42
        )
        for r in result["repeats"]:
            for score in ("C", "D", "S"):
                assert 0.0 <= r[score] <= 1.0, f"{score}={r[score]} out of [0,1]"
            assert np.isfinite(r["P"]) or r["P"] == 0.0
            assert r["P"] >= 0.0, f"P={r['P']} is negative"

    def test_reproducible(self, egfr_csv):
        r1 = score_with_repeats(egfr_csv, n_va=30, n_repeats=3, random_state=7)
        r2 = score_with_repeats(egfr_csv, n_va=30, n_repeats=3, random_state=7)
        assert r1["C_mean"] == pytest.approx(r2["C_mean"], abs=1e-9)
        assert r1["S_mean"] == pytest.approx(r2["S_mean"], abs=1e-9)

    def test_different_seeds_differ(self, egfr_csv):
        r1 = score_with_repeats(egfr_csv, n_va=50, n_repeats=4, random_state=1)
        r2 = score_with_repeats(egfr_csv, n_va=50, n_repeats=4, random_state=99)
        assert len(r1["repeats"]) == len(r2["repeats"])

    def test_settings_recorded(self, egfr_csv):
        result = score_with_repeats(
            egfr_csv, n_va=100, n_repeats=2, random_state=42
        )
        s = result["settings"]
        assert s["n_va"] == 100
        assert s["n_repeats"] == 2
        assert s["random_state"] == 42

    def test_stage_labels_valid(self, egfr_csv):
        result = score_with_repeats(
            egfr_csv, n_va=30, n_repeats=3, random_state=42
        )
        valid_stages = {"early", "early_mid", "mid", "late"}
        for r in result["repeats"]:
            assert r["stage"] in valid_stages

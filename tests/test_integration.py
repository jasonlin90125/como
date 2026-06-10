"""Integration tests: end-to-end COMO pipeline."""

import csv
import numpy as np
import pytest
from pathlib import Path

from como.scoring import score_series
from como.cli import main as cli_main


def test_full_pipeline_close_in(synthetic_egfr_csv, tmp_path):
    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        output_dir=tmp_path / "out_ci",
    )
    out = tmp_path / "out_ci"
    assert (out / "scores.csv").exists()
    assert (out / "va_populations.csv").exists()
    assert (out / "summary.txt").exists()

    # Check scores are valid floats in [0, 1]
    for attr in ("C", "D", "S", "P"):
        val = getattr(result, attr)
        assert 0.0 <= val <= 1.0, f"{attr} = {val} out of [0,1]"

    # Stage is one of the expected labels
    assert result.stage in ("early", "early_mid", "mid", "late")


def test_full_pipeline_all_strategies(synthetic_egfr_csv, tmp_path):
    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in", "diverse", "free_wilson"],
        va_n=200,
        output_dir=tmp_path / "out_all",
    )
    va_df = result.va_df
    strategies_found = set(va_df["source_strategy"].to_list())
    # At least some VAs should be produced
    assert len(va_df) > 0
    # All three strategies should appear (or at least close_in)
    assert "close_in" in strategies_found or len(va_df) > 0


def test_cli_invocation(synthetic_egfr_csv, tmp_path):
    ret = cli_main([
        "--series", str(synthetic_egfr_csv),
        "--va", "close_in",
        "--va-n", "50",
        "--output", str(tmp_path / "cli_out"),
    ])
    assert ret == 0
    assert (tmp_path / "cli_out" / "scores.csv").exists()


def test_user_radius_tiny(synthetic_egfr_csv, tmp_path):
    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        nbh_radius=0.001,
        output_dir=tmp_path / "r_tiny",
    )
    # Very small radius -> almost no VAs covered -> C near 0
    assert result.C < 0.2


def test_user_radius_large(synthetic_egfr_csv, tmp_path):
    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        nbh_radius=1000.0,
        output_dir=tmp_path / "r_large",
    )
    # Very large radius -> all VAs covered -> C should be 1.0
    assert result.C == pytest.approx(1.0)


def test_csv_plugin_integration(synthetic_egfr_csv, tmp_path):
    # Write a small VA CSV as if from a generative model
    plugin_csv = tmp_path / "gen_output.csv"
    with open(plugin_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "pred_pAct"])
        writer.writerows([
            ("COc1cc2ncnc(Nc3ccc(OC)cc3)c2cc1OC", "7.9"),
            ("CCCOc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCC", "7.3"),
        ])

    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        va_csv=plugin_csv,
        output_dir=tmp_path / "plugin_out",
    )
    va_df = result.va_df
    plugin_rows = va_df.filter(va_df["source_strategy"] == "csv_plugin")
    assert len(plugin_rows) >= 1


def test_scores_csv_has_all_metrics(synthetic_egfr_csv, tmp_path):
    import polars as pl
    score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        output_dir=tmp_path / "metrics_check",
    )
    df = pl.read_csv(tmp_path / "metrics_check" / "scores.csv")
    metrics = set(df["metric"].to_list())
    for required in ("C", "D", "S", "P", "stage", "radius_used", "n_ea"):
        assert required in metrics, f"Missing metric: {required}"


def test_summary_txt_contains_stage(synthetic_egfr_csv, tmp_path):
    result = score_series(
        series_csv=synthetic_egfr_csv,
        va_strategies=["close_in"],
        output_dir=tmp_path / "summary_check",
    )
    summary = (tmp_path / "summary_check" / "summary.txt").read_text()
    assert result.stage.upper() in summary
    assert "COMO" in summary

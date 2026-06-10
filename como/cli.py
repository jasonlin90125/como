"""Command-line interface for COMO analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="como",
        description="COMO/DeepCOMO: Compound Optimization Monitor for analog series analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m como --series series.csv --va close_in diverse --output results/
  python -m como --series series.csv --va free_wilson --core "c1ccc2c(c1)NC(=O)N2"
  python -m como --series series.csv --va-csv generated.csv --output results/
""",
    )
    parser.add_argument(
        "--series", required=True,
        help="CSV file with SMILES and pActivity columns (existing analogs).",
    )
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name [default: smiles]")
    parser.add_argument("--activity-col", default="pActivity", help="pActivity column name [default: pActivity]")
    parser.add_argument(
        "--core", default=None,
        help="Scaffold SMILES for VA generation. Auto-detected from Murcko scaffolds if omitted.",
    )
    parser.add_argument(
        "--va", nargs="+",
        choices=["close_in", "diverse", "free_wilson"],
        default=["close_in"],
        help="VA generation strategies [default: close_in].",
    )
    parser.add_argument(
        "--va-csv", default=None,
        help="CSV from generative model (plug-in hook). Must have a 'smiles' column.",
    )
    parser.add_argument(
        "--va-csv-activity-col", default=None,
        help="Column in --va-csv with predicted pActivity from the generative model. "
             "Stored as external_pred_pActivity in va_populations.csv.",
    )
    parser.add_argument("--va-n", type=int, default=1000, help="Max VAs per strategy [default: 1000]")
    parser.add_argument(
        "--nbh-radius", default="auto",
        help="NBH radius in normalized descriptor space. 'auto' uses adaptive k-NN median [default: auto].",
    )
    parser.add_argument("--output", default="results", help="Output directory [default: results]")
    parser.add_argument(
        "--fragment-lib", default=None,
        help="Custom fragment SMILES file for diverse VA strategy (one SMILES per line).",
    )
    parser.add_argument("--s-threshold", type=float, default=0.4, help="S score threshold for stage assignment [default: 0.4]")
    parser.add_argument("--p-threshold", type=float, default=0.5, help="P score threshold for stage assignment [default: 0.5]")
    parser.add_argument("--svr-c", type=float, default=10.0, help="SVR regularization parameter C [default: 10.0]")
    parser.add_argument("--svr-epsilon", type=float, default=0.1, help="SVR epsilon [default: 0.1]")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Parse nbh_radius
    if args.nbh_radius == "auto":
        nbh_radius = None
    else:
        try:
            nbh_radius = float(args.nbh_radius)
        except ValueError:
            print(f"Error: --nbh-radius must be 'auto' or a float, got {args.nbh_radius!r}", file=sys.stderr)
            return 1

    from .scoring import score_series

    score_series(
        series_csv=args.series,
        smiles_col=args.smiles_col,
        activity_col=args.activity_col,
        core=args.core,
        va_strategies=args.va,
        va_csv=args.va_csv,
        va_csv_activity_col=args.va_csv_activity_col,
        va_n=args.va_n,
        nbh_radius=nbh_radius,
        output_dir=args.output,
        s_threshold=args.s_threshold,
        p_threshold=args.p_threshold,
        svr_c=args.svr_c,
        svr_epsilon=args.svr_epsilon,
        fragment_lib=args.fragment_lib,
    )
    return 0

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
  python -m como --series series.csv --paper-mode --random-state 42 --score-repeats 10
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
        help="Column in --va-csv with predicted pActivity from the generative model.",
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
    parser.add_argument("--s-threshold", type=float, default=0.4,
                        help="S score threshold for stage assignment [default: 0.4]")
    parser.add_argument("--p-threshold", type=float, default=0.5,
                        help="P score threshold for stage assignment [default: 0.5]")
    parser.add_argument("--svr-c", type=float, default=10.0,
                        help="SVR regularization parameter C [default: 10.0]")
    parser.add_argument("--svr-epsilon", type=float, default=0.1,
                        help="SVR epsilon [default: 0.1]")

    # --- Reproducibility ---
    paper = parser.add_argument_group("Reproducibility options")
    paper.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for VA generation and SVR splits [default: 42].",
    )

    # --- Diagnostic scoring protocol ---
    scoring = parser.add_argument_group("Diagnostic scoring protocol (score_with_repeats)")
    scoring.add_argument(
        "--score-repeats", type=int, default=None,
        help="Run diagnostic scoring with this many random repeats instead of "
             "the default single-run pipeline. Reports mean ± std over repeats.",
    )
    scoring.add_argument(
        "--score-va-n", type=int, default=1000,
        help="Number of close-in VAs per repeat in diagnostic mode [default: 1000].",
    )

    # --- Paper SVR protocol ---
    svr_group = parser.add_argument_group("Paper SVR options")
    svr_group.add_argument(
        "--svr-mode", choices=["legacy", "paper", "off"], default="legacy",
        help="SVR potency model mode: legacy (EA-only CV), paper (external actives + "
             "nested CV), or off [default: legacy].",
    )
    svr_group.add_argument(
        "--external-actives", default=None,
        help="CSV with external target actives (smiles + activity columns) for paper SVR.",
    )
    svr_group.add_argument(
        "--external-activity-col", default="pActivity",
        help="Activity column in --external-actives CSV [default: pActivity].",
    )
    svr_group.add_argument(
        "--ea-train-fraction", type=float, default=0.5,
        help="Fraction of EAs used for SVR training in paper mode [default: 0.5].",
    )
    svr_group.add_argument(
        "--outer-folds", type=int, default=3,
        help="Outer CV folds in nested double CV [default: 3].",
    )
    svr_group.add_argument(
        "--inner-folds", type=int, default=3,
        help="Inner CV folds in nested double CV [default: 3].",
    )
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
            print(f"Error: --nbh-radius must be 'auto' or a float, got {args.nbh_radius!r}",
                  file=sys.stderr)
            return 1

    # --- Diagnostic scoring repeats mode ---
    if args.score_repeats is not None:
        from .scoring import score_with_repeats
        score_with_repeats(
            series_csv=args.series,
            smiles_col=args.smiles_col,
            activity_col=args.activity_col,
            core=args.core,
            n_va=args.score_va_n,
            n_repeats=args.score_repeats,
            nbh_radius=nbh_radius,
            random_state=args.random_state,
            s_threshold=args.s_threshold,
            p_threshold=args.p_threshold,
        )
        return 0

    # --- Standard single-run pipeline ---
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
        random_state=args.random_state,
        svr_mode=args.svr_mode,
        external_actives_csv=args.external_actives,
        external_activity_col=args.external_activity_col,
        ea_train_fraction=args.ea_train_fraction,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    )
    return 0

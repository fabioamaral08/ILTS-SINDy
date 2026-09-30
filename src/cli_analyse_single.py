"""CLI: for one problem and one method, plot several metrics (accuracy,
exact_recovery, trajectory_error) as a single row of noise x outlier
heatmaps in one figure. A single-(problem, method) counterpart to
cli_analyse.py, which instead builds one full problems x methods grid per
metric.

Usage:
    python cli_analyse_single.py --problem SIR --method SINDY-LTS \
        --metrics accuracy exact_recovery -o figs
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet, plot_metric_row
from pipeline.methods import list_methods
from pipeline.problems import list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Method label to load — this is just the saved results' file name, "
            f"not necessarily a registered Method (registered: {list_methods()})."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "exact_recovery"],
        choices=["accuracy", "exact_recovery", "trajectory_error", "coefficient_error"],
        help="Each metric becomes its own column of axes in the same row.",
    )
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument("-o", "--output-dir", default="figs")
    parser.add_argument(
        "--format", default="png", help="File format to save the figure as (e.g. png, pdf, svg)."
    )
    parser.add_argument(
        "--annot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show mean/std text inside each heatmap cell (--no-annot for colormap only).",
    )
    parser.add_argument(
        "--cell-size", type=float, default=0.6, help="Physical size (inches) of one heatmap cell."
    )
    args = parser.parse_args()

    result_set = ResultSet.load(
        args.problem,
        args.method,
        args.noise_levels,
        args.outlier_fractions,
        args.n_realizations,
        data_dir=args.data_dir,
        coeffs_dir=args.coeffs_dir,
    )

    out_path = f"{args.output_dir}/{args.problem}_{args.method}_metrics.{args.format}"
    plot_metric_row(
        result_set,
        metrics=args.metrics,
        annot=args.annot,
        cell_size=args.cell_size,
        output_path=out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

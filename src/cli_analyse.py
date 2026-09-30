"""CLI: build result figures (coefficient accuracy, exact-model recovery,
trajectory error) comparing a list of methods across a list of problems.

Usage:
    python cli_analyse.py --problems SIR LORENZ LV --methods SINDY ESINDY SINDY-LTS \
        --metrics accuracy exact_recovery trajectory_error -o figs
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet, plot_metric_grid
from pipeline.methods import list_methods
from pipeline.problems import list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, nargs="+", choices=list_problems())
    parser.add_argument(
        "--methods",
        required=True,
        nargs="+",
        help=(
            "Method labels to compare — these are just the saved results' file "
            f"names, not necessarily a registered Method (registered: {list_methods()})."
        ),
    )
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["accuracy", "exact_recovery"]
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument("-o", "--output-dir", default="figs")
    parser.add_argument(
        "--format", default="png", help="File format to save figures as (e.g. png, pdf, svg)."
    )
    parser.add_argument(
        "--annot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show mean/std text inside each heatmap cell (--no-annot for colormap only).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.6,
        help="Physical size (inches) of one heatmap cell.",
    )
    args = parser.parse_args()

    result_sets = [
        ResultSet.load(
            problem_name,
            method_name,
            args.noise_levels,
            args.outlier_fractions,
            args.n_realizations,
            data_dir=args.data_dir,
            coeffs_dir=args.coeffs_dir,
        )
        for problem_name in args.problems
        for method_name in args.methods
    ]

    for metric in args.metrics:
        out_path = f"{args.output_dir}/{metric}.{args.format}"
        plot_metric_grid(
            result_sets,
            metric,
            methods=args.methods,
            output_path=out_path,
            annot=args.annot,
            cell_size=args.cell_size,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

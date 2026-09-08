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
    parser.add_argument("--methods", required=True, nargs="+", choices=list_methods())
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9))
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9))
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["accuracy", "exact_recovery", "trajectory_error"]
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument("-o", "--output-dir", default="figs")
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
        out_path = f"{args.output_dir}/{metric}.png"
        plot_metric_grid(result_sets, metric, methods=args.methods, output_path=out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

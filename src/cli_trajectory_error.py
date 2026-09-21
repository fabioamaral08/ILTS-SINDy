"""CLI: for each method's saved coefficient matrix (from `cli_run_method`),
simulate the discovered system and score it by MSE against the noiseless
trajectory — for comparing discovered-model quality across methods and
problems.

Usage:
    python cli_trajectory_error.py --problems SIR LORENZ LV \
        --methods SINDY SINDY-LTS SR3 ESINDY WSINDY -o figs --format pdf
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet
from pipeline.benchmark import plot_trajectory_errors, trajectory_error_benchmark
from pipeline.methods import list_methods
from pipeline.problems import list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, nargs="+", choices=list_problems())
    parser.add_argument("--methods", required=True, nargs="+", choices=list_methods())
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="Max wall-clock seconds per simulation before it's counted as failed.",
    )
    parser.add_argument("-o", "--output-dir", default="figs")
    parser.add_argument(
        "--format", default="png", help="File format to save the figure as (e.g. png, pdf, svg)."
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

    results = trajectory_error_benchmark(result_sets, time_limit=args.time_limit)

    for r in results:
        print(
            f"{r.problem_name:8s} {r.method_name:10s} "
            f"MSE={r.mean_error:.4g} +/- {r.std_error:.4g}  "
            f"failed={r.n_failed}/{r.n_total}"
        )

    out_path = f"{args.output_dir}/trajectory_error.{args.format}"
    plot_trajectory_errors(results, output_path=out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

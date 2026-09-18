"""CLI: build per-problem execution-time boxplots comparing a list of methods.

Usage:
    python cli_time_benchmark.py --problems SIR LORENZ LV --methods SINDY ESINDY SINDY-LTS \
        -o figs
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet, plot_execution_time
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

    figs = plot_execution_time(result_sets, methods=args.methods, output_dir=args.output_dir)
    for problem_name in figs:
        print(f"Saved {args.output_dir}/{problem_name}_time_boxplot.png")


if __name__ == "__main__":
    main()

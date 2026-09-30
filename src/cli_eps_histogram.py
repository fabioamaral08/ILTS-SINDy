"""CLI: build a single figure (one axis per problem) histogramming how often
each AICc-chosen threshold was picked (extra['eps'], from SINDY-LTS-EPS /
saved SINDY-LTS-AIC runs), and print the counts per problem/method.

Usage:
    python cli_eps_histogram.py --problems SIR ROSSLER ABC VAN_DER_POL LORENZ LV \
        --methods SINDY-LTS-AIC --coeffs-dir ../eps_search -o figs --format pdf
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet, plot_eps_histogram
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
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coeffs-dir", default="coeffs")
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

    out_path = f"{args.output_dir}/eps_histogram.{args.format}"
    plot_eps_histogram(result_sets, methods=args.methods, output_path=out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

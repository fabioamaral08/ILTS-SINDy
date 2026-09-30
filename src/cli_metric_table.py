"""CLI: build a LaTeX (booktabs) table of a metric at a fixed noise level,
aggregated (mean +- std) across every outlier percentage — rows = problems,
columns = methods.

Usage:
    python cli_metric_table.py --problems SIR LORENZ LV --methods SINDY ESINDY SINDY-LTS \
        --metric accuracy --noise-level 0.1 -o figs
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.analysis import ResultSet, latex_metric_table
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
    parser.add_argument(
        "--metric",
        required=True,
        choices=["accuracy", "exact_recovery", "trajectory_error", "coefficient_error"],
    )
    parser.add_argument(
        "--noise-level",
        required=True,
        type=float,
        help="Fixed noise level (fraction, e.g. 0.1) to aggregate outlier percentages at.",
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
    args = parser.parse_args()

    if not any(np.isclose(args.noise_level, nl) for nl in args.noise_levels):
        raise SystemExit(
            f"--noise-level {args.noise_level} is not one of --noise-levels {args.noise_levels}"
        )
    noise_level = next(nl for nl in args.noise_levels if np.isclose(args.noise_level, nl))

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

    out_path = f"{args.output_dir}/{args.metric}_table_nl_{noise_level:g}.tex"
    latex_metric_table(
        result_sets, args.metric, noise_level, methods=args.methods, output_path=out_path
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

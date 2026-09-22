"""CLI: plot discovered-system trajectory error comparison from files saved
by `cli_trajectory_error_run.py` — one saved result per problem/method pair.
Pairs with no saved file are skipped.

Usage:
    python cli_trajectory_error.py --problems SIR LORENZ LV \
        --methods SINDY SINDY-LTS SR3 ESINDY WSINDY -o figs --format pdf
"""
from __future__ import annotations

import argparse

from pipeline import io as pipeline_io
from pipeline.benchmark import load_trajectory_error_result, plot_trajectory_errors
from pipeline.methods import list_methods
from pipeline.problems import list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, nargs="+", choices=list_problems())
    parser.add_argument("--methods", required=True, nargs="+", choices=list_methods())
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument("-o", "--output-dir", default="figs")
    parser.add_argument(
        "--format", default="png", help="File format to save the figure as (e.g. png, pdf, svg)."
    )
    args = parser.parse_args()

    results = []
    for problem_name in args.problems:
        for method_name in args.methods:
            path = pipeline_io.trajectory_error_path(problem_name, method_name, args.coeffs_dir)
            if not path.exists():
                print(f"Skipping {problem_name}/{method_name}: {path} not found")
                continue
            results.append(load_trajectory_error_result(path))

    if not results:
        raise SystemExit("No saved trajectory-error results found for the given problems/methods.")

    out_path = f"{args.output_dir}/trajectory_error.{args.format}"
    plot_trajectory_errors(results, output_path=out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

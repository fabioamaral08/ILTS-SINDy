"""CLI: compute discovered-system trajectory error for one problem/method
pair (coefficients from `cli_run_method`) and save it to a file. Run
`cli_trajectory_error_plot.py` separately to compare saved results across
problems/methods.

Usage:
    python cli_trajectory_error_run.py --problem SIR --method SINDY-LTS
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import io as pipeline_io
from pipeline.analysis import ResultSet
from pipeline.benchmark import save_trajectory_error_result, trajectory_error_result_set
from pipeline.methods import list_methods
from pipeline.problems import list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument("--method", required=True, choices=list_methods())
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
    parser.add_argument("-o", "--output-dir", default="coeffs")
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

    result = trajectory_error_result_set(result_set, time_limit=args.time_limit)
    print(
        f"{result.problem_name:8s} {result.method_name:10s} "
        f"MSE={result.mean_error:.4g} +/- {result.std_error:.4g}  "
        f"failed={result.n_failed}/{result.n_total}"
    )

    out_path = pipeline_io.trajectory_error_path(args.problem, args.method, args.output_dir)
    save_trajectory_error_result(out_path, result)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

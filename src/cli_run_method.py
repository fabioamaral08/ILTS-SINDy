"""CLI: run a registered identification Method over a Problem's noisy dataset
grid, grid-searching hyperparameters by trajectory prediction error.

Usage:
    python cli_run_method.py --problem SIR --method SINDY-LTS --n-realizations 100
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import io as pipeline_io
from pipeline.methods import get_method, list_methods
from pipeline.problems import get_problem, list_problems
from pipeline.runner import MethodRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--method", required=True, choices=list_methods())
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("-o", "--output-dir", default="coeffs")
    args = parser.parse_args()

    problem = get_problem(args.problem)
    method = get_method(args.method)
    dataset_path = pipeline_io.dataset_path(problem.name, args.n_realizations, args.data_dir)

    runner = MethodRunner(problem, method)
    path = runner.run_grid(
        dataset_path,
        args.noise_levels,
        args.outlier_fractions,
        args.n_realizations,
        output_dir=args.output_dir,
        eps=args.threshold
    )
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()

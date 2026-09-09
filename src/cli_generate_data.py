"""CLI: generate noisy datasets for a registered Problem.

Usage:
    python cli_generate_data.py --problem SIR --n-realizations 100 -o data
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline.datasets import DatasetGenerator
from pipeline.problems import get_problem, list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9))
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9))
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", default="data")
    args = parser.parse_args()

    problem = get_problem(args.problem)
    generator = DatasetGenerator(problem)
    path = generator.generate(
        args.noise_levels,
        args.outlier_fractions,
        args.n_realizations,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()

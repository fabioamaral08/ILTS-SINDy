"""CLI: run SINDY-LTS with threshold=None (per-state AICc threshold search,
see `lts.SINDy_LTS`/`lts.AIC`) over a Problem's noisy dataset grid, scoring
each fit by trajectory prediction error. Mirrors `cli_runsearch.py`'s p-search
script, but tests SINDY-LTS's AIC-based threshold search instead of
SINDY-LTS-SEARCH's p search; `p` is derived per grid cell from the outlier
fraction (`p = m - 3 * outlier_fraction * m`), the same way `cli_run_method.py`
does via `pipeline.runner`.

Usage:
    python cli_run_aic_search.py --problem SIR --n-realizations 100
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import io as pipeline_io
from pipeline.aic_search import MethodRunner
from pipeline.problems import get_problem, list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
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
    dataset_path = pipeline_io.dataset_path(problem.name, args.n_realizations, args.data_dir)

    runner = MethodRunner(problem)
    path = runner.run_grid(
        dataset_path,
        args.noise_levels,
        args.outlier_fractions,
        args.n_realizations,
        output_dir=args.output_dir,
    )
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()

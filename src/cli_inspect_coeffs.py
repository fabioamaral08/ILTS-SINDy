"""CLI: inspect the shape of saved coefficient matrices in a results file
against the problem's *current* feature library — useful for tracking down
"operands could not be broadcast together" errors in metrics.py, which
usually mean a results file was saved before a feature-library change (e.g.
adding/removing a bias term) and now has a different number of candidate
terms than a freshly-built true_coeff.

Usage:
    python cli_inspect_coeffs.py --problem SIR --method SINDY-LTS-AIC --coeffs-dir ../eps_search
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from pipeline import io as pipeline_io
from pipeline.problems import get_problem, list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument("--method", required=True)
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument(
        "--outlier-fractions", type=float, nargs="+", default=list(np.linspace(0, 0.2, 9)[1:])
    )
    parser.add_argument("--coeffs-dir", default="coeffs")
    args = parser.parse_args()

    problem = get_problem(args.problem)
    library = problem.feature_library()
    true_coeff = problem.true_coefficients(library)
    print(f"Current true_coefficients shape (from problem.feature_library()): {true_coeff.shape}")

    results_path = pipeline_io.results_path(
        problem.name, args.method, args.n_realizations, args.coeffs_dir
    )
    print(f"Loading {results_path}")
    results = np.load(results_path, allow_pickle=True)

    shape_counts: Counter[tuple] = Counter()
    mismatched_cells: list[tuple[float, float, tuple]] = []
    for nl in args.noise_levels:
        for op in args.outlier_fractions:
            cell = pipeline_io.load_run_cell(results, nl, op)
            for c in cell["coefficients"]:
                shape = np.asarray(c).shape
                shape_counts[shape] += 1
                if shape != true_coeff.shape:
                    mismatched_cells.append((nl, op, shape))

    print("\nCoefficient matrix shapes across the whole grid:")
    for shape, count in sorted(shape_counts.items(), key=lambda kv: -kv[1]):
        flag = "  <-- MISMATCH with current true_coeff" if shape != true_coeff.shape else ""
        print(f"  {shape}: {count} realizations{flag}")

    if mismatched_cells:
        cells = sorted(set((nl, op) for nl, op, _ in mismatched_cells))
        print(f"\n{len(mismatched_cells)} mismatched realizations across {len(cells)} grid cells.")
        if len(shape_counts) == 1:
            print(
                "Every realization has the same (wrong) shape -> this results file was almost "
                "certainly saved with an older feature library. Rerun cli_run_method.py (or "
                "cli_run_aic_search.py) to regenerate it."
            )
        else:
            print(
                "Shapes are mixed within this single file -> not just a stale-run issue; worth "
                "checking whether the library or problem definition is non-deterministic across "
                "calls (e.g. changes state on repeated .fit_transform() calls)."
            )
    else:
        print("\nNo mismatches — every saved coefficient matrix matches the current library shape.")


if __name__ == "__main__":
    main()

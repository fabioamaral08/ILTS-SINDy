"""CLI: sweep every results .npz file under a coeffs directory and check
whether the saved coefficient matrices' shape matches the problem's
*current* true_coefficients shape (i.e. its current feature library) —
tracks down "operands could not be broadcast together" errors in metrics.py,
which usually mean a results file predates a feature-library change (e.g.
adding/removing a bias term).

Discovers files by the `pipeline.io.results_path` naming convention
(<coeffs-dir>/<PROBLEM>/<PROBLEM>_<METHOD>_<N>_samples.npz) rather than
requiring a specific noise/outlier grid or method list up front.

Usage:
    python cli_check_coeff_shapes.py --coeffs-dir ../coeff
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.problems import get_problem, list_problems


def _iter_coefficient_shapes(npz_path: Path):
    """Yields the shape of every saved coefficient matrix in a results file,
    reading whatever noise/outlier keys are actually present rather than
    assuming a specific grid."""
    results = np.load(npz_path, allow_pickle=True)
    for noise_key in results.files:
        cell = results[noise_key].item()
        for key, value in cell.items():
            if not key.startswith("coeffs_"):
                continue
            for c in value:
                yield np.asarray(c).shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs-dir", default="coeffs")
    args = parser.parse_args()

    coeffs_dir = Path(args.coeffs_dir)
    npz_paths = sorted(coeffs_dir.glob("*/*.npz"))
    if not npz_paths:
        raise SystemExit(f"No .npz files found under {coeffs_dir}/*/*.npz")

    true_coeff_shapes: dict[str, tuple] = {}
    mismatched_files: list[str] = []
    skipped: list[str] = []

    for npz_path in npz_paths:
        problem_name = npz_path.parent.name
        # Anchor on the known problem name (from the directory) rather than
        # guessing the problem/method split generically — VAN_DER_POL has
        # underscores in it too, so a generic split could slice it wrong.
        filename_re = re.compile(
            rf"^{re.escape(problem_name)}_(?P<method>.+)_(?P<n>\d+)_samples\.npz$"
        )
        m = filename_re.match(npz_path.name)
        if problem_name not in list_problems() or m is None:
            skipped.append(str(npz_path))
            continue
        method_name = m.group("method")

        if problem_name not in true_coeff_shapes:
            problem = get_problem(problem_name)
            library = problem.feature_library()
            true_coeff_shapes[problem_name] = problem.true_coefficients(library).shape
        expected_shape = true_coeff_shapes[problem_name]

        shape_counts = Counter(_iter_coefficient_shapes(npz_path))
        bad = {shape: count for shape, count in shape_counts.items() if shape != expected_shape}

        label = f"{problem_name:12s} {method_name:18s}"
        if not bad:
            total = sum(shape_counts.values())
            print(f"OK       {label} {total} realizations, all {expected_shape}")
        else:
            mismatched_files.append(str(npz_path))
            breakdown = ", ".join(f"{shape}: {count}" for shape, count in shape_counts.items())
            print(f"MISMATCH {label} expected {expected_shape} -- got {{{breakdown}}}")

    print(f"\n{len(npz_paths) - len(skipped)} files checked, {len(mismatched_files)} mismatched.")
    if mismatched_files:
        print("\nMismatched files (rerun cli_run_method.py / cli_run_aic_search.py for these):")
        for path in mismatched_files:
            print(f"  {path}")
    if skipped:
        print(f"\n{len(skipped)} files skipped (didn't match the expected naming convention):")
        for path in skipped:
            print(f"  {path}")


if __name__ == "__main__":
    main()

"""CLI: plot a quantity (trajectory MSE vs. the noiseless trajectory, or
coefficient estimate error) against outlier percentage at one fixed noise
level. Rows = aggregators in --aggregators ("mean"/"median"), columns =
problems; each axis has one line per method.

--quantity trajectory_error reads the files saved by
cli_trajectory_error_run.py (one per problem/method pair); pairs with no
saved file are skipped. --quantity coefficient_error is computed directly
from the saved coefficients (same as cli_analyse.py's ResultSet) — no
separate run step needed, since it doesn't require simulating anything.

Usage:
    python cli_error_lines.py --problems SIR LORENZ LV VAN_DER_POL ABC ROSSLER \
        --methods SINDY ESINDY WSINDY SINDY-LTS --noise-level 0.05 \
        --quantity coefficient_error --aggregators mean median \
        --data-dir ../data --coeffs-dir ../coeff -o ../figs --format pdf
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import io as pipeline_io
from pipeline.analysis import ResultSet
from pipeline.benchmark import (
    coefficient_error_curve,
    load_trajectory_error_result,
    plot_trajectory_error_lines,
    trajectory_error_curve,
)
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
        "--quantity",
        choices=["trajectory_error", "coefficient_error"],
        default="trajectory_error",
        help=(
            "trajectory_error reads files saved by cli_trajectory_error_run.py; "
            "coefficient_error is computed directly from the saved coefficients."
        ),
    )
    parser.add_argument(
        "--noise-level",
        required=True,
        type=float,
        help="Fixed noise level (fraction, e.g. 0.05) to plot across outlier percentages.",
    )
    parser.add_argument(
        "--aggregators",
        nargs="+",
        choices=["mean", "median"],
        default=["mean", "median"],
        help="Each aggregator becomes its own row of axes in the same figure.",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=100,
        help="Only used for --quantity coefficient_error.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=list(np.linspace(0, 0.2, 9)[1:]),
        help="Only used for --quantity coefficient_error.",
    )
    parser.add_argument(
        "--outlier-fractions",
        type=float,
        nargs="+",
        default=list(np.linspace(0, 0.2, 9)[1:]),
        help="Only used for --quantity coefficient_error.",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Only used for --quantity coefficient_error."
    )
    parser.add_argument("--coeffs-dir", default="coeffs")
    parser.add_argument("-o", "--output-dir", default="figs")
    parser.add_argument(
        "--format", default="png", help="File format to save the figure as (e.g. png, pdf, svg)."
    )
    parser.add_argument(
        "--std-band",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shade center +- std around each line.",
    )
    args = parser.parse_args()

    curves = []
    if args.quantity == "trajectory_error":
        for problem_name in args.problems:
            for method_name in args.methods:
                path = pipeline_io.trajectory_error_path(
                    problem_name, method_name, args.coeffs_dir
                )
                if not path.exists():
                    print(f"Skipping {problem_name}/{method_name}: {path} not found")
                    continue
                result = load_trajectory_error_result(path)
                curves.extend(
                    trajectory_error_curve(result, args.noise_level, aggregator)
                    for aggregator in args.aggregators
                )
    else:
        if not any(np.isclose(args.noise_level, nl) for nl in args.noise_levels):
            raise SystemExit(
                f"--noise-level {args.noise_level} is not one of --noise-levels {args.noise_levels}"
            )
        noise_level = next(nl for nl in args.noise_levels if np.isclose(args.noise_level, nl))
        for problem_name in args.problems:
            for method_name in args.methods:
                result_set = ResultSet.load(
                    problem_name,
                    method_name,
                    args.noise_levels,
                    args.outlier_fractions,
                    args.n_realizations,
                    data_dir=args.data_dir,
                    coeffs_dir=args.coeffs_dir,
                )
                curves.extend(
                    coefficient_error_curve(result_set, noise_level, aggregator)
                    for aggregator in args.aggregators
                )

    if not curves:
        raise SystemExit("No results found for the given problems/methods.")

    out_path = (
        f"{args.output_dir}/{args.quantity}_lines_nl_{args.noise_level:g}.{args.format}"
    )
    plot_trajectory_error_lines(
        curves, methods=args.methods, output_path=out_path, std_band=args.std_band
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

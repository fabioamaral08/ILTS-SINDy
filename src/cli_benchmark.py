"""CLI: time each method's fit() call using a single, representative
hyperparameter set (no grid search) — for comparing raw execution cost.

Usage:
    python cli_benchmark.py --problems SIR LORENZ LV \
        --methods SINDY SINDY-LTS SR3 ESINDY WSINDY --n-realizations 5 -o ../figs/timing.png
"""
from __future__ import annotations

import argparse

from pipeline.benchmark import benchmark, plot_timings
from pipeline.methods import get_method, list_methods
from pipeline.problems import get_problem, list_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=list_problems(), choices=list_problems())
    parser.add_argument("--methods", nargs="+", default=list_methods(), choices=list_methods())
    parser.add_argument("--n-realizations", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--outlier-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--output", default="figs/timing.png")
    args = parser.parse_args()

    problems = [get_problem(name) for name in args.problems]
    methods = [get_method(name) for name in args.methods]

    results = benchmark(
        problems,
        methods,
        noise_level=args.noise_level,
        outlier_fraction=args.outlier_fraction,
        n_realizations=args.n_realizations,
        n_warmup=args.n_warmup,
        seed=args.seed,
    )

    for r in results:
        print(
            f"{r.problem_name:8s} {r.method_name:10s} "
            f"{r.mean_seconds:8.4f}s +/- {r.std_seconds:.4f}s  (hyperparams={r.hyperparams})"
        )

    plot_timings(results, output_path=args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

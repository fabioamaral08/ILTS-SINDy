"""Compares discovered-system trajectory error across methods and problems,
sourced from the coefficient matrices already computed by `cli_run_method`
(saved via `pipeline.io.save_run_results`, loaded through `analysis.ResultSet`)
rather than refitting anything here. For each saved coefficient matrix, pooled
across a ResultSet's whole noise/outlier grid, simulates the discovered model
and scores it by MSE against the problem's noiseless trajectory, tracking
simulations that fail to integrate or diverge as failures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from . import io
from .analysis import ResultSet, _problem_label
from .problems.base import Problem


class _SimulationTimeout(Exception):
    """Raised inside `dynamics` to abort a solve_ivp call that's run too
    long — e.g. a bad coefficient matrix makes the solver take extremely
    small steps chasing a near-blow-up instead of failing outright."""


def simulate_from_coefficients(
    problem: Problem,
    library,
    coefficients: np.ndarray,
    t: np.ndarray,
    time_limit: float = 5.0,
) -> np.ndarray | None:
    """Integrate dx/dt = library(x) @ coefficients from the problem's true
    initial condition. Returns the simulated trajectory (n_points, n_states),
    or None if the integration doesn't converge, diverges to non-finite
    values, or exceeds `time_limit` seconds of wall-clock time (all counted
    as a failed simulation)."""
    deadline = time.perf_counter() + time_limit

    def dynamics(_t, x):
        if time.perf_counter() > deadline:
            raise _SimulationTimeout()
        theta = library.transform(np.asarray(x, dtype=float).reshape(1, -1))
        return (theta @ coefficients)[0]

    try:
        sol = solve_ivp(dynamics, (t[0], t[-1]), problem.x0, t_eval=t)
    except _SimulationTimeout:
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


@dataclass
class TrajectoryErrorResult:
    problem_name: str
    method_name: str
    errors: list[float]
    n_failed: int
    n_total: int

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    @property
    def std_error(self) -> float:
        return float(np.std(self.errors)) if self.errors else float("nan")


def trajectory_error_result_set(
    result_set: ResultSet, time_limit: float = 5.0
) -> TrajectoryErrorResult:
    """Simulates every coefficient matrix saved in `result_set` (pooled across
    its whole noise/outlier grid) and scores each by MSE against the
    problem's noiseless trajectory. A simulation that fails to integrate,
    diverges to non-finite values, or exceeds `time_limit` seconds (a bad
    coefficient matrix can make the solver stall) is counted as a failure and
    excluded from the reported error stats."""
    problem = result_set.problem
    library = problem.feature_library()
    problem.true_coefficients(library)  # fits `library`'s output columns
    t, clean = problem.simulate()

    errors = []
    n_failed = 0
    n_total = 0
    for nl in result_set.noise_levels:
        for op in result_set.outlier_fractions:
            cell = io.load_run_cell(result_set.results, nl, op)
            for coeff in cell["coefficients"]:
                n_total += 1
                sim = simulate_from_coefficients(problem, library, coeff, t, time_limit=time_limit)
                if sim is None:
                    n_failed += 1
                    continue
                errors.append(float(np.mean((sim - clean) ** 2)))

    return TrajectoryErrorResult(
        problem_name=problem.name,
        method_name=result_set.method_name,
        errors=errors,
        n_failed=n_failed,
        n_total=n_total,
    )


def trajectory_error_benchmark(
    result_sets: Sequence[ResultSet], time_limit: float = 5.0
) -> list[TrajectoryErrorResult]:
    """One TrajectoryErrorResult per ResultSet (i.e. per problem/method pair
    already run via `cli_run_method`)."""
    return [trajectory_error_result_set(rs, time_limit=time_limit) for rs in result_sets]


def save_trajectory_error_result(path: str | Path, result: TrajectoryErrorResult) -> None:
    """Saves one TrajectoryErrorResult (e.g. from a single problem/method run)
    so it can be plotted later, separately, alongside other saved runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        problem_name=result.problem_name,
        method_name=result.method_name,
        errors=np.array(result.errors, dtype=float),
        n_failed=result.n_failed,
        n_total=result.n_total,
    )


def load_trajectory_error_result(path: str | Path) -> TrajectoryErrorResult:
    data = np.load(path, allow_pickle=True)
    return TrajectoryErrorResult(
        problem_name=str(data["problem_name"]),
        method_name=str(data["method_name"]),
        errors=list(data["errors"]),
        n_failed=int(data["n_failed"]),
        n_total=int(data["n_total"]),
    )


def plot_trajectory_errors(
    results: Sequence[TrajectoryErrorResult], output_path: str | Path | None = None
):
    """Grouped bar chart of mean trajectory MSE (+- std as error bars) per
    method, one bar group per problem, log-scaled y-axis. Each bar is
    annotated with its failed-simulation count (excluded from the error
    stats)."""
    problems = list(dict.fromkeys(r.problem_name for r in results))
    methods = list(dict.fromkeys(r.method_name for r in results))
    lookup = {(r.problem_name, r.method_name): r for r in results}

    x = np.arange(len(methods))
    width = 0.8 / max(len(problems), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(methods) + 2, 5))
    for i, problem_name in enumerate(problems):
        rs = [lookup.get((problem_name, m)) for m in methods]
        means = [r.mean_error if r else 0.0 for r in rs]
        stds = [r.std_error if r else 0.0 for r in rs]
        bar_x = x + i * width
        ax.bar(bar_x, means, width, yerr=stds, label=_problem_label(problem_name), capsize=3)

        for xb, r in zip(bar_x, rs):
            if r is None or r.n_failed == 0:
                continue
            y_ref = r.mean_error + r.std_error
            y = y_ref * 1.15 if np.isfinite(y_ref) and y_ref > 0 else 1e-6
            ax.text(
                xb, y, f"{r.n_failed}/{r.n_total}\nfailed", ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x + width * (len(problems) - 1) / 2)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Trajectory MSE vs. noiseless data")
    ax.set_yscale("log")
    ax.set_title("Discovered-system trajectory error (mean ± std; failed simulations excluded)")
    ax.legend(title="Problem")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig

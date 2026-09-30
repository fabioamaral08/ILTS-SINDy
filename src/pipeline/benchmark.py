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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

from . import io, metrics
from .analysis import ResultSet, _method_label, _problem_label
from .problems.base import Problem

# Distinct marker/linestyle per method (cycled), so lines in
# plot_trajectory_error_lines stay distinguishable for colorblind readers
# instead of relying on hue alone.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]

# Single hardcoded knob for plot_trajectory_error_lines' text sizing — edit
# this directly to tune it, rather than it being auto-derived from the
# number of methods (which was making it too small).
_LINE_PLOT_FONTSCALE = 1.5


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
    """`errors`/`n_failed`/`n_total` are pooled over the whole grid. The
    per-cell fields keep the grid structure (`cell_errors[i][j]` is the list
    of MSEs at noise_levels[i], outlier_fractions[j]; `cell_failed[i][j]` the
    number of failed simulations there) so results can be plotted against
    outlier percentage at a fixed noise level. They're empty for results
    saved before that structure was recorded."""

    problem_name: str
    method_name: str
    errors: list[float]
    n_failed: int
    n_total: int
    noise_levels: list[float] = field(default_factory=list)
    outlier_fractions: list[float] = field(default_factory=list)
    cell_errors: list = field(default_factory=list)
    cell_failed: list = field(default_factory=list)

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    @property
    def std_error(self) -> float:
        return float(np.std(self.errors)) if self.errors else float("nan")

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors)) if self.errors else float("nan")

    @property
    def mad_error(self) -> float:
        """Median absolute deviation — the spread statistic paired with
        `median_error`, the way `std_error` pairs with `mean_error`."""
        if not self.errors:
            return float("nan")
        return float(np.median(np.abs(np.array(self.errors) - self.median_error)))


def _score_one(problem, library, coefficients, t, clean, time_limit) -> float | None:
    """MSE of one discovered model vs. the noiseless trajectory, or None if
    its simulation failed."""
    sim = simulate_from_coefficients(problem, library, coefficients, t, time_limit=time_limit)
    if sim is None:
        return None
    return float(np.mean((sim - clean) ** 2))


def trajectory_error_result_set(
    result_set: ResultSet, time_limit: float = 5.0, n_jobs: int = -1
) -> TrajectoryErrorResult:
    """Simulates every coefficient matrix saved in `result_set` (pooled across
    its whole noise/outlier grid) and scores each by MSE against the
    problem's noiseless trajectory. A simulation that fails to integrate,
    diverges to non-finite values, or exceeds `time_limit` seconds (a bad
    coefficient matrix can make the solver stall) is counted as a failure and
    excluded from the reported error stats. Simulations are independent, so
    they run in parallel across `n_jobs` processes; `time_limit` is
    wall-clock, so keep `n_jobs` at or below the available cores or slowed
    workers may hit it spuriously."""
    problem = result_set.problem
    library = problem.feature_library()
    problem.true_coefficients(library)  # fits `library`'s output columns
    t, clean = problem.simulate()

    cells = [
        (i, j, coeff)
        for i, nl in enumerate(result_set.noise_levels)
        for j, op in enumerate(result_set.outlier_fractions)
        for coeff in io.load_run_cell(result_set.results, nl, op)["coefficients"]
    ]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(problem, library, coeff, t, clean, time_limit)
        for _, _, coeff in cells
    )

    n_noise, n_outlier = len(result_set.noise_levels), len(result_set.outlier_fractions)
    cell_errors = [[[] for _ in range(n_outlier)] for _ in range(n_noise)]
    cell_failed = [[0] * n_outlier for _ in range(n_noise)]
    for (i, j, _), score in zip(cells, scores):
        if score is None:
            cell_failed[i][j] += 1
        else:
            cell_errors[i][j].append(score)

    errors = [e for row in cell_errors for errs in row for e in errs]
    n_failed = sum(sum(row) for row in cell_failed)
    n_total = len(cells)

    return TrajectoryErrorResult(
        problem_name=problem.name,
        method_name=result_set.method_name,
        errors=errors,
        n_failed=n_failed,
        n_total=n_total,
        noise_levels=list(result_set.noise_levels),
        outlier_fractions=list(result_set.outlier_fractions),
        cell_errors=cell_errors,
        cell_failed=cell_failed,
    )


def trajectory_error_benchmark(
    result_sets: Sequence[ResultSet], time_limit: float = 5.0, n_jobs: int = -1
) -> list[TrajectoryErrorResult]:
    """One TrajectoryErrorResult per ResultSet (i.e. per problem/method pair
    already run via `cli_run_method`)."""
    return [
        trajectory_error_result_set(rs, time_limit=time_limit, n_jobs=n_jobs)
        for rs in result_sets
    ]


def save_trajectory_error_result(path: str | Path, result: TrajectoryErrorResult) -> None:
    """Saves one TrajectoryErrorResult (e.g. from a single problem/method run)
    so it can be plotted later, separately, alongside other saved runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_errors = np.empty((len(result.noise_levels), len(result.outlier_fractions)), dtype=object)
    for i, row in enumerate(result.cell_errors):
        for j, errs in enumerate(row):
            cell_errors[i, j] = list(errs)
    np.savez(
        path,
        problem_name=result.problem_name,
        method_name=result.method_name,
        errors=np.array(result.errors, dtype=float),
        n_failed=result.n_failed,
        n_total=result.n_total,
        noise_levels=np.array(result.noise_levels, dtype=float),
        outlier_fractions=np.array(result.outlier_fractions, dtype=float),
        cell_errors=cell_errors,
        cell_failed=np.array(result.cell_failed, dtype=int),
    )


def load_trajectory_error_result(path: str | Path) -> TrajectoryErrorResult:
    data = np.load(path, allow_pickle=True)
    has_cells = "cell_errors" in data.files
    return TrajectoryErrorResult(
        problem_name=str(data["problem_name"]),
        method_name=str(data["method_name"]),
        errors=list(data["errors"]),
        n_failed=int(data["n_failed"]),
        n_total=int(data["n_total"]),
        noise_levels=list(data["noise_levels"]) if has_cells else [],
        outlier_fractions=list(data["outlier_fractions"]) if has_cells else [],
        cell_errors=[list(row) for row in data["cell_errors"]] if has_cells else [],
        cell_failed=data["cell_failed"].tolist() if has_cells else [],
    )


def plot_trajectory_errors(
    results: Sequence[TrajectoryErrorResult], output_path: str | Path | None = None
) -> Figure:
    """Grouped bar chart of mean trajectory MSE (+- std as error bars) per
    method, one bar group per problem, log-scaled y-axis. Each bar is
    annotated with its failed-simulation count (excluded from the error
    stats). The problem legend is placed outside the axes, to the right,
    rather than overlapping the bars."""
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
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("Trajectory MSE vs. noiseless data")
    ax.set_yscale("log")
    ax.set_title("Discovered-system trajectory error (mean ± std; failed simulations excluded)")
    fig.legend(*ax.get_legend_handles_labels(), title="Problem", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 0.86, 1))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


_QUANTITY_LABELS = {
    "trajectory_error": "Trajectory MSE",
    "coefficient_error": "Coefficient error",
}

# Math symbol per quantity, and an accent per aggregator, combined into a
# y-axis label like "$\overline{E}(q, \eta)$" for coefficient_error/mean —
# q = outlier fraction, eta = noise level, matching the paper's notation.
_QUANTITY_SYMBOLS = {
    "trajectory_error": "\\mathrm{MSE}",
    "coefficient_error": "E",
}
_AGGREGATOR_ACCENTS = {
    "mean": "\\overline{{{symbol}}}",
    "median": "\\widetilde{{{symbol}}}",
}


@dataclass
class TrajectoryErrorCurve:
    """A `quantity` (see `_QUANTITY_LABELS`) vs. outlier fraction at one
    fixed noise level, for one (problem, method) pair. `center` is the mean
    or median (see `aggregator`) per outlier fraction; failed simulations
    (only meaningful for quantity="trajectory_error") are excluded."""

    problem_name: str
    method_name: str
    noise_level: float
    outlier_fractions: list[float]
    center: np.ndarray
    std: np.ndarray
    n_failed: np.ndarray
    n_total: np.ndarray
    aggregator: str = "mean"
    quantity: str = "trajectory_error"


_AGGREGATORS = {"mean": np.mean, "median": np.median}


def trajectory_error_curve(
    result: TrajectoryErrorResult, noise_level: float, aggregator: str = "mean"
) -> TrajectoryErrorCurve:
    """Slices a saved TrajectoryErrorResult (from cli_trajectory_error_run.py)
    at `noise_level`, giving the `aggregator` ("mean" or "median") MSE, the
    std, and failed counts per outlier fraction. Failed/diverged simulations
    are excluded from the statistics."""
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"aggregator must be one of {sorted(_AGGREGATORS)}, got {aggregator!r}")
    aggregate = _AGGREGATORS[aggregator]
    if not result.cell_errors:
        raise ValueError(
            f"{result.problem_name}/{result.method_name}: saved result has no per-cell "
            "(noise x outlier) data — rerun cli_trajectory_error_run.py to regenerate it."
        )
    matches = [i for i, nl in enumerate(result.noise_levels) if np.isclose(nl, noise_level)]
    if not matches:
        raise ValueError(
            f"{result.problem_name}/{result.method_name}: noise level {noise_level} not in "
            f"saved noise levels {result.noise_levels}."
        )
    i = matches[0]

    means, stds, n_failed, n_total = [], [], [], []
    for j in range(len(result.outlier_fractions)):
        errs = result.cell_errors[i][j]
        failed = int(result.cell_failed[i][j])
        means.append(float(aggregate(errs)) if len(errs) else float("nan"))
        stds.append(float(np.std(errs)) if len(errs) else float("nan"))
        n_failed.append(failed)
        n_total.append(len(errs) + failed)

    return TrajectoryErrorCurve(
        problem_name=result.problem_name,
        method_name=result.method_name,
        noise_level=float(result.noise_levels[i]),
        outlier_fractions=[float(op) for op in result.outlier_fractions],
        center=np.array(means),
        std=np.array(stds),
        n_failed=np.array(n_failed),
        n_total=np.array(n_total),
        aggregator=aggregator,
        quantity="trajectory_error",
    )


def coefficient_error_curve(
    result_set: ResultSet, noise_level: float, aggregator: str = "mean"
) -> TrajectoryErrorCurve:
    """Like `trajectory_error_curve`, but for the coefficient_error metric
    (see `metrics.coefficient_error`) computed directly from a ResultSet's
    already-saved coefficients — no simulation and no separate
    cli_trajectory_error_run.py pass needed, since it's cheap to compute
    on the fly. There's no notion of a "failed" coefficient, so n_failed is
    always 0."""
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"aggregator must be one of {sorted(_AGGREGATORS)}, got {aggregator!r}")
    aggregate = _AGGREGATORS[aggregator]

    library = result_set.problem.feature_library()
    true_coeff = result_set.problem.true_coefficients(library)

    centers, stds, n_total = [], [], []
    for op in result_set.outlier_fractions:
        cell = io.load_run_cell(result_set.results, noise_level, op)
        errs = [metrics.coefficient_error(true_coeff, c) for c in cell["coefficients"]]
        centers.append(float(aggregate(errs)) if errs else float("nan"))
        stds.append(float(np.std(errs)) if errs else float("nan"))
        n_total.append(len(errs))

    return TrajectoryErrorCurve(
        problem_name=result_set.problem.name,
        method_name=result_set.method_name,
        noise_level=noise_level,
        outlier_fractions=[float(op) for op in result_set.outlier_fractions],
        center=np.array(centers),
        std=np.array(stds),
        n_failed=np.zeros(len(n_total), dtype=int),
        n_total=np.array(n_total),
        aggregator=aggregator,
        quantity="coefficient_error",
    )


def plot_trajectory_error_lines(
    curves: Sequence[TrajectoryErrorCurve],
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    std_band: bool = False,
) -> Figure:
    """Rows = aggregators present in `curves` ("mean"/"median"), columns =
    problems; each axis plots the curves' quantity (trajectory MSE or
    coefficient error — see `_QUANTITY_LABELS`; line, log y-axis) against
    outlier percentage for every method, at the curves' fixed noise level.
    With only one aggregator, problems are instead laid out in a square-ish
    grid (one axis per problem, like `plot_execution_time`'s layout), since a
    single, very wide row of columns is otherwise cramped/hard to read. The
    (shared) method legend is placed outside the axes, to the right.
    `std_band` additionally shades center +- std around each line. Since a
    lower band close to (or below) 0 can't be shown on a log scale without
    heavy distortion, any axis whose std band comes near 0 is instead drawn
    with a linear y-axis."""
    problems = list(dict.fromkeys(c.problem_name for c in curves))
    aggregators = list(dict.fromkeys(c.aggregator for c in curves))
    if methods is None:
        methods = list(dict.fromkeys(c.method_name for c in curves))
    lookup = {(c.problem_name, c.method_name, c.aggregator): c for c in curves}
    noise_level = curves[0].noise_level
    quantity_label = _QUANTITY_LABELS.get(curves[0].quantity, curves[0].quantity)

    single_aggregator = len(aggregators) == 1
    if single_aggregator:
        n_rows = int(np.ceil(np.sqrt(len(problems))))
        n_cols = int(np.ceil(len(problems) / n_rows))
    else:
        n_rows, n_cols = len(aggregators), len(problems)

    scale = _LINE_PLOT_FONTSCALE
    title_fontsize = 18 * scale
    label_fontsize = 15 * scale
    tick_fontsize = 12 * scale
    suptitle_fontsize = 18 * scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    # Ordered (aggregator, problem) pairs, aggregator-major — with a single
    # aggregator this is just the problems in order, filling the square grid
    # row by row; with several aggregators, n_cols == len(problems) so this
    # lines up exactly with the old (row=aggregator, col=problem) indexing.
    pairs = [(aggregator, problem_name) for aggregator in aggregators for problem_name in problems]

    legend_handles_labels = None
    for idx, (aggregator, problem_name) in enumerate(pairs):
        row, col = divmod(idx, n_cols)
        ax = axes_flat[idx]
        print(f"{_problem_label(problem_name)} ({aggregator})")
        axis_curves = [
            (method_name, lookup[(problem_name, method_name.upper(), aggregator)])
            for method_name in methods
            if (problem_name, method_name.upper(), aggregator) in lookup
        ]

        # Decide the y-scale before plotting: log needs every value strictly
        # positive, so if the std band dips near/below 0 anywhere on this
        # axis, fall back to linear instead of clipping it up to a tiny
        # floor (which would otherwise draw a wildly stretched-out band).
        use_log = True
        if std_band and axis_curves:
            lower_bounds = np.concatenate([c.center - c.std for _, c in axis_curves])
            use_log = bool(np.all(lower_bounds > 1e-10 * np.max(np.abs(lower_bounds), initial=1.0)))

        for m_idx, method_name in enumerate(methods):
            c = lookup.get((problem_name, method_name.upper(), aggregator))
            if c is None:
                continue
            x = np.array(c.outlier_fractions) * 100
            line, = ax.plot(
                x,
                c.center,
                marker=_MARKERS[m_idx % len(_MARKERS)],
                linestyle=_LINESTYLES[m_idx % len(_LINESTYLES)],
                label=_method_label(method_name),
            )
            if std_band:
                lower = c.center - c.std
                if use_log:
                    lower = np.clip(lower, a_min=np.finfo(float).tiny, a_max=None)
                ax.fill_between(x, lower, c.center + c.std, color=line.get_color(), alpha=0.2)
            failed = ", ".join(f"{f}/{n}" for f, n in zip(c.n_failed, c.n_total))
            print(f"  {method_name}: failed per outlier level = {failed}")

        ax.set_yscale("log" if use_log else "linear")
        if single_aggregator or row == n_rows - 1:
            ax.set_xlabel("Outlier percentage (%)", fontsize=label_fontsize)
        if single_aggregator or col == 0:
            symbol = _QUANTITY_SYMBOLS.get(curves[0].quantity, quantity_label)
            accent = _AGGREGATOR_ACCENTS.get(aggregator, "{symbol}").format(symbol=symbol)
            ax.set_ylabel(f"${accent}(q, \\eta)$", fontsize=label_fontsize)
        if single_aggregator or row == 0:
            ax.set_title(_problem_label(problem_name), fontsize=title_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(alpha=0.3)
        if legend_handles_labels is None:
            legend_handles_labels = ax.get_legend_handles_labels()

    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"{quantity_label} vs. outliers (noise = {noise_level * 100:g}%)",
        fontsize=suptitle_fontsize,
    )
    fig.legend(
        *legend_handles_labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(methods),
        columnspacing=1.0,
        handletextpad=0.5,
        fontsize=label_fontsize,
        title_fontsize=title_fontsize,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig

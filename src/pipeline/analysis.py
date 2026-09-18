"""Loads (problem, method) result pairs and builds comparison figures —
coefficient accuracy, exact-model-recovery rate, and trajectory error, each as
a noise x outlier heatmap — generalized over an arbitrary list of problems and
methods (rows and columns of the figure grid).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from matplotlib.figure import Figure

from . import io, metrics
from .problems import get_problem
from .problems.base import Problem

_METRIC_LABELS = {
    "accuracy": "Coefficient accuracy",
    "exact_recovery": "Exact model recovery",
    "trajectory_error": "Trajectory error (NRMSE)",
}


def _format_metric(value: float) -> str:
    """2-decimal formatting, except near 0/1 where rounding would display a
    false exact boundary (e.g. 0.996 -> "1.00") — show more precision there
    instead of implying a perfect (or zero) score that isn't real."""
    rounded = f"{value:.2f}"
    if rounded in ("0.00", "1.00") and value not in (0.0, 1.0):
        return f"{value:.3f}"
    return rounded


@dataclass
class ResultSet:
    problem: Problem
    method_name: str
    noise_levels: list[float]
    outlier_fractions: list[float]
    n_realizations: int
    dataset: object
    results: object

    @classmethod
    def load(
        cls,
        problem_name: str,
        method_name: str,
        noise_levels: Sequence[float],
        outlier_fractions: Sequence[float],
        n_realizations: int,
        data_dir: str | Path = "data",
        coeffs_dir: str | Path = "coeffs",
    ) -> "ResultSet":
        problem = get_problem(problem_name)
        dataset_path = io.dataset_path(problem.name, n_realizations, data_dir)
        results_path = io.results_path(problem.name, method_name, n_realizations, coeffs_dir)
        dataset = np.load(dataset_path, allow_pickle=True)
        results = np.load(results_path, allow_pickle=True)
        return cls(
            problem=problem,
            method_name=method_name.upper(),
            noise_levels=list(noise_levels),
            outlier_fractions=list(outlier_fractions),
            n_realizations=n_realizations,
            dataset=dataset,
            results=results,
        )

    def metric_grid(self, metric: str) -> np.ndarray:
        library = self.problem.feature_library()
        true_coeff = self.problem.true_coefficients(library)
        grid = np.zeros((len(self.noise_levels), len(self.outlier_fractions)))
        for i, nl in enumerate(self.noise_levels):
            for j, op in enumerate(self.outlier_fractions):
                cell = io.load_run_cell(self.results, nl, op)
                if metric == "trajectory_error":
                    grid[i, j] = np.mean(cell["trajectory_error"])
                elif metric == "accuracy":
                    grid[i, j] = np.mean(
                        [metrics.coefficient_accuracy(true_coeff, c) for c in cell["coefficients"]]
                    )
                elif metric == "exact_recovery":
                    grid[i, j] = np.mean(
                        [metrics.exact_recovery(true_coeff, c) for c in cell["coefficients"]]
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")
        return grid

    def all_times(self) -> np.ndarray:
        """Flat array of per-realization fit times (seconds), pooled across
        every noise/outlier grid cell."""
        times = []
        for nl in self.noise_levels:
            for op in self.outlier_fractions:
                cell = io.load_run_cell(self.results, nl, op)
                times.extend(cell["time"])
        return np.array(times, dtype=float)


def plot_metric_grid(
    result_sets: Sequence[ResultSet],
    metric: str,
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
):
    """Rows = problems, columns = methods; each cell a noise x outlier heatmap
    of `metric`. Generalizes the SINDy-vs-noise/outlier heatmap grids used in
    the paper notebook to an arbitrary list of ResultSets."""
    problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    vmax = 1.0 if metric != "trajectory_error" else None
    n_rows, n_cols = len(problems), len(methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        sharex="col",
        sharey="row",
    )

    # Scale text with the figure's physical size (relative to a 3x3 grid)
    # so method/problem names stay legible as the grid grows.
    scale = max(0.7, min(np.sqrt(n_rows * n_cols) / 3, 2.5))
    title_fontsize = 11 * scale
    label_fontsize = 11 * scale
    tick_fontsize = 9 * scale
    annot_fontsize = 8 * scale
    suptitle_fontsize = 16 * scale

    for i, problem_name in enumerate(problems):
        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                ax.axis("off")
                continue
            grid = rs.metric_grid(metric)
            is_last_column = j == len(methods) - 1
            annot_labels = np.array([[_format_metric(v) for v in row] for row in grid])
            ax = sb.heatmap(
                grid,
                annot=annot_labels,
                fmt="",
                annot_kws={"fontsize": annot_fontsize},
                cmap="magma",
                vmin=0,
                vmax=vmax,
                ax=ax,
                cbar=is_last_column,
                cbar_kws={"label": _METRIC_LABELS.get(metric, metric)} if is_last_column else None,
            )
            ax.invert_yaxis()
            ax.tick_params(labelsize=tick_fontsize)
            if is_last_column:
                cbar = ax.collections[0].colorbar
                assert cbar is not None
                cbar.ax.tick_params(labelsize=tick_fontsize)
                cbar.ax.yaxis.label.set_fontsize(label_fontsize)

            if i == len(problems) - 1:
                ax.set_xticklabels(
                    [f"{v * 100:g}%" for v in rs.outlier_fractions], rotation=45, ha="right"
                )
                ax.set_xlabel("Outlier percentage", fontsize=label_fontsize)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_yticklabels([f"{v * 100:g}%" for v in rs.noise_levels])
                ax.set_ylabel(f"{problem_name}\nNoise level", fontsize=label_fontsize)
            else:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(method_name, fontsize=title_fontsize)

    fig.suptitle(_METRIC_LABELS.get(metric, metric), fontsize=suptitle_fontsize)

    # Reserve top margin for the suptitle proportional to its actual size
    # (in inches, plus padding) so it doesn't collide with the top row's
    # per-subplot titles as suptitle_fontsize scales with the grid.
    fig_height = 4 * n_rows
    top_margin = (suptitle_fontsize / 72) * 2.5
    top = max(0.80, 1 - top_margin / fig_height)
    fig.tight_layout(rect=(0, 0, 1, top))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_execution_time(
    result_sets: Sequence[ResultSet],
    methods: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """One figure per problem, boxplotting per-realization fit time (seconds)
    for every method, pooled across the whole noise/outlier grid."""
    problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    n_methods = len(methods)
    scale = max(0.7, min(np.sqrt(n_methods) / 2, 2.0))
    title_fontsize = 14 * scale
    label_fontsize = 11 * scale
    tick_fontsize = 10 * scale

    figs: dict[str, Figure] = {}
    for problem_name in problems:
        data = []
        labels = []
        for method_name in methods:
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                continue
            data.append(rs.all_times())
            labels.append(method_name)

        fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 2, 5))
        sb.boxplot(data=data, ax=ax)
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel("Execution time (s)", fontsize=label_fontsize)
        ax.set_title(problem_name, fontsize=title_fontsize)
        fig.tight_layout()

        if output_dir is not None:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir_path / f"{problem_name}_time_boxplot.png", bbox_inches="tight")
        figs[problem_name] = fig
    return figs

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
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig

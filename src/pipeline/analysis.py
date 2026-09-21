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

_PROBLEM_LABELS = {
    "LV": "Lotka-Volterra",
    "ROSSLER": "Rössler",
    "ABC": "ABC Flow",
    "VAN_DER_POL": "Van der Pol",
    "SIR": "SIR",
    "LORENZ": "Lorenz",
}


def _problem_label(problem_name: str) -> str:
    return _PROBLEM_LABELS.get(problem_name, problem_name)


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

    def metric_grid(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean_grid, std_grid) of `metric` over noise x outlier."""
        library = self.problem.feature_library()
        true_coeff = self.problem.true_coefficients(library)
        mean_grid = np.zeros((len(self.noise_levels), len(self.outlier_fractions)))
        std_grid = np.zeros((len(self.noise_levels), len(self.outlier_fractions)))
        for i, nl in enumerate(self.noise_levels):
            for j, op in enumerate(self.outlier_fractions):
                cell = io.load_run_cell(self.results, nl, op)
                if metric == "trajectory_error":
                    mean_grid[i, j] = np.mean(cell["trajectory_error"])
                    std_grid[i, j] = np.std(cell["trajectory_error"])
                elif metric == "accuracy":
                    mean_grid[i, j] = np.mean(
                        [metrics.coefficient_accuracy(true_coeff, c) for c in cell["coefficients"]]
                    )
                    std_grid[i, j] = np.std(
                        [metrics.coefficient_accuracy(true_coeff, c) for c in cell["coefficients"]]
                    )
                elif metric == "exact_recovery":
                    std_grid[i, j] = np.std(
                        [metrics.exact_recovery(true_coeff, c) for c in cell["coefficients"]]
                    )
                    mean_grid[i, j] = np.mean(
                        [metrics.exact_recovery(true_coeff, c) for c in cell["coefficients"]]
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")
        return mean_grid, std_grid

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
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, fig_height),
        squeeze=False,
        sharex="col",
        sharey="row",
        gridspec_kw={"wspace": 0.08, "hspace": 0.12},
    )

    # Scale text with the figure's physical size (relative to a 3x3 grid)
    # so method/problem names stay legible as the grid grows.
    scale = max(1.0, min(np.sqrt(n_rows * n_cols) / 3, 3.0))
    title_fontsize = 15 * scale
    label_fontsize = 15 * scale
    tick_fontsize = 8 * scale
    suptitle_fontsize = 20 * scale

    # Each subplot's own physical size is fixed (6in x 5in) regardless of how
    # many problems/methods there are, so the annotation font is sized from
    # the heatmap's own noise x outlier cell count — the thing that actually
    # constrains how much room each annotation has — not from the number of
    # subplots (`scale` above).
    grid_rows = len(result_sets[0].noise_levels)
    grid_cols = len(result_sets[0].outlier_fractions)
    heatmap_width_in, heatmap_height_in = 4.8, 3.9
    cell_width_pt = (heatmap_width_in / grid_cols) * 72
    cell_height_pt = (heatmap_height_in / grid_rows) * 72
    # Two lines of text ("mean" and "±std") need roughly 2.6x the font size in
    # vertical space. The "±std" line is the wider of the two (up to 5 chars,
    # e.g. "±0.12") at roughly 0.6x the font size per character; clamp to a
    # sane readable range either way.
    annot_fontsize = max(6.0, min(cell_height_pt / 2.6, cell_width_pt / (5 * 0.6), 11.0))

    last_mappable = None
    heatmap_axes = []
    for i, problem_name in enumerate(problems):
        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                ax.axis("off")
                continue
            mean_grid, std_grid = rs.metric_grid(metric)
            annot_labels = np.array(
                [
                    [f"{_format_metric(m)}\n±{s:.2f}" for m, s in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(mean_grid, std_grid)
                ]
            )
            ax = sb.heatmap(
                mean_grid,
                annot=annot_labels,
                fmt="",
                annot_kws={"fontsize": annot_fontsize},
                cmap="magma",
                vmin=0,
                vmax=vmax,
                ax=ax,
                cbar=False,
                yticklabels=[f"{v * 100:g}%" for v in rs.noise_levels]
            )
            last_mappable = ax.collections[0]
            heatmap_axes.append(ax)
            ax.invert_yaxis()
            ax.tick_params(labelsize=tick_fontsize, rotation=45)

            if i == len(problems) - 1:
                ax.set_xticks(
                    np.arange(len(rs.outlier_fractions)) + 0.5,
                    labels=[f"{v * 100:g}%" for v in rs.outlier_fractions],
                    ha="right",
                )
                ax.set_xlabel("Outlier percentage", fontsize=label_fontsize)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                # ax.set_yticks(
                #     np.arange(len(rs.noise_levels)) + 0.5,
                #     labels=[f"{v * 100:g}%" for v in rs.noise_levels],
                # )
                ax.set_ylabel(f"{_problem_label(problem_name)}\nNoise level", fontsize=label_fontsize)
            else:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(f'\n{method_name}', fontsize=title_fontsize)

    fig.suptitle(_METRIC_LABELS.get(metric, metric), fontsize=suptitle_fontsize)

    # One shared colorbar spanning every row (all heatmaps use the same
    # vmin/vmax), instead of one per row — added after tight_layout, which
    # shrinks the existing axes to make room for it. `fraction` is relative
    # to the *combined* width of every axes passed in, so a fixed fraction
    # would make the colorbar thinner and thinner as more method columns are
    # added — instead derive it from a fixed target width in inches.
    if last_mappable is not None:
        cbar = fig.colorbar(
            last_mappable, ax=axes.ravel().tolist(), label="Accuracy", aspect=50
        )
        cbar.set_label(_METRIC_LABELS.get(metric, metric), fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_execution_time(
    result_sets: Sequence[ResultSet],
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """One figure, one axis per problem (grid as square as possible), each
    boxplotting per-realization fit time (seconds) for every method, pooled
    across the whole noise/outlier grid. Also prints mean +- std per
    (problem, method)."""
    problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    n_problems = len(problems)
    n_cols = int(np.ceil(np.sqrt(n_problems)))
    n_rows = int(np.ceil(n_problems / n_cols))

    n_methods = len(methods)
    scale = max(0.7, min(np.sqrt(n_methods) / 2, 2.0))
    title_fontsize = 14 * scale
    label_fontsize = 11 * scale
    tick_fontsize = 12 * scale
    annot_fontsize = 12 * scale
    suptitle_fontsize = 18 * scale

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4, n_methods + 1.5) * n_cols, 5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, problem_name in enumerate(problems):
        ax = axes_flat[idx]
        data = []
        labels = []
        print(_problem_label(problem_name))
        for method_name in methods:
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                continue
            times = rs.all_times()
            data.append(times)
            labels.append(method_name)
            print(f"  {method_name}: {times.mean():.4g} +- {times.std():.4g} s")

        sb.boxplot(data=data, ax=ax)
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel("Execution time (s)", fontsize=label_fontsize)
        ax.set_title(_problem_label(problem_name), fontsize=title_fontsize)

        # Annotate each box with its mean +- std, above the box's max value
        # (multiplicative offset since the axis is log-scaled).
        for k, times in enumerate(data):
            ax.text(
                k,
                times.max() * 1.2,
                f"{times.mean():.3g}\n±{times.std():.3g}",
                ha="center",
                va="bottom",
                fontsize=annot_fontsize,
            )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 2.2)

    for idx in range(n_problems, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Execution time", fontsize=suptitle_fontsize)

    fig_height = 5 * n_rows
    top_margin = (suptitle_fontsize / 72) * 2.5
    top = max(0.80, 1 - top_margin / fig_height)
    fig.tight_layout(rect=(0, 0, 1, top))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig

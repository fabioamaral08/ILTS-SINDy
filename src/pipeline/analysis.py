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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from . import io, metrics
from .problems import get_problem
from .problems.base import Problem

_METRIC_LABELS = {
    "accuracy": "Coefficient accuracy",
    "exact_recovery": "Exact model recovery",
    "trajectory_error": "Trajectory error (NRMSE)",
    "coefficient_error": "Coefficient estimate error",
}

# Metrics not bounded to [0, 1] (accuracy/exact_recovery are), so heatmaps
# shouldn't cap their color scale's vmax at 1.0 for these.
_UNBOUNDED_METRICS = {"trajectory_error", "coefficient_error"}

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


_METHOD_LABELS = {
    "SINDY-LTS-SEARCH": "ILTS-SINDy (p-search)",
    "SINDY-LTS-AIC": "ILTS-SINDy ($\\lambda$-search)",
    "SINDY-LTS-SLOPE": "ILTS-SINDy ($\\lambda$-search)",
    "ESINDY": "E-SINDy",
    "WSINDY": "W-SINDy",
    "SINDY": "SINDy",
    "SINDY-LTS": "ILTS-SINDy",
}


def _method_label(method_name: str) -> str:
    return _METHOD_LABELS.get(method_name, method_name)


# Single hardcoded knobs for plot_execution_time's / plot_eps_histogram's
# text sizing — edit these directly to tune them, rather than them being
# auto-derived from the number of methods (which was making them too small).
_TIME_PLOT_FONTSCALE = 1.2
_EPS_PLOT_FONTSCALE = 1.5


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
                elif metric == "coefficient_error":
                    mean_grid[i, j] = np.mean(
                        [metrics.coefficient_error(true_coeff, c) for c in cell["coefficients"]]
                    )
                    std_grid[i, j] = np.std(
                        [metrics.coefficient_error(true_coeff, c) for c in cell["coefficients"]]
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")
        return mean_grid, std_grid

    def pooled_metric(self, metric: str, noise_level: float) -> tuple[float, float]:
        """Mean and std of `metric` at a fixed noise level, pooling every
        realization across all outlier fractions — i.e. aggregating raw
        per-realization values, not averaging each outlier fraction's own
        mean (which would weight every outlier level equally regardless of
        how much it actually varies). Same pooling approach as `all_times`."""
        library = self.problem.feature_library()
        true_coeff = self.problem.true_coefficients(library)
        values: list[float] = []
        for op in self.outlier_fractions:
            cell = io.load_run_cell(self.results, noise_level, op)
            if metric == "trajectory_error":
                values.extend(cell["trajectory_error"])
            elif metric == "accuracy":
                values.extend(
                    metrics.coefficient_accuracy(true_coeff, c) for c in cell["coefficients"]
                )
            elif metric == "exact_recovery":
                values.extend(
                    metrics.exact_recovery(true_coeff, c) for c in cell["coefficients"]
                )
            elif metric == "coefficient_error":
                values.extend(
                    metrics.coefficient_error(true_coeff, c) for c in cell["coefficients"]
                )
            else:
                raise ValueError(f"Unknown metric: {metric!r}")
        return float(np.mean(values)), float(np.std(values))

    def all_times(self) -> np.ndarray:
        """Flat array of per-realization fit times (seconds), pooled across
        every noise/outlier grid cell."""
        times = []
        for nl in self.noise_levels:
            for op in self.outlier_fractions:
                cell = io.load_run_cell(self.results, nl, op)
                if cell["time"] is None:
                    raise ValueError(
                        f"{self.problem.name}/{self.method_name}: results file has no "
                        "fit-time data (saved before time tracking was added) — "
                        "rerun cli_run_method.py to regenerate it."
                    )
                times.extend(cell["time"])
        return np.array(times, dtype=float)

    def all_eps(self) -> np.ndarray:
        """Flat array of per-state AICc-chosen thresholds (extra['eps'], see
        lts.SINDy_LTS_eps), pooled across every noise/outlier grid cell,
        realization, and state. Only meaningful for methods that record
        'eps' in extra (e.g. SINDY-LTS-EPS)."""
        eps_values = []
        for nl in self.noise_levels:
            for op in self.outlier_fractions:
                cell = io.load_run_cell(self.results, nl, op)
                extra = cell["extra"]
                if extra is None:
                    raise ValueError(
                        f"{self.problem.name}/{self.method_name}: results file has no "
                        "'extra' data at all — rerun the method to regenerate it."
                    )
                for e in extra:
                    if "eps" not in e:
                        raise ValueError(
                            f"{self.problem.name}/{self.method_name}: 'eps' not found in "
                            "extra — this method doesn't record a chosen threshold "
                            "(use SINDY-LTS-EPS)."
                        )
                    eps_values.extend(np.atleast_1d(e["eps"]))
        return np.array(eps_values, dtype=float)


def plot_metric_grid(
    result_sets: Sequence[ResultSet],
    metric: str,
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    annot: bool = True,
    cell_size: float = 0.6,
):
    """Rows = problems, columns = methods; each cell a noise x outlier heatmap
    of `metric`. Generalizes the SINDy-vs-noise/outlier heatmap grids used in
    the paper notebook to an arbitrary list of ResultSets. If `annot` is
    False, cells show only the colormap, with no mean/std text. `cell_size`
    is the physical size (inches) of one heatmap cell; each subplot's size is
    derived from it and the noise/outlier grid dimensions, rather than being
    fixed regardless of how dense that grid is."""
    problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    # Compute every cell's grid once up front (instead of inside the plotting
    # loop below) so we can also derive a single vmin/vmax shared by every
    # heatmap axis from the actual pooled data — otherwise, for unbounded
    # metrics, each axis would autoscale to its own min/max independently,
    # making the one shared colorbar meaningless for every axis but the last.
    grids = {}
    for problem_name in problems:
        for method_name in methods:
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is not None:
                grids[(problem_name, method_name)] = rs.metric_grid(metric)

    # Unbounded error metrics (trajectory_error, coefficient_error) typically
    # span several orders of magnitude across noise/outlier levels, so a
    # log-scale colorbar shows contrast at the low end that a linear scale
    # would flatten. LogNorm requires vmin > 0, so floor it at the smallest
    # *positive* value seen anywhere in the grid (any exact-zero cells are
    # clipped up to that floor just for color mapping, below).
    if metric in _UNBOUNDED_METRICS:
        all_values = np.concatenate([mean_grid.ravel() for mean_grid, _ in grids.values()])
        positive_values = all_values[all_values > 0]
        vmax = float(np.nanmax(all_values))
        vmin = float(np.nanmin(positive_values)) if positive_values.size else 1e-10
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = 0.0, 1.0
        norm = None
    n_rows, n_cols = len(problems), len(methods)

    # Each subplot's heatmap area is `cell_size` inches per cell, plus a
    # fixed margin for tick labels/axis labels/title around it — so the
    # annotation font (below) can be sized from how much room each cell
    # actually has, and the whole figure scales with both the noise/outlier
    # grid density and the number of problems/methods.
    grid_rows = len(result_sets[0].noise_levels)
    grid_cols = len(result_sets[0].outlier_fractions)
    margin_width_in, margin_height_in = 1.2, 1.1
    heatmap_width_in = cell_size * grid_cols
    heatmap_height_in = cell_size * grid_rows
    subplot_width_in = heatmap_width_in + margin_width_in
    subplot_height_in = heatmap_height_in + margin_height_in

    fig_height = subplot_height_in * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_width_in * n_cols, fig_height),
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
    cblabel_fontsize = 18 * scale
    tick_fontsize = 12 * scale
    cbtick_fontsize = 15 * scale
    suptitle_fontsize = 20 * scale

    cell_width_pt = (heatmap_width_in / grid_cols) * 72
    cell_height_pt = (heatmap_height_in / grid_rows) * 72
    # Two lines of text ("mean" and "±std") need roughly 2.6x the font size in
    # vertical space. The "±std" line is the wider of the two (up to 5 chars,
    # e.g. "±0.12") at roughly 0.6x the font size per character; clamp to a
    # sane readable range either way.
    annot_fontsize = max(5.0, min(cell_height_pt / 2.6, cell_width_pt / (5 * 0.6), 9.0))
    print(annot_fontsize)
    last_mappable = None
    heatmap_axes = []
    for i, problem_name in enumerate(problems):
        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                ax.axis("off")
                continue
            mean_grid, std_grid = grids[(problem_name, method_name)]
            if annot:
                annot_labels = np.array(
                    [
                        [f"{_format_metric(m)}\n±{s:.2f}" for m, s in zip(mean_row, std_row)]
                        for mean_row, std_row in zip(mean_grid, std_grid)
                    ]
                )
            else:
                annot_labels = False
            # Clip to the log floor only for color mapping — annot_labels
            # above already captured the true (unclipped) mean, so a genuine
            # zero-error cell still displays "0.00" even though it's drawn as
            # the darkest color rather than being masked out by LogNorm.
            color_data = np.clip(mean_grid, vmin, None) if norm is not None else mean_grid
            ax = sb.heatmap(
                color_data,
                annot=annot_labels,
                fmt="",
                annot_kws={"fontsize": annot_fontsize} if annot else None,
                cmap="magma",
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                ax=ax,
                cbar=False,
                yticklabels=[f"{v * 100:.1f}%" for v in rs.noise_levels]
            )
            if norm is not None:
                # seaborn always passes its own linear vmin/vmax into
                # pcolormesh, so a LogNorm can't be passed in directly
                # (matplotlib rejects norm + vmin/vmax together) — instead
                # swap it in on the resulting mesh afterwards, sharing the
                # same LogNorm instance across every axis for the colorbar.
                ax.collections[0].set_norm(norm)
            last_mappable = ax.collections[0]
            heatmap_axes.append(ax)
            ax.invert_yaxis()
            ax.tick_params(labelsize=tick_fontsize)
            ax.tick_params(axis="y", rotation=45)

            if i == len(problems) - 1:
                ax.set_xticks(
                    np.arange(len(rs.outlier_fractions)) + 0.5,
                    labels=[f"{v * 100:.1f}%" for v in rs.outlier_fractions],
                    rotation=45,
                    ha="right",
                )
                # "Outlier percentage" is the same for every column, so show
                # it once, centered under the bottom row, instead of
                # repeating it under every column.
                if j == n_cols // 2:
                    ax.set_xlabel("Outlier percentage", fontsize=label_fontsize)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                # The problem name identifies the row and stays on every row;
                # "Noise level" is the same for every row, so show it once,
                # centered on the middle row, instead of repeating it.
                ylabel = _problem_label(problem_name)
                if i == n_rows // 2:
                    ylabel += "\nNoise level"
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
            else:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(_method_label(method_name), fontsize=title_fontsize)

    # Reserve room for the suptitle's own rendered height (line-height is
    # roughly 1.2x the font size, to cover ascenders/descenders) plus a
    # genuine 1 em gap below it before the top row's axis titles — pinning
    # `y` to the figure's original top edge (not pushed down) so the text
    # itself doesn't encroach on the reserved gap.
    original_top = fig.subplotpars.top
    em_in = suptitle_fontsize / 72
    reserved_in = 1.2 * em_in + em_in
    new_top = original_top - reserved_in / fig_height
    fig.subplots_adjust(top=new_top)
    fig.suptitle(_METRIC_LABELS.get(metric, metric), fontsize=suptitle_fontsize, y=original_top)

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
        cbar.set_label(_METRIC_LABELS.get(metric, metric), fontsize=cblabel_fontsize)
        cbar.ax.tick_params(labelsize=cbtick_fontsize)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_metric_row(
    result_set: ResultSet,
    metrics: Sequence[str] = ("accuracy", "exact_recovery"),
    annot: bool = True,
    cell_size: float = 0.6,
    output_path: str | Path | None = None,
) -> Figure:
    """A single row of noise x outlier heatmaps, one column per metric in
    `metrics`, for one (problem, method) ResultSet — a single-(problem,
    method) counterpart to `plot_metric_grid`, which instead fixes the
    metric and varies problems/methods. Each metric gets its own colorbar
    (accuracy/exact_recovery are bounded [0,1]; trajectory_error isn't, so a
    shared colorbar wouldn't make sense)."""
    n_cols = len(metrics)
    grid_rows = len(result_set.noise_levels)
    grid_cols = len(result_set.outlier_fractions)

    # Same cell_size-driven sizing as plot_metric_grid, but each axis needs
    # its own colorbar (metrics have different scales), so the per-axis
    # margin is wider.
    margin_width_in, margin_height_in = 1.7, 1.1
    heatmap_width_in = cell_size * grid_cols
    heatmap_height_in = cell_size * grid_rows
    subplot_width_in = heatmap_width_in + margin_width_in
    subplot_height_in = heatmap_height_in + margin_height_in

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(subplot_width_in * n_cols, subplot_height_in),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    # scale = max(1.0, min(np.sqrt(n_cols) / 2, 1.))
    scale = 0.7
    title_fontsize = 25 * scale
    label_fontsize = 20 * scale
    cblabel_fontsize = 20 * scale
    tick_fontsize = 18 * scale
    cbtick_fontsize = 12 * scale
    suptitle_fontsize = 25 * scale

    cell_width_pt = (heatmap_width_in / grid_cols) * 72
    cell_height_pt = (heatmap_height_in / grid_rows) * 72
    annot_fontsize = max(5.0, min(cell_height_pt / 2.6, cell_width_pt / (5 * 0.6), 10.0))

    for j, metric in enumerate(metrics):
        ax = axes[j]
        mean_grid, std_grid = result_set.metric_grid(metric)
        vmax = None if metric in _UNBOUNDED_METRICS else 1.0
        if annot:
            annot_labels = np.array(
                [
                    [f"{_format_metric(m)}\n±{s:.2f}" for m, s in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(mean_grid, std_grid)
                ]
            )
        else:
            annot_labels = False
        ax = sb.heatmap(
            mean_grid,
            annot=annot_labels,
            fmt="",
            annot_kws={"fontsize": annot_fontsize} if annot else None,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            ax=ax,
            cbar=True,
            cbar_kws={"label": _METRIC_LABELS.get(metric, metric)},
            yticklabels=[f"{v * 100:.1f}%" for v in result_set.noise_levels],
        )
        ax.invert_yaxis()
        ax.tick_params(labelsize=tick_fontsize)
        ax.tick_params(axis="y", rotation=45)

        cbar = ax.collections[0].colorbar
        assert cbar is not None
        cbar.ax.tick_params(labelsize=cbtick_fontsize)
        cbar.set_label(_METRIC_LABELS.get(metric, metric), fontsize=cblabel_fontsize)

        ax.set_xticks(
            np.arange(len(result_set.outlier_fractions)) + 0.5,
            labels=[f"{v * 100:.1f}%" for v in result_set.outlier_fractions],
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("Outlier percentage", fontsize=label_fontsize)
        if j == 0:
            ax.set_ylabel("Noise level", fontsize=label_fontsize)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=title_fontsize)

    # Same approach as plot_metric_grid: pin the suptitle's own y to the
    # figure's *original* top margin (not matplotlib's default y=0.98) and
    # push the axes down from there by the suptitle's rendered height plus a
    # small gap — so the blank space between them is exactly that gap, not
    # whatever happens to be left between 0.98 and wherever the axes land.
    # Unlike plot_metric_grid (whose suptitle is noticeably larger than its
    # per-axis titles), title_fontsize == suptitle_fontsize here, and each
    # metric's title floats just above its axes' top edge (i.e. right at
    # new_top) — so the reserved gap has to clear both the suptitle's own
    # height *and* the axis title's, not just the suptitle's.
    # No floor here: original_top is already matplotlib's own default (not
    # a fixed constant), so clamping new_top to some absolute value can end
    # up *larger* than original_top — i.e. push the axes higher than where
    # the (unmoved) suptitle sits, putting the title inside the plot.
    original_top = fig.subplotpars.top
    suptitle_em_in = suptitle_fontsize / 72
    title_em_in = title_fontsize / 72
    reserved_in = 1.2 * suptitle_em_in + 0.3 * suptitle_em_in + 1.2 * title_em_in
    new_top = original_top - reserved_in / subplot_height_in
    fig.subplots_adjust(top=new_top)
    fig.suptitle(
        f"{_problem_label(result_set.problem.name)} — {_method_label(result_set.method_name)}",
        fontsize=suptitle_fontsize,
        y=original_top,
    )

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
    n_rows = int(np.ceil(np.sqrt(n_problems)))
    n_cols = int(np.ceil(n_problems / n_rows))

    n_methods = len(methods)
    scale = _TIME_PLOT_FONTSCALE
    title_fontsize = 18 * scale
    label_fontsize = 15 * scale
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
            labels.append(_method_label(method_name))
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


def plot_eps_histogram(
    result_sets: Sequence[ResultSet],
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """One figure, one axis per problem (grid as square as possible), bar
    chart of how often each AICc-chosen threshold (extra['eps'], see
    lts.SINDy_LTS_eps) was picked, pooled across the whole noise/outlier
    grid, every realization, and every state. AIC only ever picks among a
    small fixed set of candidate thresholds (lts.py's eps_list), so this is
    a discrete count per value rather than a continuous histogram. Only
    meaningful for methods that record 'eps' in extra (e.g. SINDY-LTS-EPS).
    Also prints the counts per (problem, method)."""
    problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    n_problems = len(problems)
    n_cols = int(np.ceil(np.sqrt(n_problems)))
    n_rows = int(np.ceil(n_problems / n_cols))

    scale = _EPS_PLOT_FONTSCALE
    title_fontsize = 18 * scale
    label_fontsize = 15 * scale
    tick_fontsize = 12 * scale
    suptitle_fontsize = 18 * scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, problem_name in enumerate(problems):
        ax = axes_flat[idx]
        print(_problem_label(problem_name))

        per_method_eps = {}
        for method_name in methods:
            rs = lookup.get((problem_name, method_name.upper()))
            if rs is None:
                continue
            per_method_eps[method_name] = rs.all_eps()
        if not per_method_eps:
            ax.axis("off")
            continue

        unique_eps = np.unique(np.concatenate(list(per_method_eps.values())))
        width = 0.8 / max(len(per_method_eps), 1)
        x_base = np.arange(len(unique_eps))

        for m_idx, (method_name, eps_values) in enumerate(per_method_eps.items()):
            counts = np.array([np.sum(np.isclose(eps_values, e)) for e in unique_eps])
            ax.bar(x_base + m_idx * width, counts, width, label=_method_label(method_name))
            breakdown = ", ".join(f"{e:.0e}={c}" for e, c in zip(unique_eps, counts))
            print(f"  {method_name}: {breakdown}")

        ax.set_xticks(x_base + width * (len(per_method_eps) - 1) / 2)
        ax.set_xticklabels(
            [f"{e:.0e}" for e in unique_eps], rotation=45, ha="right", fontsize=tick_fontsize
        )
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_xlabel("Chosen threshold (eps)", fontsize=label_fontsize)
        ax.set_ylabel("Count", fontsize=label_fontsize)
        ax.set_title(_problem_label(problem_name), fontsize=title_fontsize)
        if len(per_method_eps) > 1:
            ax.legend(fontsize=tick_fontsize)

    for idx in range(n_problems, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("AICc-chosen threshold distribution", fontsize=suptitle_fontsize)

    fig_height = 4 * n_rows
    top_margin = (suptitle_fontsize / 72) * 2.5
    top = max(0.80, 1 - top_margin / fig_height)
    fig.tight_layout(rect=(0, 0, 1, top))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


_LOWER_IS_BETTER = {"trajectory_error", "coefficient_error"}


def latex_metric_table(
    result_sets: Sequence[ResultSet],
    metric: str,
    noise_level: float,
    problems: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """LaTeX (booktabs) table of `metric` at a fixed noise level, aggregated
    (mean +- std) across every outlier percentage — rows = problems,
    columns = methods. The best method per problem (highest mean, or lowest
    for trajectory_error) is bolded. Also prints the same table to stdout."""
    if problems is None:
        problems = list(dict.fromkeys(rs.problem.name for rs in result_sets))
    if methods is None:
        methods = list(dict.fromkeys(rs.method_name for rs in result_sets))
    lookup = {(rs.problem.name, rs.method_name): rs for rs in result_sets}

    metric_label = _METRIC_LABELS.get(metric, metric)
    header = " & ".join(["Problem"] + [_method_label(m).replace("\n", " ") for m in methods])

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for problem_name in problems:
        row_stats = {
            method_name: rs.pooled_metric(metric, noise_level)
            for method_name in methods
            if (rs := lookup.get((problem_name, method_name.upper()))) is not None
        }
        best_method = None
        if row_stats:
            pick = min if metric in _LOWER_IS_BETTER else max
            best_method = pick(row_stats, key=lambda m: row_stats[m][0])

        row = [_problem_label(problem_name)]
        for method_name in methods:
            if method_name not in row_stats:
                row.append("--")
                continue
            mean, std = row_stats[method_name]
            value = f"{_format_metric(mean)} \\pm {std:.2f}"
            cell = f"$\\mathbf{{{value}}}$" if method_name == best_method else f"${value}$"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = (
        f"\\caption{{{metric_label} at noise level {noise_level * 100:g}\\%, "
        f"aggregated across all outlier percentages; best method per problem in bold.}}"
    )
    lines.append(caption)
    lines.append(f"\\label{{tab:{metric}_noise_{noise_level:g}}}")
    lines.append(r"\end{table}")
    table = "\n".join(lines)

    print(table)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table, encoding="utf-8")
    return table

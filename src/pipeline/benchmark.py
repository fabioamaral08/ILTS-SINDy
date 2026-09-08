"""Times each Method's `fit` call using a single representative hyperparameter
combination (`Method.default_hyperparams`, not the full grid searched by
`pipeline.runner`), for comparing raw execution cost across methods and
problems. Timed once per independent noisy realization (not repeats of the
same data), so the reported std reflects data-dependent variability in fit
time, not just measurement noise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .methods.base import Method
from .noise import add_noise
from .problems.base import Problem


@dataclass
class TimingResult:
    problem_name: str
    method_name: str
    hyperparams: dict
    seconds: list[float]

    @property
    def mean_seconds(self) -> float:
        return float(np.mean(self.seconds))

    @property
    def std_seconds(self) -> float:
        return float(np.std(self.seconds))


def time_method(
    method: Method,
    problem: Problem,
    data_realizations: Sequence[np.ndarray],
    t: np.ndarray,
    library=None,
    n_warmup: int = 1,
) -> TimingResult:
    """Times one `method.fit(data, t, library, **default_hyperparams)` call
    per realization in `data_realizations` (independent noisy draws, not
    repeats of the same data), so the reported std reflects how fit time
    varies with the data itself (e.g. outlier count/placement for LTS-style
    methods) rather than pure measurement noise. The first `n_warmup`
    realizations are fit once, untimed, to absorb first-call overhead, and
    excluded from the timed/reported set."""
    library = library if library is not None else problem.feature_library()
    hyperparams = method.default_hyperparams(problem)

    warmup, timed = data_realizations[:n_warmup], data_realizations[n_warmup:]
    for data in warmup:
        method.fit(data, t, library, **hyperparams)

    seconds = []
    for data in timed:
        start = time.perf_counter()
        method.fit(data, t, library, **hyperparams)
        seconds.append(time.perf_counter() - start)

    return TimingResult(
        problem_name=problem.name,
        method_name=method.name,
        hyperparams=hyperparams,
        seconds=seconds,
    )


def benchmark(
    problems: Sequence[Problem],
    methods: Sequence[Method],
    noise_level: float = 0.0,
    outlier_fraction: float = 0.0,
    n_realizations: int = 5,
    n_warmup: int = 1,
    seed: int | None = None,
) -> list[TimingResult]:
    """Times every (problem, method) pair on `n_realizations` independent
    noisy realizations of that problem (plus `n_warmup` untimed ones), so the
    same realizations are reused across methods for a fair comparison."""
    if seed is not None:
        np.random.seed(seed)

    results = []
    for problem in problems:
        _, clean = problem.simulate()
        t = problem.time_vector()
        data_realizations = [
            add_noise(clean, noise_level, outlier_fraction)[0]
            for _ in range(n_warmup + n_realizations)
        ]
        for method in methods:
            results.append(time_method(method, problem, data_realizations, t, n_warmup=n_warmup))
    return results


def plot_timings(results: Sequence[TimingResult], output_path: str | Path | None = None):
    """Grouped bar chart of mean fit time per method, one bar group per
    problem, log-scaled y-axis (fit times commonly span orders of magnitude
    across methods)."""
    problems = list(dict.fromkeys(r.problem_name for r in results))
    methods = list(dict.fromkeys(r.method_name for r in results))
    lookup = {(r.problem_name, r.method_name): r for r in results}

    x = np.arange(len(methods))
    width = 0.8 / max(len(problems), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(methods) + 2, 5))
    for i, problem_name in enumerate(problems):
        means = [lookup[(problem_name, m)].mean_seconds if (problem_name, m) in lookup else 0.0 for m in methods]
        stds = [lookup[(problem_name, m)].std_seconds if (problem_name, m) in lookup else 0.0 for m in methods]
        ax.bar(x + i * width, means, width, yerr=stds, label=problem_name, capsize=3)

    ax.set_xticks(x + width * (len(problems) - 1) / 2)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Fit time (s)")
    ax.set_yscale("log")
    ax.set_title("Method execution time (fixed default hyperparameters, no grid search)")
    ax.legend(title="Problem")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig

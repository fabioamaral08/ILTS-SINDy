"""Runs SINDY-LTS with threshold=None (per-state AICc threshold search, see
`lts.SINDy_LTS`/`lts.AIC`) over a Problem's noisy dataset grid, scoring each
fit by trajectory prediction error and saving the best result per grid cell /
realization. Mirrors `pipeline.inliers_search`'s p-search runner, but
exercises SINDY-LTS's AIC-based threshold search instead of
SINDY-LTS-SEARCH's internal p search — `p` still has to be supplied here
since (unlike SINDY-LTS-SEARCH) SINDY-LTS doesn't search it, so it's derived
per grid cell from the outlier fraction the same way `pipeline.runner` does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

import lts
from . import io, metrics
from .methods import get_method
from .problems.base import Problem

# Distinct from the "SINDY-LTS" method name so a fixed-threshold run and an
# AIC-search run of the same method never collide on the same results file.
_SAVE_NAME = "SINDY-LTS-AIC"


@dataclass
class RunResult:
    coefficients: np.ndarray
    hyperparams: dict
    trajectory_error: float
    time: float
    extra: dict = field(default_factory=dict)


class MethodRunner:
    def __init__(self, problem: Problem, library=None):
        self.problem = problem
        self.method = get_method("SINDY-LTS")
        self.library = library if library is not None else problem.feature_library()

    def _score(self, coefficients: np.ndarray, data: np.ndarray, t: np.ndarray) -> float:
        x0 = data[0]
        t_eval = np.linspace(*self.problem.t_span, data.shape[0])
        sim = lts.simulate(coefficients, self.library, x0, self.problem.t_span, t_eval)
        return metrics.trajectory_error(data, sim)

    def run_single(self, data: np.ndarray, t: np.ndarray, p: int, hyperparams: dict) -> RunResult:
        hyperparams = {**hyperparams, "p": p}
        fit_result = self.method.fit(data, t, self.library, None, **hyperparams)
        score = self._score(fit_result.coefficients, data, t)
        return RunResult(
            coefficients=fit_result.coefficients,
            hyperparams=hyperparams,
            trajectory_error=score,
            time=fit_result.time,
            extra=fit_result.extra,
        )

    def run_grid(
        self,
        dataset_path: str | Path,
        noise_levels: Sequence[float],
        outlier_fractions: Sequence[float],
        n_realizations: int,
        output_dir: str | Path = "coeffs",
        n_jobs: int = -1,
    ) -> Path:
        dataset = np.load(dataset_path, allow_pickle=True)
        t = self.problem.time_vector()
        hyperparams = {"threshold": None}

        cells: list[tuple[float, float]] = [
            (nl, op) for nl in noise_levels for op in outlier_fractions
        ]
        tasks: list[tuple[float, float, np.ndarray, int]] = []
        for noise_level, outlier_fraction in cells:
            data_realizations, _ = io.load_dataset_cell(dataset, noise_level, outlier_fraction)
            m = data_realizations[0].shape[0]
            p = int(m - (3 * outlier_fraction * m))
            for data in data_realizations:
                tasks.append((noise_level, outlier_fraction, data, p))

        # AIC search reruns SINDy per candidate threshold on top of the
        # existing per-realization ILTS fit, so each task is noticeably
        # slower than a fixed-threshold run — parallelize across processes
        # the same way inliers_search/runner do.
        flat_results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_single)(data, t, p, hyperparams) for _, _, data, p in tasks
        )

        results: dict[float, dict[float, list]] = {
            nl: {op: [] for op in outlier_fractions} for nl in noise_levels
        }
        for (noise_level, outlier_fraction, _, _), run_result in zip(tasks, flat_results):
            results[noise_level][outlier_fraction].append(run_result)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = io.results_path(self.problem.name, _SAVE_NAME, n_realizations, output_dir)
        io.save_run_results(path, results)
        return path

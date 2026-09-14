"""Runs an identification Method on a Problem's noisy datasets, grid-searching
hyperparameters by trajectory prediction error and saving the best result per
grid cell / realization.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import pysindy as ps

import numpy as np
from joblib import Parallel, delayed

import lts
from . import io, metrics
from .methods import get_method
from .problems.base import Problem


@dataclass
class RunResult:
    coefficients: np.ndarray
    hyperparams: dict
    trajectory_error: float
    extra: dict = field(default_factory=dict)


class MethodRunner:
    def __init__(self, problem: Problem, library=None):
        self.problem = problem
        self.method =get_method('SINDY-LTS-SEARCH')
        self.library = library if library is not None else problem.feature_library()

    def _score(self, coefficients: np.ndarray, data: np.ndarray, t: np.ndarray) -> float:
    
        ## Compute the Akaike information criterion (AIC)
        D = self.library.fit_transform(data)
        data_dot = ps.FiniteDifference()._differentiate(data, t=t)
        RSS = metrics.derivative_error(data_dot, coefficients, D)
        l = data.shape[0]
        k = np.count_nonzero(coefficients)
        AIC = l * np.log(RSS/l) + 2*k
        AIC += (2*(k+1)*(k+2))/(l-k-2)
        return AIC


    def run_single(self, data: np.ndarray, t: np.ndarray, hyperparams) -> RunResult:
        fit_result = self.method.fit(data, t, self.library, **hyperparams)
        score = self._score(fit_result.coefficients, data, t)
        result = RunResult(
            coefficients=fit_result.coefficients,
            hyperparams=hyperparams,
            trajectory_error=score,
            extra=fit_result.extra
        )
        return result

    def run_grid(
        self,
        dataset_path: str | Path,
        noise_levels: Sequence[float],
        outlier_fractions: Sequence[float],
        n_realizations: int,
        hyperparams: dict,
        output_dir: str | Path = "coeffs",
        n_jobs: int = -1,
    ) -> Path:
        dataset = np.load(dataset_path, allow_pickle=True)
        t = self.problem.time_vector()

        # Each (noise_level, outlier_fraction, realization) grid-search is
        # independent, and a single one can take seconds (it simulates every
        # hyperparameter candidate to score it) — at the scale of a full
        # 8x8 noise/outlier grid with 100 realizations this is the difference
        # between minutes and hours, so run them in parallel processes.
        cells: list[tuple[float, float]] = [
            (nl, op) for nl in noise_levels for op in outlier_fractions
        ]
        tasks: list[tuple[float, float, np.ndarray]] = []
        for noise_level, outlier_fraction in cells:
            data_realizations, _ = io.load_dataset_cell(dataset, noise_level, outlier_fraction)
            for data in data_realizations:
                tasks.append((noise_level, outlier_fraction, data))

        flat_results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_single)(data, t, hyperparams) for _, _, data in tasks
        )

        results: dict[float, dict[float, list]] = {nl: {op: [] for op in outlier_fractions} for nl in noise_levels}
        for (noise_level, outlier_fraction, _), run_result in zip(tasks, flat_results):
            results[noise_level][outlier_fraction].append(run_result)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = io.results_path(self.problem.name, self.method.name, n_realizations, output_dir)
        io.save_run_results(path, results)
        return path

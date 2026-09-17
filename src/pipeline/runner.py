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
from .methods.base import Method
from .problems.base import Problem


@dataclass
class RunResult:
    coefficients: np.ndarray
    extra: dict = field(default_factory=dict)


class MethodRunner:
    def __init__(self, problem: Problem, method: Method, library=None):
        self.problem = problem
        self.method = method
        self.library = library if library is not None else problem.feature_library()


    def run_single(self, data: np.ndarray, t: np.ndarray, eps: float = 0.1,p: int | float = 0.8, **hyperparams) -> RunResult:
        hyperparams['p'] = p
        hyperparams["trimming_fraction"] = data.shape[0]/float(p)
        fit_result = self.method.fit(data, t, self.library,threshold = eps, **hyperparams)
        result = RunResult(
                    coefficients=fit_result.coefficients,
                    extra=fit_result.extra,
                )
        return result

    def run_grid(
        self,
        dataset_path: str | Path,
        noise_levels: Sequence[float],
        outlier_fractions: Sequence[float],
        n_realizations: int,
        output_dir: str | Path = "coeffs",
        eps: float = 0.1,
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
        tasks: list[tuple[float, float, np.ndarray, int]] = []
        for noise_level, outlier_fraction in cells:
            data_realizations, _ = io.load_dataset_cell(dataset, noise_level, outlier_fraction)
            m = data_realizations[0].shape[0]
            p = int(m - (3*outlier_fraction*m))
            for data in data_realizations:
                tasks.append((noise_level, outlier_fraction, data, p))

        flat_results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_single)(data, t, eps, p) for _, _, data, p in tasks
        )

        results: dict[float, dict[float, list]] = {nl: {op: [] for op in outlier_fractions} for nl in noise_levels}
        for (noise_level, outlier_fraction, _, _), run_result in zip(tasks, flat_results):
            results[noise_level][outlier_fraction].append(run_result)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = io.results_path(self.problem.name, self.method.name, n_realizations, output_dir)
        io.save_run_results(path, results)
        return path

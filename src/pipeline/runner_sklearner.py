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
    hyperparams: dict
    trajectory_error: float
    extra: dict = field(default_factory=dict)

def sindy_aic_scorer(estimator, X, y=None):
    """
    Custom scorer to calculate -AIC for a PySINDy model.
    Returns negative AIC so GridSearchCV maximizes it (which minimizes AIC).
    """
    # 1. Calculate k: number of non-zero parameters in the SINDy model
    # For E-SINDy, this evaluates the aggregated ensemble coefficients
    k = np.count_nonzero(estimator.coefficients())
    
    # 2. Get predictions (x_dot_pred)
    x_dot_pred = estimator.predict(X)
    
    # 3. Get true derivatives (x_dot_true)
    if y is not None:
        x_dot_true = y
    else:
        # If true derivatives aren't passed, use SINDy's internal differentiator
        x_dot_true = estimator.differentiate(X)
        
    # 4. Calculate Residual Sum of Squares (RSS)
    rss = np.sum((x_dot_true - x_dot_pred) ** 2)
    
    # 5. Calculate n (number of samples)
    n = X.shape[0]
    
    # 6. Compute AIC (adding a tiny epsilon to avoid log(0) if RSS is exactly 0)
    eps = np.finfo(float).eps
    aic = n * np.log((rss / n) + eps) + 2 * k
    
    return -aic

class MethodRunner:
    def __init__(self, problem: Problem, method: Method, library=None):
        self.problem = problem
        self.method = method
        self.library = library if library is not None else problem.feature_library()


    def run_single(self, data: np.ndarray, t: np.ndarray) -> RunResult:
        fit_result = self.method.grid_fit(data, t, self.library, scorer= sindy_aic_scorer)
        best = RunResult(
                            coefficients=fit_result.best_estimator_.coefficients().T,
                            hyperparams=fit_result.best_params_,
                            trajectory_error=fit_result.best_score_,
                            extra=field(default_factory=dict),
                        )
        return best

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
            delayed(self.run_single)(data, t) for _, _, data in tasks
        )

        results: dict[float, dict[float, list]] = {nl: {op: [] for op in outlier_fractions} for nl in noise_levels}
        for (noise_level, outlier_fraction, _), run_result in zip(tasks, flat_results):
            results[noise_level][outlier_fraction].append(run_result)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = io.results_path(self.problem.name, self.method.name, n_realizations, output_dir)
        io.save_run_results(path, results)
        return path

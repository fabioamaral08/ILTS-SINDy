from __future__ import annotations

import numpy as np
import pysindy as ps

from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class EnsembleSINDyMethod(Method):
    """Bagging ensemble of STLSQ SINDy fits."""

    name = "ESINDY"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
                "threshold": [x for x in np.logspace(1, -5, 10)],
                "n_models": [x for x in (10, 20, 50, 100)]
                }

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps(), "n_models": 20}

    def fit(self, data: np.ndarray, t: np.ndarray, library, **hyperparams) -> FitResult:
        threshold = hyperparams["threshold"]
        n_models = hyperparams["n_models"]
        dt = t[1] - t[0]
        opt = ps.EnsembleOptimizer(opt=ps.STLSQ(threshold=threshold), bagging=True, n_models=n_models)
        model = ps.SINDy(optimizer=opt, feature_library=library)
        model.fit(data, dt)
        return FitResult(coefficients=model.coefficients().T)

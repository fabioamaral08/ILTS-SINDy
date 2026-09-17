from __future__ import annotations

import numpy as np
import pysindy as ps

from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method

from sklearn.model_selection import GridSearchCV


@register_method
class SR3Method(Method):
    """SINDy with the SR3 trimming optimizer."""

    name = "SR3"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {"trimming_fraction": [0.0, 0.05, 0.1, 0.2, 0.3]}

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"trimming_fraction": 0.1}

    def fit(self, data: np.ndarray, t: np.ndarray, library, **hyperparams) -> FitResult:
        trimming_fraction = hyperparams["trimming_fraction"]
        dt = t[1] - t[0]
        opt = ps.SR3(trimming_fraction=trimming_fraction)
        model = ps.SINDy(optimizer=opt, feature_library=library)
        model.fit(data, dt)
        return FitResult(coefficients=model.coefficients().T)

    def grid_fit(self, data, t, library, scorer = 'neg_mean_squared_error'):
        param_grid = {
            "optimizer__reg_weight_lam": [x for x in np.logspace(0,-5,6)],  # Base sparsity threshold
            "optimizer__relax_coeff_nu": [x for x in np.logspace(0,-5,6)],  # Base sparsity threshold
            "optimizer__trimming_fraction": [x for x in np.linspace(0,0.3,7)],  # Base sparsity threshold
        }
        opt = ps.SR3()
        model = ps.SINDy(optimizer=opt, feature_library=library)

        search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=5, 
            scoring=scorer,
            n_jobs=-1 # Uses all available CPU cores for speed
        )

        search.fit(data, t=t)
        return search

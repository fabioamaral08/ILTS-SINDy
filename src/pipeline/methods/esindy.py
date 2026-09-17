from __future__ import annotations

import numpy as np
import pysindy as ps

from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method
from sklearn.model_selection import GridSearchCV

@register_method
class EnsembleSINDyMethod(Method):
    """Bagging ensemble of STLSQ SINDy fits."""

    name = "ESINDY"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
                "threshold": [x for x in np.logspace(0,-5,6)],
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


    def def grid_fit(self, data, t, library, scorer = 'neg_mean_squared_error'):
        opt = ps.EnsembleOptimizer(opt=ps.STLSQ(), bagging=True)
        param_grid = {
            "optimizer__opt__threshold": [x for x in np.logspace(0,-5,6)],  # Base sparsity threshold
            "optimizer__opt__alpha": [0, 0.01, 0.05, 0.1],             # Ridge penalty on STLSQ
            "optimizer__n_models": [20, 50, 100],                        # Number of ensemble models
        }
        model = ps.SINDy(optimizer=opt,  feature_library=library)

        search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=5, 
            scoring=scorer,
            n_jobs=-1 # Uses all available CPU cores for speed
        )

        search.fit(data, t=t)
        return search

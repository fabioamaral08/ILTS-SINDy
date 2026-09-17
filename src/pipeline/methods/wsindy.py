from __future__ import annotations

import numpy as np
import pysindy as ps

from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method
from sklearn.model_selection import GridSearchCV


@register_method
class WeakSINDyMethod(Method):
    """SINDy over a weak (integral) formulation of the candidate library."""

    name = "WSINDY"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
            "threshold": [x for x in np.logspace(0,-5,6)],
            "K": [50, 100, 200],
        }

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps(), "K": 100}

    def fit(self, data: np.ndarray, t: np.ndarray, library,threshold, **hyperparams) -> FitResult:
        dt = t[1] - t[0]
        t_train = np.arange(data.shape[0]) * dt
        weak_lib = ps.WeakPDELibrary(
            function_library=library, spatiotemporal_grid=t_train
        )
        opt = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=opt, feature_library=weak_lib)
        model.fit(data, dt)
        return FitResult(coefficients=model.coefficients().T)

from __future__ import annotations

import numpy as np
import pysindy as ps

import lts
from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class SINDyLTSMethod(Method):
    """Outlier-robust SINDy via Iterative Least Trimmed Squares (ILTS)."""

    name = "SINDY-LTS"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
            "threshold": [x for x in np.logspace(0,-5,6)],
            "p_fraction": [0.80, 0.85, 0.90, 0.95, 0.99],
        }

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps(), "p_fraction": 0.90}

    def fit(self, data: np.ndarray, t: np.ndarray, library, threshold, **hyperparams) -> FitResult:
        p = hyperparams["p"]
        x_dot = ps.FiniteDifference()._differentiate(data, t=t)
        D = np.array(library.fit_transform(data))
        coeff, trusted_order = lts.SINDy_LTS(x_dot, D, p=p, threshold=threshold)
        return FitResult(coefficients=coeff, extra={"trusted_order": trusted_order, "p": p})

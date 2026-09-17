from __future__ import annotations

import numpy as np
import pysindy as ps

import lts
from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class SINDyLTSSEARCHMethod(Method):
    """Outlier-robust SINDy via Iterative Least Trimmed Squares (ILTS)."""

    name = "SINDY-LTS-SEARCH"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
            "threshold": [x for x in np.logspace(0,-5,6)]
        }

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps()}

    def fit(self, data: np.ndarray, t: np.ndarray, library, **hyperparams) -> FitResult:
        eps = hyperparams["threshold"]
        x_dot = ps.FiniteDifference()._differentiate(data, t=t)
        D = np.array(library.fit_transform(data))
        coeff, trusted_order, pvalues = lts.SINDy_LTS_search(x_dot, D, p=None, threshold=eps,)
        return FitResult(coefficients=coeff, extra={"trusted_order": trusted_order, "p": pvalues})

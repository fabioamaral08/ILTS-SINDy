from __future__ import annotations

import time

import numpy as np
import pysindy as ps

import lts
from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class SINDyMethod(Method):
    """Standard SINDy: least squares + iterative small-coefficient thresholding."""

    name = "SINDY"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
            "threshold": [x for x in np.logspace(0,-5,6)],
        }


    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps()}

    def fit(self, data: np.ndarray, t: np.ndarray, library, threshold, **hyperparams) -> FitResult:
        x_dot = ps.FiniteDifference()._differentiate(data, t=t)
        D = np.array(library.fit_transform(data))
        start = time.perf_counter()
        coeff = lts.SINDy(x_dot, D, threshold=threshold)
        elapsed = time.perf_counter() - start
        return FitResult(coefficients=coeff, time=elapsed)

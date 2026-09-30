from __future__ import annotations

import time

import numpy as np
import pysindy as ps

import lts
from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class SINDyLTSSlopeMethod(Method):
    """Same as SINDY-LTS (outlier-robust SINDy via ILTS), but for
    threshold=None picks the per-state threshold by a slope/elbow criterion
    instead of AICc (see lts.SINDy_LTS_eps_slope), recording the chosen
    value in extra['eps']."""

    name = "SINDY-LTS-SLOPE"

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
        start = time.perf_counter()
        coeff, trusted_order, eps, _ = lts.SINDy_LTS_eps_slope(x_dot, D, p=p, threshold=threshold)
        elapsed = time.perf_counter() - start
        return FitResult(
            coefficients=coeff, time=elapsed, extra={"trusted_order": trusted_order, "p": p, "eps": eps}
        )

from __future__ import annotations

import time

import numpy as np
import pysindy as ps

import lts
from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method


@register_method
class SINDyLTSEpsMethod(Method):
    """Same as SINDY-LTS (outlier-robust SINDy via ILTS), but also records
    the per-state threshold actually used to fit each coefficient — the
    fixed threshold, or the AICc-chosen eps (see lts.AIC) when
    threshold=None — in extra['eps']."""

    name = "SINDY-LTS-EPS"

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
        coeff, trusted_order, eps = lts.SINDy_LTS_eps(x_dot, D, p=p, threshold=threshold)
        elapsed = time.perf_counter() - start
        return FitResult(
            coefficients=coeff, time=elapsed, extra={"trusted_order": trusted_order, "p": p, "eps": eps}
        )

from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from .base import Problem


@register_problem
class LorenzProblem(Problem):
    name = "VAN_DER_POL"
    state_names = ["x1", "x2"]

    def dynamics(self, t, x, mu):
        x, y = x
        dx = y
        dy = mu * (1 - x**2) * y - x
        return [dx, dy]

    @property
    def true_params(self):
        return (2.0,)

    @property
    def x0(self):
        return [2.0,0.0]

    @property
    def t_span(self):
        return (0, 100)

    def feature_library(self):
        return ps.PolynomialLibrary(degree=3, include_bias=False)

    def default_eps(self) -> float:
        return 1e-1

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)))
        mu, = self.true_params
        # Feature order:  x, y, x^2, xy, y^2, ...
        true_coeff[1, 0] = 1.0  # dx/dt = y
        true_coeff[0, 1] = -1.0  # dy/dt = mu * y - mu*y*x^2 - x
        true_coeff[1, 1] = mu
        true_coeff[6, 1] = -mu
        return true_coeff

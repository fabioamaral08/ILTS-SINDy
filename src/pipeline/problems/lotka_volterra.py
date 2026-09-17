from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from .base import Problem


@register_problem
class LotkaVolterraProblem(Problem):
    name = "LV"
    state_names = ["prey", "predator"]

    def dynamics(self, t, x, alpha, beta):
        x, y = x
        dx = alpha * x - beta * x * y
        dy = beta * x * y - 2 * alpha * y
        return [dx, dy]

    @property
    def true_params(self):
        return (1.0, 0.1)

    @property
    def x0(self):
        return [1.0, 2.0]

    @property
    def t_span(self):
        return (0, 30)

    def feature_library(self):
        return ps.PolynomialLibrary(degree=3)

    def default_eps(self) -> float:
        return 5e-2

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)))
        alpha, beta = self.true_params
        # Feature order:  x, y, x^2, xy, y^2, ...
        true_coeff[1, 0] = alpha  # dx/dt = alpha*x - beta*x*y
        true_coeff[4, 0] = -beta
        true_coeff[4, 1] = beta  # dy/dt = beta*x*y - 2*alpha*y
        true_coeff[2, 1] = -2 * alpha
        return true_coeff

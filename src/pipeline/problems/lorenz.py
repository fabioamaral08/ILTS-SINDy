from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from .base import Problem


@register_problem
class LorenzProblem(Problem):
    name = "LORENZ"
    state_names = ["x", "y", "z"]

    def dynamics(self, t, x, sigma, rho, beta):
        x, y, z = x
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    @property
    def true_params(self):
        return (10.0, 28.0, 8.0 / 3.0)

    @property
    def x0(self):
        return [-8.0, 7.0, 27.0]

    @property
    def t_span(self):
        return (0, 20)

    def feature_library(self):
        return ps.PolynomialLibrary(degree=3)

    def default_eps(self) -> float:
        return 1e-1

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)), dtype=float)
        sigma, rho, beta = self.true_params
        # Feature order: x, y, z, x^2, xy, xz, y^2, yz, z^2, ...
        true_coeff[1, 0] = -sigma  # dx/dt = sigma*(y - x)
        true_coeff[2, 0] = sigma
        true_coeff[1, 1] = rho  # dy/dt = x*(rho - z) - y
        true_coeff[2, 1] = -1
        true_coeff[6, 1] = -1
        true_coeff[3, 2] = -beta  # dz/dt = x*y - beta*z
        true_coeff[5, 2] = 1
        return true_coeff

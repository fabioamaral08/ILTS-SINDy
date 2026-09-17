from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from .base import Problem


@register_problem
class LorenzProblem(Problem):
    name = "ROSSLER"
    state_names = ["x", "y", "z"]

    def dynamics(self, t, x, a, b, c):
        x, y, z = x
        dx = -y -z
        dy = x + a*y
        dz = b + z * (x - c)
        return [dx, dy, dz]

    @property
    def true_params(self):
        return (0.2, 0.2, 5.7)

    @property
    def x0(self):
        return [0.1, 0.1, 0.1,]

    @property
    def t_span(self):
        return (0, 100)

    def feature_library(self):
        return ps.PolynomialLibrary(degree=3)

    def default_eps(self) -> float:
        return 1e-1

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)))
        a, b, c = self.true_params
        # Feature order: 1, x, y, x^2, xy, y^2, ...
        true_coeff[3, 0] = -1.0  # dx/dt = -y -z
        true_coeff[4, 0] = -1.0  
        true_coeff[2, 1] = 1.0  # dy/dt = x + a*y
        true_coeff[3, 1] = a
        true_coeff[1, 2] = b #dz/dy = b + zx - zc
        true_coeff[8, 2] = 1.0 
        true_coeff[4, 2] = -c 
        return true_coeff

from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from .base import Problem


@register_problem
class LorenzProblem(Problem):
    name = "ABC"
    state_names = ["x", "y", "z"]

    def dynamics(self, t, x, A, B, C, w):
        x, y, z = x
        dx = A * np.sin(w*z) + C *np.cos(w*y)
        dy = B * np.sin(w*x) + A *np.cos(w*z)
        dz = C * np.sin(w*y) + B *np.cos(w*x)

        return [dx, dy, dz]

    @property
    def true_params(self):
        return (2.0, 3.0, 1.0, 1.0)

    @property
    def x0(self):
        return [0.5, 0.2, 1.0]

    @property
    def t_span(self):
        return (0, 20)

    def feature_library(self):
        return ps.FourierLibrary(n_frequencies=2)

    def default_eps(self) -> float:
        return 1e-1

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)), dtype=float)
        A, B, C, _ = self.true_params
        # Feature order: 1, x, y, z, x^2, xy, xz, y^2, yz, z^2, ...
        true_coeff[4,0] = A # dx/dt = A sin(z) + C cos(y)
        true_coeff[3,0] = C
        true_coeff[0,1] = B # dy/dt = B sin(x) + A cos(z)
        true_coeff[5,1] = A
        true_coeff[2,2] = C # dz/dt = C sin(y) + B cos(x)
        true_coeff[1,2] = B
        return true_coeff

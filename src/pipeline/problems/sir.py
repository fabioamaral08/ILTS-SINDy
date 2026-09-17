from __future__ import annotations

import numpy as np
import pysindy as ps

from . import register_problem
from ..sklearn_compat import clonable
from .base import Problem

_ClonableCustomLibrary = clonable(ps.CustomLibrary)


@register_problem
class SIRProblem(Problem):
    name = "SIR"
    state_names = ["S", "I", "R"]

    def dynamics(self, t, x, beta, gamma):
        S, I, _ = x
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    @property
    def true_params(self):
        return (0.3, 0.1)

    @property
    def x0(self):
        return [0.99, 0.01, 0.0]

    @property
    def t_span(self):
        return (0, 100)

    def feature_library(self):
        functions = [
            lambda x: x,  # identity
            lambda x, y: x * y,  # pairwise product
        ]
        function_names = [
            lambda x: f"{x}",
            lambda x, y: f"{x}{y}",
        ]
        return _ClonableCustomLibrary(library_functions=functions, function_names=function_names)

    def true_coefficients(self, library) -> np.ndarray:
        library.fit_transform(np.array([self.x0, self.x0]))
        n_features = len(library.get_feature_names(self.state_names))
        true_coeff = np.zeros((n_features, len(self.state_names)))
        beta, gamma = self.true_params
        # Feature order: identity(S), identity(I), identity(R),
        #                product(S,I), product(S,R), product(I,R)
        true_coeff[3, 0] = -beta  # dS/dt = -beta*S*I
        true_coeff[3, 1] = beta  # dI/dt = beta*S*I - gamma*I
        true_coeff[1, 1] = -gamma
        true_coeff[1, 2] = gamma  # dR/dt = gamma*I
        return true_coeff

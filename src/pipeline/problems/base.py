"""Abstract base class for ODE test problems used by the pipeline.

To add a new problem: subclass `Problem`, implement the abstract members,
decorate the class with `@register_problem` (see `pipeline.problems`), and
add one import line to `pipeline/problems/__init__.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps


class Problem(ABC):
    name: str
    state_names: list[str]
    n_points: int = 1001

    @abstractmethod
    def dynamics(self, t, x, *params):
        """Right-hand side of the ODE: dx/dt = dynamics(t, x, *true_params)."""

    @property
    @abstractmethod
    def true_params(self) -> tuple:
        """Parameter values passed to `dynamics` when generating data."""

    @property
    @abstractmethod
    def x0(self) -> list[float]:
        """Initial condition."""

    @property
    @abstractmethod
    def t_span(self) -> tuple[float, float]:
        """Integration interval (t0, tf)."""

    @abstractmethod
    def feature_library(self)-> ps.feature_library.base.BaseFeatureLibrary:
        """Return a fresh (unfitted) pysindy feature library for this problem."""

    @abstractmethod
    def true_coefficients(self, library) -> np.ndarray:
        """Ground-truth coefficient matrix matching `library`'s output
        columns, shape (n_features, n_states). May fit `library` as a side
        effect."""

    def time_vector(self) -> np.ndarray:
        return np.linspace(*self.t_span, self.n_points)

    def simulate(self, t_eval: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the clean model. Returns (t, y) with y of shape
        (n_points, n_states)."""
        if t_eval is None:
            t_eval = self.time_vector()
        sol = solve_ivp(
            self.dynamics, self.t_span, self.x0, t_eval=t_eval, args=self.true_params
        )
        return t_eval, sol.y.T

    def default_eps(self) -> float:
        """Coefficient-thresholding scale used to center method hyperparameter
        grids; problems may override with a case-specific value."""
        return 1e-2

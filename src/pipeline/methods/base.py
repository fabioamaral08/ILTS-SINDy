"""Abstract base class for system-identification methods used by the pipeline.

To add a new method: subclass `Method`, implement `hyperparameter_grid` and
`fit`, decorate the class with `@register_method` (see `pipeline.methods`),
and add one import line to `pipeline/methods/__init__.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..problems.base import Problem


@dataclass
class FitResult:
    coefficients: np.ndarray
    extra: dict = field(default_factory=dict)


class Method(ABC):
    name: str


    @abstractmethod
    def fit(self, data: np.ndarray, t: np.ndarray, library, **hyperparams) -> FitResult:
        """Fit the model to `data` (n_samples, n_states) sampled at times `t`
        using the given feature `library`, returning coefficients with shape
        (n_features, n_states)."""

    
    def hyperparameter_grid(self) -> dict[str, list]:
        """Candidate values per hyperparameter name; the runner grid-searches
        their cartesian product and keeps the combination with the lowest
        trajectory prediction error."""
        return {"threshold": [x for x in np.logspace(1, -5, 10)]}



    def default_hyperparams(self, problem: Problem) -> dict:
        """A single, representative hyperparameter combination — used to
        time or sanity-check a method without paying for the full grid
        search. Defaults to the first candidate of each
        `hyperparameter_grid` entry; override for a more meaningful default."""
        grid = self.hyperparameter_grid(problem)
        return {name: values[0] for name, values in grid.items()}

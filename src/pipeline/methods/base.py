"""Abstract base class for system-identification methods used by the pipeline.

To add a new method: subclass `Method`, implement `hyperparameter_grid` and
`fit`, decorate the class with `@register_method` (see `pipeline.methods`),
and add one import line to `pipeline/methods/__init__.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..problems.base import Problem



@dataclass
class FitResult:
    coefficients: np.ndarray
    time: float
    extra: dict = field(default_factory=dict)

class Method(ABC):
    name: str


    @abstractmethod
    def fit(self, data: np.ndarray, t: np.ndarray, library, threshold, **hyperparams) -> FitResult:
        """Fit the model to `data` (n_samples, n_states) sampled at times `t`
        using the given feature `library`, returning coefficients with shape
        (n_features, n_states)."""
    
    def hyperparameter_grid(self) -> dict[str, list]:
        """Candidate values per hyperparameter name; the runner grid-searches
        their cartesian product and keeps the combination with the lowest
        trajectory prediction error."""
        return {"threshold": [x for x in np.logspace(1, -5, 10)]}

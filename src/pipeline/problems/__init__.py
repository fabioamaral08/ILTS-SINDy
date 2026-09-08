"""Registry for ODE test problems.

To add a new problem: create a module in this package with a `Problem`
subclass decorated with `@register_problem`, then add one import line at the
bottom of this file so the module executes and the class registers itself.
"""
from __future__ import annotations

from .base import Problem

_REGISTRY: dict[str, type[Problem]] = {}


def register_problem(cls: type[Problem]) -> type[Problem]:
    key = cls.name.upper()
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"Problem name '{cls.name}' is already registered")
    _REGISTRY[key] = cls
    return cls


def get_problem(name: str) -> Problem:
    try:
        return _REGISTRY[name.upper()]()
    except KeyError:
        raise KeyError(
            f"Unknown problem '{name}'. Available problems: {list_problems()}"
        ) from None


def list_problems() -> list[str]:
    return sorted(_REGISTRY)


# Import concrete problems so they register themselves.
from . import sir, lorenz, lotka_volterra

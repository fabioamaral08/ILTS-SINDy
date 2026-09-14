"""Registry for identification methods.

To add a new method: create a module in this package with a `Method`
subclass decorated with `@register_method`, then add one import line at the
bottom of this file so the module executes and the class registers itself.
"""
from __future__ import annotations

from .base import FitResult, Method

_REGISTRY: dict[str, type[Method]] = {}


def register_method(cls: type[Method]) -> type[Method]:
    key = cls.name.upper()
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"Method name '{cls.name}' is already registered")
    _REGISTRY[key] = cls
    return cls


def get_method(name: str) -> Method:
    try:
        return _REGISTRY[name.upper()]()
    except KeyError:
        raise KeyError(
            f"Unknown method '{name}'. Available methods: {list_methods()}"
        ) from None


def list_methods() -> list[str]:
    return sorted(_REGISTRY)


# Import concrete methods so they register themselves.
from . import sindy, sindy_lts, sr3, esindy, wsindy, sindy_lts_search

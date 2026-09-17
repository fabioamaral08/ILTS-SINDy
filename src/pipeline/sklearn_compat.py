"""Workaround for a pysindy 2.1.0 bug affecting some feature-library classes
(seen so far on WeakPDELibrary and CustomLibrary): their __init__ accepts
constructor parameters without assigning them as instance attributes. That
breaks sklearn's get_params()/clone(), which GridSearchCV relies on for
every nested estimator param -- even params that aren't part of the search
grid -- so any SINDy model built on one of these libraries fails as soon as
it goes through GridSearchCV.
"""
from __future__ import annotations

import inspect


def clonable(library_cls):
    """Return a subclass of `library_cls` whose __init__ stores every
    constructor parameter as `self.<name>`, so sklearn's get_params()/
    clone() work on it.
    """
    base_init = library_cls.__init__
    sig = inspect.signature(base_init)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        raise TypeError(
            f"{library_cls.__name__}.__init__ uses *args/**kwargs; "
            "clonable() can't wrap it generically"
        )

    def __init__(self, **kwargs):
        bound = sig.bind_partial(self, **kwargs)
        bound.apply_defaults()
        base_init(self, **{k: v for k, v in bound.arguments.items() if k != "self"})
        for name, value in bound.arguments.items():
            if name != "self":
                setattr(self, name, value)

    __init__.__signature__ = inspect.Signature(parameters=params)

    return type(f"Clonable{library_cls.__name__}", (library_cls,), {"__init__": __init__})

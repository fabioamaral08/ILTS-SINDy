from __future__ import annotations

import numpy as np
import pysindy as ps

from ..problems.base import Problem
from . import register_method
from .base import FitResult, Method
from sklearn.model_selection import GridSearchCV


class _ClonableWeakPDELibrary(ps.WeakPDELibrary):
    """Workaround for a pysindy 2.1.0 bug: WeakPDELibrary.__init__ doesn't
    assign several of its own constructor params (is_uniform, periodic,
    num_pts_per_domain, ...) as instance attributes, which breaks sklearn's
    get_params()/clone() (needed by GridSearchCV) even when they're left at
    their defaults. sklearn also requires __init__ to spell out every
    parameter (no *args/**kwargs), so this mirrors the parent class's full
    signature instead of passing through varargs, and re-assigns every
    param as self.<name> after construction so none of them are missing.
    """

    def __init__(
        self,
        function_library=None,
        derivative_order=0,
        spatiotemporal_grid=None,
        include_bias=False,
        include_interaction=True,
        K=100,
        H_xt=None,
        p=4,
        num_pts_per_domain=None,
        implicit_terms=False,
        multiindices=None,
        differentiation_method=ps.FiniteDifference,
        diff_kwargs=None,
        is_uniform=None,
        periodic=None,
    ):
        super().__init__(
            function_library=function_library,
            derivative_order=derivative_order,
            spatiotemporal_grid=spatiotemporal_grid,
            include_bias=include_bias,
            include_interaction=include_interaction,
            K=K,
            H_xt=H_xt,
            p=p,
            num_pts_per_domain=num_pts_per_domain,
            implicit_terms=implicit_terms,
            multiindices=multiindices,
            differentiation_method=differentiation_method,
            diff_kwargs=diff_kwargs if diff_kwargs is not None else {},
            is_uniform=is_uniform,
            periodic=periodic,
        )
        self.function_library = function_library
        self.derivative_order = derivative_order
        self.spatiotemporal_grid = spatiotemporal_grid
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.K = K
        self.H_xt = H_xt
        self.p = p
        self.num_pts_per_domain = num_pts_per_domain
        self.implicit_terms = implicit_terms
        self.multiindices = multiindices
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs if diff_kwargs is not None else {}
        self.is_uniform = is_uniform
        self.periodic = periodic


@register_method
class WeakSINDyMethod(Method):
    """SINDy over a weak (integral) formulation of the candidate library."""

    name = "WSINDY"

    def hyperparameter_grid(self) -> dict[str, list]:
        return {
            "threshold": [x for x in np.logspace(0,-5,6)],
            "K": [50, 100, 200],
        }

    def default_hyperparams(self, problem: Problem) -> dict:
        return {"threshold": problem.default_eps(), "K": 100}

    def fit(self, data: np.ndarray, t: np.ndarray, library, **hyperparams) -> FitResult:
        threshold = hyperparams["threshold"]
        K = hyperparams["K"]
        dt = t[1] - t[0]
        t_train = np.arange(data.shape[0]) * dt
        weak_lib = _ClonableWeakPDELibrary(
            function_library=library, spatiotemporal_grid=t_train, K=K
        )
        opt = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=opt, feature_library=weak_lib)
        model.fit(data, dt)
        return FitResult(coefficients=model.coefficients().T)


    def grid_fit(self, data, t, library, scorer = 'neg_mean_squared_error'):
        param_grid = {
            "optimizer__threshold": [x for x in np.logspace(0,-5,6)],  # Base sparsity threshold
            "optimizer__alpha": [0, 0.01, 0.05, 0.1],                  # Ridge penalty on STLSQ
            "feature_library__K": [50, 100, 200],                        # number of test functions
            "feature_library__p": [2, 4, 8],                        # test function degree
        }
        weak_lib = _ClonableWeakPDELibrary(
                    function_library=library, spatiotemporal_grid=t
                )
        model = ps.SINDy(optimizer=ps.STLSQ(),  feature_library=weak_lib)

        search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=5, 
            scoring= scorer,
            n_jobs=-1 # Uses all available CPU cores for speed
        )

        search.fit(data, t=t)
        return search

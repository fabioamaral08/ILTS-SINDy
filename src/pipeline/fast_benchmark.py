"""Faster version of the trajectory-error scoring in `pipeline.benchmark`.

The slow part of `benchmark.simulate_from_coefficients` is calling pysindy's
`library.transform` on a single point at every right-hand-side evaluation
(sklearn input validation each time). Here that call is replaced by a small
NumPy evaluator built from the fitted library (polynomial exponent tables,
or direct sin/cos for Fourier). The evaluator is verified against
`library.transform` on random points before use; if no fast path matches (e.g.
the custom SIR library), it falls back to `library.transform`, so results are
always the same as the slow version — just faster when a fast path exists.

Results use the same `TrajectoryErrorResult` and save format as
`pipeline.benchmark`, so the plotting scripts work unchanged.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

from . import io
from .analysis import ResultSet
from .benchmark import TrajectoryErrorResult, _SimulationTimeout


def _polynomial_evaluator(library, n_states: int):
    powers = np.asarray(library.powers_, dtype=float)  # (n_features, n_states)
    if powers.ndim != 2 or powers.shape[1] != n_states:
        raise ValueError("unexpected powers_ shape")

    def features(x: np.ndarray) -> np.ndarray:
        return np.prod(x[None, :] ** powers, axis=1)

    return features


def _fourier_evaluators(library, n_states: int):
    """Yields one candidate evaluator per plausible feature ordering; the
    caller keeps the first that matches library.transform."""
    n_freq = int(library.n_frequencies)
    kinds = [
        k
        for k, include in (
            ("sin", getattr(library, "include_sin", True)),
            ("cos", getattr(library, "include_cos", True)),
        )
        if include
    ]
    axes = {"f": range(1, n_freq + 1), "s": range(n_states), "k": kinds}
    for perm in itertools.permutations("fsk"):
        items = [dict(zip(perm, combo)) for combo in itertools.product(*(axes[a] for a in perm))]
        mult = np.array([it["f"] for it in items], dtype=float)
        sidx = np.array([it["s"] for it in items], dtype=int)
        is_sin = np.array([it["k"] == "sin" for it in items])

        def features(x, mult=mult, sidx=sidx, is_sin=is_sin):
            arg = mult * x[sidx]
            return np.where(is_sin, np.sin(arg), np.cos(arg))

        yield features


def make_fast_features(library, n_states: int, seed: int = 0):
    """Returns (features_fn, description). `features_fn(x)` maps a 1D state to
    the library's feature vector; a fast evaluator is used only if it matches
    `library.transform` on random points, otherwise `library.transform` itself
    is wrapped (correct but slow)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-2.0, 2.0, size=(8, n_states))
    reference = np.asarray(library.transform(points))

    def matches(fn) -> bool:
        try:
            return all(np.allclose(fn(p), reference[k], rtol=1e-9, atol=1e-12) for k, p in enumerate(points))
        except Exception:
            return False

    candidates = []
    if hasattr(library, "powers_"):
        try:
            candidates.append(("polynomial exponent table", _polynomial_evaluator(library, n_states)))
        except Exception:
            pass
    if hasattr(library, "n_frequencies"):
        candidates.extend(("fourier sin/cos", fn) for fn in _fourier_evaluators(library, n_states))

    for description, fn in candidates:
        if matches(fn):
            return fn, description

    def slow(x: np.ndarray) -> np.ndarray:
        return np.asarray(library.transform(x.reshape(1, -1)))[0]

    return slow, "library.transform (no fast path matched)"


def fast_simulate_from_coefficients(
    problem, features, coefficients: np.ndarray, t: np.ndarray, time_limit: float = 5.0
) -> np.ndarray | None:
    """Same contract as `benchmark.simulate_from_coefficients` (None on
    failure/divergence/timeout), using a precomputed feature evaluator."""
    deadline = time.perf_counter() + time_limit

    def dynamics(_t, x):
        if time.perf_counter() > deadline:
            raise _SimulationTimeout()
        return features(x) @ coefficients

    try:
        sol = solve_ivp(dynamics, (t[0], t[-1]), problem.x0, t_eval=t)
    except _SimulationTimeout:
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


def _score_one(problem, features, coefficients, t, clean, time_limit) -> float | None:
    sim = fast_simulate_from_coefficients(problem, features, coefficients, t, time_limit)
    if sim is None:
        return None
    return float(np.mean((sim - clean) ** 2))


def fast_trajectory_error_result_set(
    result_set: ResultSet, time_limit: float = 5.0, n_jobs: int = -1
) -> tuple[TrajectoryErrorResult, str]:
    """Drop-in for `benchmark.trajectory_error_result_set` using the fast
    feature evaluator. Returns (result, description of the evaluator used)."""
    problem = result_set.problem
    library = problem.feature_library()
    problem.true_coefficients(library)  # fits `library`
    t, clean = problem.simulate()
    features, description = make_fast_features(library, len(problem.x0))

    cells = [
        (i, j, coeff)
        for i, nl in enumerate(result_set.noise_levels)
        for j, op in enumerate(result_set.outlier_fractions)
        for coeff in io.load_run_cell(result_set.results, nl, op)["coefficients"]
    ]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(problem, features, coeff, t, clean, time_limit)
        for _, _, coeff in cells
    )

    n_noise, n_outlier = len(result_set.noise_levels), len(result_set.outlier_fractions)
    cell_errors = [[[] for _ in range(n_outlier)] for _ in range(n_noise)]
    cell_failed = [[0] * n_outlier for _ in range(n_noise)]
    for (i, j, _), score in zip(cells, scores):
        if score is None:
            cell_failed[i][j] += 1
        else:
            cell_errors[i][j].append(score)

    result = TrajectoryErrorResult(
        problem_name=problem.name,
        method_name=result_set.method_name,
        errors=[e for row in cell_errors for errs in row for e in errs],
        n_failed=sum(sum(row) for row in cell_failed),
        n_total=len(cells),
        noise_levels=list(result_set.noise_levels),
        outlier_fractions=list(result_set.outlier_fractions),
        cell_errors=cell_errors,
        cell_failed=cell_failed,
    )
    return result, description

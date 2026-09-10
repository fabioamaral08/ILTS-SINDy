"""Pure evaluation metrics for identified models: coefficient accuracy,
exact-recovery, and trajectory prediction error."""
from __future__ import annotations

import numpy as np


def coefficient_accuracy(
    true_coeff: np.ndarray, estimated_coeff: np.ndarray, tolerance: float = 1e-3
) -> float:
    """(TP + TN) / total over the binary support (nonzero vs zero) of each
    coefficient matrix entry."""
    true_binary = np.abs(true_coeff) > tolerance
    estimated_binary = np.abs(estimated_coeff) > tolerance
    return float(np.mean(true_binary == estimated_binary))


def exact_recovery(
    true_coeff: np.ndarray, estimated_coeff: np.ndarray, tolerance: float = 1e-3
) -> bool:
    """Whether the estimated support (which coefficients are nonzero) exactly
    matches the true support."""
    return coefficient_accuracy(true_coeff, estimated_coeff, tolerance) == 1.0


def trajectory_error(true_traj: np.ndarray, sim_traj: np.ndarray, kind: str = "nrmse") -> float:
    """RMSE (kind="rmse") or RMSE normalized by the true trajectory's RMS
    (kind="nrmse", default) between two trajectories of the same shape
    (n_samples, n_states)."""
    rmse = float(np.sqrt(np.mean((true_traj - sim_traj) ** 2)))
    if kind == "rmse":
        return rmse
    if kind == "nrmse":
        scale = float(np.sqrt(np.mean(true_traj**2)))
        return rmse / scale if scale > 0 else rmse
    raise ValueError(f"Unknown trajectory_error kind: {kind!r}")

def derivative_error(data_dot: np.ndarray, coefficients: np.ndarray, Theta: np.ndarray) -> float:
    """NRMSE between true derivatives and those reconstructed as
    Theta @ coefficients, normalized by the true derivatives' RMS."""
    RSS = float(np.sum((data_dot - Theta @ coefficients ) ** 2))
    return RSS
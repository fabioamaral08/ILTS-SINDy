"""Centralizes the .npz file/key conventions shared by the data-generation
and method-runner stages, so no other module hand-writes a noise/outlier key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _noise_key(noise_level: float) -> str:
    return f"noise_{noise_level:g}"


def _data_key(outlier_percent: float) -> str:
    return f"data_{outlier_percent:g}"


def _outlier_index_key(outlier_percent: float) -> str:
    return f"outInd_{outlier_percent:g}"


def _coeffs_key(outlier_percent: float) -> str:
    return f"coeffs_{outlier_percent * 100:g}"


def _hyperparams_key(outlier_percent: float) -> str:
    return f"hyperparams_{outlier_percent * 100:g}"


def _trajectory_error_key(outlier_percent: float) -> str:
    return f"trajectory_error_{outlier_percent * 100:g}"


def _extra_key(outlier_percent: float) -> str:
    return f"extra_{outlier_percent * 100:g}"

def _time_key(outlier_percent: float) -> str:
    return f"time_{outlier_percent * 100:g}"


def dataset_path(problem_name: str, n_realizations: int, output_dir: str | Path) -> Path:
    return Path(output_dir) / f"{problem_name.upper()}_{n_realizations}_data.npz"


def results_path(
    problem_name: str, method_name: str, n_realizations: int, output_dir: str | Path
) -> Path:
    return Path(output_dir) /  f"{problem_name.upper()}" / f"{problem_name.upper()}_{method_name.upper()}_{n_realizations}_samples.npz"


def save_dataset(path: Path, grid: dict[float, dict[float, dict]]) -> None:
    """`grid`: noise_level -> outlier_percent -> {"data": array, "outlier_index": array}."""
    save_dct = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    for noise_level, by_outlier in grid.items():
        result_dct = {}
        for outlier_percent, payload in by_outlier.items():
            result_dct[_data_key(outlier_percent)] = payload["data"]
            result_dct[_outlier_index_key(outlier_percent)] = payload["outlier_index"]
        save_dct[_noise_key(noise_level)] = result_dct
    np.savez(path, allow_pickle=True, **save_dct)


def load_dataset_cell(
    dataset: np.lib.npyio.NpzFile, noise_level: float, outlier_percent: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (data, outlier_index) for one grid cell.

    `data` has shape (n_realizations, n_samples, n_states).
    """
    cell = dataset[_noise_key(noise_level)].item()
    return cell[_data_key(outlier_percent)], cell[_outlier_index_key(outlier_percent)]


def save_run_results(path: Path, grid: dict[float, dict[float, list]]) -> None:
    """`grid`: noise_level -> outlier_percent -> list[RunResult]."""
    save_dct = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    for noise_level, by_outlier in grid.items():
        result_dct = {}
        for outlier_percent, run_results in by_outlier.items():
            result_dct[_coeffs_key(outlier_percent)] = np.array(
                [r.coefficients for r in run_results]
            )
            result_dct[_extra_key(outlier_percent)] = np.array(
                [r.extra for r in run_results], dtype=object
            )
            result_dct[_time_key(outlier_percent)] = np.array(
                [r.time for r in run_results], dtype=object
            )
        save_dct[_noise_key(noise_level)] = result_dct
    np.savez(path, allow_pickle=True, **save_dct)


def load_run_cell(
    results: np.lib.npyio.NpzFile, noise_level: float, outlier_percent: float
) -> dict:
    """Returns dict with keys 'coefficients', 'hyperparams', 'trajectory_error', 'extra'."""
    cell = results[_noise_key(noise_level)].item()
    op = outlier_percent
    return {
        "coefficients": cell[_coeffs_key(op)],
        "time": cell[_time_key(op)],
        "extra": cell[_extra_key(op)],
    }

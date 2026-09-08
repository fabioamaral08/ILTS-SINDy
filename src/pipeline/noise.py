"""Outlier-injection noise model shared by the data-generation pipeline.

Deliberately a small, self-contained copy of the logic in
`gen_heatmap_data.addNoise` rather than an import from that module, since
`gen_heatmap_data.py` is kept as a standalone legacy script.
"""
from __future__ import annotations

import numpy as np


def add_noise(
    data: np.ndarray, noise_level: float, outlier_percent: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inject RMS-scaled Gaussian outliers into a copy of `data`.

    Parameters
        data: shape (n_samples, n_states)
        noise_level: outlier scale, relative to each state's RMS value
        outlier_percent: fraction of time points to corrupt

    Returns
        data_noisy: shape (n_samples, n_states)
        outlier_mask: boolean array, shape (n_samples,)
    """
    data = data.copy()
    n_samples = data.shape[0]
    n_outliers = int(n_samples * outlier_percent)
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[:n_outliers] = True
    np.random.shuffle(outlier_mask)

    for j in range(data.shape[1]):
        col = data[:, j]
        rms = float(np.sqrt(np.mean(col**2)))
        scale = noise_level * rms
        col[outlier_mask] += np.random.normal(0, scale, size=n_outliers)

    return data, outlier_mask

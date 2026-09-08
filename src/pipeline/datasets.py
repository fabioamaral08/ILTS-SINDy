"""Generates noisy/outlier-corrupted realizations of a Problem's trajectory
over a noise-level x outlier-fraction grid, and saves them as a .npz dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import io
from .noise import add_noise
from .problems.base import Problem


class DatasetGenerator:
    def __init__(self, problem: Problem):
        self.problem = problem

    def generate(
        self,
        noise_levels: Sequence[float],
        outlier_fractions: Sequence[float],
        n_realizations: int,
        output_dir: str | Path = "data",
        seed: int | None = None,
    ) -> Path:
        if seed is not None:
            np.random.seed(seed)

        _, clean_data = self.problem.simulate()

        grid: dict[float, dict[float, dict]] = {}
        for noise_level in noise_levels:
            grid[noise_level] = {}
            for outlier_fraction in outlier_fractions:
                data_list = []
                index_list = []
                for _ in range(n_realizations):
                    noisy, outlier_mask = add_noise(clean_data, noise_level, outlier_fraction)
                    data_list.append(noisy)
                    index_list.append(np.argwhere(outlier_mask))
                grid[noise_level][outlier_fraction] = {
                    "data": np.array(data_list),
                    "outlier_index": np.array(index_list, dtype=object),
                }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = io.dataset_path(self.problem.name, n_realizations, output_dir)
        io.save_dataset(path, grid)
        return path

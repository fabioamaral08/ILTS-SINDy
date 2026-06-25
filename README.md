# LTS-SINDy

Outlier-robust Sparse Identification of Nonlinear Dynamics (SINDy) via Iterative Least Trimmed Squares (ILTS).

This repository contains the reference implementation and experiments accompanying the paper introducing **ILTS-SINDy**: a SINDy variant that discovers governing equations of dynamical systems from time-series data contaminated with outliers, by iteratively trimming high-residual (outlying) samples before fitting a sparse regression model.

## Method

Standard SINDy fits a sparse linear model `x_dot = D(x) @ Xi` over a library of candidate functions `D(x)`, using least squares plus sequential thresholding. This is sensitive to outliers in the data or in the numerically estimated derivatives, since a single corrupted sample can distort the regression and the resulting sparsity pattern.

ILTS-SINDy addresses this by wrapping the regression in a **Least Trimmed Squares** loop (`ilts` in [src/lts.py](src/lts.py)):

1. Fit an initial least-squares model on all points.
2. Keep only the `p` samples with the smallest residuals (the "trusted" set).
3. Refit on the trusted set, recompute residuals over the *full* dataset, and re-select the `p` best points.
4. Repeat until the trimmed residual stops improving.

The final trusted subset is then passed to a sparsifying SINDy step (`SINDy_LTS` / `SINDy` in [src/lts.py](src/lts.py)), which performs ridge regression with iterative small-coefficient thresholding (STLSQ-style) to recover a sparse set of active terms per state dimension.

## Repository structure

```
src/
  lts.py              Core ILTS algorithm, sparsifying SINDy step, and ODE simulation utilities
  gen_heatmap_data.py  Generates noisy/outlier-corrupted trajectories for SIR, Lorenz, and Lotka-Volterra
  run_heatmap.py       Runs SINDy / E-SINDy / SR3 / ILTS over a noise x outlier-percentage grid
data/
  *_heatmap_100_data.npz   Pre-generated noisy datasets (100 samples per grid cell) for each test system
examples/
  coeffs_heatmap/      Recovered coefficients for each method/system/grid-point combination
  heatmap_results.ipynb  Notebook that aggregates results into the paper's accuracy/recall heatmaps and figures
```

## Test systems

Experiments are run on three benchmark ODE systems (defined in [src/gen_heatmap_data.py](src/gen_heatmap_data.py)):

- **SIR** epidemic model
- **Lorenz** system
- **Lotka-Volterra** predator-prey model

For each system, trajectories are corrupted with a grid of noise levels and outlier percentages (8 x 8 grid, 0-20% each), and 100 noisy realizations are generated per grid cell.

## Installation

Requires Python 3.9+ and the following packages:

```
pip install numpy scipy scikit-learn joblib pysindy matplotlib jupyter
```

## Usage

### 1. Generate noisy datasets

```bash
cd src
python gen_heatmap_data.py --case SIR --numSamples 100 -o ../data
```

`--case` is one of `SIR`, `LORENZ`, `LV`. This writes `data/<CASE>_heatmap_<numSamples>_data.npz`.

### 2. Run a solver over the noise/outlier grid

```bash
python run_heatmap.py --case SIR --solver SINDY-LTS --pmult 3 --numSamples 100
```

- `--solver`: `SINDY` (standard STLSQ), `SINDY-LTS` (ours, ILTS), `SR3`, or `ESINDY` (bagging ensemble SINDy).
- `--pmult`: for `SINDY-LTS`, multiplier on the expected number of outliers used to set the trimmed sample size `p`.

This writes coefficient results to `coeffs_heatmap/<CASE>_<SOLVER>_<PMULT>P_<numSamples>_samples.npz`.

### 3. Reproduce paper figures

Open [examples/heatmap_results.ipynb](examples/heatmap_results.ipynb) to load the files in `examples/coeffs_heatmap/` and `data/`, compute identification accuracy/recall, and regenerate the heatmaps and outlier-detection figures used in the paper.

### Library usage

```python
import numpy as np
from src.lts import SINDy_LTS, print_model

# x_dot: (n_samples, n_states) finite-difference derivatives
# D: (n_samples, n_features) candidate function library evaluated at x
Xi, trusted_order = SINDy_LTS(x_dot, D, p=900, eps=0.1)
print_model(Xi, feature_names)
```

`p` is the number of trusted (inlier) samples to keep; `eps` is the coefficient threshold below which terms are pruned.

## License

MIT — see [LICENSE](LICENSE).

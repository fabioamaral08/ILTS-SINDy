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
  gen_heatmap_data.py  Legacy: generates noisy/outlier-corrupted trajectories for SIR, Lorenz, Lotka-Volterra
  run_heatmap.py       Legacy: runs SINDy / E-SINDy / SR3 / WSINDy / ILTS over a noise x outlier-percentage grid
  cli_generate_data.py CLI: generate a dataset for a registered problem (pipeline/)
  cli_run_method.py    CLI: run a registered method over a dataset's noise/outlier grid (pipeline/)
  cli_analyse.py        CLI: build comparison figures across problems/methods (pipeline/)
  cli_benchmark.py       CLI: time each method's fit() with fixed default hyperparameters (pipeline/)
  pipeline/            Modular test pipeline (see "Modular pipeline" below)
    problems/           Problem base class + registry; one file per ODE test case (SIR, Lorenz, LV)
    methods/             Method base class + registry; one file per identification algorithm
    noise.py               Outlier-injection noise model
    io.py                    .npz key conventions shared by datasets.py and runner.py
    datasets.py               Builds noisy realizations for a Problem and saves them
    runner.py                  Grid-searches a Method's hyperparameters by trajectory error and saves results
    metrics.py                  Coefficient accuracy, exact recovery, trajectory error
    analysis.py                  Loads results and builds the comparison heatmaps
    benchmark.py                 Times a Method's fit() with a single fixed hyperparameter set
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

## Modular pipeline

`src/pipeline/` runs the same three stages (generate data → run a method → analyse results) through two small extension points instead of hardcoded `if/elif` chains, so adding a new ODE test case or a new identification method is one new file plus a one-line import — nothing else needs to change:

- **`pipeline.problems.Problem`** — an ODE test case: right-hand side, true parameters, initial condition, time span, pysindy feature library, and ground-truth coefficients. Built-in: `SIR`, `LORENZ`, `LV` (`src/pipeline/problems/{sir,lorenz,lotka_volterra}.py`).
- **`pipeline.methods.Method`** — an identification algorithm: the hyperparameter grid to search and a `fit(data, t, library, **hyperparams)` call. Built-in: `SINDY`, `SINDY-LTS` (ILTS, ours), `SR3`, `ESINDY`, `WSINDY`.

Both are looked up by name through a registry (`pipeline.problems.get_problem` / `pipeline.methods.get_method`), which the CLIs use to populate `--problem`/`--method` choices automatically.

### 1. Generate noisy datasets

```bash
cd src
python cli_generate_data.py --problem SIR --n-realizations 100 -o ../data
```

Writes `data/<PROBLEM>_<n-realizations>_data.npz`. `--noise-levels` and `--outlier-fractions` default to an 8-point 0–20% grid each; override with e.g. `--noise-levels 0 0.05 0.1`.

### 2. Run a method over the noise/outlier grid

```bash
python cli_run_method.py --problem SIR --method SINDY-LTS --n-realizations 100
```

For every grid cell and realization, this grid-searches the method's hyperparameters (see each `pipeline/methods/*.py`'s `hyperparameter_grid`), scoring each candidate by simulating it and comparing to the observed trajectory (trajectory prediction error, not the known ground truth — so the same runner works on data without a known equation), and keeps the best. Grid points are run in parallel (`joblib`, `n_jobs=-1` by default). Writes `coeffs/<PROBLEM>_<METHOD>_<n-realizations>_samples.npz`.

This trajectory-error grid search is more expensive than the legacy scripts' fixed-hyperparameter runs, especially for chaotic systems like Lorenz — trim the candidate lists in `pipeline/methods/*.py` if a full run is too slow.

### 3. Build comparison figures

```bash
python cli_analyse.py --problems SIR LORENZ LV --methods SINDY ESINDY SINDY-LTS -o ../figs
```

Loads each `(problem, method)` result pair and saves one heatmap figure per metric (`--metrics`, default `accuracy exact_recovery trajectory_error`) to `<output-dir>/<metric>.png` — rows are problems, columns are methods, generalizing the paper notebook's hardcoded heatmap grids to an arbitrary list.

### 4. Benchmark execution time

```bash
python cli_benchmark.py --problems SIR LORENZ LV --methods SINDY SINDY-LTS SR3 ESINDY WSINDY -o ../figs/timing.png
```

Times each method's `fit()` call using a single fixed, representative hyperparameter set (`Method.default_hyperparams`) rather than the grid searched in step 2 — this measures a method's raw execution cost, not the cost of tuning it. Prints a `problem, method, mean +/- std` table and saves a grouped bar chart (log-scaled) to `<output>`. Each method is timed once per independent noisy realization (`--n-realizations`, default 5; `--noise-level`/`--outlier-fraction`, default clean data) rather than repeated on the same data, so the reported std reflects how fit time actually varies with the data (e.g. outlier count/placement for LTS-style methods), not just measurement noise. `--n-warmup` (default 1) realizations are fit once, untimed, first to absorb first-call overhead.

### Adding a new problem or method

- **Problem**: add `src/pipeline/problems/my_system.py` with a `Problem` subclass decorated `@register_problem`, then add `my_system` to the import line at the bottom of `src/pipeline/problems/__init__.py`.
- **Method**: add `src/pipeline/methods/my_method.py` with a `Method` subclass decorated `@register_method`, then add `my_method` to the import line at the bottom of `src/pipeline/methods/__init__.py`.

Both then show up automatically in the CLIs' `--problem`/`--method` choices.

## Legacy heatmap scripts

The original, non-modular scripts used for the paper's experiments are kept as-is.

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

## Library usage

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

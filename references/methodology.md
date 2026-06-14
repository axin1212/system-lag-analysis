# System Lag Analysis Methodology

## Objective

Detect whether an industrial time-series system has actionable lag: external variables or target inertia that can support early prediction or early control.

## Stage 1: Lead-Lag Scan

For target `y`, control variable `u_j`, and lag `k`:

```text
x_level(j,k,t) = u_j(t-k)
x_diff(j,k,t) = delta u_j(t-k)
z_level(t) = y(t)
z_diff(t) = delta y(t)
```

Evaluate both relationships:

```text
x_level(j,k,t) -> z_level(t)
x_diff(j,k,t) -> z_diff(t)
```

Metrics:

- Pearson absolute correlation for linear response.
- Spearman absolute correlation for monotonic nonlinear response.
- Mutual information for general nonlinear dependency.
- Rolling-window stability for temporal robustness.

Combined score:

```text
score(j,k) =
  0.30 * abs_pearson
+ 0.25 * abs_spearman
+ 0.25 * mi_norm
+ 0.20 * stability
- boundary_penalty
```

The boundary penalty is applied when the best lag equals `max_lag_steps`; the result should be treated as incomplete because the real lag might be beyond the search range.

## Stage 2: Prediction Gain Validation

Stage 2 tests whether the best lag candidates improve out-of-sample prediction.

Target-history baseline:

```text
y(t) = a
     + sum_i phi_i y(t-i)
     + sum_i psi_i delta y(t-i)
     + e(t)
```

Single-variable candidate model:

```text
y(t) = a
     + sum_i phi_i y(t-i)
     + sum_i psi_i delta y(t-i)
     + sum_l beta_l u_j(t-l)
     + sum_l gamma_l delta u_j(t-l)
     + e(t)
```

where `l` is a local window around the Stage 1 best lag, normally `best_lag - 2` through `best_lag + 2` clipped to `1..max_lag_steps`.

Combined top-variable model:

```text
y_hat(t) = f(
  y(t-1..p),
  delta y(t-1..p),
  selected u_j(t-l),
  selected delta u_j(t-l)
)
```

## Models

Linear:

- Ridge: stable linear baseline with L2 regularization.
- ElasticNet: sparse linear baseline with L1/L2 regularization.

Nonlinear:

- HistGradientBoostingRegressor: fast boosted-tree model for nonlinear response.
- ExtraTreesRegressor: randomized tree ensemble for robust nonlinear checks.

## Hyperparameter Search

The script first runs a small fixed grid so every candidate receives a reproducible baseline. If time remains, it uses Optuna TPE Bayesian optimization. If Optuna is unavailable, it falls back to bounded random search.

Ridge / ElasticNet:

```text
alpha: loguniform(1e-4, 100)
l1_ratio: uniform(0.05, 0.95)  # ElasticNet only
```

HistGradientBoosting:

```text
learning_rate: loguniform(0.01, 0.20)
max_iter: int(80, 500)
max_leaf_nodes: int(7, 63)
min_samples_leaf: int(20, 200)
l2_regularization: loguniform(1e-4, 10)
max_bins: categorical(64, 128, 255)
```

ExtraTrees:

```text
n_estimators: int(100, 600)
max_depth: categorical(None, 4, 6, 8, 12)
min_samples_leaf: int(2, 80)
min_samples_split: int(4, 120)
max_features: uniform(0.3, 1.0)
bootstrap: categorical(true, false)
```

Ranges are intentionally conservative. Industrial process data is often noisy and downsampled, so very deep trees or tiny leaves can overfit transient artifacts.

## Validation

Use `TimeSeriesSplit` by default. Do not shuffle rows.

Primary metrics:

- `delta_r2 = R2(candidate) - R2(baseline)`
- `rmse_reduction = 1 - RMSE(candidate) / RMSE(baseline)`
- `mae_reduction = 1 - MAE(candidate) / MAE(baseline)`

## Verdicts

Clearly actionable system lag:

- Stage 1 score is strong and stable.
- Best lag is not only a boundary hit.
- Stage 2 prediction gain is positive and material.
- Shifted overlay is visually plausible.

Possible lag, needs review:

- Stage 1 is strong but Stage 2 is weak, unavailable, or inconsistent.

No stable actionable lag found:

- Stage 1 is weak or Stage 2 consistently fails to improve prediction.

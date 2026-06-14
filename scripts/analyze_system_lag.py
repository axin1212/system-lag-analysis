#!/usr/bin/env python3
"""Lightweight system lag analysis for industrial time-series data."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class Stage1Record:
    variable: str
    lag: int
    relation: str
    pearson: float
    spearman: float
    mutual_info: float
    mutual_info_norm: float
    stability: float
    combined_score: float
    boundary_hit: bool


@dataclass
class Stage2Record:
    variable: str
    best_lag: int
    lag_window: list[int]
    naive_r2: float
    naive_rmse: float
    naive_mae: float
    baseline_r2: float
    baseline_rmse: float
    baseline_mae: float
    y_only_delta_r2_vs_naive: float
    y_only_rmse_reduction_vs_naive: float
    y_only_mae_reduction_vs_naive: float
    best_model: str
    best_search: str
    candidate_r2: float
    candidate_rmse: float
    candidate_mae: float
    delta_r2: float
    rmse_reduction: float
    mae_reduction: float
    best_params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze actionable system lag in time-series data.")
    parser.add_argument("--data", required=True, help="CSV or Parquet file path.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--controls", required=True, help="Comma-separated candidate control/process columns.")
    parser.add_argument("--timestamp", default=None, help="Optional timestamp column.")
    parser.add_argument("--downsample-rule", default=None, help="Pandas resample rule, such as 30s, 1min, 5min.")
    parser.add_argument("--downsample-factor", type=int, default=None, help="Row-based downsampling factor for data without timestamp.")
    parser.add_argument("--no-downsample", action="store_true", help="Use only when the user explicitly chose no downsampling.")
    parser.add_argument("--max-lag-steps", type=int, default=20)
    parser.add_argument("--stage1-top-k", type=int, default=20)
    parser.add_argument("--stage2-budget-minutes", type=float, default=15.0)
    parser.add_argument("--stage2-max-rows", type=int, default=20000, help="Maximum recent rows used by Stage 2 model validation.")
    parser.add_argument("--history-lags", type=int, default=5)
    parser.add_argument("--lag-window-radius", type=int, default=2)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--html-top-heatmap", type=int, default=50)
    parser.add_argument("--html-top-detail", type=int, default=20)
    parser.add_argument("--html-top-overlay", type=int, default=5)
    parser.add_argument("--overlay-max-points", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.max_lag_steps < 1:
        raise SystemExit("--max-lag-steps must be >= 1")
    if not args.no_downsample and args.downsample_rule is None and args.downsample_factor is None:
        raise SystemExit(
            "Downsampling must be explicitly confirmed. Pass --downsample-rule, "
            "--downsample-factor, or --no-downsample."
        )
    if args.downsample_rule and not args.timestamp:
        raise SystemExit("--downsample-rule requires --timestamp")
    if args.downsample_factor is not None and args.downsample_factor < 1:
        raise SystemExit("--downsample-factor must be >= 1")
    if args.stage2_max_rows < 500:
        raise SystemExit("--stage2-max-rows must be >= 500")
    return args


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Parquet.")


def robust_scale(series: pd.Series) -> pd.Series:
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    if not np.isfinite(iqr) or iqr == 0:
        std = series.std()
        iqr = std if np.isfinite(std) and std != 0 else 1.0
    return (series - median) / iqr


def prepare_data(df: pd.DataFrame, args: argparse.Namespace, controls: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [args.target] + controls
    if args.timestamp:
        cols = [args.timestamp] + cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[cols].copy()
    for col in [args.target] + controls:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    meta: dict[str, Any] = {
        "rows_before": int(len(data)),
        "downsample": "none" if args.no_downsample else None,
    }

    if args.timestamp:
        data[args.timestamp] = pd.to_datetime(data[args.timestamp], errors="coerce")
        data = data.dropna(subset=[args.timestamp]).sort_values(args.timestamp)
        data = data.set_index(args.timestamp)
        if args.downsample_rule:
            data = data.resample(args.downsample_rule).mean(numeric_only=True)
            data = data.interpolate(method="time", limit=3, limit_direction="both")
            meta["downsample"] = f"time:{args.downsample_rule}"
        else:
            data = data.interpolate(method="time", limit=3, limit_direction="both")
    elif args.downsample_factor and args.downsample_factor > 1:
        data = data.iloc[:: args.downsample_factor].copy()
        meta["downsample"] = f"row_factor:{args.downsample_factor}"
    elif args.no_downsample:
        meta["downsample"] = "none"

    data = data.dropna(subset=[args.target])
    usable = [args.target] + [c for c in controls if c in data.columns]
    data = data[usable]
    data = data.dropna(how="all")
    meta["rows_after"] = int(len(data))
    meta["controls_after_validation"] = [c for c in controls if c in data.columns]
    return data, meta


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 8:
        return 0.0
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return 0.0
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    if not np.isfinite(value):
        return 0.0
    return float(value)


def safe_mi(a: pd.Series, b: pd.Series, random_state: int, max_samples: int = 5000) -> float:
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 30:
        return 0.0
    if len(pair) > max_samples:
        pair = pair.iloc[np.linspace(0, len(pair) - 1, max_samples).astype(int)]
    x = pair.iloc[:, 0].to_numpy().reshape(-1, 1)
    y = pair.iloc[:, 1].to_numpy()
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    try:
        return float(mutual_info_regression(x, y, random_state=random_state)[0])
    except Exception:
        return 0.0


def rolling_stability(a: pd.Series, b: pd.Series, method: str = "pearson", n_windows: int = 5) -> float:
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < n_windows * 12:
        return 0.0
    boundaries = np.linspace(0, len(pair), n_windows + 1).astype(int)
    corrs: list[float] = []
    signs: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = pair.iloc[start:end]
        if len(chunk) < 8:
            continue
        corr = safe_corr(chunk.iloc[:, 0], chunk.iloc[:, 1], method)
        if corr == 0:
            continue
        corrs.append(abs(corr))
        signs.append(1 if corr > 0 else -1)
    if not corrs:
        return 0.0
    sign_consistency = max(signs.count(1), signs.count(-1)) / len(signs)
    magnitude = float(np.mean(corrs))
    coverage = len(corrs) / n_windows
    return float(np.clip(0.55 * magnitude + 0.30 * sign_consistency + 0.15 * coverage, 0.0, 1.0))


def stage1_scan(
    data: pd.DataFrame,
    target: str,
    controls: list[str],
    max_lag: int,
    random_state: int,
) -> pd.DataFrame:
    y = data[target]
    dy = y.diff()
    rows: list[dict[str, Any]] = []

    for variable in controls:
        u = data[variable]
        du = u.diff()
        for lag in range(1, max_lag + 1):
            level_x = u.shift(lag)
            diff_x = du.shift(lag)
            metrics: dict[str, dict[str, float]] = {}
            for relation, x, z in (
                ("level", level_x, y),
                ("diff", diff_x, dy),
            ):
                pearson = abs(safe_corr(x, z, "pearson"))
                spearman = abs(safe_corr(x, z, "spearman"))
                mi = safe_mi(x, z, random_state=random_state)
                stability = rolling_stability(x, z, "pearson")
                preliminary = 0.35 * pearson + 0.30 * spearman + 0.35 * stability
                metrics[relation] = {
                    "pearson": pearson,
                    "spearman": spearman,
                    "mutual_info": mi,
                    "stability": stability,
                    "preliminary": preliminary,
                }
            best_relation = max(metrics, key=lambda name: metrics[name]["preliminary"])
            row = metrics[best_relation].copy()
            row.update(
                {
                    "variable": variable,
                    "lag": lag,
                    "relation": best_relation,
                    "boundary_hit": lag == max_lag,
                }
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    mi_max = float(result["mutual_info"].max())
    result["mutual_info_norm"] = 0.0 if mi_max <= 0 else result["mutual_info"] / mi_max
    result["combined_score"] = (
        0.30 * result["pearson"]
        + 0.25 * result["spearman"]
        + 0.25 * result["mutual_info_norm"]
        + 0.20 * result["stability"]
        - np.where(result["boundary_hit"], 0.03, 0.0)
    )
    return result.sort_values(["combined_score", "stability"], ascending=False).reset_index(drop=True)


def best_per_variable(stage1: pd.DataFrame) -> pd.DataFrame:
    if stage1.empty:
        return stage1
    idx = stage1.groupby("variable")["combined_score"].idxmax()
    return stage1.loc[idx].sort_values("combined_score", ascending=False).reset_index(drop=True)


def make_supervised_frame(
    data: pd.DataFrame,
    target: str,
    variable: str | None,
    lag_window: list[int],
    history_lags: int,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    frame = pd.DataFrame(index=data.index)
    y = data[target]
    dy = y.diff()
    baseline_cols: list[str] = []
    candidate_cols: list[str] = []
    for lag in range(1, history_lags + 1):
        c1 = f"{target}_lag_{lag}"
        c2 = f"{target}_diff_lag_{lag}"
        frame[c1] = y.shift(lag)
        frame[c2] = dy.shift(lag)
        baseline_cols.extend([c1, c2])
    if variable is not None:
        u = data[variable]
        du = u.diff()
        for lag in lag_window:
            c1 = f"{variable}_lag_{lag}"
            c2 = f"{variable}_diff_lag_{lag}"
            frame[c1] = u.shift(lag)
            frame[c2] = du.shift(lag)
            candidate_cols.extend([c1, c2])
    frame["target"] = y
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    y_out = frame["target"]
    x_cols = baseline_cols + candidate_cols
    return frame[x_cols], y_out, baseline_cols, candidate_cols


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}


def cv_score_model(
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    splits: int,
) -> dict[str, float]:
    n = len(x)
    if n < 80:
        split = max(int(n * 0.7), 1)
        train_idx = np.arange(0, split)
        test_idx = np.arange(split, n)
        split_iter = [(train_idx, test_idx)]
    else:
        usable_splits = min(splits, max(2, n // 80))
        split_iter = TimeSeriesSplit(n_splits=usable_splits).split(x)
    preds: list[float] = []
    actuals: list[float] = []
    for train_idx, test_idx in split_iter:
        if len(test_idx) == 0:
            continue
        model = clone(estimator)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        preds.extend(np.asarray(pred).tolist())
        actuals.extend(y.iloc[test_idx].to_numpy().tolist())
    return regression_metrics(np.asarray(actuals), np.asarray(preds))


def cv_score_naive_persistence(
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    splits: int,
) -> dict[str, float]:
    lag1_col = f"{target}_lag_1"
    if lag1_col not in x.columns:
        raise ValueError(f"Naive persistence requires `{lag1_col}` in supervised features.")
    n = len(x)
    if n < 80:
        split = max(int(n * 0.7), 1)
        split_iter = [(np.arange(0, split), np.arange(split, n))]
    else:
        usable_splits = min(splits, max(2, n // 80))
        split_iter = TimeSeriesSplit(n_splits=usable_splits).split(x)
    preds: list[float] = []
    actuals: list[float] = []
    for _, test_idx in split_iter:
        if len(test_idx) == 0:
            continue
        preds.extend(x.iloc[test_idx][lag1_col].to_numpy().tolist())
        actuals.extend(y.iloc[test_idx].to_numpy().tolist())
    return regression_metrics(np.asarray(actuals), np.asarray(preds))


def model_candidates(random_state: int) -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    return {
        "ridge": (
            make_pipeline(StandardScaler(), Ridge()),
            [
                {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
            ],
        ),
        "elastic_net": (
            make_pipeline(StandardScaler(), ElasticNet(max_iter=5000, random_state=random_state)),
            [
                {"elasticnet__alpha": [0.01, 0.1, 1.0], "elasticnet__l1_ratio": [0.2, 0.7]},
            ],
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            [
                {
                    "learning_rate": [0.05, 0.12],
                    "max_iter": [160],
                    "max_leaf_nodes": [15, 31],
                    "min_samples_leaf": [40, 120],
                    "l2_regularization": [0.01],
                    "max_bins": [128],
                }
            ],
        ),
        "extra_trees": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            [
                {
                    "n_estimators": [160, 320],
                    "max_depth": [6, 10],
                    "min_samples_leaf": [8, 40],
                    "min_samples_split": [20],
                    "max_features": [0.5, 1.0],
                    "bootstrap": [False],
                }
            ],
        ),
    }


def evaluate_param_grid(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int,
    random_state: int,
    deadline: float,
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    best = ("", "", {}, {"r2": -np.inf, "rmse": np.inf, "mae": np.inf})
    for model_name, (estimator, grids) in model_candidates(random_state).items():
        for params in ParameterGrid(grids):
            if time.monotonic() >= deadline:
                return best
            model = clone(estimator).set_params(**params)
            metrics = cv_score_model(model, x, y, cv_splits)
            if metrics["rmse"] < best[3]["rmse"]:
                best = (model_name, "grid", dict(params), metrics)
    return best


def sample_random_params(model_name: str, rng: random.Random) -> dict[str, Any]:
    if model_name == "ridge":
        return {"ridge__alpha": 10 ** rng.uniform(-4, 2)}
    if model_name == "elastic_net":
        return {
            "elasticnet__alpha": 10 ** rng.uniform(-4, 2),
            "elasticnet__l1_ratio": rng.uniform(0.05, 0.95),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": 10 ** rng.uniform(math.log10(0.01), math.log10(0.20)),
            "max_iter": rng.randint(80, 500),
            "max_leaf_nodes": rng.randint(7, 63),
            "min_samples_leaf": rng.randint(20, 200),
            "l2_regularization": 10 ** rng.uniform(-4, 1),
            "max_bins": rng.choice([64, 128, 255]),
        }
    return {
        "n_estimators": rng.randint(100, 600),
        "max_depth": rng.choice([None, 4, 6, 8, 12]),
        "min_samples_leaf": rng.randint(2, 80),
        "min_samples_split": rng.randint(4, 120),
        "max_features": rng.uniform(0.3, 1.0),
        "bootstrap": rng.choice([True, False]),
    }


def evaluate_random_or_optuna(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int,
    random_state: int,
    deadline: float,
    current_best: tuple[str, str, dict[str, Any], dict[str, float]],
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    try:
        import optuna  # type: ignore
    except Exception:
        return evaluate_random_search(x, y, cv_splits, random_state, deadline, current_best)

    model_defs = model_candidates(random_state)
    names = list(model_defs.keys())

    def objective(trial: Any) -> float:
        model_name = trial.suggest_categorical("model", names)
        estimator = clone(model_defs[model_name][0])
        if model_name == "ridge":
            params = {"ridge__alpha": trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)}
        elif model_name == "elastic_net":
            params = {
                "elasticnet__alpha": trial.suggest_float("en_alpha", 1e-4, 100.0, log=True),
                "elasticnet__l1_ratio": trial.suggest_float("en_l1_ratio", 0.05, 0.95),
            }
        elif model_name == "hist_gradient_boosting":
            params = {
                "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.20, log=True),
                "max_iter": trial.suggest_int("hgb_max_iter", 80, 500),
                "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 7, 63),
                "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 20, 200),
                "l2_regularization": trial.suggest_float("hgb_l2", 1e-4, 10.0, log=True),
                "max_bins": trial.suggest_categorical("hgb_max_bins", [64, 128, 255]),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("et_n_estimators", 100, 600),
                "max_depth": trial.suggest_categorical("et_max_depth", [None, 4, 6, 8, 12]),
                "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 2, 80),
                "min_samples_split": trial.suggest_int("et_min_samples_split", 4, 120),
                "max_features": trial.suggest_float("et_max_features", 0.3, 1.0),
                "bootstrap": trial.suggest_categorical("et_bootstrap", [True, False]),
            }
        estimator.set_params(**params)
        metrics = cv_score_model(estimator, x, y, cv_splits)
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", metrics)
        return metrics["rmse"]

    remaining = max(deadline - time.monotonic(), 1.0)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, timeout=remaining, n_trials=200, show_progress_bar=False)
    if not study.trials:
        return current_best
    trial = study.best_trial
    metrics = trial.user_attrs["metrics"]
    if metrics["rmse"] < current_best[3]["rmse"]:
        return (trial.user_attrs["model_name"], "optuna_tpe", trial.user_attrs["params"], metrics)
    return current_best


def evaluate_random_search(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int,
    random_state: int,
    deadline: float,
    current_best: tuple[str, str, dict[str, Any], dict[str, float]],
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    rng = random.Random(random_state)
    best = current_best
    model_defs = model_candidates(random_state)
    names = list(model_defs.keys())
    while time.monotonic() < deadline:
        model_name = rng.choice(names)
        estimator = clone(model_defs[model_name][0])
        params = sample_random_params(model_name, rng)
        estimator.set_params(**params)
        metrics = cv_score_model(estimator, x, y, cv_splits)
        if metrics["rmse"] < best[3]["rmse"]:
            best = (model_name, "random", params, metrics)
    return best


def stage2_validate(
    data: pd.DataFrame,
    target: str,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
) -> list[Stage2Record]:
    records: list[Stage2Record] = []
    top = candidates.head(args.stage1_top_k)
    if top.empty or args.stage2_budget_minutes <= 0:
        return records
    total_deadline = time.monotonic() + args.stage2_budget_minutes * 60
    per_candidate_budget = max((args.stage2_budget_minutes * 60) / max(len(top), 1), 10.0)

    for _, row in top.iterrows():
        if time.monotonic() >= total_deadline:
            break
        variable = str(row["variable"])
        best_lag = int(row["lag"])
        lag_window = [
            lag
            for lag in range(best_lag - args.lag_window_radius, best_lag + args.lag_window_radius + 1)
            if 1 <= lag <= args.max_lag_steps
        ]
        x_candidate, y_candidate, base_cols, _ = make_supervised_frame(
            data, target, variable, lag_window, args.history_lags
        )
        if len(x_candidate) < 50:
            continue
        if len(x_candidate) > args.stage2_max_rows:
            x_candidate = x_candidate.tail(args.stage2_max_rows)
            y_candidate = y_candidate.loc[x_candidate.index]

        base_estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        naive = cv_score_naive_persistence(x_candidate, y_candidate, target, args.cv_splits)
        baseline = cv_score_model(base_estimator, x_candidate[base_cols], y_candidate, args.cv_splits)

        candidate_deadline = min(total_deadline, time.monotonic() + per_candidate_budget)
        best = evaluate_param_grid(
            x_candidate,
            y_candidate,
            args.cv_splits,
            args.random_state,
            candidate_deadline,
        )
        if time.monotonic() < candidate_deadline:
            best = evaluate_random_or_optuna(
                x_candidate,
                y_candidate,
                args.cv_splits,
                args.random_state,
                candidate_deadline,
                best,
            )
        model_name, search_name, params, metrics = best
        records.append(
            Stage2Record(
                variable=variable,
                best_lag=best_lag,
                lag_window=lag_window,
                naive_r2=naive["r2"],
                naive_rmse=naive["rmse"],
                naive_mae=naive["mae"],
                baseline_r2=baseline["r2"],
                baseline_rmse=baseline["rmse"],
                baseline_mae=baseline["mae"],
                y_only_delta_r2_vs_naive=baseline["r2"] - naive["r2"],
                y_only_rmse_reduction_vs_naive=1 - baseline["rmse"] / naive["rmse"] if naive["rmse"] else 0.0,
                y_only_mae_reduction_vs_naive=1 - baseline["mae"] / naive["mae"] if naive["mae"] else 0.0,
                best_model=model_name,
                best_search=search_name,
                candidate_r2=metrics["r2"],
                candidate_rmse=metrics["rmse"],
                candidate_mae=metrics["mae"],
                delta_r2=metrics["r2"] - baseline["r2"],
                rmse_reduction=1 - metrics["rmse"] / baseline["rmse"] if baseline["rmse"] else 0.0,
                mae_reduction=1 - metrics["mae"] / baseline["mae"] if baseline["mae"] else 0.0,
                best_params=params,
            )
        )
    return records


def classify_verdict(best_stage1: pd.DataFrame, stage2: list[Stage2Record]) -> str:
    if best_stage1.empty:
        return "no_stable_actionable_lag_found"
    top = best_stage1.iloc[0]
    stage1_strong = float(top["combined_score"]) >= 0.35 and float(top["stability"]) >= 0.35
    boundary = bool(top["boundary_hit"])
    gains = [r for r in stage2 if r.variable == top["variable"]]
    stage2_good = bool(gains and gains[0].delta_r2 >= 0.03 and gains[0].rmse_reduction >= 0.03)
    if stage1_strong and stage2_good and not boundary:
        return "clearly_actionable_system_lag"
    if stage1_strong or stage2_good:
        return "possible_lag_needs_review"
    return "no_stable_actionable_lag_found"


def downsample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx]


METRIC_GUIDE_TEXT = (
    "Metric Guide: abs Pearson = linear leading relationship; "
    "abs Spearman = monotonic leading relationship; "
    "MI norm = nonlinear dependency strength; "
    "stability = robustness across time windows; "
    "combined = weighted overall lead-lag score."
)


def write_plotly_outputs(
    output_dir: Path,
    data: pd.DataFrame,
    target: str,
    stage1: pd.DataFrame,
    best_vars: pd.DataFrame,
    stage2: list[Stage2Record],
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (output_dir / "plots" / "PLOTLY_NOT_AVAILABLE.txt").write_text(
            "Plotly is required for interactive HTML reports.\n"
            "Install it in the analysis environment, for example:\n"
            "  uv pip install plotly\n"
            f"\nOriginal import error: {exc}\n",
            encoding="utf-8",
        )
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    heatmap_vars = best_vars.head(args.html_top_heatmap)["variable"].tolist()
    heatmap_data = stage1[stage1["variable"].isin(heatmap_vars)]
    pivot = heatmap_data.pivot_table(index="variable", columns="lag", values="combined_score", aggfunc="max")
    pivot = pivot.reindex(index=heatmap_vars, columns=list(range(1, args.max_lag_steps + 1))).fillna(0)

    detail_vars = best_vars.head(args.html_top_detail)["variable"].tolist()
    detail = go.Figure()
    for variable in detail_vars:
        sub = stage1[stage1["variable"] == variable].sort_values("lag")
        if sub.empty:
            continue
        detail.add_trace(
            go.Scatter(
                x=sub["lag"],
                y=sub["combined_score"],
                mode="lines+markers",
                name=f"{variable} combined",
                hovertemplate=(
                    f"variable={variable}<br>"
                    "lag=%{x}<br>"
                    "combined=%{y:.3f}<extra></extra>"
                ),
            )
        )
    detail.update_layout(
        title="Top Variable Combined Score Curves",
        xaxis_title="Lag step",
        yaxis_title="Combined score",
    )

    explorer = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Lead-Lag Heatmap", "Top Variable Combined Score Curves"),
        vertical_spacing=0.22,
        row_heights=[0.58, 0.42],
    )
    explorer.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar={
                "title": {"text": "combined<br>score"},
                "x": 1.01,
                "y": 0.78,
                "len": 0.42,
                "thickness": 16,
            },
            hovertemplate="variable=%{y}<br>lag=%{x}<br>combined=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for tr in detail.data:
        explorer.add_trace(tr, row=2, col=1)
    explorer.update_layout(
        height=max(900, 520 + 22 * len(heatmap_vars)),
        width=1280,
        title="System Lag Explorer",
        margin={"l": 170, "r": 190, "t": 155, "b": 110},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.13,
            "xanchor": "left",
            "x": 0.0,
        },
        annotations=[
            *list(explorer.layout.annotations),
            {
                "text": METRIC_GUIDE_TEXT,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.12,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#334155"},
                "bgcolor": "#f8fafc",
                "bordercolor": "#cbd5e1",
                "borderwidth": 1,
                "borderpad": 6,
            },
        ],
    )
    explorer.update_xaxes(title_text="Lag step", row=1, col=1)
    explorer.update_yaxes(title_text="Variable", automargin=True, row=1, col=1)
    explorer.update_xaxes(title_text="Lag step", row=2, col=1)
    explorer.update_yaxes(title_text="Combined score", row=2, col=1)
    explorer.write_html(plots_dir / "lag_explorer.html", include_plotlyjs="directory", full_html=True)

    stage2_df = pd.DataFrame([asdict(r) for r in stage2])
    gain = go.Figure()
    if not stage2_df.empty:
        gain.add_trace(
            go.Bar(
                x=stage2_df["variable"],
                y=stage2_df["y_only_rmse_reduction_vs_naive"],
                name="Y-only RMSE reduction vs naive",
            )
        )
        gain.add_trace(go.Bar(x=stage2_df["variable"], y=stage2_df["delta_r2"], name="delta R2"))
        gain.add_trace(go.Bar(x=stage2_df["variable"], y=stage2_df["rmse_reduction"], name="full RMSE reduction vs Y-only"))
    gain.update_layout(
        title="Stage 2 Prediction Gain",
        xaxis_title="Variable",
        yaxis_title="Gain",
        barmode="group",
    )

    overlay_vars = best_vars.head(args.html_top_overlay)["variable"].tolist()
    overlay = go.Figure()
    traces_per_overlay = 2
    for idx, variable in enumerate(overlay_vars):
        best = best_vars[best_vars["variable"] == variable].iloc[0]
        lag = int(best["lag"])
        relation = str(best["relation"])
        if relation == "diff":
            plot_df = pd.DataFrame(
                {
                    "target": robust_scale(data[target].diff()),
                    "driver": robust_scale(data[variable].diff().shift(lag)),
                },
                index=data.index,
            ).dropna()
            target_label = f"delta {target}"
            driver_label = f"delta {variable}(t-{lag})"
        else:
            plot_df = pd.DataFrame(
                {
                    "target": robust_scale(data[target]),
                    "driver": robust_scale(data[variable].shift(lag)),
                },
                index=data.index,
            ).dropna()
            target_label = target
            driver_label = f"{variable}(t-{lag})"
        plot_df = downsample_for_plot(plot_df, args.overlay_max_points)
        x_values = plot_df.index.astype(str).tolist()
        visible = idx == 0
        overlay.add_trace(go.Scatter(x=x_values, y=plot_df["target"], mode="lines", name=target_label, visible=visible))
        overlay.add_trace(go.Scatter(x=x_values, y=plot_df["driver"], mode="lines", name=driver_label, visible=visible))
    overlay_buttons = []
    for idx, variable in enumerate(overlay_vars):
        visible = [False] * (len(overlay_vars) * traces_per_overlay)
        visible[idx * traces_per_overlay] = True
        visible[idx * traces_per_overlay + 1] = True
        overlay_buttons.append({"label": variable, "method": "update", "args": [{"visible": visible}, {"title": variable}]})
    overlay.update_layout(
        title="Shifted Overlay Viewer",
        xaxis_title="Time or row index",
        yaxis_title="Robust-scaled value",
        updatemenus=[{"buttons": overlay_buttons, "direction": "down"}] if overlay_buttons else [],
    )

    model_page = make_subplots(rows=2, cols=1, subplot_titles=("Stage 2 Prediction Gain", "Shifted Overlay Viewer"))
    for tr in gain.data:
        model_page.add_trace(tr, row=1, col=1)
    for tr in overlay.data:
        model_page.add_trace(tr, row=2, col=1)
    model_page.update_layout(height=950, title="System Lag Model Gain", updatemenus=overlay.layout.updatemenus)
    model_page.write_html(plots_dir / "model_gain.html", include_plotlyjs="directory", full_html=True)

    verdict = meta.get("verdict", "")
    top_rows = best_vars.head(10).to_html(index=False, float_format=lambda v: f"{v:.4f}")
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>System Lag Analysis</title></head>
<body>
<h1>System Lag Analysis</h1>
<p><strong>Verdict:</strong> {verdict}</p>
<p><strong>Downsample:</strong> {meta.get("downsample")} | <strong>Rows:</strong> {meta.get("rows_after")}</p>
<p><a href="lag_explorer.html">Open lag explorer</a></p>
<p><a href="model_gain.html">Open model gain and shifted overlays</a></p>
<h2>Top Variables</h2>
{top_rows}
</body>
</html>
"""
    (plots_dir / "index.html").write_text(html, encoding="utf-8")


def write_reports(
    output_dir: Path,
    meta: dict[str, Any],
    stage1: pd.DataFrame,
    best_vars: pd.DataFrame,
    stage2: list[Stage2Record],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_vars.to_csv(output_dir / "lag_rankings.csv", index=False)
    result = {
        "metadata": meta,
        "stage1_best_by_variable": best_vars.to_dict(orient="records"),
        "stage2": [asdict(r) for r in stage2],
    }
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    top = best_vars.head(5)
    lines = [
        "# System Lag Analysis Summary",
        "",
        f"Verdict: `{meta.get('verdict')}`",
        f"Rows after preparation: `{meta.get('rows_after')}`",
        f"Downsample: `{meta.get('downsample')}`",
        f"Max lag steps: `{meta.get('max_lag_steps')}`",
        "",
        "## Top Lag Candidates",
        "",
    ]
    if top.empty:
        lines.append("No stable lag candidates were found.")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['variable']}`: best lag `{int(row['lag'])}`, "
                f"relation `{row['relation']}`, score `{float(row['combined_score']):.4f}`, "
                f"stability `{float(row['stability']):.4f}`, boundary hit `{bool(row['boundary_hit'])}`."
            )
    lines.extend(["", "## Stage 2 Target Inertia", ""])
    if not stage2:
        lines.append("Stage 2 did not run or did not produce valid model results.")
    else:
        first = stage2[0]
        lines.append(
            f"- Naive persistence RMSE `{first.naive_rmse:.4f}`, "
            f"Y-only baseline RMSE `{first.baseline_rmse:.4f}`, "
            f"Y-only RMSE reduction vs naive `{first.y_only_rmse_reduction_vs_naive:.4f}`, "
            f"Y-only delta R2 vs naive `{first.y_only_delta_r2_vs_naive:.4f}`."
        )
    lines.extend(["", "## Stage 2 X-Lag Prediction Gain", ""])
    if not stage2:
        lines.append("Stage 2 did not run or did not produce valid model results.")
    else:
        for rec in stage2[:10]:
            lines.append(
                f"- `{rec.variable}`: delta R2 `{rec.delta_r2:.4f}`, "
                f"RMSE reduction `{rec.rmse_reduction:.4f}`, model `{rec.best_model}`, search `{rec.best_search}`."
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    if not controls:
        raise SystemExit("--controls must contain at least one column")

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw = load_data(Path(args.data).expanduser())
    data, meta = prepare_data(raw, args, controls)
    controls = [c for c in controls if c in data.columns and c != args.target]
    if not controls:
        raise SystemExit("No valid control columns remain after validation.")
    if len(data) < max(80, args.max_lag_steps + args.history_lags + 20):
        raise SystemExit("Not enough rows after preparation for lag analysis.")

    start = time.monotonic()
    stage1 = stage1_scan(data, args.target, controls, args.max_lag_steps, args.random_state)
    best_vars = best_per_variable(stage1)
    stage1_seconds = time.monotonic() - start

    stage2_start = time.monotonic()
    stage2 = stage2_validate(data, args.target, best_vars, args)
    stage2_seconds = time.monotonic() - stage2_start

    verdict = classify_verdict(best_vars, stage2)
    meta.update(
        {
            "target": args.target,
            "controls": controls,
            "max_lag_steps": args.max_lag_steps,
            "history_lags": args.history_lags,
            "stage2_max_rows": args.stage2_max_rows,
            "stage1_seconds": round(stage1_seconds, 3),
            "stage2_seconds": round(stage2_seconds, 3),
            "verdict": verdict,
        }
    )
    write_reports(output_dir, meta, stage1, best_vars, stage2)
    write_plotly_outputs(output_dir, data, args.target, stage1, best_vars, stage2, args, meta)
    print(f"System lag analysis complete: {output_dir}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()

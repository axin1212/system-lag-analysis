---
name: system-lag-analysis
description: Analyze industrial time-series data for actionable system lag, lead-lag relationships, early prediction potential, and early control opportunities. Use when the user wants to judge whether a new process case has hysteresis, delayed response, lagged external drivers, target inertia, or whether candidate control/process variables can predict a target ahead of time without running a heavy forecastability workflow.
---

# System Lag Analysis

Use this skill to run a lightweight, evidence-driven lag analysis before expensive forecastability or control modeling.

## Required First Question

Before running the script, ask the user to confirm the downsampling frequency. Do not silently choose it.

Ask: "What downsampling frequency should I use for this data, such as `30s`, `1min`, `5min`, or `none`?"

If the user is unsure, inspect the timestamp spacing if available, recommend a frequency, and wait for confirmation. If there is no usable timestamp, ask for a row downsampling factor such as `1`, `2`, `5`, or `10`.

## What This Skill Does

Answer three questions:

1. Does the system show an actionable lag structure?
2. Which external variables lead the target, and by how many steps or minutes?
3. Does adding those lagged variables improve prediction beyond target-history-only baselines?

Do not claim causality from correlation or model gain. Phrase conclusions as "leading evidence", "prediction gain", or "candidate early-control signal" unless a separate causal analysis supports stronger wording.

## Default Workflow

1. Confirm downsampling frequency with the user.
2. Ask for the data file, target column, candidate control columns, and optional timestamp column.
3. Classify candidate variables before running: keep actionable inputs and external disturbances; exclude same-time analyzer sibling variables and other same-source state/quality measurements unless the user explicitly wants diagnostic coupling rather than actionable control lag.
4. Run the script in `scripts/analyze_system_lag.py`.
5. Review `summary.md`, `lag_rankings.csv`, `results.json`, and the Plotly HTML pages under `plots/`.
6. If the best lag is at `max_lag_steps`, tell the user the search hit the boundary and recommend rerunning with a larger lag range.

## Variable Eligibility

Before passing values to `--controls`, separate variables into roles:

- Actionable inputs: valve position, flow, air, feed, setpoint, manipulated variable. Use these for actionable system-lag analysis.
- External disturbances: load, inlet composition, ambient condition. Use these for early prediction, but do not call them controllable drivers.
- Diagnostic/state/quality variables: online analyzer outputs, lab-like quality indicators, downstream measurements, and same-time analyzer sibling variables collected from the same analyzer group as the target. Exclude these from actionable system-lag analysis by default.

If an online analyzer produces five values and one is the target, do not put the other four same-time analyzer sibling values into `--controls` for control-lag screening. They can dominate Stage 1 or Stage 2 because they share measurement timing, sample path, analyzer delay, or common process state with the target. If the user asks to include them, run a separate diagnostic-coupling analysis and label the result as state coupling or diagnostic leading evidence, not an early-control signal.

## CLI

```bash
python ~/.codex/skills/system-lag-analysis/scripts/analyze_system_lag.py \
  --data case.parquet \
  --target TI101 \
  --controls FIC101,PIC201,TIC301 \
  --timestamp time \
  --downsample-rule 1min \
  --max-lag-steps 20 \
  --stage2-max-rows 20000 \
  --stage2-budget-minutes 15 \
  --output-dir outputs/system_lag
```

For row-based data without a timestamp:

```bash
python ~/.codex/skills/system-lag-analysis/scripts/analyze_system_lag.py \
  --data case.csv \
  --target TI101 \
  --controls FIC101,PIC201,TIC301 \
  --downsample-factor 5 \
  --max-lag-steps 20 \
  --stage2-budget-minutes 15 \
  --output-dir outputs/system_lag
```

Use `--no-downsample` only after the user explicitly chooses no downsampling.

## Defaults

- `max_lag_steps`: 20
- `stage1_top_k`: 20
- `stage2_budget_minutes`: 15
- `stage2_max_rows`: 20000 recent rows
- `history_lags`: 5
- `lag_window_radius`: 2
- `cv_splits`: 3
- Heatmap HTML: top 50 variables
- Detail HTML: top 20 variables shown together as overview curves
- Overlay HTML: top 5 variables, capped to 3000 plotted points per series

## Method Summary

Stage 1 scans lag candidates for each control variable:

```text
u_j(t-k)     -> y(t)
delta u_j(t-k) -> delta y(t)
```

for `k = 1..max_lag_steps`, using Pearson, Spearman, mutual information, and rolling-window stability. This stage is designed to complete in a few minutes.

Stage 2 validates whether the strongest lag candidates improve prediction:

```text
Baseline:
y(t) = a + sum_i phi_i y(t-i) + sum_i psi_i delta y(t-i) + e(t)

Candidate:
y(t) = a + sum_i phi_i y(t-i) + sum_i psi_i delta y(t-i)
     + sum_l beta_l u_j(t-l) + sum_l gamma_l delta u_j(t-l) + e(t)
```

It compares linear baselines (Ridge / ElasticNet) and lightweight nonlinear models (HistGradientBoosting / ExtraTrees). It first runs a small fixed grid for reproducible results, then uses Optuna TPE Bayesian optimization if available and time remains. If Optuna is not installed, it falls back to bounded random search.

Stage 2 defaults to the most recent 20000 supervised rows so large 1-minute datasets remain bounded. Increase `--stage2-max-rows` only when the user explicitly wants deeper validation and accepts longer runtime.

Read `references/methodology.md` when you need the detailed equations, search spaces, and interpretation rules.

## Outputs

```text
output-dir/
  summary.md
  results.json
  lag_rankings.csv
  plots/
    index.html
    lag_explorer.html
    model_gain.html
    plotly.min.js
```

`index.html` is the entry page. `lag_explorer.html` contains the lead-lag heatmap, a plain-language metric guide, and top-variable combined-score curves without variable dropdowns. `model_gain.html` contains Stage 2 model gain charts and shifted overlays.

## Interpretation Rules

Use "clearly actionable system lag" only when:

- Stage 1 score is strong and stable across windows.
- The best lag is not only a boundary hit.
- Stage 2 improves prediction beyond target-history-only baseline.
- Shifted overlay visually supports the lag relationship.

Use "possible lag, needs review" when Stage 1 is strong but Stage 2 gain is weak, unstable, or not run.

Use "no stable actionable lag found" when Stage 1 scores are weak or Stage 2 consistently fails to improve prediction.

When recommending next steps, say whether the case should proceed to FDE's heavier `forecastability` workflow, deeper causal discovery, or manual process review.

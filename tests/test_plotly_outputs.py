from argparse import Namespace
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_system_lag import Stage2Record, rolling_stability, write_plotly_outputs


def test_rolling_stability_keeps_dataframe_chunks() -> None:
    left = pd.Series(range(100), dtype=float)
    right = left.shift(1).bfill()

    value = rolling_stability(left, right)

    assert 0.0 <= value <= 1.0


def test_lag_explorer_is_readable_overview_without_dropdown(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "target": [1.0, 1.2, 1.4, 1.1, 1.5, 1.8, 2.0, 1.9],
            "flow_a": [0.8, 0.9, 1.1, 1.3, 1.4, 1.5, 1.7, 1.8],
            "flow_b": [2.0, 1.8, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9],
        }
    )
    stage1 = pd.DataFrame(
        [
            {
                "variable": variable,
                "lag": lag,
                "relation": "level",
                "pearson": 0.2 + lag / 100,
                "spearman": 0.18 + lag / 100,
                "mutual_info": 0.1 + lag / 100,
                "mutual_info_norm": 0.15 + lag / 100,
                "stability": 0.3 + lag / 100,
                "combined_score": score + lag / 100,
                "boundary_hit": False,
            }
            for variable, score in [("flow_a", 0.45), ("flow_b", 0.35)]
            for lag in range(1, 4)
        ]
    )
    best_vars = pd.DataFrame(
        [
            {"variable": "flow_a", "lag": 2, "relation": "level", "combined_score": 0.47, "stability": 0.32},
            {"variable": "flow_b", "lag": 1, "relation": "level", "combined_score": 0.36, "stability": 0.31},
        ]
    )
    stage2 = [
        Stage2Record(
            variable="flow_a",
            best_lag=2,
            lag_window=[1, 2, 3],
            baseline_r2=0.1,
            baseline_rmse=1.0,
            baseline_mae=0.8,
            best_model="ridge",
            best_search="grid",
            candidate_r2=0.13,
            candidate_rmse=0.96,
            candidate_mae=0.76,
            delta_r2=0.03,
            rmse_reduction=0.04,
            mae_reduction=0.05,
            best_params={},
        )
    ]
    args = Namespace(
        html_top_heatmap=10,
        html_top_detail=10,
        html_top_overlay=5,
        overlay_max_points=100,
        max_lag_steps=3,
    )

    write_plotly_outputs(
        tmp_path,
        data,
        "target",
        stage1,
        best_vars,
        stage2,
        args,
        {"verdict": "possible_lag_needs_review", "downsample": "time:1min", "rows_after": len(data)},
    )

    html = (tmp_path / "plots" / "lag_explorer.html").read_text(encoding="utf-8")
    assert "updatemenus" not in html
    assert "Metric Guide" in html
    assert "abs Pearson" in html
    assert "linear leading relationship" in html
    assert "flow_a combined" in html
    assert "flow_b combined" in html

"""
Combinatorial Ablation Scenarios Heatmap

Generates a Sharpe Ratio heatmap for the structured ablation scenarios
from the results

To use it --> see README.md

Optional arguments:
  --asset   Filter to a single asset
  --metric  Metric to display: sharpe_ratio_mean (default) |
   total_return_mean | sortino_ratio_mean | max_drawdown_mean | win_rate_mean
"""

import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

JSON_DIR = os.path.join(PROJECT_ROOT, "results", "json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")

# Ordered ablation scenario families to show as rows 
ABLATION_FAMILIES = [
    "baseline",
    "ablation_trend",
    "ablation_momentum",
    "ablation_volatility",
    "ablation_statistical",
    "ablation_all",
    "pure_trend",
    "pure_momentum",
    "pure_volatility",
]

FAMILY_DISPLAY = {
    "baseline": "Baseline (OHLC)",
    "ablation_trend": "Base + Trend",
    "ablation_momentum": "Base + Momentum",
    "ablation_volatility": "Base + Volatility",
    "ablation_statistical": "Base + Statistical",
    "ablation_all": "Base + ALL",
    "pure_trend": "Pure Trend",
    "pure_momentum": "Pure Momentum",
    "pure_volatility": "Pure Volatility",
}

ALGORITHMS = ["DQN", "PPO", "A2C"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_results(asset_filter: str = None) -> pd.DataFrame:
    # Load all *_results.json files into a DataFrame
    rows = []
    for algo in ALGORITHMS:
        pattern = os.path.join(JSON_DIR, f"*_{algo}_results.json")
        for fpath in glob.glob(pattern):
            with open(fpath) as f:
                data = json.load(f)

            if asset_filter and data.get("asset") != asset_filter:
                continue

            fam = data.get("feature_family", "")
            if fam not in ABLATION_FAMILIES:
                continue

            rows.append({
                "algorithm": algo,
                "asset": data.get("asset", ""),
                "feature_family": fam,
                "sharpe_ratio_mean": data.get("sharpe_ratio_mean", np.nan),
                "total_return_mean": data.get("total_return_mean", np.nan),
                "sortino_ratio_mean": data.get("sortino_ratio_mean", np.nan),
                "max_drawdown_mean": data.get("max_drawdown_mean", np.nan),
                "win_rate_mean": data.get("win_rate_mean", np.nan),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heatmap generator
# ---------------------------------------------------------------------------

def generate_heatmap(df: pd.DataFrame, metric: str, output_path: str, asset_label: str):
    # Create and save a heatmap: rows = ablation groups, columns = algorithms
    pivot = df.pivot_table(
        index="feature_family",
        columns="algorithm",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(
        [f for f in ABLATION_FAMILIES if f in pivot.index],
        axis=0,
    )
    pivot = pivot.reindex(
        [a for a in ALGORITHMS if a in pivot.columns],
        axis=1,
    )

    row_labels = [FAMILY_DISPLAY.get(f, f) for f in pivot.index]
    col_labels = pivot.columns.tolist()

    n_rows, n_cols = pivot.shape
    fig_w = max(7, 2.5 * n_cols)
    fig_h = max(5, 0.6 * n_rows + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vals = pivot.values.astype(float)
    finite_vals = vals[~np.isnan(vals)]
    if len(finite_vals) == 0:
        print("  No valid data to plot.")
        plt.close()
        return

    vmax = max(abs(finite_vals.max()), abs(finite_vals.min()))
    vmax = vmax if vmax > 0 else 1.0

    cmap = plt.cm.RdYlGn
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=13, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)

    # Annotate each cell
    for i in range(n_rows):
        for j in range(n_cols):
            val = vals[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.65 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color,
                )

    # Draw separator line between "ablation" and "pure" groups
    pure_start = next(
        (i for i, f in enumerate(pivot.index) if f.startswith("pure_")), None
    )
    if pure_start is not None and pure_start > 0:
        ax.axhline(pure_start - 0.5, color="white", linewidth=2.5, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    metric_label = metric.replace("_mean", "").replace("_", " ").title()
    cbar.set_label(metric_label, fontsize=11)

    title_asset = f" — {asset_label}" 
    ax.set_title(
        f"Ablation Scenarios — {metric_label}{title_asset}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved heatmap -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    asset_filter = args.asset if args.asset else None
    asset_label = asset_filter if asset_filter else "All assets"

    print(f"\nLoading ablation JSON results from {JSON_DIR} ...")
    df = load_json_results(asset_filter)

    if df.empty:
        print(
            "  No matching JSON results found.\n"
            "  Run DQN.py / PPO.py / A2C.py first to generate results, "
            "or check that --asset matches an asset name in the JSON files."
        )
        return

    print(f"  Loaded {len(df)} rows  "
          f"(families: {df['feature_family'].unique().tolist()})")

    # Filename suffix for asset
    suffix = f"_{asset_filter}" if asset_filter else "_all_assets"
    out_path = os.path.join(OUTPUT_DIR, f"ablation_scenarios_heatmap{suffix}.png")

    generate_heatmap(df, args.metric, out_path, asset_label)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Sharpe Ratio heatmap for the structured ablation "
            "scenarios using existing results from DQN/PPO/A2C."
        )
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help=(
            "Filter to a single asset (e.g. Gold, Bitcoin). "
            "If omitted, values are averaged across all assets."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio_mean",
        choices=[
            "sharpe_ratio_mean",
            "total_return_mean",
            "sortino_ratio_mean",
            "max_drawdown_mean",
            "win_rate_mean",
        ],
        help="Metric to display in the heatmap (default: sharpe_ratio_mean).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

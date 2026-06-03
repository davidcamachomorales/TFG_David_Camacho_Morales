"""
Combinatorial ablation experiment

Instead of evaluating individual feature families in isolation, this script
generates combinations of feature families and benchmarks DQN, PPO and A2C
on any supported asset to reveal which multi-family compositions perform best.


To use it ->
# Recommended quick run (Gold, families individually, 1 seed)
python experiments/combinatorial_ablation.py 
    --asset Gold 
    --max-combination-size 1 
    --timesteps 5000 
    --seeds 42 
    --algorithms DQN PPO A2C 
    --top-k 10

# Full combinatorial (Gold, pairs + triples, 3 seeds — takes longer)
python experiments/combinatorial_ablation.py 
    --asset Gold 
    --max-combination-size 3 
    --timesteps 50000 
    --seeds 42 123 456 
    --algorithms DQN PPO A2C 
    --top-k 10

# Another asset
python experiments/combinatorial_ablation.py 
    --asset Bitcoin 
    --max-combination-size 2 
    --timesteps 5000 
    --seeds 42 
    --algorithms DQN PPO A2C 
    --top-k 10


"""

import argparse
import itertools
import json
import math
import os
import sys
import traceback
import warnings
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup 
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data_utils import load_dataset, prepare_features, split_train_val
from src.experiment_utils import (
    aggregate_seed_results,
    get_family_members,
    get_transaction_cost,
    set_all_seeds,
)
from src.trading_env_improved import ImprovedTradingEnv

# ---------------------------------------------------------------------------
# Asset -> file mapping
# ---------------------------------------------------------------------------
ASSET_FILES: Dict[str, str] = {
    "Gold": "Gold.csv",
    "Silver": "Silver.csv",
    "Nvidia": "Nvidia.csv",
    "Apple": "Apple.csv",
    "Google": "Google.csv",
    "Inditex": "Inditex.csv",
    "Bitcoin": "Bitcoin.csv",
    "Ethereum": "Ethereum.csv",
    "TetherUSDT": "TetherUSDT.csv",
    "S&P_500_Vanguard": "S&P_500_Vanguard.csv",
}

# Crypto assets have a slightly later end date in the main dataset
CRYPTO_ASSETS = {"Bitcoin", "Ethereum", "TetherUSDT"}

# Fixed train/val dates — same convention as the main scenarios
START_TRAIN = "2023-06-01"
END_TRAIN = "2025-06-30"
START_VAL = "2025-07-01"
END_VAL_DEFAULT = "2026-03-13"       # traditional assets
END_VAL_CRYPTO = "2026-03-15"        # crypto assets

# 12 individual feature families without baseline and any ablation group
INDIVIDUAL_FAMILIES: List[str] = [
    "SMA",
    "EMA",
    "MACD",
    "RSI",
    "SO",
    "BB",
    "ATR",
    "RV",
    "lagged",
    "difference_and_change",
    "temporal_decomposition",
    "time_delay_embedding",
]

USE_DETERMINISTIC = True


# ---------------------------------------------------------------------------
# Hyperparameters match the main DQN / PPO / A2C scripts
# ---------------------------------------------------------------------------

def _make_env_fn(df: pd.DataFrame, transaction_cost: float, seed: int, rank: int):
    #Returns a thunk that builds an ImprovedTradingEnv
    def _init():
        set_all_seeds(seed + rank, USE_DETERMINISTIC)
        return ImprovedTradingEnv(df, transaction_cost=transaction_cost)
    return _init


def _build_and_train_model(
    algo: str,
    train_env,
    seed: int,
    total_timesteps: int,
):
    # DQN learning_starts is capped at min(1000, total_timesteps // 4) so that
    # short runs used in this complementary experiment can still start learning.
    # In the main experiments DQN always uses learning_starts=1000 because
    # total_timesteps is large enough (≥100 000) for that to be safe.
    if algo == "DQN":
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            learning_starts=min(1000, total_timesteps // 4),
            buffer_size=50000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            seed=seed,
        )
    elif algo == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
        )
    elif algo == "A2C":
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            verbose=0,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    model.learn(total_timesteps=total_timesteps)
    return model


# ---------------------------------------------------------------------------
# Single-seed training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_single_seed(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    transaction_cost: float,
    seed: int,
    algo: str,
    total_timesteps: int,
) -> Dict:
    #Train `algo` for one seed and return a raw metrics dict.
    set_all_seeds(seed, USE_DETERMINISTIC)

    train_env = DummyVecEnv([_make_env_fn(df_train, transaction_cost, seed, 0)])
    val_env = DummyVecEnv([_make_env_fn(df_val, transaction_cost, seed, 0)])

    model = _build_and_train_model(algo, train_env, seed, total_timesteps)

    # Full evaluation episode on the validation set
    obs = val_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = val_env.step(action)

    m = info[0]
    metrics = {
        "seed": seed,
        "total_return": m["total_return"],
        "total_return_before_costs": m["total_return_before_costs"],
        "sharpe_ratio": m["sharpe_ratio"],
        "sharpe_ratio_before_costs": m["sharpe_ratio_before_costs"],
        "sortino_ratio": m["sortino_ratio"],
        "max_drawdown": m["max_drawdown"],
        "win_rate": m["win_rate"],
        "win_rate_before_costs": m["win_rate_before_costs"],
        "total_transaction_costs": m["total_transaction_costs"],
        "trade_count": m["trade_count"],
        "trade_frequency": m["trade_frequency"],
        "final_net_worth": m["net_worth"],
        "final_net_worth_before_costs": m["net_worth_before_costs"],
    }

    train_env.close()
    val_env.close()
    del model

    return metrics


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def get_combined_features(
    families: Tuple[str, ...], feature_family_cfg: dict
) -> List[str]:
    #Return deduplicated list of feature columns for a tuple of family names.
    combined: List[str] = []
    seen: set = set()
    for fam in families:
        members = get_family_members(fam, feature_family_cfg)
        if members is None:
            continue
        for col in members:
            if col not in seen:
                combined.append(col)
                seen.add(col)
    return combined


def scale_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame, features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #MinMaxScaler fitted on train, applied to both splits
    df_tr = df_train.copy()
    df_vl = df_val.copy()

    for feat in features:
        df_tr[feat] = df_tr[feat].replace([np.inf, -np.inf], np.nan).ffill()
        df_vl[feat] = df_vl[feat].replace([np.inf, -np.inf], np.nan).ffill()

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_tr[feat] = scaler.fit_transform(df_tr[[feat]])
        df_vl[feat] = scaler.transform(df_vl[[feat]])

    return df_tr, df_vl


# ---------------------------------------------------------------------------
# Combination generator
# ---------------------------------------------------------------------------

def generate_combinations(
    families: List[str], max_size: int
) -> List[Tuple[str, ...]]:
    #All combinations of `families` with size in [1, max_size], small first
    result: List[Tuple[str, ...]] = []
    for size in range(1, max_size + 1):
        result.extend(itertools.combinations(families, size))
    return result


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------

def _save_top_k_table_png(df_summary: pd.DataFrame, top_k: int, output_path: str, asset: str):
    #Compute global average across algorithms
    global_avg = (
        df_summary.groupby("combo_name")
        .agg(
            avg_sharpe=("sharpe_mean", "mean"),
            avg_return=("return_mean", "mean"),
            avg_sortino=("sortino_mean", "mean"),
            avg_max_dd=("max_drawdown_mean", "mean"),
            n_features=("n_features", "first"),
        )
        .sort_values("avg_sharpe", ascending=False)
        .head(top_k)
        .reset_index()
    )

    global_avg.insert(0, "Rank", range(1, len(global_avg) + 1))
    global_avg = global_avg.rename(columns={
        "combo_name": "Combination",
        "n_features": "# Features",
        "avg_sharpe": "Sharpe",
        "avg_return": "Return",
        "avg_sortino": "Sortino",
        "avg_max_dd": "Max DD",
    })
    global_avg["Return"] = (global_avg["Return"] * 100).round(2).astype(str) + "%"
    global_avg["Max DD"] = (global_avg["Max DD"] * 100).round(2).astype(str) + "%"
    global_avg["Sharpe"] = global_avg["Sharpe"].round(3)
    global_avg["Sortino"] = global_avg["Sortino"].round(3)

    display_cols = ["Rank", "Combination", "# Features", "Sharpe", "Return", "Sortino", "Max DD"]
    display_df = global_avg[display_cols]

    n_rows, n_cols = display_df.shape
    col_widths = [0.5, 3.0, 0.8, 0.9, 0.9, 0.9, 0.9]
    fig_width = sum(col_widths) + 0.4
    fig_height = 1.4 + n_rows * 0.46

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(
        f"Top-{top_k} Combinations — Avg Sharpe across algorithms ({asset})",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    header_color = "#1A3A5C"
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.09)

    row_colors = ["#EAF2FB", "#FDFEFE"]
    for i in range(n_rows):
        bg = row_colors[i % 2]
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_height(0.07)
            if j == 1:
                cell.set_text_props(ha="left")

    table.auto_set_column_width(list(range(n_cols)))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _save_heatmap_png(df_summary: pd.DataFrame, top_k: int, output_path: str, asset: str):
    #Heatmap: rows = top-K combos, columns = algorithms, value = Sharpe mean.
    global_order = (
        df_summary.groupby("combo_name")["sharpe_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    pivot = df_summary[df_summary["combo_name"].isin(global_order)].pivot_table(
        index="combo_name", columns="algorithm", values="sharpe_mean", aggfunc="mean"
    )
    pivot = pivot.reindex(global_order)  # keep Sharpe-descending order

    n_rows, n_cols = pivot.shape
    fig_w = max(6, 2 * n_cols)
    fig_h = max(5, 0.45 * n_rows + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()),
               abs(pivot.values[~np.isnan(pivot.values)].min())) if pivot.size else 1
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, label="Sharpe Ratio (mean)")
    ax.set_title(
        f"Combinatorial Ablation Heatmap — Sharpe Ratio ({asset})\n"
        f"Top-{top_k} combinations x algorithms",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_combinatorial_ablation(args):
    #Resolve asset config
    asset = args.asset
    if asset not in ASSET_FILES:
        raise ValueError(
            f"Unknown asset '{asset}'. "
            f"Valid options: {list(ASSET_FILES.keys())}"
        )
    raw_file = ASSET_FILES[asset]
    transaction_cost = get_transaction_cost(asset)
    end_val = END_VAL_CRYPTO if asset in CRYPTO_ASSETS else END_VAL_DEFAULT

    output_dir = os.path.join(PROJECT_ROOT, "results", "combinatorial_ablation", asset)
    os.makedirs(output_dir, exist_ok=True)

    # Load feature family config
    with open(os.path.join(PROJECT_ROOT, "config", "feature_family.json")) as f:
        feature_family_cfg = json.load(f)

    #Load and prepare full feature dataset
    print(f"\nLoading {asset} dataset and computing all features...")
    df_raw = load_dataset(raw_file)
    df_all_features = prepare_features(df_raw)

    df_train_full, df_val_full = split_train_val(
        df_all_features, START_TRAIN, END_TRAIN, START_VAL, end_val
    )

    #Combination generation
    combinations = generate_combinations(INDIVIDUAL_FAMILIES, args.max_combination_size)
    total_combinations = len(combinations)
    total_runs = total_combinations * len(args.algorithms) * len(args.seeds)

    #Pre-run summary to debug
    #print("\n" + "=" * 65)
    #print("  COMBINATORIAL ABLATION — EXPERIMENT SETUP")
    #print("=" * 65)
    #print(f"  Asset used          : {asset}")
    #print(f"  CSV file            : {raw_file}")
    #print(f"  Transaction cost    : {transaction_cost * 100:.2f}%  per trade")
    #print(f"  Train period        : {START_TRAIN}  ->  {END_TRAIN}  ({len(df_train_full)} samples)")
    #print(f"  Val   period        : {START_VAL}  ->  {end_val}  ({len(df_val_full)} samples)")
    #print(f"  Individual families : {len(INDIVIDUAL_FAMILIES)}")
    #print(f"  Max combination sz  : {args.max_combination_size}")
    #print(f"  Total combinations  : {total_combinations}")
    #print(f"  Algorithms          : {args.algorithms}")
    #print(f"  Seeds               : {args.seeds}")
    #print(f"  Timesteps per run   : {args.timesteps}")
    #print(f"  Total training runs : {total_runs}")
    #print(f"  Output directory    : {output_dir}")
    #print("=" * 65)

    if total_runs > 1000:
        print(
            f"\n  Warning: {total_runs} training runs requested. "
            "This may take a very long time. "
            "Consider reducing --max-combination-size, --seeds or --timesteps.\n"
        )

    #Main loop
    per_seed_rows: List[Dict] = []      # one row per combo x algo x seed
    summary_rows: List[Dict] = []       # one row per combo x algo (aggregated)

    for combo_idx, combo in enumerate(combinations):
        combo_name = "+".join(combo)
        features = get_combined_features(combo, feature_family_cfg)

        cols_needed = [c for c in features if c in df_train_full.columns]
        missing = set(features) - set(cols_needed)
        if missing:
            print(f"  Warning: {len(missing)} feature(s) not in dataframe — "
                  f"skipped: {missing}")
        if not cols_needed:
            print(f"[{combo_idx+1}/{total_combinations}] {combo_name}: "
                  "no valid features, skipping.")
            continue

        df_train_combo = df_train_full[cols_needed + ["close"]].copy()
        df_val_combo = df_val_full[cols_needed + ["close"]].copy()
        df_train_s, df_val_s = scale_features(df_train_combo, df_val_combo, cols_needed)

        print(f"\n[{combo_idx+1}/{total_combinations}] {combo_name}  "
              f"({len(cols_needed)} features)")

        for algo in args.algorithms:
            algo_seed_results: List[Dict] = []

            for seed in args.seeds:
                try:
                    metrics = train_and_evaluate_single_seed(
                        df_train=df_train_s,
                        df_val=df_val_s,
                        transaction_cost=transaction_cost,
                        seed=seed,
                        algo=algo,
                        total_timesteps=args.timesteps,
                    )
                    print(
                        f"  [{algo}] seed={seed} -> "
                        f"Return={metrics['total_return']*100:.2f}%  "
                        f"Sharpe={metrics['sharpe_ratio']:.3f}"
                    )

                    per_seed_rows.append({
                        "asset": asset,
                        "algorithm": algo,
                        "combo_name": combo_name,
                        "families": json.dumps(list(combo)),
                        "combo_size": len(combo),
                        "n_features": len(cols_needed),
                        "seed": seed,
                        "total_return": metrics["total_return"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "sortino_ratio": metrics["sortino_ratio"],
                        "max_drawdown": metrics["max_drawdown"],
                        "win_rate": metrics["win_rate"],
                        "trade_count": metrics["trade_count"],
                        "total_transaction_costs": metrics["total_transaction_costs"],
                        "final_net_worth": metrics["final_net_worth"],
                    })
                    algo_seed_results.append(metrics)

                except Exception as e:
                    print(f"  [{algo}] seed={seed} Error: {e}")
                    traceback.print_exc()
                    continue

            if not algo_seed_results:
                continue

            agg = aggregate_seed_results(algo_seed_results)
            summary_rows.append({
                "asset": asset,
                "algorithm": algo,
                "combo_name": combo_name,
                "families": json.dumps(list(combo)),
                "combo_size": len(combo),
                "n_features": len(cols_needed),
                "n_seeds": len(algo_seed_results),
                "sharpe_mean": agg["sharpe_ratio_mean"],
                "sharpe_std": agg["sharpe_ratio_std"],
                "return_mean": agg["total_return_mean"],
                "return_std": agg["total_return_std"],
                "sortino_mean": agg["sortino_ratio_mean"],
                "max_drawdown_mean": agg["max_drawdown_mean"],
                "win_rate_mean": agg["win_rate_mean"],
                "trade_count_mean": agg["trade_count_mean"],
                "transaction_costs_mean": agg["total_transaction_costs_mean"],
            })

    #Persist results
    df_per_seed = pd.DataFrame(per_seed_rows)
    df_summary = pd.DataFrame(summary_rows)

    # CSV - per seed
    per_seed_csv = os.path.join(output_dir, "combinatorial_results.csv")
    df_per_seed.to_csv(per_seed_csv, index=False)

    # JSON — per seed
    per_seed_json = os.path.join(output_dir, "combinatorial_results.json")
    with open(per_seed_json, "w") as f:
        json.dump(per_seed_rows, f, indent=2)

    # CSV — aggregated summary
    summary_csv = os.path.join(output_dir, "combinatorial_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    # To debug
    #print(f"\n  Saved per-seed CSV  -> {per_seed_csv}")
    #print(f"  Saved per-seed JSON -> {per_seed_json}")
    #print(f"  Saved summary CSV   -> {summary_csv}")

    if df_summary.empty:
        print("\n  No results to visualise. Exiting.")
        return

    #Top-K table PNG
    top_k_png = os.path.join(output_dir, "top_10_combinations.png")
    try:
        _save_top_k_table_png(df_summary, args.top_k, top_k_png, asset)
        print(f"  Saved top-K table   -> {top_k_png}")
    except Exception as e:
        print(f"  Could not generate top-K table: {e}")

    #Heatmap PNG
    heatmap_png = os.path.join(output_dir, "combinatorial_heatmap.png")
    try:
        _save_heatmap_png(df_summary, args.top_k, heatmap_png, asset)
        print(f"  Saved heatmap       -> {heatmap_png}")
    except Exception as e:
        print(f"  Could not generate heatmap: {e}")

    #Text summary report
    global_ranking = (
        df_summary.groupby("combo_name")
        .agg(
            avg_sharpe=("sharpe_mean", "mean"),
            avg_return=("return_mean", "mean"),
            avg_sortino=("sortino_mean", "mean"),
            avg_max_dd=("max_drawdown_mean", "mean"),
            n_features=("n_features", "first"),
        )
        .sort_values("avg_sharpe", ascending=False)
        .head(args.top_k)
        .reset_index()
    )

    report_lines = [
        "=" * 70,
        "COMBINATORIAL ABLATION — SUMMARY REPORT",
        f"  Asset               : {asset}",
        f"  CSV file            : {raw_file}",
        f"  Transaction cost    : {transaction_cost * 100:.2f}%",
        f"  Train period        : {START_TRAIN} -> {END_TRAIN}",
        f"  Val period          : {START_VAL} -> {end_val}",
        f"  Algorithms          : {args.algorithms}",
        f"  Seeds               : {args.seeds}",
        f"  Timesteps           : {args.timesteps}",
        f"  Max combination sz  : {args.max_combination_size}",
        f"  Total combinations  : {total_combinations}",
        f"  Total training runs : {total_runs}",
        "=" * 70,
        f"\nTop {args.top_k} combinations (average Sharpe across algorithms):\n",
        global_ranking.to_string(index=False),
        "",
    ]
    for algo in args.algorithms:
        algo_df = (
            df_summary[df_summary["algorithm"] == algo]
            .sort_values("sharpe_mean", ascending=False)
            .head(5)[["combo_name", "n_features", "sharpe_mean",
                       "return_mean", "win_rate_mean", "sortino_mean"]]
        )
        report_lines.append(f"--- {algo} — Top 5 ---")
        report_lines.append(algo_df.to_string(index=False))
        report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"  Saved report        -> {report_path}")
    print("\n" + report_text)
    


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combinatorial ablation of feature families — TFG DRL Benchmark. "
            "Complementary experiment; does not affect the main ablation scenarios."
        )
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="Gold",
        choices=list(ASSET_FILES.keys()),
        help=(
            "Asset to run the experiment on. Default: Gold "
            "(recommended as baseline asset for the complementary study)."
        ),
    )
    parser.add_argument(
        "--max-combination-size",
        type=int,
        default=3,
        help="Maximum number of families per combination (default: 3).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Training timesteps per run (default: 5000).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds (default: 42).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["DQN", "PPO", "A2C"],
        choices=["DQN", "PPO", "A2C"],
        help="Algorithms to benchmark (default: DQN PPO A2C).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top combinations to show in visuals and report (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_combinatorial_ablation(args)

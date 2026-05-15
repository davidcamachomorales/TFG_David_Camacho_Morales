"""
Read JSON results from DQN.py, PPO.py and A2C.py
and produce comparative tables, boxplots, heatmaps and statistical tests

To use it->
    cd experiments
    python analyze_results.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non interactive backend
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
# Configuration
#---------------------------------------------------------------

RESULTS_DIR = f'{PROJECT_ROOT}/results'
JSON_DIR = os.path.join(RESULTS_DIR, 'json')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ALGORITHMS = ['DQN', 'PPO', 'A2C']

# Category mapping for feature families
FAMILY_CATEGORY = {
    'SMA': 'trend', 'EMA': 'trend', 'MACD': 'trend',
    'RSI': 'momentum', 'SO': 'momentum',
    'BB': 'volatility', 'ATR': 'volatility', 'RV': 'volatility',
    'lagged': 'statistical', 'difference_and_change': 'statistical',
    'temporal_decomposition': 'statistical', 'time_delay_embedding': 'statistical',
    'baseline': 'baseline',
    'ablation_trend': 'ablation', 'ablation_momentum': 'ablation',
    'ablation_volatility': 'ablation', 'ablation_statistical': 'ablation',
    'ablation_all': 'ablation',
    'pure_trend': 'pure_ablation', 'pure_momentum': 'pure_ablation',
    'pure_volatility': 'pure_ablation',
}

# Display-friendly names
FAMILY_DISPLAY = {
    'SMA': 'SMA', 'EMA': 'EMA', 'MACD': 'MACD',
    'RSI': 'RSI', 'SO': 'Stochastic Osc.',
    'BB': 'Bollinger Bands', 'ATR': 'ATR', 'RV': 'Rolling Vol.',
    'lagged': 'Lagged', 'difference_and_change': 'Diff & Change',
    'temporal_decomposition': 'Temporal Decomp.', 'time_delay_embedding': 'TDE',
    'baseline': 'Baseline (OHLC)',
    'ablation_trend': 'Base+Trend', 'ablation_momentum': 'Base+Momentum',
    'ablation_volatility': 'Base+Volatility', 'ablation_statistical': 'Base+Statistical',
    'ablation_all': 'Base+ALL',
    'pure_trend': 'Pure Trend', 'pure_momentum': 'Pure Momentum',
    'pure_volatility': 'Pure Volatility',
}

# Color palette
CATEGORY_COLORS = {
    'baseline': '#6C757D',
    'trend': '#0D6EFD',
    'momentum': '#198754',
    'volatility': '#DC3545',
    'statistical': '#FFC107',
    'ablation': '#6F42C1',
    'pure_ablation': '#D63384',
}


#---------------------------------------------------------------
# Load data
#---------------------------------------------------------------

def load_all_results():
    # Load all json results into a single DF
    all_rows = []
    
    for algo in ALGORITHMS:
        pattern = os.path.join(JSON_DIR, f'*_{algo}_results.json')
        files = glob.glob(pattern)
        
        if not files:
            print(f"  No results found for {algo}")
            continue
        
        print(f"  {algo}: {len(files)} result files found")
        
        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)
            
            row = {
                'algorithm': algo,
                'scenario_id': data['scenario_id'],
                'asset': data['asset'],
                'feature_family': data['feature_family'],
                'n_features': data['n_features'],
                'category': FAMILY_CATEGORY.get(data['feature_family'], 'unknown'),
                'family_display': FAMILY_DISPLAY.get(data['feature_family'], data['feature_family']),
                'n_seeds': data['n_seeds'],
            }
            
            # Extract mean/std for key metrics
            for metric in ['total_return', 'sharpe_ratio', 'sortino_ratio',
                           'max_drawdown', 'win_rate', 'trade_count',
                           'final_net_worth', 'total_transaction_costs']:
                row[f'{metric}_mean'] = data.get(f'{metric}_mean', np.nan)
                row[f'{metric}_std'] = data.get(f'{metric}_std', np.nan)
                row[f'{metric}_values'] = data.get(f'{metric}_values', [])
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows loaded: {len(df)}")
    return df


#---------------------------------------------------------------
# Summary tables
#---------------------------------------------------------------

def generate_summary_tables(df):
    #Generate summary tables grouped by family and algorithm
    
    print("\n" + "-"*80)
    print("Table 1 — average Sharpe Ratio by family and algorithm")
    print("-"*80)
    
    pivot = df.pivot_table(
        values='sharpe_ratio_mean',
        index='feature_family',
        columns='algorithm',
        aggfunc='mean'
    ).round(3)
    
    # Add category column
    pivot['Category'] = pivot.index.map(lambda x: FAMILY_CATEGORY.get(x, ''))
    pivot = pivot.sort_values(['Category', 'sharpe_ratio_mean' if 'DQN' not in pivot.columns else pivot.columns[0]], 
                               ascending=[True, False])
    
    print(pivot.to_string())
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'table_sharpe_by_family.csv'))
    
    print("\n" + "-"*80)
    print("Table 2 — average return by family and algorithm")
    print("-"*80)
    
    pivot_ret = df.pivot_table(
        values='total_return_mean',
        index='feature_family',
        columns='algorithm',
        aggfunc='mean'
    ).round(4)
    
    print(pivot_ret.to_string())
    pivot_ret.to_csv(os.path.join(OUTPUT_DIR, 'table_return_by_family.csv'))
    
    # Table 3 overall ranking
    print("\n" + "-"*80)
    print("Table 3 — global ranking of families by average Sharpe Ratio across all algorithms and assets")
    print("-"*80)
    
    ranking = df.groupby('feature_family').agg(
        sharpe_mean=('sharpe_ratio_mean', 'mean'),
        sharpe_std=('sharpe_ratio_mean', 'std'),
        return_mean=('total_return_mean', 'mean'),
        win_rate_mean=('win_rate_mean', 'mean'),
        max_dd_mean=('max_drawdown_mean', 'mean'),
        n_scenarios=('scenario_id', 'count'),
    ).sort_values('sharpe_mean', ascending=False).round(4)
    
    ranking['category'] = ranking.index.map(lambda x: FAMILY_CATEGORY.get(x, ''))
    print(ranking.to_string())
    ranking.to_csv(os.path.join(OUTPUT_DIR, 'table_ranking_global.csv'))
    
    return pivot, ranking
#########################################

def generate_paper_comparison_table(df):
    # Generate academic comparison table for baseline, individuals, and top 3 ablations
    
    print("\n" + "-"*80)
    print("Table 4 — Academic Paper Comparison Table (Top Features vs Baseline)")
    print("-"*80)
    
    # Identify individual features and top ablations based on global Sharpe Ratio
    individual_cats = ['trend', 'momentum', 'volatility', 'statistical']
    ablation_cats = ['ablation', 'pure_ablation']
    
    ranking = df.groupby('feature_family')['sharpe_ratio_mean'].mean().reset_index()
    ranking['category'] = ranking['feature_family'].map(lambda x: FAMILY_CATEGORY.get(x, 'unknown'))
    
    all_individuals = ranking[ranking['category'].isin(individual_cats)]['feature_family'].tolist()
        
    top_ablations = ranking[ranking['category'].isin(ablation_cats)] \
        .sort_values('sharpe_ratio_mean', ascending=False).head(3)['feature_family'].tolist()
        
    # Maintain a logical order for individuals
    families_order = ['SMA', 'EMA', 'MACD', 'RSI', 'SO', 'BB', 'ATR', 'RV', 'lagged', 'difference_and_change', 'temporal_decomposition', 'time_delay_embedding']
    ordered_individuals = [f for f in families_order if f in all_individuals]
    
    selected_families = ['baseline'] + ordered_individuals + top_ablations
    
    print(f"  Selected rows: {selected_families}")
    
    # Filter dataframe
    df_filtered = df[df['feature_family'].isin(selected_families)].copy()
    
    # Create pivot table for the selected metrics
    metrics = {
        'total_return_mean': 'Return (%)',
        'sharpe_ratio_mean': 'Sharpe',
        'win_rate_mean': 'Win Rate (%)',
        'max_drawdown_mean': 'Max DD (%)'
    }
    
    # Scale metrics that are percentages
    df_filtered['total_return_mean'] = df_filtered['total_return_mean'] * 100
    df_filtered['win_rate_mean'] = df_filtered['win_rate_mean'] * 100
    df_filtered['max_drawdown_mean'] = df_filtered['max_drawdown_mean'] * 100
    
    pivot = df_filtered.pivot_table(
        index='feature_family',
        columns='algorithm',
        values=list(metrics.keys()),
        aggfunc='mean'
    )
    
    # Reorder index to match selected families order
    pivot = pivot.reindex(selected_families)
    
    # Rename display labels for rows
    pivot.index = pivot.index.map(lambda x: FAMILY_DISPLAY.get(x, x))
    pivot.index.name = 'Feature Family'
    
    # Reorder and rename columns
    pivot = pivot.swaplevel(0, 1, axis=1)
    
    metric_order = list(metrics.keys())
    sorted_cols = []
    for algo in ALGORITHMS:
        if algo in pivot.columns.get_level_values(0):
            for m in metric_order:
                if m in pivot.columns.get_level_values(1):
                    sorted_cols.append((algo, m))
                    
    pivot = pivot[sorted_cols]
    pivot.rename(columns=metrics, level=1, inplace=True)
    pivot = pivot.round(2)
    
    print(pivot.to_string())
    
    path = os.path.join(OUTPUT_DIR, 'table_academic_comparison.csv')
    pivot.to_csv(path)
    print(f"\n Saved: {path}")
    
    return pivot
#######################################

#---------------------------------------------------------------
# Graphics
#---------------------------------------------------------------

def plot_boxplot_sharpe_by_family(df):
    #Boxplot of Sharpe ratio for each family, grouped by algorithm
    
    # Get individual families only
    individual = df[df['category'].isin(['baseline', 'trend', 'momentum', 'volatility', 'statistical'])]
    
    families_order = ['baseline', 'SMA', 'EMA', 'MACD', 'RSI', 'SO',
                      'BB', 'ATR', 'RV', 'lagged', 'difference_and_change',
                      'temporal_decomposition', 'time_delay_embedding']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    fig.suptitle('Sharpe Ratio by feature family for individual features', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for ax, algo in zip(axes, ALGORITHMS):
        algo_data = individual[individual['algorithm'] == algo]
        
        box_data = []
        labels = []
        colors = []
        for fam in families_order:
            fam_data = algo_data[algo_data['feature_family'] == fam]
            if len(fam_data) > 0:
                box_data.append(fam_data['sharpe_ratio_mean'].values)
                labels.append(FAMILY_DISPLAY.get(fam, fam))
                colors.append(CATEGORY_COLORS.get(FAMILY_CATEGORY.get(fam, ''), '#999'))
        
        if not box_data:
            continue
            
        bp = ax.boxplot(box_data, patch_artist=True, labels=labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(algo, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio' if ax == axes[0] else '')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'boxplot_sharpe_individual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_comparison(df):
    #Bar chart comparing baseline vs ablation groups
    
    ablation_families = ['baseline', 'ablation_trend', 'ablation_momentum',
                         'ablation_volatility', 'ablation_statistical', 'ablation_all']
    
    abl = df[df['feature_family'].isin(ablation_families)]
    
    if len(abl) == 0:
        print(" No ablation data found...")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.suptitle('Incremental Ablation: Baseline vs Combined Groups',
                 fontsize=16, fontweight='bold', y=1.02)
    
    x_labels = [FAMILY_DISPLAY.get(f, f) for f in ablation_families]
    bar_colors = ['#6C757D', '#0D6EFD', '#198754', '#DC3545', '#FFC107', '#6F42C1']
    
    for ax, algo in zip(axes, ALGORITHMS):
        algo_data = abl[abl['algorithm'] == algo]
        
        means = []
        stds = []
        for fam in ablation_families:
            fam_data = algo_data[algo_data['feature_family'] == fam]
            if len(fam_data) > 0:
                means.append(fam_data['sharpe_ratio_mean'].mean())
                stds.append(fam_data['sharpe_ratio_mean'].std())
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x_labels, means, yerr=stds, capsize=4, color=bar_colors, 
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(algo, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio mean' if ax == axes[0] else '')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'barplot_ablation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {path}")


def plot_pure_vs_baseline(df):
    # Compare ablation without baseline vs ablation with baseline.
    
    pairs = [
        ('pure_trend', 'ablation_trend', 'Trend'),
        ('pure_momentum', 'ablation_momentum', 'Momentum'),
        ('pure_volatility', 'ablation_volatility', 'Volatility'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('How important is the baseline? Comparison Pure vs Baseline+Group',
                 fontsize=16, fontweight='bold', y=1.02)
    
    for ax, algo in zip(axes, ALGORITHMS):
        algo_data = df[df['algorithm'] == algo]
        
        x = np.arange(len(pairs))
        width = 0.35
        
        pure_means = []
        base_means = []
        for pure_fam, base_fam, _ in pairs:
            pure_d = algo_data[algo_data['feature_family'] == pure_fam]
            base_d = algo_data[algo_data['feature_family'] == base_fam]
            pure_means.append(pure_d['sharpe_ratio_mean'].mean() if len(pure_d) > 0 else 0)
            base_means.append(base_d['sharpe_ratio_mean'].mean() if len(base_d) > 0 else 0)
        
        ax.bar(x - width/2, pure_means, width, label='Pure (no baseline)', 
               color='#D63384', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, base_means, width, label='With Baseline', 
               color='#6F42C1', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels([p[2] for p in pairs])
        ax.set_title(algo, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio mean' if ax == axes[0] else '')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'barplot_pure_vs_baseline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {path}")


def plot_heatmap_sharpe(df):
    #Heatmap -> rows=families, columns=assets, color=Sharpe Ratio
    
    for algo in ALGORITHMS:
        algo_data = df[df['algorithm'] == algo]
        
        # individual families + baseline
        individual = algo_data[algo_data['category'].isin(
            ['baseline', 'trend', 'momentum', 'volatility', 'statistical'])]
        
        if len(individual) == 0:
            continue
        
        pivot = individual.pivot_table(
            values='sharpe_ratio_mean',
            index='feature_family',
            columns='asset',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([FAMILY_DISPLAY.get(f, f) for f in pivot.index])
        
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 1.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                            color=color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im, label='Sharpe Ratio')
        ax.set_title(f'Heatmap Sharpe Ratio — {algo}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f'heatmap_sharpe_{algo}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {path}")


def plot_category_summary(df):
    #Bar chart comparing average Sharpe by category across algorithms
    
    # only individual families
    individual = df[df['category'].isin(['baseline', 'trend', 'momentum', 'volatility', 'statistical'])]
    
    if len(individual) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = ['baseline', 'trend', 'momentum', 'volatility', 'statistical']
    x = np.arange(len(categories))
    width = 0.25
    
    for i, algo in enumerate(ALGORITHMS):
        means = []
        for cat in categories:
            cat_data = individual[(individual['algorithm'] == algo) & (individual['category'] == cat)]
            means.append(cat_data['sharpe_ratio_mean'].mean() if len(cat_data) > 0 else 0)
        
        ax.bar(x + i*width, means, width, label=algo, alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel('Sharpe Ratio (average)', fontsize=12)
    ax.set_title('Sharpe Ratio by feature category and algorithm', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'barplot_category_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {path}")


#---------------------------------------------------------------
# Statistical tests
#---------------------------------------------------------------


def run_statistical_tests(df):
    #Run Wilcoxon signed-rank tests comparing each family vs baseline
    
    print("\n" + "-"*80)
    print("Statistical test — Wilcoxon signed-rank comparing each family vs baseline")
    print("-"*80)
    
    results = []
    
    for algo in ALGORITHMS:
        algo_data = df[df['algorithm'] == algo]
        baseline_data = algo_data[algo_data['feature_family'] == 'baseline']
        
        if len(baseline_data) == 0:
            continue
        
        baseline_sharpes = baseline_data.set_index('asset')['sharpe_ratio_mean']
        
        families = [f for f in algo_data['feature_family'].unique() if f != 'baseline']
        
        for fam in sorted(families):
            fam_data = algo_data[algo_data['feature_family'] == fam]
            fam_sharpes = fam_data.set_index('asset')['sharpe_ratio_mean']
            
            # Align on common assets
            common = baseline_sharpes.index.intersection(fam_sharpes.index)
            if len(common) < 3:
                continue
            
            b = baseline_sharpes.loc[common].values
            f_vals = fam_sharpes.loc[common].values
            
            try:
                stat, p_value = stats.wilcoxon(f_vals, b, alternative='two-sided')
                direction = 'better' if np.mean(f_vals) > np.mean(b) else 'worse'
                sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else ''
            except Exception:
                p_value = np.nan
                direction = 'N/A'
                sig = ''
                stat = np.nan
            
            results.append({
                'algorithm': algo,
                'family': fam,
                'category': FAMILY_CATEGORY.get(fam, ''),
                'mean_sharpe_family': np.mean(f_vals),
                'mean_sharpe_baseline': np.mean(b),
                'dif': np.mean(f_vals) - np.mean(b),
                'direction': direction,
                'p_value': p_value,
                'significance': sig,
                'n_assets': len(common),
            })
    
    df_stats = pd.DataFrame(results)
    if len(df_stats) > 0:
        df_stats = df_stats.sort_values(['algorithm', 'p_value'])
        print(df_stats.to_string(index=False))
        df_stats.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests_wilcoxon.csv'), index=False)
        print(f"\n Saved: {os.path.join(OUTPUT_DIR, 'statistical_tests_wilcoxon.csv')}")
    
    return df_stats


#---------------------------------------------------------------
# Main
#---------------------------------------------------------------

if __name__ == '__main__':
    print("-" * 80)
    print(" Results — TFG DRL Trading Benchmark")
    print("-" * 80)
    
    # Load data
    print("\n Loading results...")
    df = load_all_results()
    
    if len(df) == 0:
        print("\n Results not found")
        exit(1)
    
    # Save CSV
    df.to_csv(os.path.join(OUTPUT_DIR, 'all_results_consolidated.csv'), index=False)
    print(f"  Data saved in {OUTPUT_DIR}/all_results_consolidated.csv")
    
    # Summary tables
    print("\n Generating tables...")
    pivot, ranking = generate_summary_tables(df)
    ###############
    paper_table = generate_paper_comparison_table(df)
    ##################
    
    # Plots
    print("\n Generating graphics....")
    plot_boxplot_sharpe_by_family(df)
    plot_ablation_comparison(df)
    plot_pure_vs_baseline(df)
    plot_heatmap_sharpe(df)
    plot_category_summary(df)
    
    # Statistical tests
    print("\n Executing statistical tests...")
    df_stats = run_statistical_tests(df)
    
    # Full summary
    print("\n" + "-" * 80)
    print(" Full analysis")
    print("-" * 80)
    print(f"\n Generated files: {os.path.abspath(OUTPUT_DIR)}")
    print(f" Graphics in: {os.path.abspath(PLOT_DIR)}")
    print(f"\n Tables CSV:")
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.csv'))):
        print(f"    - {os.path.basename(f)}")
    print(f"\n Graphics PNG:")
    for f in sorted(glob.glob(os.path.join(PLOT_DIR, '*.png'))):
        print(f"    - {os.path.basename(f)}")

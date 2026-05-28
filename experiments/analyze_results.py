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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
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
                           'final_net_worth', 'total_transaction_costs',
                           'trade_frequency']:
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
    print("Table 1.5 — average Sortino Ratio by family and algorithm")
    print("-"*80)
    
    pivot_sortino = df.pivot_table(
        values='sortino_ratio_mean',
        index='feature_family',
        columns='algorithm',
        aggfunc='mean'
    ).round(3)
    
    pivot_sortino['Category'] = pivot_sortino.index.map(lambda x: FAMILY_CATEGORY.get(x, ''))
    # Calculate average of available algorithms for sorting
    algo_cols = [c for c in ALGORITHMS if c in pivot_sortino.columns]
    pivot_sortino['Mean_Sortino'] = pivot_sortino[algo_cols].mean(axis=1)
    pivot_sortino = pivot_sortino.sort_values(['Category', 'Mean_Sortino'], ascending=[True, False])
    
    # Print and save pivot table without Mean_Sortino column
    pivot_sortino_print = pivot_sortino.drop(columns=['Mean_Sortino'])
    print(pivot_sortino_print.to_string())
    pivot_sortino_print.to_csv(os.path.join(OUTPUT_DIR, 'table_sortino_by_family.csv'))
    
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
    print("Table 3 — global ranking of families by metrics across all algorithms and assets")
    print("-"*80)
    
    ranking = df.groupby('feature_family').agg(
        sharpe_mean=('sharpe_ratio_mean', 'mean'),
        sharpe_std=('sharpe_ratio_mean', 'std'),
        sortino_mean=('sortino_ratio_mean', 'mean'),
        sortino_std=('sortino_ratio_mean', 'std'),
        return_mean=('total_return_mean', 'mean'),
        win_rate_mean=('win_rate_mean', 'mean'),
        max_dd_mean=('max_drawdown_mean', 'mean'),
        trade_count_mean=('trade_count_mean', 'mean'),
    ).sort_values('sharpe_mean', ascending=False).round(4)
    
    ranking['category'] = ranking.index.map(lambda x: FAMILY_CATEGORY.get(x, ''))
    print(ranking.to_string())
    ranking.to_csv(os.path.join(OUTPUT_DIR, 'table_ranking_global.csv'))
    
    return pivot, ranking
#########################################

def generate_paper_comparison_table(df):
    # Generate academic comparison table for baseline, individuals, and top 3 ablations
    
    print("\n" + "-"*80)
    print("Table 4 — Comparison Table (Top Features vs Baseline)")
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
        'sortino_ratio_mean': 'Sortino',
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
    
    # --- Render as a styled matplotlib table plot ---
    # Flatten multi-level columns for display
    flat_cols = [f"{algo}\n{metric}" for algo, metric in pivot.columns]
    flat_df = pivot.copy()
    flat_df.columns = flat_cols
    flat_df = flat_df.reset_index()

    n_rows, n_cols = flat_df.shape
    col_widths = [2.2] + [1.15] * (n_cols - 1)
    fig_width = sum(col_widths) + 0.5
    fig_height = 1.2 + n_rows * 0.45

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title('Comparison Table — Top Features vs Baseline',
                 fontsize=14, fontweight='bold', pad=18)

    cell_text = flat_df.values.tolist()
    col_labels = flat_df.columns.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        cell.set_height(0.08)

    # Display names back to categories for row colouring
    display_to_cat = {}
    for fam, display in FAMILY_DISPLAY.items():
        display_to_cat[display] = FAMILY_CATEGORY.get(fam, 'unknown')

    cat_row_colors = {
        'baseline':      '#E9ECEF',
        'trend':         '#D0E2FF',
        'momentum':      '#D1E7DD',
        'volatility':    '#F8D7DA',
        'statistical':   '#FFF3CD',
        'ablation':      '#E2D9F3',
        'pure_ablation': '#F7D6E6',
    }

    for i in range(n_rows):
        family_name = cell_text[i][0]
        cat = display_to_cat.get(family_name, 'unknown')
        bg = cat_row_colors.get(cat, '#FFFFFF')

        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_height(0.06)
            if j == 0:
                cell.set_text_props(fontweight='bold', ha='left')

    table.auto_set_column_width(list(range(n_cols)))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'table_comparison_results.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved plot: {path}")

    return pivot
#######################################

#---------------------------------------------------------------
# Graphics
#---------------------------------------------------------------

def plot_boxplot_sharpe_by_family(df):
    #Boxplot of Sharpe ratio for each family and ablation group, grouped by algorithm

    # Include all categories: individual families, ablation groups and pure groups
    all_data = df[df['category'].isin(
        ['baseline', 'trend', 'momentum', 'volatility', 'statistical',
         'ablation', 'pure_ablation']
    )]

    # Ordered list: individuals first, then ablation groups, then pure groups
    families_order = [
        'baseline', 'SMA', 'EMA', 'MACD', 'RSI', 'SO',
        'BB', 'ATR', 'RV', 'lagged', 'difference_and_change',
        'temporal_decomposition', 'time_delay_embedding',
        # incremental ablation groups
        'ablation_trend', 'ablation_momentum', 'ablation_volatility',
        'ablation_statistical', 'ablation_all',
        # pure groups (no baseline)
        'pure_trend', 'pure_momentum', 'pure_volatility',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(32, 7), sharey=True)
    fig.suptitle('Sharpe Ratio by feature family — Individual, Ablation and Pure groups',
                 fontsize=16, fontweight='bold', y=1.02)

    for ax, algo in zip(axes, ALGORITHMS):
        algo_data = all_data[all_data['algorithm'] == algo]

        box_data = []
        labels = []
        colors = []
        positions = []
        pos = 1
        separator_positions = []   # x-positions where we draw a vertical separator

        for fam in families_order:
            # Add a small gap before the ablation and pure sections
            if fam == 'ablation_trend':
                separator_positions.append(pos - 0.5)
                pos += 0.5
            elif fam == 'pure_trend':
                separator_positions.append(pos - 0.5)
                pos += 0.5

            fam_data = algo_data[algo_data['feature_family'] == fam]
            if len(fam_data) > 0:
                box_data.append(fam_data['sharpe_ratio_mean'].values)
                labels.append(FAMILY_DISPLAY.get(fam, fam))
                colors.append(CATEGORY_COLORS.get(FAMILY_CATEGORY.get(fam, ''), '#999'))
                positions.append(pos)
            pos += 1

        if not box_data:
            continue

        bp = ax.boxplot(box_data, patch_artist=True, positions=positions,
                        widths=0.6, manage_ticks=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Vertical separators between sections
        for sep_x in separator_positions:
            ax.axvline(x=sep_x, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax.set_xlim(0, pos)
        ax.set_title(algo, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio' if ax == axes[0] else '')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'boxplot_sharpe_individual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_boxplot_sortino_by_family(df):
    #Boxplot of Sortino ratio for each family and ablation group, grouped by algorithm

    # Include all categories: individual families, ablation groups and pure groups
    all_data = df[df['category'].isin(
        ['baseline', 'trend', 'momentum', 'volatility', 'statistical',
         'ablation', 'pure_ablation']
    )]

    # Ordered list: individuals first, then ablation groups, then pure groups (matching Sharpe)
    families_order = [
        'baseline', 'SMA', 'EMA', 'MACD', 'RSI', 'SO',
        'BB', 'ATR', 'RV', 'lagged', 'difference_and_change',
        'temporal_decomposition', 'time_delay_embedding',
        # incremental ablation groups
        'ablation_trend', 'ablation_momentum', 'ablation_volatility',
        'ablation_statistical', 'ablation_all',
        # pure groups (no baseline)
        'pure_trend', 'pure_momentum', 'pure_volatility',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(32, 7), sharey=True)
    fig.suptitle('Sortino Ratio by feature family — Individual, Ablation and Pure groups',
                 fontsize=16, fontweight='bold', y=1.02)

    for ax, algo in zip(axes, ALGORITHMS):
        algo_data = all_data[all_data['algorithm'] == algo]

        box_data = []
        labels = []
        colors = []
        positions = []
        pos = 1
        separator_positions = []   # x-positions where we draw a vertical separator

        for fam in families_order:
            # Add a small gap before the ablation and pure sections
            if fam == 'ablation_trend':
                separator_positions.append(pos - 0.5)
                pos += 0.5
            elif fam == 'pure_trend':
                separator_positions.append(pos - 0.5)
                pos += 0.5

            fam_data = algo_data[algo_data['feature_family'] == fam]
            if len(fam_data) > 0:
                # Handle possible NaN values in Sortino
                vals = fam_data['sortino_ratio_mean'].dropna().values
                if len(vals) > 0:
                    box_data.append(vals)
                    labels.append(FAMILY_DISPLAY.get(fam, fam))
                    colors.append(CATEGORY_COLORS.get(FAMILY_CATEGORY.get(fam, ''), '#999'))
                    positions.append(pos)
            pos += 1

        if not box_data:
            continue

        bp = ax.boxplot(box_data, patch_artist=True, positions=positions,
                        widths=0.6, manage_ticks=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Vertical separators between sections
        for sep_x in separator_positions:
            ax.axvline(x=sep_x, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax.set_xlim(0, pos)
        ax.set_title(algo, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sortino Ratio' if ax == axes[0] else '')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'boxplot_sortino_individual.png')
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
    #Heatmap -> rows=families (individual + ablation), columns=assets, color=Sharpe Ratio

    # Row order: individuals first, then ablation groups, then pure groups
    row_order = [
        'baseline', 'SMA', 'EMA', 'MACD', 'RSI', 'SO',
        'BB', 'ATR', 'RV', 'lagged', 'difference_and_change',
        'temporal_decomposition', 'time_delay_embedding',
        'ablation_trend', 'ablation_momentum', 'ablation_volatility',
        'ablation_statistical', 'ablation_all',
        'pure_trend', 'pure_momentum', 'pure_volatility',
    ]

    for algo in ALGORITHMS:
        algo_data = df[df['algorithm'] == algo]

        # Include all categories
        all_fams = algo_data[algo_data['category'].isin(
            ['baseline', 'trend', 'momentum', 'volatility', 'statistical',
             'ablation', 'pure_ablation']
        )]

        if len(all_fams) == 0:
            continue

        pivot = all_fams.pivot_table(
            values='sharpe_ratio_mean',
            index='feature_family',
            columns='asset',
            aggfunc='mean'
        )

        # Reindex rows in the canonical order (only rows that exist in results)
        ordered_rows = [r for r in row_order if r in pivot.index]
        pivot = pivot.reindex(ordered_rows)

        n_rows = len(pivot.index)
        fig_height = max(8, 0.55 * n_rows + 2)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Symmetric colour scale
        vals = pivot.values.astype(float)
        finite_vals = vals[~np.isnan(vals)]
        vmax = max(abs(finite_vals.max()), abs(finite_vals.min())) if len(finite_vals) else 1
        cmap = plt.cm.RdYlGn
        im = ax.imshow(vals, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([FAMILY_DISPLAY.get(f, f) for f in pivot.index], fontsize=9)

        # Annotate cells
        for i in range(n_rows):
            for j in range(len(pivot.columns)):
                val = vals[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=text_color, fontsize=7.5, fontweight='bold')

        # Draw separator lines between individual / ablation / pure sections
        ablation_start = next(
            (i for i, f in enumerate(pivot.index) if f == 'ablation_trend'), None)
        pure_start = next(
            (i for i, f in enumerate(pivot.index) if f == 'pure_trend'), None)
        for sep in [ablation_start, pure_start]:
            if sep is not None and sep > 0:
                ax.axhline(sep - 0.5, color='white', linewidth=2, linestyle='--')

        plt.colorbar(im, label='Sharpe Ratio')
        ax.set_title(
            f'Heatmap Sharpe Ratio — {algo}\n'
            f'(individual families | ablation groups | pure groups)',
            fontsize=13, fontweight='bold'
        )

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f'heatmap_sharpe_{algo}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {path}")


def plot_category_summary(df):
    #Bar chart comparing average Sharpe by category across algorithms
    #Includes individual, ablation and pure_ablation categories

    all_data = df[df['category'].isin(
        ['baseline', 'trend', 'momentum', 'volatility', 'statistical',
         'ablation', 'pure_ablation']
    )]

    if len(all_data) == 0:
        return

    categories = ['baseline', 'trend', 'momentum', 'volatility', 'statistical',
                  'ablation', 'pure_ablation']
    cat_labels = ['Baseline', 'Trend', 'Momentum', 'Volatility', 'Statistical',
                  'Ablation\n(Base+Group)', 'Pure Ablation\n(no Baseline)']

    x = np.arange(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, algo in enumerate(ALGORITHMS):
        means = []
        for cat in categories:
            cat_data = all_data[(all_data['algorithm'] == algo) & (all_data['category'] == cat)]
            means.append(cat_data['sharpe_ratio_mean'].mean() if len(cat_data) > 0 else 0)

        ax.bar(x + i * width, means, width, label=algo, alpha=0.8)

    # Separator line between individual and ablation sections
    ax.axvline(x=4.5 + width, color='grey', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(4.6 + width, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.1,
            '← individual  |  combined →', fontsize=8, color='grey', va='top')

    ax.set_xticks(x + width)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel('Sharpe Ratio (average)', fontsize=12)
    ax.set_title('Sharpe Ratio by feature category and algorithm\n'
                 '(individual families, ablation groups and pure groups)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
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
    plot_boxplot_sortino_by_family(df)
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

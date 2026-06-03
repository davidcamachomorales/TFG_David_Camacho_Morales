"""
Create 3 graphics one for each algorithm with the evolution of an investment
1000€ for each feature family

To use it --> see README.md
"""
import pandas as pd
import numpy as np
import json
import os
import random
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.trading_env_improved import ImprovedTradingEnv
from src.data_utils import load_dataset, prepare_features, split_train_val
from sklearn.preprocessing import MinMaxScaler

#----------------------------------------------------------------
# Configuration
#----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--asset', default='Gold', help='Asset to use')
parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
parser.add_argument('--seed', type=int, default=42, help='Seed to reproduce')
args = parser.parse_args()

ASSET = args.asset
TIMESTEPS = args.timesteps
SEED = args.seed
INITIAL_INVESTMENT = 1000.0

# Map asset -> file CSV
asset_files = {
    'Gold': 'Gold.csv', 'Silver': 'Silver.csv', 'Nvidia': 'Nvidia.csv',
    'Apple': 'Apple.csv', 'Google': 'Google.csv', 'Inditex': 'Inditex.csv',
    'Bitcoin': 'Bitcoin.csv', 'Ethereum': 'Ethereum.csv',
    'TetherUSDT': 'TetherUSDT.csv', 'S&P_500_Vanguard': 'S&P_500_Vanguard.csv'
}

# Dates are same as scenarios_config
crypto_assets = ['Bitcoin', 'Ethereum', 'TetherUSDT']
END_VAL = '15/03/2026' if ASSET in crypto_assets else '13/03/2026'

# Transaction cost
transaction_cost = 0.0002 if ASSET in crypto_assets else 0.0001

# Colours
FAMILY_COLORS = {
    'SMA': '#e6194b',
    'EMA': '#3cb44b',
    'RSI': '#4363d8',
    'MACD': '#f58231',
    'BB': '#911eb4',
    'SO': '#42d4f4',
    'ATR': '#f032e6',
    'RV': '#bfef45',
    'lagged': '#fabed4',
    'difference_and_change': '#469990',
    'temporal_decomposition': '#dcbef0',
    'time_delay_embedding': '#9A6324',
    'baseline': '#808000',
    'ablation_statistical': '#000075',  
    'ablation_all': '#000000',          
    'pure_trend': '#800000',            
}

#----------------------------------------------------------------
# Functions
#----------------------------------------------------------------

# Setting all seeds as in algorithms codes
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_algorithm(algo_name, env, seed):
    #Return configured model for each algorithm
    if algo_name == 'DQN':
        return DQN(
            "MlpPolicy", env,
            learning_rate=1e-4, learning_starts=1000, buffer_size=50000,
            batch_size=64, gamma=0.99, target_update_interval=500,
            exploration_fraction=0.3, exploration_initial_eps=1.0,
            exploration_final_eps=0.05, train_freq=4, gradient_steps=1,
            verbose=0, seed=seed
        )
    elif algo_name == 'PPO':
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            verbose=0, seed=seed
        )
    elif algo_name == 'A2C':
        return A2C(
            "MlpPolicy", env,
            learning_rate=7e-4, n_steps=5, gamma=0.99,
            gae_lambda=1.0, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0, seed=seed
        )


def evaluate_and_collect_equity(model, df_val, transaction_cost):
    # Evaluate the model and take the net worth curve
    
    env = ImprovedTradingEnv(df_val, transaction_cost=transaction_cost)
    obs, info = env.reset()
    
    # The initial balance of the env is 1000 (= INITIAL_INVESTMENT)
    dates = [info['current_date']]
    net_worths = [env.net_worth]
    
    done = False
    while not done:
        # The model expects observations with shape (1, obs_dim) for the VecEnv
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        action = action.item() if hasattr(action, 'item') else action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dates.append(info['current_date'])
        net_worths.append(env.net_worth)
    
    equity_curve = np.array(net_worths)
    
    return dates, equity_curve


def plot_equity_curves(algo_name, family_curves, asset_name):
    
    # Generates a plot with all equity curves for an algorithm
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for family_name, (dates, equity) in family_curves.items():
        color = FAMILY_COLORS.get(family_name, '#000000')
        ax.plot(dates, equity, label=family_name, color=color, linewidth=1.5, alpha=0.85)
        
        # Put name at the end of the line
        final_value = equity[-1]
        final_date = dates[-1]
        ax.annotate(
            f'{family_name} ({final_value:.0f}€)',
            xy=(final_date, final_value),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=7.5,
            fontweight='bold',
            color=color,
            va='center'
        )
    
    # Horizontal line at 1000€ as initial investment
    ax.axhline(y=INITIAL_INVESTMENT, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.annotate('Initial investment (1000€)', xy=(dates[0], INITIAL_INVESTMENT),
                xytext=(10, -15), textcoords='offset points',
                fontsize=8, color='gray', alpha=0.7)
    
    # Format
    ax.set_title(f'{algo_name} — Evolution of 1000€ by Feature Family ({asset_name})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Investment value (€)', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=45)
    
    # Adjust right margin for the labels
    plt.subplots_adjust(right=0.82)
    
    # Save
    os.makedirs(f'{PROJECT_ROOT}/results/plot', exist_ok=True)
    filepath = f'{PROJECT_ROOT}/results/plot/{algo_name}_{asset_name}_equity_curves.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" Saved: {filepath}")
    return filepath


#----------------------------------------------------------------
# Main
#----------------------------------------------------------------

print("-" * 70)
print(f"Equity curves —> {ASSET}")
print(f"Timesteps: {TIMESTEPS} | Seed: {SEED} | Investment: {INITIAL_INVESTMENT}€")
print("-" * 70)

#  Load data and features
print(f"\n Loading data for {ASSET}...")
raw_file = asset_files[ASSET]
df_raw = load_dataset(raw_file)
df_with_features = prepare_features(df_raw)
print(f"  {len(df_with_features)} rows, {len(df_with_features.columns)} columns")

# Load feature families
with open(f'{PROJECT_ROOT}/config/feature_family.json') as f:
    feature_family = json.load(f)

families = {}
for fam in feature_family['Families']:
    members = [m['Name'] for m in fam['Members']]
    # Verify that all features exist in the DF
    if all(m in df_with_features.columns for m in members):
        families[fam['Name']] = members
    else:
        missing = [m for m in members if m not in df_with_features.columns]
        print(f" Skipping family '{fam['Name']}' — missing features: {missing}")

# Keep only individual features and top 3 ablations to avoid visual overload
allowed_families = [
    'baseline', 'SMA', 'EMA', 'MACD', 'RSI', 'SO', 'BB', 'ATR', 'RV', 
    'lagged', 'difference_and_change', 'temporal_decomposition', 'time_delay_embedding',
    'ablation_statistical', 'ablation_all', 'pure_trend'
]
families = {k: v for k, v in families.items() if k in allowed_families}


# For each algorithm train and test each family
algorithms = ['DQN', 'PPO', 'A2C']

for algo_name in algorithms:
    print(f"\n{'-'*70}")
    print(f" Algorithm: {algo_name}")
    print(f"{'-'*70}")
    
    family_curves = {}
    
    for fam_name, fam_features in families.items():
        print(f"\n  → {fam_name} ({len(fam_features)} features)...", end=" ")
        
        set_all_seeds(SEED)
        
        # Prepare data with only this family + close
        df = df_with_features[fam_features + ['close']].copy()
        
        df_train, df_val = split_train_val(
            df, '01/06/2023', '30/06/2025', '01/07/2025', END_VAL
        )
        
        if len(df_train) < 50 or len(df_val) < 10:
            print(" Not enough data, skipping")
            continue
        
        # Normalize features
        for feature in fam_features:
            df_train[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_val[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train[feature].ffill(inplace=True)
            df_val[feature].ffill(inplace=True)
            df_train[feature].bfill(inplace=True)
            df_val[feature].bfill(inplace=True)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_train[feature] = scaler.fit_transform(df_train[feature].values.reshape(-1, 1))
            df_val[feature] = scaler.transform(df_val[feature].values.reshape(-1, 1))
        
        # Create env and train
        def make_env(df):
            def _init():
                return ImprovedTradingEnv(df, transaction_cost=transaction_cost)
            return _init
        
        train_env = DummyVecEnv([make_env(df_train)])
        model = get_algorithm(algo_name, train_env, SEED)
        model.learn(total_timesteps=TIMESTEPS)
        
        # Evaluate and collect curve
        dates, equity = evaluate_and_collect_equity(model, df_val, transaction_cost)
        family_curves[fam_name] = (dates, equity)
        
        final_val = equity[-1]
        pct = (final_val / INITIAL_INVESTMENT - 1) * 100
        print(f"Final: {final_val:.0f}€ ({pct:+.1f}%)")
        
        # Clean
        train_env.close()
        del model
    
    # Generate plot
    print(f"\n Generating plot {algo_name}...")
    plot_equity_curves(algo_name, family_curves, ASSET)

print(f"\n{'-'*70}")
print("All plots generated in results/plot/")
print(f"{'-'*70}")

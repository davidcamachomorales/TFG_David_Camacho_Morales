import pandas as pd
import numpy as np
import json
import os
import random
import torch
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Uses improved environment
# Import utility functions
from src.experiment_utils import (
    set_all_seeds,
    get_family_members,
    get_scenario_data,
    get_transaction_cost,
    aggregate_seed_results
)

from src.trading_env_improved import ImprovedTradingEnv
from src.data_utils import load_dataset, prepare_features, split_train_val, normalize_dataframe, normalize_val_with_scaler

from sklearn.preprocessing import MinMaxScaler

import traceback

# Seed per scenario, min 3 to be reliable, no more because of execution time
N_SEEDS = 3 

# Specific seeds to include in the paper
SEEDS = [42, 123, 456, 789, 1000, 2024, 3141, 5926, 8675, 3090, 7777, 9999, 1337, 4444, 5555, 6666, 8888, 1111, 2222, 3333][:N_SEEDS]

# Use deterministic mode. Slower but more reproducible
USE_DETERMINISTIC = True

# Unique environment for deterministic
# False to train faster
USE_SINGLE_ENV = True

print(f"Running {N_SEEDS} seeds per scenario: {SEEDS}")
print(f"Deterministic mode: {USE_DETERMINISTIC}")
print(f"Single environment: {USE_SINGLE_ENV}")

timeframe = '1d'

# Load scenario config
scenario_config = pd.read_csv(f'{PROJECT_ROOT}/config/scenarios_config_{timeframe}_ablation.csv')

# Format date columns
scenario_config['start_train_date'] = pd.to_datetime(scenario_config['start_train_date'], format='%d/%m/%Y')
scenario_config['end_train_date'] = pd.to_datetime(scenario_config['end_train_date'], format='%d/%m/%Y')
scenario_config['start_val_date'] = pd.to_datetime(scenario_config['start_val_date'], format='%d/%m/%Y')
scenario_config['end_val_date'] = pd.to_datetime(scenario_config['end_val_date'], format='%d/%m/%Y')

# Load feature family json
with open(f'{PROJECT_ROOT}/config/feature_family.json') as f:
    feature_family = json.load(f)

def train_and_evaluate_single_seed(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    list_features: List[str],
    transaction_cost: float,
    seed: int,
    scenario_id: int,
    total_timesteps: int,
    verbose: bool = False
) -> Dict:
    # Train and evaluate DQN for a single seed
    # Returns dict with every metric for this seed
    # Set all seeds
    set_all_seeds(seed, USE_DETERMINISTIC)
    
    # Create environments
    def make_env(df, rank):
        def _init():
            set_all_seeds(seed + rank, USE_DETERMINISTIC)  # Slightly different seed per env
            return ImprovedTradingEnv(df, transaction_cost=transaction_cost)
        return _init
    
    if USE_SINGLE_ENV:
        n_envs = 1
        train_env = DummyVecEnv([make_env(df_train, 0)])
    else:
        n_envs = 4
        train_env = SubprocVecEnv([make_env(df_train, i) for i in range(n_envs)])
    
    val_env = DummyVecEnv([make_env(df_val, 0)])
    
    # Create and train model
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        learning_starts=1000,
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
        seed=seed
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate
    obs = val_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = val_env.step(action)
    
    # Extract metrics
    metrics = {
        'seed': seed,
        'total_return': info[0]['total_return'],
        'total_return_before_costs': info[0]['total_return_before_costs'],
        'sharpe_ratio': info[0]['sharpe_ratio'],
        'sharpe_ratio_before_costs': info[0]['sharpe_ratio_before_costs'],
        'sortino_ratio': info[0]['sortino_ratio'],
        'max_drawdown': info[0]['max_drawdown'],
        'win_rate': info[0]['win_rate'],
        'win_rate_before_costs': info[0]['win_rate_before_costs'],
        'total_transaction_costs': info[0]['total_transaction_costs'],
        'trade_count': info[0]['trade_count'],
        'trade_frequency': info[0]['trade_frequency'],
        'final_net_worth': info[0]['net_worth'],
        'final_net_worth_before_costs': info[0]['net_worth_before_costs'],
    }
    
    # Cleanup
    train_env.close()
    val_env.close()
    del model
    
    if verbose:
        print(f"\n Seed {seed}: Return={metrics['total_return']*100:.2f}%, Sharpe={metrics['sharpe_ratio']:.3f}")
    
    return metrics

#----------------------------------------------------------------
# Select scenarios to process
#----------------------------------------------------------------

# list_scenario_id = list(range(1, 181))  # All 180 scenarios (130 individual + 50 ablation)
# this one is a smaller test with ablation but alwais with baseliners in combinations

list_scenario_id = list(range(1, 211))  # All 210 scenarios (130 individual + 50 ablation + 30 pure ablation)

total_runs = len(list_scenario_id) * N_SEEDS
print(f"Processing {len(list_scenario_id)} scenarios × {N_SEEDS} seeds = {total_runs} total training runs")
print(f"Estimated time: {total_runs * 2 / 60:.1f} hours (assuming ~2 min per run)")

# Create results directories
os.makedirs(f'{PROJECT_ROOT}/results/df', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/json', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/plot', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/tensorboard', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/models', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/logs', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/results/checkpoints', exist_ok=True)

# debug
# print("Results directories created")

#----------------------------------------------------------------
# Main processing loop
#----------------------------------------------------------------

batch_summary = []
all_seed_results = []  # Store individual seed results for statistical tests

for idx, scenario_id in enumerate(list_scenario_id):
    print("\n" + "-"*80)
    print(f"Scenario {scenario_id} ({idx+1}/{len(list_scenario_id)})")
    print("-"*80)
    
    try:
        # Get scenario data
        scenario_data = get_scenario_data(scenario_id, scenario_config)
        print(f"Asset: {scenario_data['asset']}, Feature Family: {scenario_data['feature_family']}")
        
        transaction_cost = get_transaction_cost(scenario_data['asset'])
        
        #---------------------------------------------------------------
        # Load data and compute features
        #---------------------------------------------------------------
        df_raw = load_dataset(scenario_data['raw_file'])
        df_with_features = prepare_features(df_raw)
        
        # Get the feature columns for this family
        list_features = get_family_members(scenario_data['feature_family'], feature_family)
        
        if list_features is None:
            print(f"Feature family '{scenario_data['feature_family']}' not found, skipping...")
            continue
        
        # Keep only the features + close
        df = df_with_features[list_features + ['close']]
        
        # Split train/val
        df_train, df_val = split_train_val(
            df,
            scenario_data['start_train_date'],
            scenario_data['end_train_date'],
            scenario_data['start_val_date'],
            scenario_data['end_val_date']
        )
        
        print(f"Train: {len(df_train)} samples, Val: {len(df_val)} samples")
        
        if len(df_train) < 50 or len(df_val) < 10:
            print(f"Skipping because not enough data...")
            continue
        
        # Scale features (using training data statistics)
        for feature in list_features:
            df_train[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_val[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train[feature].ffill(inplace=True)
            df_val[feature].ffill(inplace=True)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_train[feature] = scaler.fit_transform(df_train[feature].values.reshape(-1, 1))
            df_val[feature] = scaler.transform(df_val[feature].values.reshape(-1, 1))
        
        total_timesteps = max(100000, len(df_train) * 50)
        
        #---------------------------------------------------------------
        # Run multiple seeds
        #---------------------------------------------------------------
        print(f"\nRunning {N_SEEDS} seeds...")
        seed_results = []
        
        for seed_idx, seed in enumerate(SEEDS):
            print(f"  Seed {seed_idx+1}/{N_SEEDS} (seed={seed})...", end=" ")
            
            metrics = train_and_evaluate_single_seed(
                df_train=df_train,
                df_val=df_val,
                list_features=list_features,
                transaction_cost=transaction_cost,
                seed=seed,
                scenario_id=scenario_id,
                total_timesteps=total_timesteps,
                verbose=False
            )
            
            # Add scenario metadata
            metrics['scenario_id'] = scenario_id
            metrics['asset'] = scenario_data['asset']
            metrics['feature_family'] = scenario_data['feature_family']
            
            seed_results.append(metrics)
            all_seed_results.append(metrics)
            
            print(f"Return={metrics['total_return']*100:.2f}%")
        
        #---------------------------------------------------------------
        # Aggregated results
        #---------------------------------------------------------------
        aggregated = aggregate_seed_results(seed_results)
        aggregated['scenario_id'] = scenario_id
        aggregated['asset'] = scenario_data['asset']
        aggregated['feature_family'] = scenario_data['feature_family']
        aggregated['transaction_cost_pct'] = transaction_cost * 100
        aggregated['n_features'] = len(list_features)
        
        batch_summary.append(aggregated)

        # Uncomment to see the summary
        '''
        # Print summary
        print(f"\n" + "-"*60)
        print(f"Results with n={N_SEEDS} seeds")
        print("-"*60)
        print(f"Total Return:  {aggregated['total_return_mean']*100:>7.2f}% ± {aggregated['total_return_std']*100:.2f}%")
        print(f"Sharpe Ratio:  {aggregated['sharpe_ratio_mean']:>7.3f} ± {aggregated['sharpe_ratio_std']:.3f}")
        print(f"Win Rate:      {aggregated['win_rate_mean']*100:>7.2f}% ± {aggregated['win_rate_std']*100:.2f}%")
        print(f"Trade Count:   {aggregated['trade_count_mean']:>7.0f} ± {aggregated['trade_count_std']:.0f}")
        print("-"*60)
        '''
        # Save results in json for later analysis
        save_dir_json = f"{PROJECT_ROOT}/results/json"
        os.makedirs(save_dir_json, exist_ok=True)
        with open(f"{save_dir_json}/{scenario_id}_DQN_results.json", 'w') as f:
            json_safe = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in aggregated.items()}
            json.dump(json_safe, f, indent=2)
        
        # Save per-seed results for analysis
        save_dir_per_seed = f"{PROJECT_ROOT}/results/json_per_seed"
        os.makedirs(save_dir_per_seed, exist_ok=True)
        pd.DataFrame(seed_results).to_csv(
            f"{save_dir_per_seed}/{scenario_id}_per_seed.csv",
            index=False
        )
        
        print(f"Scenario {scenario_id} complete!")
        
    except Exception as e:
        print(f"\n Error in scenario {scenario_id}: {str(e)}")
        traceback.print_exc()
        continue

# Uncomment to see the final summary
'''
print("\n" + "-"*80)
print("Processing complete")
print("-"*80)
print(f"Processed {len(batch_summary)} scenarios successfully")
print(f"Total training runs: {len(all_seed_results)}")
'''

#----------------------------------------------------------------
# Results analysis
#----------------------------------------------------------------

# Summary DF
df_batch_summary = pd.DataFrame(batch_summary)
# Save summary
df_batch_summary.to_csv(f'{PROJECT_ROOT}/results/DQN_batch_summary.csv', index=False)
# Save all seed results 
df_all_seeds = pd.DataFrame(all_seed_results)
df_all_seeds.to_csv(f'{PROJECT_ROOT}/results/DQN_all_seed_results.csv', index=False)
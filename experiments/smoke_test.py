"""
Smoke test to verify the full pipeline works for DQN, PPO and A2C
with only 1 scenario, 1 seed, and few timesteps

To use it go to README.md
"""
import pandas as pd
import numpy as np
import json
import os
import random
import torch
import warnings
warnings.filterwarnings("ignore")

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
# Load data and features, scenario 1 with Gold + SMA
#----------------------------------------------------------------
print("-" * 60)
print("Smoke test — verifying full pipeline")
print("-" * 60)

print("\n[1/4] Loading Gold data...")
df_raw = load_dataset("Gold.csv")
print(f"  Raw data: {len(df_raw)} rows, columns: {list(df_raw.columns)}")

print("\n[2/4] Computing features...")
df_with_features = prepare_features(df_raw)
print(f"  With features: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")

# Use SMA family
with open(f'{PROJECT_ROOT}/config/feature_family.json') as f:
    feature_family = json.load(f)

list_features = None
for family in feature_family['Families']:
    if family['Name'] == 'SMA':
        list_features = [m['Name'] for m in family['Members']]
        break

print(f"  Features SMA: {list_features}")

# Verify that the features exist in the DF
missing = [f for f in list_features if f not in df_with_features.columns]
if missing:
    print(f"\n  Features not found in DF: {missing}")
    print(f"  Available columns starting with TI_SMA: {[c for c in df_with_features.columns if 'SMA' in c]}")
    exit(1)

df = df_with_features[list_features + ['close']]

#----------------------------------------------------------------
# Split train/val
#----------------------------------------------------------------
print("\n[3/4] Splitting train/val...")
df_train, df_val = split_train_val(
    df,
    start_train='01/06/2023',
    end_train='30/06/2025',
    start_val='01/07/2025',
    end_val='13/03/2026'
)
print(f"  Train: {len(df_train)} samples")
print(f"  Val:   {len(df_val)} samples")

# Normalize
for feature in list_features:
    df_train[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_val[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train[feature].ffill(inplace=True)
    df_val[feature].ffill(inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train[feature] = scaler.fit_transform(df_train[feature].values.reshape(-1, 1))
    df_val[feature] = scaler.transform(df_val[feature].values.reshape(-1, 1))

#----------------------------------------------------------------
# Train and evaluate the 3 algorithms
#----------------------------------------------------------------
print("\n[4/4] Training the 3 algorithms with reduced timesteps...")

SEED = 42
TIMESTEPS = 5000  # not many so it runs fast

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

transaction_cost = 0.0001  # Gold = traditional asset

def make_env(df):
    def _init():
        return ImprovedTradingEnv(df, transaction_cost=transaction_cost)
    return _init

algorithms = {
    'DQN': DQN(
        "MlpPolicy",
        DummyVecEnv([make_env(df_train)]),
        learning_rate=1e-4,
        learning_starts=500,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        verbose=0,
        seed=SEED
    ),
    'PPO': PPO(
        "MlpPolicy",
        DummyVecEnv([make_env(df_train)]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=SEED
    ),
    'A2C': A2C(
        "MlpPolicy",
        DummyVecEnv([make_env(df_train)]),
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        verbose=0,
        seed=SEED
    ),
}

results = {}

for algo_name, model in algorithms.items():
    print(f"\n  --- {algo_name} ---")
    
    # Train
    model.learn(total_timesteps=TIMESTEPS)
    print(f"  Training completed ({TIMESTEPS} timesteps)")
    
    # Evaluate
    val_env = DummyVecEnv([make_env(df_val)])
    obs = val_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = val_env.step(action)
    
    results[algo_name] = {
        'total_return': info[0]['total_return'],
        'sharpe_ratio': info[0]['sharpe_ratio'],
        'win_rate': info[0]['win_rate'],
        'trade_count': info[0]['trade_count'],
        'final_net_worth': info[0]['net_worth'],
    }
    
    print(f"  Return: {info[0]['total_return']*100:.2f}%")
    print(f"  Sharpe: {info[0]['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {info[0]['win_rate']*100:.1f}%")
    print(f"  Trades: {info[0]['trade_count']}")
    
    val_env.close()
    model.get_env().close()
    del model

#----------------------------------------------------------------
# Summary
#----------------------------------------------------------------
print("\n" + "-" * 60)
print("Smoke test summary")
print("-" * 60)
print(f"{'Algorithm':<10} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10} {'Trades':>8}")
print("-" * 48)
for algo, r in results.items():
    print(f"{algo:<10} {r['total_return']*100:>9.2f}% {r['sharpe_ratio']:>10.3f} {r['win_rate']*100:>9.1f}% {r['trade_count']:>8}")

print("\nSmoke test completed successfully")
print("The 3 algorithms train and evaluate with no errors")

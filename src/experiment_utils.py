import pandas as pd
import numpy as np
import json
import os
import random
import torch
from typing import List, Dict

def set_all_seeds(seed: int, use_deterministic: bool = True):
    # Set all random seeds for reproducibility
    # Cuda operations are deterministic when use_deterministic is True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed

def get_family_members(family_name, data):
    # Parse the json data if it is a string, otherwise use it as is
    if isinstance(data, str):
        data = json.loads(data)
    
    # Find the family with the name
    for family in data['Families']:
        if family['Name'] == family_name:
            # Return a list of member names
            return [member['Name'] for member in family['Members']]
    
    # If family not found, return None or an empty list
    return None

def get_scenario_data(scenario_name, scenario_config):
    # Get the scenario row
    scenario = scenario_config[scenario_config['scenario'] == scenario_name].iloc[0]
    
    # Get the data for the scenario
    scenario_data = {
        'asset': scenario['asset'],
        'feature_family': scenario['feature_family'],
        'start_train_date': scenario['start_train_date'],
        'end_train_date': scenario['end_train_date'],
        'start_val_date': scenario['start_val_date'],
        'end_val_date': scenario['end_val_date'],
        'raw_file': scenario['raw_file'],
    }
    
    return scenario_data

def get_transaction_cost(asset_name):
    # Determine transaction cost based on asset type
    # Crypto is 0.02% from Binance maker fee, traditional assets are 0.01%
    crypto_assets = ['Bitcoin', 'Ethereum', 'TetherUSDT']
    if asset_name in crypto_assets:
        return 0.0002  # 0.02% crypto
    else:
        return 0.0001  # 0.01% traditional assets

def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    # Aggregate results from multiple seeds into mean and std
    # Returns mean, std, min, max and list of values for statistical tests
    df_seeds = pd.DataFrame(seed_results)
    
    aggregated = {
        'n_seeds': len(seed_results),
        'seeds_used': [r['seed'] for r in seed_results],
    }
    
    metrics_to_aggregate = [
        'total_return', 'total_return_before_costs',
        'sharpe_ratio', 'sharpe_ratio_before_costs',
        'sortino_ratio', 'max_drawdown',
        'win_rate', 'win_rate_before_costs',
        'total_transaction_costs', 'trade_count', 'trade_frequency',
        'final_net_worth', 'final_net_worth_before_costs'
    ]
    
    for metric in metrics_to_aggregate:
        values = df_seeds[metric].values
        aggregated[f'{metric}_mean'] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values))
        aggregated[f'{metric}_min'] = float(np.min(values))
        aggregated[f'{metric}_max'] = float(np.max(values))
        aggregated[f'{metric}_values'] = values.tolist()
    
    return aggregated

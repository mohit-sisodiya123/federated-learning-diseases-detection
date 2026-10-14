# server/fedavg.py
"""
Federated Averaging (FedAvg) aggregation function.
Given a list of client model state_dicts and their sample counts,
it returns the weighted average state_dict.
"""

import torch
from collections import OrderedDict

def fedavg(updates: list[tuple[dict, int]]) -> OrderedDict:
    """
    Aggregate client updates using weighted average.
    
    Args:
        updates: list of (state_dict, num_samples) from each client
    
    Returns:
        averaged_state_dict: aggregated model parameters
    """
    # Compute total number of samples
    total_samples = sum(num_samples for _, num_samples in updates)
    
    # Initialize the averaged state dict with zeros
    avg_state = OrderedDict()
    # We use the first client's state_dict as a template to know parameter shapes
    first_state = updates[0][0]
    for key in first_state.keys():
        avg_state[key] = torch.zeros_like(first_state[key])
    
    # Sum weighted parameters
    for state_dict, num_samples in updates:
        weight = num_samples / total_samples
        for key in state_dict.keys():
            avg_state[key] += weight * state_dict[key]
    
    return avg_state
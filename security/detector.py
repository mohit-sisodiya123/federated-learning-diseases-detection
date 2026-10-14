# security/detector.py
"""
Malicious-client detection using cosine similarity of model updates.
"""

import numpy as np
import torch
from collections import OrderedDict


def flatten_state_dict(state_dict: OrderedDict) -> torch.Tensor:
    """
    Flatten all tensors inside a state_dict into one long 1D vector.
    """
    parts = []
    for key in state_dict.keys():
        parts.append(state_dict[key].flatten())
    return torch.cat(parts)


def compute_update(client_state: OrderedDict,
                   global_state: OrderedDict) -> OrderedDict:
    """
    Compute the model update:
        update = client_parameters - global_parameters
    """
    update = OrderedDict()
    for key in client_state.keys():
        update[key] = client_state[key] - global_state[key]
    return update


def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D vectors.
    """
    norm_u = torch.norm(u)
    norm_v = torch.norm(v)
    if norm_u.item() == 0.0 or norm_v.item() == 0.0:
        return 0.0
    return float(torch.dot(u, v) / (norm_u * norm_v))


def calculate_similarity_matrix(updates: list) -> np.ndarray:
    """
    Given a list of update state_dicts, return an (n x n) similarity matrix.
    """
    flat_updates = [flatten_state_dict(u) for u in updates]
    n = len(flat_updates)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(flat_updates[i], flat_updates[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


def detect_suspicious_clients(updates: list, sensitivity: float = 2.5):
    """
    Detect suspicious clients based on average cosine similarity (Phase 3).
    """
    n = len(updates)
    sim_matrix = calculate_similarity_matrix(updates)

    avg_similarities = []
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        avg_similarities.append(float(np.mean(others)) if others else 1.0)
    avg_similarities = np.array(avg_similarities)

    median_score = float(np.median(avg_similarities))
    mad = float(np.median(np.abs(avg_similarities - median_score)))

    if mad < 1e-6:
        mad = 1e-3

    threshold = median_score - sensitivity * mad

    suspicious_ids = [i for i in range(n) if avg_similarities[i] < threshold]

    return suspicious_ids, avg_similarities, sim_matrix, threshold


def detect_malicious_clients_simple(updates: list, threshold: float = 0.5):
    """
    Simple detection method requested in Phase 4:
    1. Compute the mean (average) update of all clients.
    2. Compare each client's update to this mean update using cosine similarity.
    3. If similarity < threshold, flag the client as suspicious.
    """
    # Compute the mean update across all clients
    mean_update = OrderedDict()
    for key in updates[0].keys():
        stacked = torch.stack([u[key] for u in updates])
        mean_update[key] = torch.mean(stacked, dim=0)

    # Flatten the mean update into a single vector
    flat_mean = flatten_state_dict(mean_update)

    suspicious = []
    similarities = []

    for i, update in enumerate(updates):
        flat_update = flatten_state_dict(update)
        sim = cosine_similarity(flat_update, flat_mean)
        similarities.append(sim)

        if sim < threshold:
            suspicious.append(i)

    return suspicious, similarities

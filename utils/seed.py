# utils/seed.py
"""
Set random seeds for Python, NumPy and PyTorch to ensure reproducibility.
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set all seeds to the given value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations on GPU (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
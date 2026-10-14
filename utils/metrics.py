# utils/metrics.py
"""
Functions to evaluate model performance.
Includes accuracy, precision, recall, F1, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

def evaluate_model(model: torch.nn.Module, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Evaluate the model on the test set.
    
    Returns:
        metrics: dict with accuracy, precision, recall, f1, confusion matrix
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    
    y_true = y_test.numpy().flatten()
    y_pred = preds.numpy().flatten()
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics

def compute_loss(model: torch.nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    """Compute binary cross-entropy loss on test set."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_test)
    return loss.item()
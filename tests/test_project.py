# tests/test_project.py
"""
Simple sanity tests for the project. Run with:
    python -m tests.test_project
"""

import torch
from utils.seed import set_seed
from utils.config import SEED, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_CLIENTS
from models.classifier import SimpleClassifier
from security.detector import (
    compute_update, cosine_similarity, detect_malicious_clients_simple
)
from server.fedavg import fedavg
from collections import OrderedDict


def test_model_forward():
    set_seed(SEED)
    model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    x = torch.randn(4, INPUT_DIM)
    y = model(x)
    assert y.shape == (4, 1), f"Unexpected output shape {y.shape}"
    print("  [OK] model forward pass")


def test_cosine_similarity():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([-1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    assert abs(cosine_similarity(a, c) + 1.0) < 1e-6
    print("  [OK] cosine similarity")


def test_compute_update():
    a = OrderedDict([("w", torch.tensor([2.0, 3.0]))])
    b = OrderedDict([("w", torch.tensor([1.0, 1.0]))])
    u = compute_update(a, b)
    assert torch.allclose(u["w"], torch.tensor([1.0, 2.0]))
    print("  [OK] compute_update")


def test_detector_flags_outlier():
    # 4 similar updates, 1 reversed
    u_normal = OrderedDict([("w", torch.tensor([1.0, 1.0]))])
    u_bad = OrderedDict([("w", torch.tensor([-1.0, -1.0]))])
    updates = [u_normal, u_normal, u_bad, u_normal, u_normal]
    suspicious, sims = detect_malicious_clients_simple(updates, threshold=0.5)
    assert 2 in suspicious, f"Expected client 2 to be flagged, got {suspicious}"
    print("  [OK] detector flags outlier")


def test_fedavg_weighted_average():
    s1 = OrderedDict([("w", torch.tensor([0.0, 0.0]))])
    s2 = OrderedDict([("w", torch.tensor([1.0, 1.0]))])
    result = fedavg([(s1, 1), (s2, 3)])
    # Weighted average: (1*0 + 3*1) / 4 = 0.75
    assert torch.allclose(result["w"], torch.tensor([0.75, 0.75]))
    print("  [OK] fedavg weighted average")


def main():
    print("Running tests...")
    test_model_forward()
    test_cosine_similarity()
    test_compute_update()
    test_detector_flags_outlier()
    test_fedavg_weighted_average()
    print("All tests passed.")


if __name__ == "__main__":
    main()
# experiments/run_phase2.py
"""
Phase 2 experiment:
- Run normal Federated Learning (no malicious clients)
- Run Federated Learning with one malicious client performing model poisoning
- Compare accuracy, loss, and other metrics
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils.config import (
    SEED, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    MALICIOUS_CLIENT_IDS, ATTACK_SCALE, RESULTS_DIR
)
from utils.seed import set_seed
from data.dataset import load_and_preprocess
from models.classifier import SimpleClassifier
from clients.client import Client
from server.fedavg import fedavg
from utils.metrics import evaluate_model, compute_loss

def run_federated_learning(
    X_train, y_train, X_test, y_test,
    num_clients, malicious_ids, attack_scale
):
    """
    Run federated learning with optional malicious clients.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        num_clients: number of clients
        malicious_ids: list of client indices that are malicious
        attack_scale: scale for poisoning attack (only used if there are malicious clients)
    
    Returns:
        final_model: the final global model
        history: dict with accuracy and loss per round
    """
    set_seed(SEED)
    
    # Split training data among clients
    num_samples_per_client = len(X_train) // num_clients
    client_data_indices = []
    perm = torch.randperm(len(X_train))
    for i in range(num_clients):
        start = i * num_samples_per_client
        end = start + num_samples_per_client if i < num_clients-1 else len(X_train)
        indices = perm[start:end]
        client_data_indices.append(indices)
    
    # Create clients
    clients = []
    for i in range(num_clients):
        idx = client_data_indices[i]
        is_malicious = i in malicious_ids
        client = Client(
            client_id=i,
            X_train=X_train[idx],
            y_train=y_train[idx],
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            is_malicious=is_malicious,
            attack_scale=attack_scale
        )
        clients.append(client)
    
    # Initialize global model
    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    history = {"accuracy": [], "loss": []}
    
    for round_num in range(1, NUM_ROUNDS+1):
        print(f"  Round {round_num}")
        updates = []
        for client in clients:
            state_dict, num_samples = client.train(global_model)
            updates.append((state_dict, num_samples))
        
        # Aggregate using FedAvg (server does not know which are malicious)
        new_state = fedavg(updates)
        global_model.load_state_dict(new_state)
        
        # Evaluate
        acc = evaluate_model(global_model, X_test, y_test)["accuracy"]
        loss = compute_loss(global_model, X_test, y_test)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"    Global accuracy: {acc:.4f}, loss: {loss:.4f}")
    
    return global_model, history

def save_plots(history_normal, history_malicious, results_dir):
    """Save accuracy and loss comparison plots."""
    rounds = range(1, NUM_ROUNDS+1)
    
    # Accuracy comparison
    plt.figure(figsize=(8,5))
    plt.plot(rounds, history_normal["accuracy"], marker='o', label="Normal FL")
    plt.plot(rounds, history_malicious["accuracy"], marker='s', label="FL + Malicious Client")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: Normal FL vs FL with Malicious Client")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "phase2_accuracy_comparison.png"))
    plt.close()
    
    # Loss comparison
    plt.figure(figsize=(8,5))
    plt.plot(rounds, history_normal["loss"], marker='o', label="Normal FL")
    plt.plot(rounds, history_malicious["loss"], marker='s', label="FL + Malicious Client")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Loss Comparison: Normal FL vs FL with Malicious Client")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "phase2_loss_comparison.png"))
    plt.close()

def main():
    print("Starting Phase 2: Malicious Client and Model Poisoning Simulation")
    set_seed(SEED)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Experiment A: Normal FL (no malicious clients)
    print("\n=== Experiment A: Normal Federated Learning (0 malicious) ===")
    normal_model, history_normal = run_federated_learning(
        X_train, y_train, X_test, y_test,
        num_clients=NUM_CLIENTS,
        malicious_ids=[],
        attack_scale=0.0
    )
    normal_metrics = evaluate_model(normal_model, X_test, y_test)
    normal_loss = compute_loss(normal_model, X_test, y_test)
    print(f"Normal FL final metrics:")
    print(f"  Accuracy: {normal_metrics['accuracy']:.4f}")
    print(f"  Precision: {normal_metrics['precision']:.4f}")
    print(f"  Recall: {normal_metrics['recall']:.4f}")
    print(f"  F1-score: {normal_metrics['f1']:.4f}")
    print(f"  Loss: {normal_loss:.4f}")
    
    # Experiment B: FL with malicious client(s)
    print("\n=== Experiment B: Federated Learning with Malicious Client(s) ===")
    print(f"Malicious client IDs: {MALICIOUS_CLIENT_IDS}")
    print(f"Attack scale: {ATTACK_SCALE}")
    malicious_model, history_malicious = run_federated_learning(
        X_train, y_train, X_test, y_test,
        num_clients=NUM_CLIENTS,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE
    )
    malicious_metrics = evaluate_model(malicious_model, X_test, y_test)
    malicious_loss = compute_loss(malicious_model, X_test, y_test)
    print(f"FL with malicious client(s) final metrics:")
    print(f"  Accuracy: {malicious_metrics['accuracy']:.4f}")
    print(f"  Precision: {malicious_metrics['precision']:.4f}")
    print(f"  Recall: {malicious_metrics['recall']:.4f}")
    print(f"  F1-score: {malicious_metrics['f1']:.4f}")
    print(f"  Loss: {malicious_loss:.4f}")
    
    # Save plots
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_plots(history_normal, history_malicious, RESULTS_DIR)
    
    # Save text summary
    summary_path = os.path.join(RESULTS_DIR, "phase2_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("Phase 2 Comparison: Normal FL vs FL with Malicious Client\n")
        f.write("="*60 + "\n\n")
        f.write(f"Malicious Client IDs: {MALICIOUS_CLIENT_IDS}\n")
        f.write(f"Attack Scale: {ATTACK_SCALE}\n\n")
        f.write("Normal FL:\n")
        f.write(f"  Accuracy: {normal_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {normal_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {normal_metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {normal_metrics['f1']:.4f}\n")
        f.write(f"  Loss: {normal_loss:.4f}\n\n")
        f.write("FL with Malicious Client:\n")
        f.write(f"  Accuracy: {malicious_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {malicious_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {malicious_metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {malicious_metrics['f1']:.4f}\n")
        f.write(f"  Loss: {malicious_loss:.4f}\n")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Phase 2 completed successfully.")

if __name__ == "__main__":
    main()
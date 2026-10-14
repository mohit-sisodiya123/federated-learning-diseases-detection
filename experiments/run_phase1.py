# experiments/run_phase1.py
"""
Phase 1 experiment:
- Loads and preprocesses dataset
- Trains a centralized model as baseline
- Simulates Federated Learning with 5 clients (no malicious clients)
- Compares performance and saves plots
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
    LEARNING_RATE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, RESULTS_DIR
)
from utils.seed import set_seed
from data.dataset import load_and_preprocess
from models.classifier import SimpleClassifier
from clients.client import Client
from server.fedavg import fedavg
from utils.metrics import evaluate_model, compute_loss

def train_centralized(X_train, y_train, X_test, y_test):
    """Train a single model on all training data (centralized baseline)."""
    set_seed(SEED)
    model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train for a number of epochs (same total epochs as FL for fair comparison)
    total_epochs = NUM_ROUNDS * LOCAL_EPOCHS  # 10 rounds * 3 local epochs = 30
    for epoch in range(total_epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

def run_federated_learning(X_train, y_train, X_test, y_test):
    """
    Simulate Federated Learning with multiple clients.
    Returns:
        global_model: final global model
        history: dict with accuracy and loss per round
    """
    set_seed(SEED)
    
    # Split training data among clients randomly
    num_samples_per_client = len(X_train) // NUM_CLIENTS
    client_data_indices = []
    perm = torch.randperm(len(X_train))  # shuffle indices
    for i in range(NUM_CLIENTS):
        start = i * num_samples_per_client
        end = start + num_samples_per_client if i < NUM_CLIENTS-1 else len(X_train)
        indices = perm[start:end]
        client_data_indices.append(indices)
    
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        idx = client_data_indices[i]
        client = Client(
            client_id=i,
            X_train=X_train[idx],
            y_train=y_train[idx],
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
    
    # Initialize global model
    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    history = {"accuracy": [], "loss": []}
    
    # Federated rounds
    for round_num in range(1, NUM_ROUNDS+1):
        print(f"\n--- Federated Round {round_num} ---")
        updates = []
        for client in clients:
            print(f"  Client {client.client_id} training on {client.num_samples} samples...")
            state_dict, num_samples = client.train(global_model)
            updates.append((state_dict, num_samples))
        
        # Aggregate using FedAvg
        new_state = fedavg(updates)
        global_model.load_state_dict(new_state)
        
        # Evaluate global model
        acc = evaluate_model(global_model, X_test, y_test)["accuracy"]
        loss = compute_loss(global_model, X_test, y_test)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"  Global model test accuracy: {acc:.4f}, loss: {loss:.4f}")
    
    return global_model, history

def plot_results(history, central_acc, central_loss):
    """Generate and save plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    rounds = range(1, NUM_ROUNDS+1)
    
    # Accuracy plot
    plt.figure(figsize=(8,5))
    plt.plot(rounds, history["accuracy"], marker='o', label="Federated")
    plt.axhline(y=central_acc, color='r', linestyle='--', label="Centralized")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs FL Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_rounds.png"))
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(rounds, history["loss"], marker='o', label="Federated")
    plt.axhline(y=central_loss, color='r', linestyle='--', label="Centralized")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Loss vs FL Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "loss_vs_rounds.png"))
    plt.close()
    
    print(f"\nPlots saved in {RESULTS_DIR}/")

def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    import seaborn as sns  # optional, but we can use matplotlib only
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Federated Model)")
    plt.colorbar()
    tick_marks = [0,1]
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # Add numbers
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved in {RESULTS_DIR}/confusion_matrix.png")

def main():
    print("Starting Phase 1: Centralized vs Federated Learning")
    set_seed(SEED)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 1. Centralized baseline
    print("\nTraining centralized model...")
    central_model = train_centralized(X_train, y_train, X_test, y_test)
    central_metrics = evaluate_model(central_model, X_test, y_test)
    central_loss = compute_loss(central_model, X_test, y_test)
    print(f"Centralized model - Accuracy: {central_metrics['accuracy']:.4f}, Loss: {central_loss:.4f}")
    
    # 2. Federated Learning
    print("\nStarting Federated Learning simulation...")
    global_model, history = run_federated_learning(X_train, y_train, X_test, y_test)
    
    # Final evaluation
    final_metrics = evaluate_model(global_model, X_test, y_test)
    final_loss = compute_loss(global_model, X_test, y_test)
    print("\nFinal Federated Model:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1-score: {final_metrics['f1']:.4f}")
    print(f"Loss: {final_loss:.4f}")
    print("Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    
    # Plot results
    plot_results(history, central_metrics['accuracy'], central_loss)
    plot_confusion_matrix(final_metrics['confusion_matrix'])
    
    print("\nPhase 1 completed successfully.")

if __name__ == "__main__":
    main()
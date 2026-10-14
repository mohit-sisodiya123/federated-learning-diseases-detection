# experiments/run_phase4.py
"""
Phase 4 test: Simple malicious-client detection using a fixed threshold.
This script runs ONE federated round and tests the detection logic.
It does not perform the full experiment/metrics/plots (that is Phase 5).
"""

import torch
from utils.config import (
    SEED, NUM_CLIENTS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    MALICIOUS_CLIENT_IDS, ATTACK_SCALE,
    DETECTION_ENABLED, SIMILARITY_THRESHOLD
)
from utils.seed import set_seed
from data.dataset import load_and_preprocess
from models.classifier import SimpleClassifier
from clients.client import Client
from security.detector import compute_update, detect_malicious_clients_simple
from server.fedavg import fedavg


def main():
    set_seed(SEED)
    print("--- Phase 4: Simple Malicious Client Detection Test ---")
    print(f"Malicious clients (ground truth): {MALICIOUS_CLIENT_IDS}")
    print(f"Detection enabled: {DETECTION_ENABLED}, Threshold: {SIMILARITY_THRESHOLD}")
    print()

    # 1. Load and split data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    num_samples = len(X_train) // NUM_CLIENTS

    clients = []
    for i in range(NUM_CLIENTS):
        start = i * num_samples
        end = start + num_samples if i < NUM_CLIENTS - 1 else len(X_train)
        client = Client(
            client_id=i,
            X_train=X_train[start:end],
            y_train=y_train[start:end],
            is_malicious=(i in MALICIOUS_CLIENT_IDS),
            attack_scale=ATTACK_SCALE
        )
        clients.append(client)

    # 2. Initialize global model
    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    global_state = global_model.state_dict()

    # 3. Each client trains locally
    client_states = []
    client_counts = []
    for client in clients:
        state, count = client.train(global_model)
        client_states.append(state)
        client_counts.append(count)

    # 4. Compute updates (client_params - global_params)
    updates = [compute_update(s, global_state) for s in client_states]

    # 5. Detect malicious clients
    if DETECTION_ENABLED:
        suspicious, similarities = detect_malicious_clients_simple(
            updates, threshold=SIMILARITY_THRESHOLD
        )

        # Print per-client result
        for i, sim in enumerate(similarities):
            status = "SUSPICIOUS" if i in suspicious else "ACCEPTED"
            print(f"Client {i} -> similarity: {sim:.2f} -> {status}")

        print(f"\nDetected malicious clients: {suspicious}")
        accepted = [i for i in range(NUM_CLIENTS) if i not in suspicious]
        print(f"Accepted clients: {accepted}")

        # Safety fallback: if all clients are suspicious, use all of them
        if len(accepted) == 0:
            print("All clients suspicious! Fallback: using all updates.")
            accepted = list(range(NUM_CLIENTS))

        print("Performing FedAvg using accepted clients...")

        # 6. Filter updates and perform FedAvg
        filtered_states = [client_states[i] for i in accepted]
        filtered_counts = [client_counts[i] for i in accepted]
        updates_for_fedavg = list(zip(filtered_states, filtered_counts))

        new_state = fedavg(updates_for_fedavg)
        global_model.load_state_dict(new_state)
        print("Global model updated.")

    else:
        print("Detection disabled. Using all updates.")
        new_state = fedavg(list(zip(client_states, client_counts)))
        global_model.load_state_dict(new_state)
        print("Global model updated.")

    # 7. Final summary
    print("\n--- Phase 4 Summary ---")
    print(f"Total clients: {NUM_CLIENTS}")
    print(f"Malicious clients (ground truth): {len(MALICIOUS_CLIENT_IDS)}")
    print(f"Detected suspicious clients: {len(suspicious) if DETECTION_ENABLED else 0}")
    print(f"Accepted clients: {len(accepted) if DETECTION_ENABLED else NUM_CLIENTS}")
    print("Detection completed successfully.")


if __name__ == "__main__":
    main()
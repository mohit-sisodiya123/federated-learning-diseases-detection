# experiments/run_phase3.py
"""
Phase 3 experiment: Malicious-client detection using cosine similarity.

Runs three scenarios with the SAME data split and SAME seed:
  A) Normal FL (no malicious clients, no detection)
  B) FL with a malicious client (no detection) -> shows the attack
  C) FL with a malicious client + detection     -> shows the defense

Compares accuracy, loss, and detection metrics.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

from utils.config import (
    SEED, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    MALICIOUS_CLIENT_IDS, ATTACK_SCALE, DETECTION_SENSITIVITY, RESULTS_DIR
)
from utils.seed import set_seed
from data.dataset import load_and_preprocess
from models.classifier import SimpleClassifier
from clients.client import Client
from server.fedavg import fedavg
from security.detector import (
    compute_update, detect_suspicious_clients, flatten_state_dict
)
from utils.metrics import evaluate_model, compute_loss


def split_data_among_clients(X_train, y_train, seed: int):
    """
    Split the training data into NUM_CLIENTS random, non-overlapping shards.
    Using a fixed seed ensures every experiment uses the SAME split.
    """
    set_seed(seed)
    num_samples_per_client = len(X_train) // NUM_CLIENTS
    perm = torch.randperm(len(X_train))
    shards = []
    for i in range(NUM_CLIENTS):
        start = i * num_samples_per_client
        end = start + num_samples_per_client if i < NUM_CLIENTS - 1 else len(X_train)
        idx = perm[start:end]
        shards.append((X_train[idx], y_train[idx]))
    return shards


def run_experiment(X_train, y_train, X_test, y_test,
                   malicious_ids, attack_scale, use_detection):
    """
    Run one full federated-learning simulation.

    Args:
        malicious_ids : list of client IDs that will poison their updates
        attack_scale  : how strongly the malicious clients reverse/scale
        use_detection : if True, run the detector before FedAvg

    Returns:
        final_model     : trained global model
        history         : {'accuracy': [...], 'loss': [...]}
        detection_log   : list of dicts (one per round) with detection info
    """
    set_seed(SEED)

    # ---- Split data (same across all experiments) ----
    shards = split_data_among_clients(X_train, y_train, SEED)

    # ---- Create clients ----
    clients = []
    for i in range(NUM_CLIENTS):
        Xi, yi = shards[i]
        clients.append(Client(
            client_id=i,
            X_train=Xi,
            y_train=yi,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            is_malicious=(i in malicious_ids),
            attack_scale=attack_scale,
        ))

    # ---- Global model ----
    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    history = {"accuracy": [], "loss": []}
    detection_log = []  # one entry per round

    # ---- Federated rounds ----
    for round_num in range(1, NUM_ROUNDS + 1):
        # 1. Each client trains locally and returns its parameters
        client_states = []
        sample_counts = []
        for client in clients:
            state, count = client.train(global_model)
            client_states.append(state)
            sample_counts.append(count)

        # 2. Optionally run detection
        if use_detection:
            # Compute the per-client update vectors (client - global)
            global_state = global_model.state_dict()
            updates = [compute_update(s, global_state) for s in client_states]

            suspicious, avg_sims, sim_matrix, threshold = detect_suspicious_clients(
                updates, sensitivity=DETECTION_SENSITIVITY
            )
            print(f"    Detected suspicious clients: {suspicious}  "
                  f"(threshold={threshold:.4f})")

            detection_log.append({
                "round": round_num,
                "suspicious": suspicious,
                "avg_similarities": avg_sims,
                "sim_matrix": sim_matrix,
                "threshold": threshold,
            })

            # Keep only non-suspicious updates
            filtered_states = []
            filtered_counts = []
            for i, (s, c) in enumerate(zip(client_states, sample_counts)):
                if i not in suspicious:
                    filtered_states.append(s)
                    filtered_counts.append(c)

            # Safety fallback: if EVERY client is flagged, use all of them.
            if len(filtered_states) == 0:
                print("    All clients flagged -> falling back to using all updates.")
                filtered_states = client_states
                filtered_counts = sample_counts

            updates_for_fedavg = list(zip(filtered_states, filtered_counts))
        else:
            updates_for_fedavg = list(zip(client_states, sample_counts))

        # 3. FedAvg over (possibly filtered) updates
        new_state = fedavg(updates_for_fedavg)
        global_model.load_state_dict(new_state)

        # 4. Evaluate global model
        acc = evaluate_model(global_model, X_test, y_test)["accuracy"]
        loss = compute_loss(global_model, X_test, y_test)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"    Round {round_num}: accuracy={acc:.4f}, loss={loss:.4f}")

    return global_model, history, detection_log


def compute_detection_metrics(ground_truth_ids, detected_ids):
    """
    Compare the detector's output with the ground truth.
    Returns a dict with TP, FP, FN, TN and Precision/Recall/F1.
    """
    ground_truth = [1 if i in ground_truth_ids else 0 for i in range(NUM_CLIENTS)]
    detected = [1 if i in detected_ids else 0 for i in range(NUM_CLIENTS)]

    tp = sum(1 for g, d in zip(ground_truth, detected) if g == 1 and d == 1)
    fp = sum(1 for g, d in zip(ground_truth, detected) if g == 0 and d == 1)
    fn = sum(1 for g, d in zip(ground_truth, detected) if g == 1 and d == 0)
    tn = sum(1 for g, d in zip(ground_truth, detected) if g == 0 and d == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def plot_accuracy_comparison(hist_normal, hist_attack, hist_detected, path):
    plt.figure(figsize=(8, 5))
    rounds = range(1, NUM_ROUNDS + 1)
    plt.plot(rounds, hist_normal["accuracy"], marker='o', label="Normal FL")
    plt.plot(rounds, hist_attack["accuracy"], marker='s', label="FL + Attack (no detection)")
    plt.plot(rounds, hist_detected["accuracy"], marker='^', label="FL + Attack + Detection")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Round: Normal vs Attack vs Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_loss_comparison(hist_normal, hist_attack, hist_detected, path):
    plt.figure(figsize=(8, 5))
    rounds = range(1, NUM_ROUNDS + 1)
    plt.plot(rounds, hist_normal["loss"], marker='o', label="Normal FL")
    plt.plot(rounds, hist_attack["loss"], marker='s', label="FL + Attack (no detection)")
    plt.plot(rounds, hist_detected["loss"], marker='^', label="FL + Attack + Detection")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Loss per Round: Normal vs Attack vs Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_detection_scores(avg_similarities, suspicious_ids, path):
    plt.figure(figsize=(8, 5))
    client_ids = [f"Client {i}" for i in range(len(avg_similarities))]
    bars = plt.bar(client_ids, avg_similarities)
    # Mark suspicious bars with a hatch pattern (still uses default colors)
    for i, bar in enumerate(bars):
        if i in suspicious_ids:
            bar.set_hatch("//")
            bar.set_edgecolor("black")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Average Cosine Similarity per Client (last round)")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_similarity_matrix(sim_matrix, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(NUM_CLIENTS), [f"C{i}" for i in range(NUM_CLIENTS)])
    plt.yticks(range(NUM_CLIENTS), [f"C{i}" for i in range(NUM_CLIENTS)])
    plt.title("Client Update Cosine Similarity Matrix")
    # write values inside cells
    for i in range(NUM_CLIENTS):
        for j in range(NUM_CLIENTS):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if sim_matrix[i, j] < 0.5 else "black",
                     fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_detection_confusion_matrix(tp, fp, fn, tn, path):
    # Rows = actual, Columns = predicted
    # Order: [Normal, Malicious]
    cm = np.array([[tn, fp],
                   [fn, tp]])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted Normal", "Predicted Malicious"])
    plt.yticks([0, 1], ["Actual Normal", "Actual Malicious"])
    plt.title("Detection Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    print("=" * 70)
    print("Phase 3: Malicious-client detection using cosine similarity")
    print("=" * 70)
    print(f"Ground truth malicious clients: {MALICIOUS_CLIENT_IDS}")
    print(f"Attack scale: {ATTACK_SCALE}")
    print(f"Detection sensitivity: {DETECTION_SENSITIVITY}")

    set_seed(SEED)

    # Load dataset ONCE (same for all experiments)
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- Experiment A: Normal FL ----------
    print("\n--- Experiment A: Normal FL (no malicious clients) ---")
    model_a, hist_a, _ = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=[], attack_scale=0.0, use_detection=False
    )
    metrics_a = evaluate_model(model_a, X_test, y_test)
    loss_a = compute_loss(model_a, X_test, y_test)

    # ---------- Experiment B: Malicious, NO detection ----------
    print("\n--- Experiment B: FL + Malicious, NO detection ---")
    model_b, hist_b, _ = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE,
        use_detection=False
    )
    metrics_b = evaluate_model(model_b, X_test, y_test)
    loss_b = compute_loss(model_b, X_test, y_test)

    # ---------- Experiment C: Malicious + Detection ----------
    print("\n--- Experiment C: FL + Malicious + Detection ---")
    model_c, hist_c, det_log = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE,
        use_detection=True
    )
    metrics_c = evaluate_model(model_c, X_test, y_test)
    loss_c = compute_loss(model_c, X_test, y_test)

    # Use the last round's detection output for metrics
    last_det = det_log[-1] if det_log else {"suspicious": [], "avg_similarities": np.zeros(NUM_CLIENTS),
                                            "sim_matrix": np.zeros((NUM_CLIENTS, NUM_CLIENTS))}
    detected_ids = last_det["suspicious"]
    det_metrics = compute_detection_metrics(MALICIOUS_CLIENT_IDS, detected_ids)

    # ---------- Print Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, m, l in [
        ("Normal FL", metrics_a, loss_a),
        ("FL + Attack (no detection)", metrics_b, loss_b),
        ("FL + Attack + Detection", metrics_c, loss_c),
    ]:
        print(f"{name}")
        print(f"   Accuracy : {m['accuracy']:.4f}")
        print(f"   Precision: {m['precision']:.4f}")
        print(f"   Recall   : {m['recall']:.4f}")
        print(f"   F1-score : {m['f1']:.4f}")
        print(f"   Loss     : {l:.4f}")

    print("\nDetection results (last round):")
    print(f"   Ground truth malicious: {MALICIOUS_CLIENT_IDS}")
    print(f"   Detected as suspicious: {detected_ids}")
    print(f"   True Positives : {det_metrics['tp']}")
    print(f"   False Positives: {det_metrics['fp']}")
    print(f"   False Negatives: {det_metrics['fn']}")
    print(f"   True Negatives : {det_metrics['tn']}")
    print(f"   Detection Precision: {det_metrics['precision']:.4f}")
    print(f"   Detection Recall   : {det_metrics['recall']:.4f}")
    print(f"   Detection F1       : {det_metrics['f1']:.4f}")

    # ---------- Plots ----------
    plot_accuracy_comparison(
        hist_a, hist_b, hist_c,
        os.path.join(RESULTS_DIR, "phase3_accuracy_comparison.png")
    )
    plot_loss_comparison(
        hist_a, hist_b, hist_c,
        os.path.join(RESULTS_DIR, "phase3_loss_comparison.png")
    )
    plot_detection_scores(
        last_det["avg_similarities"], detected_ids,
        os.path.join(RESULTS_DIR, "phase3_detection_scores.png")
    )
    plot_similarity_matrix(
        last_det["sim_matrix"],
        os.path.join(RESULTS_DIR, "phase3_similarity_matrix.png")
    )
    plot_detection_confusion_matrix(
        det_metrics["tp"], det_metrics["fp"],
        det_metrics["fn"], det_metrics["tn"],
        os.path.join(RESULTS_DIR, "phase3_detection_confusion_matrix.png")
    )

    # ---------- Save results text file ----------
    results_path = os.path.join(RESULTS_DIR, "phase3_results.txt")
    with open(results_path, "w") as f:
        f.write("PHASE 3 RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Ground-truth malicious clients: {MALICIOUS_CLIENT_IDS}\n")
        f.write(f"Attack scale: {ATTACK_SCALE}\n")
        f.write(f"Detection sensitivity: {DETECTION_SENSITIVITY}\n\n")

        f.write("Normal FL:\n")
        f.write(f"  Accuracy : {metrics_a['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics_a['precision']:.4f}\n")
        f.write(f"  Recall   : {metrics_a['recall']:.4f}\n")
        f.write(f"  F1-score : {metrics_a['f1']:.4f}\n")
        f.write(f"  Loss     : {loss_a:.4f}\n\n")

        f.write("FL + Malicious Clients (no detection):\n")
        f.write(f"  Accuracy : {metrics_b['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics_b['precision']:.4f}\n")
        f.write(f"  Recall   : {metrics_b['recall']:.4f}\n")
        f.write(f"  F1-score : {metrics_b['f1']:.4f}\n")
        f.write(f"  Loss     : {loss_b:.4f}\n\n")

        f.write("FL + Malicious Clients + Detection:\n")
        f.write(f"  Accuracy : {metrics_c['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics_c['precision']:.4f}\n")
        f.write(f"  Recall   : {metrics_c['recall']:.4f}\n")
        f.write(f"  F1-score : {metrics_c['f1']:.4f}\n")
        f.write(f"  Loss     : {loss_c:.4f}\n\n")

        f.write("Detection performance (last round):\n")
        f.write(f"  Detected suspicious clients: {detected_ids}\n")
        f.write(f"  True Positives : {det_metrics['tp']}\n")
        f.write(f"  False Positives: {det_metrics['fp']}\n")
        f.write(f"  False Negatives: {det_metrics['fn']}\n")
        f.write(f"  True Negatives : {det_metrics['tn']}\n")
        f.write(f"  Detection Precision: {det_metrics['precision']:.4f}\n")
        f.write(f"  Detection Recall   : {det_metrics['recall']:.4f}\n")
        f.write(f"  Detection F1       : {det_metrics['f1']:.4f}\n")

    print(f"\nResults saved in {RESULTS_DIR}/")
    print("Phase 3 completed successfully.")


if __name__ == "__main__":
    main()
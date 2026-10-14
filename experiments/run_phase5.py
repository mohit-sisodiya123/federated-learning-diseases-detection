# experiments/run_phase5.py
"""
PHASE 5 — Experiments, Evaluation and Comparison.

Runs three experiments with the same data, model and seed:
  A) Normal FL (no malicious clients)
  B) FL + Malicious clients (no detection)
  C) FL + Malicious clients + Detection

Saves a CSV, several graphs, and classification confusion matrices,
then prints a simple automatic summary.
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.config import (
    SEED, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    MALICIOUS_CLIENT_IDS, ATTACK_SCALE,
    SIMILARITY_THRESHOLD, RESULTS_DIR
)
from utils.seed import set_seed
from data.dataset import load_and_preprocess
from models.classifier import SimpleClassifier
from clients.client import Client
from server.fedavg import fedavg
from security.detector import (
    compute_update, detect_malicious_clients_simple
)
from utils.metrics import evaluate_model, compute_loss


# Where graphs will be saved
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


# ---------------------------------------------------------------------
# Data splitting (same as Phase 3 — same seed = identical split)
# ---------------------------------------------------------------------
def split_data_among_clients(X_train, y_train, seed: int):
    """Randomly split training data into NUM_CLIENTS shards."""
    set_seed(seed)
    n_per = len(X_train) // NUM_CLIENTS
    perm = torch.randperm(len(X_train))
    shards = []
    for i in range(NUM_CLIENTS):
        start = i * n_per
        end = start + n_per if i < NUM_CLIENTS - 1 else len(X_train)
        idx = perm[start:end]
        shards.append((X_train[idx], y_train[idx]))
    return shards


# ---------------------------------------------------------------------
# One complete FL simulation
# ---------------------------------------------------------------------
def run_one_experiment(
    X_train, y_train, X_test, y_test,
    malicious_ids, attack_scale, use_detection
):
    """
    Run one full federated learning simulation.

    Returns:
        final_model     : trained global model
        history         : {"accuracy": [...], "loss": [...]}
        detection_history : list of dicts (only when use_detection=True)
    """
    set_seed(SEED)
    shards = split_data_among_clients(X_train, y_train, SEED)

    # Create clients
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
            attack_scale=attack_scale
        ))

    # Fresh global model for every experiment
    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    history = {"accuracy": [], "loss": []}
    detection_history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        # --- 1. Local training ---
        client_states = []
        counts = []
        for c in clients:
            state, n = c.train(global_model)
            client_states.append(state)
            counts.append(n)

        # --- 2. Optional detection before FedAvg ---
        if use_detection:
            global_state = global_model.state_dict()
            updates = [compute_update(s, global_state) for s in client_states]

            suspicious, sims = detect_malicious_clients_simple(
                updates, threshold=SIMILARITY_THRESHOLD
            )
            print(f"  Round {rnd}: suspicious={suspicious}, "
                  f"similarities={[round(float(s), 3) for s in sims]}")

            accepted = [i for i in range(NUM_CLIENTS) if i not in suspicious]
            # Safety fallback
            if len(accepted) == 0:
                print("    All clients flagged -> fallback: use all.")
                accepted = list(range(NUM_CLIENTS))

            updates_for_fedavg = list(zip(
                [client_states[i] for i in accepted],
                [counts[i] for i in accepted]
            ))
            detection_history.append({
                "round": rnd,
                "suspicious": suspicious,
                "similarities": sims,
                "accepted": accepted
            })
        else:
            updates_for_fedavg = list(zip(client_states, counts))

        # --- 3. FedAvg ---
        new_state = fedavg(updates_for_fedavg)
        global_model.load_state_dict(new_state)

        # --- 4. Evaluate ---
        acc = evaluate_model(global_model, X_test, y_test)["accuracy"]
        loss = compute_loss(global_model, X_test, y_test)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"    accuracy={acc:.4f}, loss={loss:.4f}")

    return global_model, history, detection_history


# ---------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------
def compute_detection_metrics(ground_truth_ids, detected_ids):
    """Compare detector output with ground truth."""
    gt = set(ground_truth_ids)
    det = set(detected_ids)
    tp = len(gt & det)       # malicious AND detected
    fp = len(det - gt)       # normal   AND detected
    fn = len(gt - det)       # malicious AND missed
    tn = NUM_CLIENTS - tp - fp - fn  # normal AND accepted

    detection_rate = tp / len(gt) if len(gt) > 0 else 0.0
    legitimate = NUM_CLIENTS - len(gt)
    fpr = fp / legitimate if legitimate > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "detection_rate": detection_rate, "fpr": fpr}


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_accuracy_comparison(hist_a, hist_b, hist_c, path):
    rounds = range(1, NUM_ROUNDS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, hist_a["accuracy"], marker='o', label="Normal FL")
    plt.plot(rounds, hist_b["accuracy"], marker='s', label="FL + Malicious")
    plt.plot(rounds, hist_c["accuracy"], marker='^',
             label="FL + Malicious + Detection")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Communication Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_f1_comparison(results, path):
    names = [r["name"] for r in results]
    f1s = [r["f1"] for r in results]
    plt.figure(figsize=(7, 5))
    plt.bar(names, f1s)
    plt.ylabel("F1-score")
    plt.title("F1-score Comparison Across Experiments")
    plt.ylim(0, 1)
    for i, v in enumerate(f1s):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_detection_summary(det_metrics, path):
    labels = ["Total Malicious", "Detected", "Missed", "False Positives"]
    total_mal = det_metrics["tp"] + det_metrics["fn"]
    values = [total_mal, det_metrics["tp"],
              det_metrics["fn"], det_metrics["fp"]]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Number of Clients")
    plt.title("Malicious Client Detection Summary (last round)")
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 5: Experiments, Evaluation and Comparison")
    print("=" * 70)
    set_seed(SEED)

    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Config: clients={NUM_CLIENTS}, rounds={NUM_ROUNDS}, "
          f"malicious={MALICIOUS_CLIENT_IDS}, attack_scale={ATTACK_SCALE}, "
          f"threshold={SIMILARITY_THRESHOLD}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = []

    # ---------- Experiment A: Normal FL ----------
    print("\n>>> Experiment A: Normal Federated Learning")
    model_a, hist_a, _ = run_one_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=[], attack_scale=0.0, use_detection=False
    )
    m_a = evaluate_model(model_a, X_test, y_test)
    l_a = compute_loss(model_a, X_test, y_test)
    all_results.append({
        "name": "Normal FL",
        "accuracy": m_a["accuracy"],
        "precision": m_a["precision"],
        "recall": m_a["recall"],
        "f1": m_a["f1"],
        "loss": l_a,
        "confusion_matrix": m_a["confusion_matrix"],
        "malicious": 0, "detected": 0, "missed": 0, "fp": 0,
        "detection_rate": 0.0, "fpr": 0.0
    })

    # ---------- Experiment B: FL + Malicious (no detection) ----------
    print("\n>>> Experiment B: FL + Malicious (No Detection)")
    model_b, hist_b, _ = run_one_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE,
        use_detection=False
    )
    m_b = evaluate_model(model_b, X_test, y_test)
    l_b = compute_loss(model_b, X_test, y_test)
    all_results.append({
        "name": "FL + Malicious",
        "accuracy": m_b["accuracy"],
        "precision": m_b["precision"],
        "recall": m_b["recall"],
        "f1": m_b["f1"],
        "loss": l_b,
        "confusion_matrix": m_b["confusion_matrix"],
        "malicious": len(MALICIOUS_CLIENT_IDS),
        "detected": 0,
        "missed": len(MALICIOUS_CLIENT_IDS),
        "fp": 0,
        "detection_rate": 0.0, "fpr": 0.0
    })

    # ---------- Experiment C: FL + Malicious + Detection ----------
    print("\n>>> Experiment C: FL + Malicious + Detection")
    model_c, hist_c, det_hist = run_one_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE,
        use_detection=True
    )
    m_c = evaluate_model(model_c, X_test, y_test)
    l_c = compute_loss(model_c, X_test, y_test)

    last_det = det_hist[-1] if det_hist else {"suspicious": []}
    det_metrics = compute_detection_metrics(
        MALICIOUS_CLIENT_IDS, last_det["suspicious"]
    )

    all_results.append({
        "name": "FL + Detection",
        "accuracy": m_c["accuracy"],
        "precision": m_c["precision"],
        "recall": m_c["recall"],
        "f1": m_c["f1"],
        "loss": l_c,
        "confusion_matrix": m_c["confusion_matrix"],
        "malicious": len(MALICIOUS_CLIENT_IDS),
        "detected": det_metrics["tp"],
        "missed": det_metrics["fn"],
        "fp": det_metrics["fp"],
        "detection_rate": det_metrics["detection_rate"],
        "fpr": det_metrics["fpr"]
    })

    # ---------- Save CSV ----------
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Accuracy", "Precision", "Recall",
                         "F1_Score", "Loss", "Malicious_Clients",
                         "Detected_Malicious", "Missed_Malicious",
                         "False_Positives", "Detection_Rate", "FPR"])
        for r in all_results:
            writer.writerow([
                r["name"],
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['loss']:.4f}",
                r["malicious"],
                r["detected"],
                r["missed"],
                r["fp"],
                f"{r['detection_rate']:.4f}",
                f"{r['fpr']:.4f}"
            ])
    print(f"\nCSV saved: {csv_path}")

    # ---------- Plots ----------
    plot_accuracy_comparison(
        hist_a, hist_b, hist_c,
        os.path.join(PLOTS_DIR, "accuracy_vs_rounds.png")
    )
    plot_f1_comparison(
        all_results,
        os.path.join(PLOTS_DIR, "f1_comparison.png")
    )
    plot_detection_summary(
        det_metrics,
        os.path.join(PLOTS_DIR, "detection_summary.png")
    )

    plot_confusion_matrix(
        m_a["confusion_matrix"], "Normal Federated Learning",
        os.path.join(PLOTS_DIR, "cm_normal_fl.png")
    )
    plot_confusion_matrix(
        m_b["confusion_matrix"], "FL with Malicious Clients",
        os.path.join(PLOTS_DIR, "cm_malicious.png")
    )
    plot_confusion_matrix(
        m_c["confusion_matrix"], "FL with Detection",
        os.path.join(PLOTS_DIR, "cm_detection.png")
    )
    print(f"Plots saved in {PLOTS_DIR}/")

    # ---------- Final table ----------
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Experiment':<22}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for r in all_results:
        print(f"{r['name']:<22}{r['accuracy']:>8.4f}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}")

    # ---------- Auto summary ----------
    print("\n" + "=" * 70)
    print("AUTOMATIC SUMMARY (based on actual results)")
    print("=" * 70)
    print(f"Normal FL:  accuracy = {all_results[0]['accuracy']:.4f}, "
          f"F1 = {all_results[0]['f1']:.4f}.")
    print(f"FL + Malicious (no detection): accuracy = "
          f"{all_results[1]['accuracy']:.4f}, F1 = {all_results[1]['f1']:.4f}.")
    print(f"FL + Detection: accuracy = {all_results[2]['accuracy']:.4f}, "
          f"F1 = {all_results[2]['f1']:.4f}.")
    print(f"Detector caught {det_metrics['tp']} / "
          f"{len(MALICIOUS_CLIENT_IDS)} malicious client(s) "
          f"(detection rate = {det_metrics['detection_rate']:.2f}).")
    print(f"False positives = {det_metrics['fp']} "
          f"(FPR = {det_metrics['fpr']:.2f}).")

    delta_f1 = all_results[2]["f1"] - all_results[1]["f1"]
    if delta_f1 > 0.01:
        print(f"Detection improved F1 by {delta_f1:.4f} vs. no detection.")
    elif delta_f1 < -0.01:
        print(f"Detection reduced F1 by {-delta_f1:.4f} vs. no detection.")
    else:
        print("Detection produced no significant F1 change.")

    print("\nPhase 5 completed successfully.")


if __name__ == "__main__":
    main()
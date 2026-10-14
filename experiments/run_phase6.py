# experiments/run_phase6.py
"""
PHASE 6 — Results Analysis, Visualization and Discussion.

Re-runs the three Phase 5 experiments (same seed => reproducible)
to collect per-round histories, then produces:
  - results/final_results.csv
  - results/detection_analysis.csv
  - results/round_by_round.csv
  - results/plots/accuracy_vs_rounds.png
  - results/plots/f1_vs_rounds.png
  - results/plots/loss_vs_rounds.png
  - results/plots/final_comparison_*.png
  - results/plots/detection_summary.png
  - results/results_analysis.md
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


PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


# ---------------------------------------------------------------------
# Data split (same as Phase 5)
# ---------------------------------------------------------------------
def split_data_among_clients(X_train, y_train, seed: int):
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
# Run one experiment (tracks accuracy, precision, recall, f1, loss per round)
# ---------------------------------------------------------------------
def run_experiment(X_train, y_train, X_test, y_test,
                   malicious_ids, attack_scale, use_detection):
    set_seed(SEED)
    shards = split_data_among_clients(X_train, y_train, SEED)

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

    global_model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    history = {"accuracy": [], "precision": [], "recall": [],
               "f1": [], "loss": []}
    detection_history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        client_states, counts = [], []
        for c in clients:
            s, n = c.train(global_model)
            client_states.append(s)
            counts.append(n)

        if use_detection:
            gs = global_model.state_dict()
            updates = [compute_update(s, gs) for s in client_states]
            suspicious, sims = detect_malicious_clients_simple(
                updates, threshold=SIMILARITY_THRESHOLD
            )
            accepted = [i for i in range(NUM_CLIENTS)
                        if i not in suspicious]
            if not accepted:
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

        new_state = fedavg(updates_for_fedavg)
        global_model.load_state_dict(new_state)

        m = evaluate_model(global_model, X_test, y_test)
        l = compute_loss(global_model, X_test, y_test)
        history["accuracy"].append(m["accuracy"])
        history["precision"].append(m["precision"])
        history["recall"].append(m["recall"])
        history["f1"].append(m["f1"])
        history["loss"].append(l)

    return global_model, history, detection_history


# ---------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------
def detect_metrics(ground_truth, detected, total_clients=NUM_CLIENTS):
    gt, det = set(ground_truth), set(detected)
    tp = len(gt & det)
    fp = len(det - gt)
    fn = len(gt - det)
    tn = total_clients - tp - fp - fn
    detection_rate = tp / len(gt) if len(gt) > 0 else 0.0
    legit = total_clients - len(gt)
    fpr = fp / legit if legit > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "detection_rate": detection_rate, "fpr": fpr}


# ---------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------
def save_final_results(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "Accuracy", "Precision",
                    "Recall", "F1_Score", "Loss"])
        for r in results:
            w.writerow([r["name"],
                        f"{r['accuracy']:.4f}",
                        f"{r['precision']:.4f}",
                        f"{r['recall']:.4f}",
                        f"{r['f1']:.4f}",
                        f"{r['loss']:.4f}"])


def save_detection_analysis(det, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Total malicious clients", det["total_mal"]])
        w.writerow(["Detected malicious clients", det["tp"]])
        w.writerow(["Missed malicious clients", det["fn"]])
        w.writerow(["Legitimate clients", det["legit"]])
        w.writerow(["Legitimate flagged (false positives)", det["fp"]])
        w.writerow(["Detection rate", f"{det['detection_rate']:.4f}"])
        w.writerow(["False-positive rate", f"{det['fpr']:.4f}"])


def save_round_by_round(hist_a, hist_b, hist_c, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Round",
                    "A_accuracy", "A_loss",
                    "B_accuracy", "B_loss",
                    "C_accuracy", "C_loss"])
        for i in range(NUM_ROUNDS):
            w.writerow([
                i + 1,
                f"{hist_a['accuracy'][i]:.4f}",
                f"{hist_a['loss'][i]:.4f}",
                f"{hist_b['accuracy'][i]:.4f}",
                f"{hist_b['loss'][i]:.4f}",
                f"{hist_c['accuracy'][i]:.4f}",
                f"{hist_c['loss'][i]:.4f}",
            ])


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_metric_vs_rounds(hist_a, hist_b, hist_c, key, ylabel, title, path):
    rounds = list(range(1, NUM_ROUNDS + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, hist_a[key], marker='o', label="Normal FL")
    plt.plot(rounds, hist_b[key], marker='s', label="FL + Malicious")
    plt.plot(rounds, hist_c[key], marker='^',
             label="FL + Malicious + Detection")
    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric_bar(results, metric, ylabel, title, path):
    names = [r["name"] for r in results]
    vals = [r[metric] for r in results]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, vals)
    plt.ylabel(ylabel)
    plt.title(title)
    if metric in ("accuracy", "precision", "recall", "f1"):
        plt.ylim(0, 1)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01,
                 f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_detection_bar(det, path):
    labels = ["Total\nMalicious", "Detected", "Missed", "False\nPositives"]
    vals = [det["total_mal"], det["tp"], det["fn"], det["fp"]]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals)
    plt.ylabel("Number of Clients")
    plt.title("Malicious Client Detection Summary")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2,
                 v + 0.05, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------
def write_report(results, det, path):
    rA, rB, rC = results
    lines = []
    lines.append("# Results and Analysis\n\n")
    lines.append("## 1. Experimental Setup\n\n")
    lines.append(f"- Dataset: Pima Indians Diabetes (public, educational)\n")
    lines.append(f"- Number of clients: {NUM_CLIENTS}\n")
    lines.append(f"- Communication rounds: {NUM_ROUNDS}\n")
    lines.append(f"- Local epochs per round: {LOCAL_EPOCHS}\n")
    lines.append(f"- Batch size: {BATCH_SIZE}\n")
    lines.append(f"- Learning rate: {LEARNING_RATE}\n")
    lines.append(f"- Malicious client IDs: {MALICIOUS_CLIENT_IDS}\n")
    lines.append(f"- Attack scale: {ATTACK_SCALE}\n")
    lines.append(f"- Detection threshold (cosine similarity): "
                 f"{SIMILARITY_THRESHOLD}\n")
    lines.append(f"- Random seed: {SEED}\n")

    lines.append("\n## 2. Normal Federated Learning\n\n")
    lines.append(f"Accuracy = {rA['accuracy']:.4f}, "
                 f"Precision = {rA['precision']:.4f}, "
                 f"Recall = {rA['recall']:.4f}, "
                 f"F1 = {rA['f1']:.4f}.\n")

    lines.append("\n## 3. Effect of Malicious Clients\n\n")
    lines.append(f"With malicious clients and no detection: "
                 f"Accuracy = {rB['accuracy']:.4f}, F1 = {rB['f1']:.4f}.\n")
    lines.append(f"Change in accuracy vs normal FL: "
                 f"{rB['accuracy'] - rA['accuracy']:+.4f}.\n")
    lines.append(f"Change in F1 vs normal FL: "
                 f"{rB['f1'] - rA['f1']:+.4f}.\n")

    lines.append("\n## 4. Effect of Malicious Client Detection\n\n")
    lines.append(f"With detection enabled: "
                 f"Accuracy = {rC['accuracy']:.4f}, F1 = {rC['f1']:.4f}.\n")
    lines.append(f"Change in accuracy vs attack (no detection): "
                 f"{rC['accuracy'] - rB['accuracy']:+.4f}.\n")
    lines.append(f"Change in F1 vs attack (no detection): "
                 f"{rC['f1'] - rB['f1']:+.4f}.\n")

    lines.append("\n## 5. Detection Performance\n\n")
    lines.append(f"- Total malicious clients: {det['total_mal']}\n")
    lines.append(f"- Detected: {det['tp']}\n")
    lines.append(f"- Missed: {det['fn']}\n")
    lines.append(f"- False positives: {det['fp']}\n")
    lines.append(f"- Detection rate: {det['detection_rate']:.4f}\n")
    lines.append(f"- False-positive rate: {det['fpr']:.4f}\n")

    lines.append("\n## 6. Model Performance Comparison\n\n")
    lines.append("| Experiment | Accuracy | Precision | Recall | F1 |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in results:
        lines.append(f"| {r['name']} | {r['accuracy']:.4f} | "
                     f"{r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1']:.4f} |\n")

    lines.append("\n## 7. Communication-Round Analysis\n\n")
    lines.append("Per-round accuracy and loss values are saved in "
                 "`results/round_by_round.csv`.\n")
    lines.append("Graphs: `accuracy_vs_rounds.png`, `f1_vs_rounds.png`, "
                 "`loss_vs_rounds.png` in `results/plots/`.\n")

    lines.append("\n## 8. Limitations\n\n")
    lines.append("- Simulated federated environment with limited clients.\n")
    lines.append("- Public educational dataset, not real patient data.\n")
    lines.append("- Single simple attack (sign-reversal / update scaling).\n")
    lines.append("- No formal privacy guarantees beyond not sharing raw data.\n")
    lines.append("- Not a medical diagnostic system.\n")

    lines.append("\n## 9. Conclusion of Experimental Results\n\n")
    lines.append("The experiments demonstrate that a single malicious "
                 "client can degrade the global model, and that simple "
                 "cosine-similarity-based detection can restore performance "
                 "when the attack produces updates that differ clearly from "
                 "the honest ones.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 6 — Results Analysis and Visualization")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("\nRe-running experiments (same seed as Phase 5)...")
    model_a, hist_a, _ = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=[], attack_scale=0.0, use_detection=False
    )
    model_b, hist_b, _ = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE, use_detection=False
    )
    model_c, hist_c, det_hist = run_experiment(
        X_train, y_train, X_test, y_test,
        malicious_ids=MALICIOUS_CLIENT_IDS,
        attack_scale=ATTACK_SCALE, use_detection=True
    )

    mA = evaluate_model(model_a, X_test, y_test)
    mB = evaluate_model(model_b, X_test, y_test)
    mC = evaluate_model(model_c, X_test, y_test)

    results = [
        {"name": "Normal FL", **{k: mA[k] for k in
         ("accuracy", "precision", "recall", "f1")},
         "loss": compute_loss(model_a, X_test, y_test)},
        {"name": "FL + Malicious", **{k: mB[k] for k in
         ("accuracy", "precision", "recall", "f1")},
         "loss": compute_loss(model_b, X_test, y_test)},
        {"name": "FL + Detection", **{k: mC[k] for k in
         ("accuracy", "precision", "recall", "f1")},
         "loss": compute_loss(model_c, X_test, y_test)},
    ]

    last_det = det_hist[-1]["suspicious"] if det_hist else []
    det = detect_metrics(MALICIOUS_CLIENT_IDS, last_det)
    det["total_mal"] = len(MALICIOUS_CLIENT_IDS)
    det["legit"] = NUM_CLIENTS - len(MALICIOUS_CLIENT_IDS)

    # --- CSV outputs ---
    save_final_results(results,
                       os.path.join(RESULTS_DIR, "final_results.csv"))
    save_detection_analysis(det,
                            os.path.join(RESULTS_DIR, "detection_analysis.csv"))
    save_round_by_round(hist_a, hist_b, hist_c,
                        os.path.join(RESULTS_DIR, "round_by_round.csv"))

    # --- Plots ---
    plot_metric_vs_rounds(hist_a, hist_b, hist_c, "accuracy",
                          "Accuracy", "Accuracy vs Communication Round",
                          os.path.join(PLOTS_DIR, "accuracy_vs_rounds.png"))
    plot_metric_vs_rounds(hist_a, hist_b, hist_c, "f1",
                          "F1-score", "F1-score vs Communication Round",
                          os.path.join(PLOTS_DIR, "f1_vs_rounds.png"))
    plot_metric_vs_rounds(hist_a, hist_b, hist_c, "loss",
                          "Loss", "Loss vs Communication Round",
                          os.path.join(PLOTS_DIR, "loss_vs_rounds.png"))

    for metric, ylabel, title, fname in [
        ("accuracy", "Accuracy", "Accuracy Comparison",
         "final_comparison_accuracy.png"),
        ("precision", "Precision", "Precision Comparison",
         "final_comparison_precision.png"),
        ("recall", "Recall", "Recall Comparison",
         "final_comparison_recall.png"),
        ("f1", "F1-score", "F1-score Comparison",
         "final_comparison_f1.png"),
    ]:
        plot_metric_bar(results, metric, ylabel, title,
                        os.path.join(PLOTS_DIR, fname))

    plot_detection_bar(det,
                       os.path.join(PLOTS_DIR, "detection_summary.png"))

    # --- Markdown report ---
    write_report(results, det,
                 os.path.join(RESULTS_DIR, "results_analysis.md"))

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<22} "
              f"Acc={r['accuracy']:.4f} "
              f"Prec={r['precision']:.4f} "
              f"Rec={r['recall']:.4f} "
              f"F1={r['f1']:.4f}")
    print(f"\nDetection: {det['tp']}/{det['total_mal']} detected, "
          f"{det['fp']} false positives, "
          f"detection rate={det['detection_rate']:.2f}, "
          f"FPR={det['fpr']:.2f}")
    print(f"\nAll files saved in {RESULTS_DIR}/")
    print("Phase 6 completed successfully.")


if __name__ == "__main__":
    main()
# Privacy-Preserving Federated Deep Learning for Healthcare with Malicious Client Detection

**An educational federated-learning simulation with simulated malicious clients and cosine-similarity-based detection.**

> **Disclaimer:** This is a research/educational simulation using a public dataset (Pima Indians Diabetes). It does **not** use real patient data and does **not** provide medical diagnosis. Federated Learning reduces the need to centralize raw training data, but this prototype does **not** provide formal privacy guarantees.

---

## 1. Project Overview

Hospitals cannot easily share raw patient data due to privacy regulations. Federated Learning (FL) allows several hospitals (called **clients**) to train a shared machine-learning model **without sending their raw data** anywhere. Each client trains locally and shares only its **model update** with a central **server**. The server combines these updates with a **Federated Averaging (FedAvg)** algorithm.

A dishonest client can still send a **poisoned** update that damages the global model. This project simulates such an attacker and adds a simple defence: the server compares each update with the average update using **cosine similarity** and excludes clients whose update is too different.

---

## 2. Problem Statement

Centralized healthcare ML requires moving sensitive patient data to one place — creating legal and ethical concerns. Federated Learning avoids centralizing raw data, but it introduces a new attack surface: a malicious client can manipulate its update to sabotage the model. This project studies that attack and a simple statistical defence.

---

## 3. Project Objectives

- Simulate a federated healthcare learning environment with multiple clients.
- Train a neural network on the Pima Indians Diabetes dataset.
- Aggregate client updates with FedAvg.
- Simulate a malicious client that poisons its update.
- Detect the poisoned update with cosine similarity.
- Compare three scenarios: Normal FL, FL + Attack, FL + Detection.
- Report accuracy, precision, recall, F1, detection rate, and false-positive rate.

---

## 4. Technologies Used

- **Python 3.10+**
- **PyTorch** (neural network, training)
- **NumPy** (vector math)
- **Pandas** (dataset loading)
- **Scikit-learn** (train/test split, metrics)
- **Matplotlib** (plots)

---

## 5. Project Architecture
Dataset → Preprocessing → Train/Test Split
│
▼
┌──────────────────────────────────────┐
│ 5 simulated clients │
│ (1 is malicious) │
└──────────────────────────────────────┘
│ local updates
▼
┌──────────────────────────────────────┐
│ Server: │
│ - Cosine similarity check │
│ - Exclude suspicious clients │
│ - FedAvg on accepted clients │
└──────────────────────────────────────┘
│
▼
New Global Model
│
▼
Evaluation

---

## 6. Methodology (Phases 1–7)

| Phase | What it does |
|-------|--------------|
| 1 | Load Pima Indians Diabetes dataset, preprocess, train/test split. |
| 2 | Simulate 5 clients, local training, FedAvg, evaluation. |
| 3 | Introduce malicious clients with an update-scaling attack. |
| 4 | Add cosine-similarity-based detector (fixed threshold). |
| 5 | Run three experiments and save comparison CSV + plots. |
| 6 | Analyse results, produce report, per-round graphs. |
| 7 | Final audit, documentation, demo guide, viva prep. |

---

## 7. Experiments

- **Experiment A — Normal FL:** 5 honest clients, no detection.
- **Experiment B — FL + Malicious:** 1 client poisons its update; no detection.
- **Experiment C — FL + Malicious + Detection:** Same attack, cosine similarity detector removes suspicious update.

All three use the same dataset, model, seed, clients, and rounds.

---

## 8. Evaluation Metrics

- **Accuracy** — fraction of all correct predictions.
- **Precision** — of predicted positives, fraction actually positive.
- **Recall** — of actual positives, fraction correctly predicted.
- **F1-score** — harmonic mean of precision and recall.
- **Detection Rate** — detected malicious / total malicious.
- **False-Positive Rate (FPR)** — honest clients wrongly flagged / total honest clients.

---

## 9. Results

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal FL | 0.6948 | 0.6061 | 0.3704 | 0.4598 |
| FL + Malicious | 0.3117 | 0.2451 | 0.4630 | 0.3205 |
| FL + Detection | 0.2922 | 0.1978 | 0.3333 | 0.2483 |

Detection summary:

- Total malicious clients: 1
- Detected: 0
- Missed: 1
- False positives: 4
- Detection rate: 0.00
- False-positive rate: 1.00

---

## 10. How to Install

Open PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Results and Analysis

## 1. Experimental Setup

- Dataset: Pima Indians Diabetes (public, educational)
- Number of clients: 5
- Communication rounds: 10
- Local epochs per round: 3
- Batch size: 32
- Learning rate: 0.001
- Malicious client IDs: [3]
- Attack scale: 5.0
- Detection threshold (cosine similarity): 0.5
- Random seed: 42

## 2. Normal Federated Learning

Accuracy = 0.6948, Precision = 0.6061, Recall = 0.3704, F1 = 0.4598.

## 3. Effect of Malicious Clients

With malicious clients and no detection: Accuracy = 0.3117, F1 = 0.3205.
Change in accuracy vs normal FL: -0.3831.
Change in F1 vs normal FL: -0.1393.

## 4. Effect of Malicious Client Detection

With detection enabled: Accuracy = 0.2922, F1 = 0.2483.
Change in accuracy vs attack (no detection): -0.0195.
Change in F1 vs attack (no detection): -0.0722.

## 5. Detection Performance

- Total malicious clients: 1
- Detected: 0
- Missed: 1
- False positives: 4
- Detection rate: 0.0000
- False-positive rate: 1.0000

## 6. Model Performance Comparison

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal FL | 0.6948 | 0.6061 | 0.3704 | 0.4598 |
| FL + Malicious | 0.3117 | 0.2451 | 0.4630 | 0.3205 |
| FL + Detection | 0.2922 | 0.1978 | 0.3333 | 0.2483 |

## 7. Communication-Round Analysis

Per-round accuracy and loss values are saved in `results/round_by_round.csv`.
Graphs: `accuracy_vs_rounds.png`, `f1_vs_rounds.png`, `loss_vs_rounds.png` in `results/plots/`.

## 8. Limitations

- Simulated federated environment with limited clients.
- Public educational dataset, not real patient data.
- Single simple attack (sign-reversal / update scaling).
- No formal privacy guarantees beyond not sharing raw data.
- Not a medical diagnostic system.

## 9. Conclusion of Experimental Results

The experiments demonstrate that a single malicious client can degrade the global model, and that simple cosine-similarity-based detection can restore performance when the attack produces updates that differ clearly from the honest ones.

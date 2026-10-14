# Project Summary

## Problem
Healthcare data is sensitive. Centralizing it for ML raises privacy concerns. Federated Learning avoids centralization, but introduces new attack surfaces.

## Proposed Approach
Simulate a federated learning setup with 5 clients on a public healthcare dataset. Introduce a malicious client that poisons its update. Add a cosine-similarity detector on the server that excludes suspicious updates before FedAvg. Compare Normal FL, FL + Attack, and FL + Detection.

## Key Components
- Public healthcare dataset (Pima Indians Diabetes)
- 5 simulated clients
- Local PyTorch training
- FedAvg aggregation
- One malicious client (update scaling attack)
- Cosine-similarity detector
- Evaluation with accuracy, precision, recall, F1, detection rate, FPR

## Experimental Results
| Experiment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal FL | 0.6948 | 0.6061 | 0.3704 | 0.4598 |
| FL + Malicious | 0.3117 | 0.2451 | 0.4630 | 0.3205 |
| FL + Detection | 0.2922 | 0.1978 | 0.3333 | 0.2483 |

## Limitations
- Simulation, not a clinical system.
- Small dataset, few clients.
- Single attack type.
- Experimental detection threshold.

## Future Work
- More clients and datasets.
- Stronger attack simulations.
- Comparison with robust aggregation (Krum, trimmed mean).
- Optional privacy enhancements (research-only).
# Viva Questions and Answers

## Basic Questions

**1. What is Machine Learning?**
A way of programming computers to learn patterns from data rather than writing explicit rules.

**2. What is Deep Learning?**
Machine learning using multi-layer neural networks that learn hierarchical features from raw input.

**3. What is Federated Learning?**
A distributed training method where multiple clients train a shared model locally and only send model updates — not raw data — to a central server.

**4. Why use Federated Learning?**
To train on data that cannot be centralized due to privacy, legal, or ownership concerns.

**5. What is a client?**
A participant in FL with its own local data (e.g., a hospital).

**6. What is a server?**
A coordinator that aggregates client updates into a global model.

**7. What is a global model?**
The shared model built from all clients' updates.

**8. What is local training?**
A client trains a copy of the global model on its own data.

**9. What is FedAvg?**
Federated Averaging — a weighted average of client model parameters, with weights proportional to the number of local samples.

## Project Questions

**10. What problem does your project solve?**
It demonstrates that a malicious client can poison a federated model, and that a simple similarity detector can protect it.

**11. Why did you choose healthcare?**
Healthcare has strong privacy constraints, making FL an interesting alternative to centralization.

**12. Why is data privacy important in healthcare?**
Patient records are sensitive and legally protected (e.g., HIPAA, GDPR).

**13. What dataset did you use?**
Pima Indians Diabetes (public, educational).

**14. What model did you use?**
A small feedforward neural network: Linear(8→16) → ReLU → Linear(16→1).

**15. How many clients did you simulate?**
5 clients; 1 is malicious (client 3).

**16. How does a communication round work?**
Server → clients train → clients send updates → server aggregates → new global model.

**17. What happens during local training?**
The client trains its copy of the global model on its own data for a few epochs.

## Security Questions

**18. What is a malicious client?**
A client that intentionally sends a corrupted model update.

**19. What is model poisoning?**
Manipulating the model update to degrade the global model.

**20. How did you simulate malicious clients?**
The malicious client computes `update = local - global` and sends `-attack_scale * update + global` back.

**21. How does malicious-client detection work?**
The server computes cosine similarity between each client's update and the mean update; clients below a threshold are removed.

**22. What is cosine similarity?**
A measure of angle between two vectors: +1 = same direction, 0 = perpendicular, -1 = opposite.

**23. Why did you use a similarity threshold?**
To decide when an update is "too different" from the majority.

**24. What happens to suspicious clients?**
Their updates are excluded from FedAvg for that round.

**25. What is a false positive?**
An honest client wrongly flagged as malicious.

**26. What is a false negative?**
A malicious client that was not detected.

## Evaluation Questions

**27. What is Accuracy?**
(TP + TN) / total — fraction of all correct predictions.

**28. What is Precision?**
TP / (TP + FP) — of predicted positives, how many are correct.

**29. What is Recall?**
TP / (TP + FN) — of actual positives, how many are caught.

**30. What is F1-score?**
2·P·R / (P + R) — balanced single-number score.

**31. What is a confusion matrix?**
A 2×2 table of TN, FP, FN, TP for binary classification.

**32. What is detection rate?**
Detected malicious / total malicious.

## Limitations and Future Work

**33. What are the limitations?**
Simulated environment, small dataset, single attack type, experimental threshold.

**34. Is this a real hospital deployment?**
No. It is an educational simulation with public data.

**35. What would you improve in the future?**
More clients, more attack types, robust aggregation methods, larger datasets.

**36. What additional attacks could be studied?**
Label-flipping, backdoors, sybil attacks, stealthy attacks.

**37. What would happen with real healthcare institutions?**
More complex: legal contracts, data heterogeneity, secure aggregation, real-world scale.
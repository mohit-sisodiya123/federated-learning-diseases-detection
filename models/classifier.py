# models/classifier.py
"""
A simple feedforward neural network for binary classification.
Architecture:
    Input (8 features) -> Linear(8,16) -> ReLU -> Linear(16,1)
"""

import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # first linear layer
        self.relu = nn.ReLU()                         # activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # output layer (logits)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)   # no sigmoid here because we use BCEWithLogitsLoss
        return x
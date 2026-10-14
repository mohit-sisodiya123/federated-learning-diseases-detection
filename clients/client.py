# clients/client.py
"""
Client class for Federated Learning.
Each client holds a local dataset and can train a model on it.
It returns model parameters and the number of samples after training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.classifier import SimpleClassifier

class Client:
    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 3
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_samples = len(X_train)
        
        # Create DataLoader for local training
        self.dataset = TensorDataset(X_train, y_train)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, global_model: nn.Module) -> tuple[dict, int]:
        """
        Train a local copy of the global model on the client's data.
        
        Args:
            global_model: the current global model (parameters will be copied)
        
        Returns:
            model_state_dict: the trained model parameters
            num_samples: number of local training samples
        """
        # Create a local copy of the model
        local_model = SimpleClassifier()
        local_model.load_state_dict(global_model.state_dict())
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.learning_rate)
        
        # Local training loop
        local_model.train()
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.dataloader:
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Return the updated model parameters and number of samples
        return local_model.state_dict(), self.num_samples
    # clients/client.py
"""
Client class for Federated Learning.
Each client holds a local dataset and can train a model on it.
Optionally, a client can be malicious and poison its model update.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.classifier import SimpleClassifier

class Client:
    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 3,
        is_malicious: bool = False,
        attack_scale: float = 1.0
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.is_malicious = is_malicious
        self.attack_scale = attack_scale
        self.num_samples = len(X_train)
        
        # Create DataLoader for local training
        self.dataset = TensorDataset(X_train, y_train)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, global_model: nn.Module) -> tuple[dict, int]:
        """
        Train a local copy of the global model on the client's data.
        If the client is malicious, its model update is poisoned before returning.
        
        Args:
            global_model: the current global model (parameters will be copied)
        
        Returns:
            model_state_dict: the (possibly poisoned) trained model parameters
            num_samples: number of local training samples
        """
        # Create a local copy of the model
        local_model = SimpleClassifier()
        local_model.load_state_dict(global_model.state_dict())
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.learning_rate)
        
        # Local training loop
        local_model.train()
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.dataloader:
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Return the trained model parameters, but if malicious, poison the update
        if self.is_malicious:
            # Calculate the update (difference between local and global)
            global_state = global_model.state_dict()
            local_state = local_model.state_dict()
            # Compute poisoned update: -attack_scale * (local - global)
            poisoned_state = {}
            for key in global_state.keys():
                update = local_state[key] - global_state[key]
                poisoned_update = -self.attack_scale * update
                # Reconstruct the final parameters as global + poisoned_update
                poisoned_state[key] = global_state[key] + poisoned_update
            return poisoned_state, self.num_samples
        else:
            return local_model.state_dict(), self.num_samples
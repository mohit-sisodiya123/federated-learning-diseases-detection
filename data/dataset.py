# data/dataset.py
"""
Dataset loading and preprocessing for the Pima Indians Diabetes dataset.
This file downloads the data if not present, cleans missing values,
and returns train/test tensors ready for PyTorch.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.config import (
    DATA_DIR, DATASET_FILENAME, DATASET_URL, COLUMN_NAMES, SEED
)
from utils.seed import set_seed

def download_dataset():
    """Download the dataset if it does not exist in the data folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, DATASET_FILENAME)
    if not os.path.exists(filepath):
        print(f"Downloading dataset to {filepath}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, filepath)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset. Please manually download it from {DATASET_URL} "
                f"and place it as '{filepath}'."
            ) from e
    else:
        print(f"Dataset found at {filepath}")
    return filepath

def load_and_preprocess():
    """
    Load the dataset, clean missing values, split into train/test,
    standardize features, and convert to PyTorch tensors.
    
    Returns:
        X_train, X_test (torch.Tensor): feature tensors
        y_train, y_test (torch.Tensor): label tensors (float for BCEWithLogitsLoss)
    """
    set_seed(SEED)
    filepath = download_dataset()
    
    # Read CSV with column names
    df = pd.read_csv(filepath, names=COLUMN_NAMES)
    
    # Handle invalid zeros: in Pima dataset, some columns cannot be zero.
    # We replace zeros with NaN then impute with median.
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_columns:
        df[col] = df[col].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    
    # Separate features and target
    X = df.drop("Outcome", axis=1).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)  # 0 or 1
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Standardize features (fit only on training data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (n,1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
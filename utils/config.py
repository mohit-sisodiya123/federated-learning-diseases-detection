# utils/config.py
"""
Central configuration for the federated learning simulation.
All important settings are here so you can change them easily.
"""

# Random seed for reproducibility
SEED = 42

# Dataset settings
DATA_DIR = "data"                     # folder where dataset is stored
DATASET_FILENAME = "pima-indians-diabetes.data.csv"
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Training hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model architecture
INPUT_DIM = 8   # number of features
HIDDEN_DIM = 16 # hidden layer size
OUTPUT_DIM = 1  # binary classification

# Results folder
RESULTS_DIR = "results"
# utils/config.py
"""
Central configuration for the federated learning simulation.
All important settings are here so you can change them easily.
"""

# Random seed for reproducibility
SEED = 42

# Dataset settings
DATA_DIR = "data"                     # folder where dataset is stored
DATASET_FILENAME = "pima-indians-diabetes.data.csv"
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Training hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model architecture
INPUT_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 1

# Phase 2: Malicious client settings
# List of client IDs that will be malicious (0-indexed)
MALICIOUS_CLIENT_IDS = [3]   # Client 3 will be malicious
# Attack scale: how much to amplify and reverse the update
ATTACK_SCALE = 5.0

# Results folder
RESULTS_DIR = "results"
# Phase 3: Malicious-client detection sensitivity
DETECTION_SENSITIVITY = 2.5
# Phase 4: Simple detection settings
DETECTION_ENABLED = True
SIMILARITY_THRESHOLD = 0.5
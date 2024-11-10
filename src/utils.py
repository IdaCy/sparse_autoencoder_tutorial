# src/utils.py
import os
import torch

def load_data(filepath="data/sample_activations.pt"):
    # Define the path relative to the root of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_filepath = os.path.join(project_root, filepath)

    print(f"Attempting to load data from: {full_filepath}")  # Debugging line

    if not os.path.exists(full_filepath):
        raise FileNotFoundError(f"Data file not found at path: {full_filepath}")

    # Load and return the data
    return torch.load(full_filepath)

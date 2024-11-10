# src/train.py

import torch
from torch.optim import Adam
from src.autoencoder import SparseAutoencoder, sparse_loss
from src.utils import load_data

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


def train_autoencoder(data, input_dim=100, hidden_dim=20, epochs=50, l1_lambda=1e-3):
    model = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = Adam(model.parameters(), lr=0.001)
    data = data.to(torch.float32)

    for epoch in range(epochs):
        model.train()
        encoded, reconstructed = model(data)
        loss = sparse_loss(reconstructed, data, encoded, l1_lambda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "sparse_autoencoder.pth")
    print("Model saved as sparse_autoencoder.pth")

if __name__ == "__main__":
    data = load_data()
    train_autoencoder(data)

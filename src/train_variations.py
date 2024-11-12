import torch
from torch.optim import Adam
from src.autoencoder_variations import DeepSparseAutoencoder, ConvSparseAutoencoder
from src.autoencoder_basic import sparse_loss
from src.utils import load_data

def train_autoencoder(model_type="deep", epochs=50, l1_lambda=1e-3):
    data = load_data()
    
    if model_type == "deep":
        model = DeepSparseAutoencoder(input_dim=100, hidden_dims=[50, 20])
    elif model_type == "conv":
        data = data.view(-1, 1, 10, 10)  # Reshape for ConvSparseAutoencoder
        model = ConvSparseAutoencoder(input_channels=1, hidden_channels=16)
    else:
        raise ValueError("Invalid model_type. Choose 'deep' or 'conv'.")

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

    torch.save(model.state_dict(), f"{model_type}_sparse_autoencoder.pth")
    print(f"Model saved as {model_type}_sparse_autoencoder.pth")

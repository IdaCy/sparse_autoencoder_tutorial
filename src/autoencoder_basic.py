import torch
import torch.nn as nn

class BasicSparseAutoencoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=20):
        super(BasicSparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def sparse_loss(reconstructed, original, encoded, l1_lambda=1e-3):
    mse_loss = nn.MSELoss()(reconstructed, original)
    l1_penalty = l1_lambda * torch.mean(torch.abs(encoded))
    return mse_loss + l1_penalty

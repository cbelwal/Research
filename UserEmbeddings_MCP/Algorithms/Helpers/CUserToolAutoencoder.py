"""
Shared autoencoder used to compress user-tool interaction vectors.
"""
import torch.nn as nn


class CUserToolAutoencoder(nn.Module):
    def __init__(self, numberOfTools: int, embeddingDimensions: int):
        super().__init__()
        hiddenDimensions = min(
            256,
            max(embeddingDimensions * 2, min(numberOfTools, 64)),
        )

        self.encoder = nn.Sequential(
            nn.Linear(numberOfTools, hiddenDimensions),
            nn.ReLU(),
            nn.Linear(hiddenDimensions, embeddingDimensions),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embeddingDimensions, hiddenDimensions),
            nn.ReLU(),
            nn.Linear(hiddenDimensions, numberOfTools),
            nn.Sigmoid(),
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return reconstructed, embeddings

"""
Generate user embeddings with a shared autoencoder.

The encoder consumes each complete user-tool vector and produces the user
embedding. The decoder reconstructs the original vector, causing the shared
embedding space to preserve tool-usage patterns across all users.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

topRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)

from Algorithms.Helpers.CUserToolAutoencoder import CUserToolAutoencoder
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix

MAX_EPOCHS = 5000
LEARNING_RATE = 2e-2
BATCH_SIZE = 256
MIN_TARGET_LOSS = 5e-5
ACTIVE_TOOL_WEIGHT = 10.0
ACTIVE_TOOL_THRESHOLD = 1e-4
RANDOM_SEED = 1


def _weighted_reconstruction_loss(reconstructed, target, reduction: str = "mean"):
    weights = torch.where(
        target > ACTIVE_TOOL_THRESHOLD,
        ACTIVE_TOOL_WEIGHT,
        1.0,
    )
    losses = weights * (reconstructed - target).pow(2)
    if reduction == "none":
        return losses.mean(dim=1)
    return losses.mean()


def Alg_2_AutoEncoder(
    embeddingDimensions: int = 8,
    testData: IUserToolMatrix = None,
):
    if testData is None:
        raise ValueError("testData is required")
    if embeddingDimensions <= 0:
        raise ValueError("embeddingDimensions must be greater than zero")

    MAT_u_tau = testData.get_MAT_u_tau().float()
    if MAT_u_tau.ndim != 2:
        raise ValueError("The user-tool matrix must be two-dimensional")
    if MAT_u_tau.shape != (testData.NumberOfUsers, testData.NumberOfTools):
        raise ValueError(
            "The user-tool matrix shape does not match NumberOfUsers and NumberOfTools"
        )
    if testData.NumberOfUsers <= 0 or testData.NumberOfTools <= 0:
        raise ValueError("The user-tool matrix must contain users and tools")

    torch.manual_seed(RANDOM_SEED)
    model = CUserToolAutoencoder(
        numberOfTools=testData.NumberOfTools,
        embeddingDimensions=embeddingDimensions,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    trainingLoader = DataLoader(
        TensorDataset(MAT_u_tau),
        batch_size=min(BATCH_SIZE, testData.NumberOfUsers),
        shuffle=True,
        generator=generator,
    )

    print("Starting Shared Autoencoder Training")
    model.train()
    for _ in tqdm(range(MAX_EPOCHS)):
        epochLoss = 0.0
        for (batch,) in trainingLoader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = _weighted_reconstruction_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item() * batch.shape[0]

        epochLoss /= testData.NumberOfUsers
        if epochLoss < MIN_TARGET_LOSS:
            break

    MAT_E = torch.zeros(testData.NumberOfUsers, embeddingDimensions)
    loss_for_each_user = torch.zeros(testData.NumberOfUsers)
    inferenceLoader = DataLoader(
        TensorDataset(MAT_u_tau),
        batch_size=min(BATCH_SIZE, testData.NumberOfUsers),
        shuffle=False,
    )

    model.eval()
    offset = 0
    with torch.no_grad():
        for (batch,) in inferenceLoader:
            reconstructed, embeddings = model(batch)
            batchSize = batch.shape[0]
            MAT_E[offset:offset + batchSize] = embeddings
            loss_for_each_user[offset:offset + batchSize] = (
                _weighted_reconstruction_loss(
                    reconstructed,
                    batch,
                    reduction="none",
                )
            )
            offset += batchSize

    return MAT_E, loss_for_each_user

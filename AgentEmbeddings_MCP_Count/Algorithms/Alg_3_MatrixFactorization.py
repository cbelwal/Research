"""
Generate agent embeddings with truncated singular value decomposition.

The agent-tool matrix is factorized into agent and tool latent factors. The agent
factors are returned as embeddings in one shared coordinate system.
"""
import os
import sys

import torch
from sklearn.decomposition import TruncatedSVD

topRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)

from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix

RANDOM_SEED = 1
SVD_ITERATIONS = 7


def Alg_3_MatrixFactorization(
    embeddingDimensions: int = 8,
    testData: IAgentToolMatrix = None,
):
    if testData is None:
        raise ValueError("testData is required")
    if embeddingDimensions <= 0:
        raise ValueError("embeddingDimensions must be greater than zero")

    agentToolMatrix = testData.get_MAT_a_tau().float()
    if agentToolMatrix.ndim != 2:
        raise ValueError("The agent-tool matrix must be two-dimensional")
    if agentToolMatrix.shape != (testData.NumberOfAgents, testData.NumberOfTools):
        raise ValueError(
            "The agent-tool matrix shape does not match NumberOfAgents and NumberOfTools"
        )
    if not torch.isfinite(agentToolMatrix).all():
        raise ValueError("The agent-tool matrix must contain only finite values")

    maximumDimensions = min(testData.NumberOfAgents, testData.NumberOfTools)
    if embeddingDimensions > maximumDimensions:
        raise ValueError(
            f"embeddingDimensions cannot exceed the matrix rank bound "
            f"({maximumDimensions})"
        )

    factorizer = TruncatedSVD(
        n_components=embeddingDimensions,
        n_iter=SVD_ITERATIONS,
        random_state=RANDOM_SEED,
    )
    matrixArray = agentToolMatrix.numpy()
    embeddingsArray = factorizer.fit_transform(matrixArray)
    reconstructedArray = factorizer.inverse_transform(embeddingsArray)

    MAT_E = torch.from_numpy(embeddingsArray).float()
    reconstructed = torch.from_numpy(reconstructedArray).float()
    loss_for_each_agent = (reconstructed - agentToolMatrix).pow(2).mean(dim=1)

    return MAT_E, loss_for_each_agent
